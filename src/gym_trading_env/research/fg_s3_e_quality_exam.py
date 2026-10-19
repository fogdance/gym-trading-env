"""Unified exam commands over native source, ActorPolicy and ExamSession."""
from __future__ import annotations

import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys

from .fg_s3_e_quality_exam_contract import validate_plan,plan_hash,identity,contract_window
from .fg_s3_e_quality_course_preflight import _local_path
from .fg_replay import read_manifest,write_json,signed,digest
from .fg_s3_e_quality_foundation import PROJECT,execution_identity


def read_plan(path):
    from dreamerv3.configuration import read_config_yaml
    raw=read_config_yaml(_local_path(path))
    def resolve(value):
        if not isinstance(value,str) or not value.strip():raise ValueError('EXAM_EXPLICIT_NONEMPTY_PATH_REQUIRED')
        p=Path(value).expanduser()
        return str(_local_path(p if p.is_absolute() else PROJECT/p))
    raw['output_dir']=resolve(raw.get('output_dir'))
    source=raw.get('source',{})
    for key in ('manifest','env_receipt'):
        if source.get(key) is not None:source[key]=resolve(source[key])
    for actor in raw.get('actors',{}).values():
        for key in ('run','checkpoint'):actor[key]=resolve(actor.get(key))
    return validate_plan(raw)


def _output(plan):
    from e_data_v1.io import independent_output
    out=_local_path(plan['output_dir'])
    inputs=[a['run'] for a in plan['actors'].values()]
    inputs += [x for k,x in plan['source'].items() if k!='kind' and x is not None]
    independent_output(out,inputs=inputs,allow_existing=True)
    return out


def precheck(plan):
    from .fg_s3_e_quality_exam_source import validation_members
    from embodied.run.fg_s3_e_quality_exam import inspect_exam_actor
    p=validate_plan(plan);_output(p);models={};blockers=[]
    if p['exam_mode']=='DEV_EXAM':
        rows,_,_,_=validation_members(p)
        source=dict(status='VALIDATION_METADATA_VERIFIED',contracts=[r['contract'] for r in rows],
            native_profile='VERIFIED_DURING_EXAM_SOURCE_PREPARATION')
        if not any(name in p['actors'] for name in p['baselines']):blockers.append('FORMAL_TRAINED_ACTOR_NOT_PROVIDED')
    else:source=dict(status='ENGINEERING_SOURCE_ONLY',kind=p['source']['kind'])
    for name,a in p['actors'].items():
        proof,_,_=inspect_exam_actor(a['kind'],a['run'],a['checkpoint'],engineering=p['exam_mode']=='SYNTHETIC_DEV')
        models[name]=proof
    return signed(dict(protocol='FG_S3_E_QUALITY_EXAM_PRECHECK_V1',plan_hash=plan_hash(p),
        status='EXAM_PLAN_READY' if not blockers else 'EXAM_PLAN_BLOCKED',source=source,models=models,
        blockers=blockers,training_admitted=False,research_admitted=False,sealed_test_ready=False))


def create_plan(config):
    p=read_plan(config);out=_output(p)
    if out.exists():raise ValueError('EXAM_PLAN_OUTPUT_MUST_BE_NEW')
    receipt=precheck(p)
    if receipt['blockers']:raise ValueError('EXAM_PRECHECK_BLOCKED: '+','.join(receipt['blockers']))
    out.mkdir(parents=True);write_json(out/'PLAN.json',p);write_json(out/'PRECHECK.json',receipt)
    return receipt


def source_for(plan,contract,*,existing):
    from .fg_s3_e_quality_exam_source import prepare_validation,prepare_finite_train
    folder=Path(plan['output_dir'])/'sources'/contract
    window=contract_window(plan,contract)
    kind=plan['source']['kind']
    if kind=='VALIDATION':return prepare_validation(plan,contract,folder,existing=existing)
    if kind=='FINITE_TRAIN':return prepare_finite_train(plan,contract,folder,existing=existing)
    if kind!='BUILTIN_SYNTHETIC':raise ValueError('EXAM_UNKNOWN_SOURCE_KIND')
    if folder.exists()!=existing:raise ValueError('EXAM_BUILTIN_SOURCE_NEW_OUTPUT_OR_EXPLICIT_REOPEN_REQUIRED')
    from .fg_s3_e_quality_exam_synthetic import native_source
    if (contract not in ('CZCE.FG2701','CZCE.FG2705') or window['start_day']<'2026-01-08'
            or window['end_day']>'2026-01-14'):
        raise ValueError('EXAM_BUILTIN_FIXTURE_CONTRACT_OR_DATES_UNSUPPORTED')
    if existing:
        for name in ('source.json','env.yaml','contract.yaml','directions.yaml','market.csv'):_local_path(folder/name)
        saved=json.loads((folder/'source.json').read_text());receipt=saved['receipt']
        profile=saved['profile'];expected=str(folder/'market.csv')
        if (receipt.get('scope')!='BUILTIN_SYNTHETIC_ONLY' or receipt.get('role')!='synthetic'
                or receipt.get('contract')!=contract or profile['sources'][contract]['market']['path']!=expected
                or set(receipt['frozen_inputs'])!={str(folder/n) for n in ('env.yaml','contract.yaml','directions.yaml','market.csv')}):
            raise ValueError('EXAM_BUILTIN_REOPEN_CANNOT_SUBSTITUTE_EXTERNAL_MARKET')
        import yaml
        raw=yaml.safe_load((folder/'env.yaml').read_text())
        if raw['trading']['data_path']!=expected:raise ValueError('EXAM_BUILTIN_REOPEN_MARKET_PATH_CHANGED')
    result=native_source(folder,contract=contract,**window)
    result.verify()
    return result


def _job_workers(plan,jobs):
    # One known-idle 31-GiB host. Estimate from immutable model/source sizes;
    # no repeated free/nvidia-smi queries or arbitrary fixed 1/2/4 workers.
    largest=768<<20
    for a in plan['actors'].values():
        largest=max(largest,(768<<20)+4*(Path(a['checkpoint'])/'agent.pkl').stat().st_size)
    return max(1,min(len(jobs),os.cpu_count() or 1,(28<<30)//largest))


def execute_job(plan,job,*,resume=False,stop_at=None):
    from .fg_s3_e_quality_exam_compare import run_case,policy_jobs
    plan=validate_plan(plan)
    if job not in policy_jobs(plan):raise ValueError('EXAM_JOB_NOT_PREREGISTERED_IN_PLAN')
    source=source_for(plan,job['contract'],existing=True)
    return run_case(plan,source,Path(plan['output_dir'])/job['contract']/job['label'],job,
        resume=resume,stop_at=stop_at)


def run(config=None,*,directory=None,resume=False,stop_at=None):
    from .fg_s3_e_quality_exam_compare import policy_jobs
    p=read_plan(config) if config is not None else validate_plan(json.loads((_local_path(directory)/'PLAN.json').read_text()))
    root=_output(p)
    if directory is not None and _local_path(directory)!=root:raise ValueError('EXAM_RESUME_PLAN_OUTPUT_MISMATCH')
    check=precheck(p)
    if check['blockers']:raise ValueError('EXAM_PRECHECK_BLOCKED: '+','.join(check['blockers']))
    if stop_at is not None and (type(stop_at) is not int or stop_at<1):raise ValueError('EXAM_POSITIVE_STOP_MINUTES_REQUIRED')
    if resume:
        if not root.exists() or not (root/'SOURCE_CODE_IDENTITY.json').exists():raise ValueError('EXAM_RESUME_EXISTING_SESSION_REQUIRED')
        if read_manifest(root/'SOURCE_CODE_IDENTITY.json')['code']!=execution_identity():raise ValueError('EXAM_RESUME_CODE_CHANGED')
        if (root/'EXAM_RECEIPT.json').exists() and read_manifest(root/'EXAM_RECEIPT.json')['status']=='EXAM_COMPLETE':
            raise ValueError('EXAM_ALREADY_COMPLETE_USE_READ_ONLY_VERIFY')
    else:
        if root.exists():
            if set(x.name for x in root.iterdir())!={'PLAN.json','PRECHECK.json'} or json.loads((root/'PLAN.json').read_text())!=p:
                raise ValueError('EXAM_RUN_OUTPUT_MUST_BE_NEW_OR_UNUSED_PLAN')
        else:
            root.mkdir(parents=True);write_json(root/'PLAN.json',p);write_json(root/'PRECHECK.json',check)
        write_json(root/'SOURCE_CODE_IDENTITY.json',signed(dict(code=execution_identity(),plan_hash=plan_hash(p))))
        for contract in p['contracts']:source_for(p,contract,existing=False)
    jobs=policy_jobs(p);logs=root/'logs';logs.mkdir(exist_ok=True)
    def work(job):
        target=root/job['contract']/job['label']
        restore=resume and target.exists()
        stem=job['contract']+'__'+job['label']+('__resume' if restore else '__start')
        index=0;log=logs/(stem+'.log')
        while log.exists():index+=1;log=logs/(stem+f'_{index}.log')
        argv=[sys.executable,'-B','-m','gym_trading_env.research.fg_s3_e_quality_exam_worker',
            str(root/'PLAN.json'),json.dumps(job,separators=(',',':')),'resume' if restore else 'start',
            str(stop_at) if stop_at is not None else 'end']
        env={**os.environ,'PYTHONPATH':os.pathsep.join((str(PROJECT),str(PROJECT/'src'),str(PROJECT.parent/'dreamerv3'))),
            'JAX_PLATFORMS':'cpu','CUDA_VISIBLE_DEVICES':'','OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'}
        with log.open('x') as stream:r=subprocess.run(argv,env=env,stdout=stream,stderr=subprocess.STDOUT)
        receipt=dict(job=job,argv=argv,cwd=str(PROJECT),exit_code=r.returncode,log=str(log))
        write_json(log.with_suffix('.command.json'),receipt)
        if r.returncode:raise ValueError('EXAM_CASE_FAILED: '+str(log))
        return receipt
    workers=_job_workers(p,jobs)
    with concurrent.futures.ThreadPoolExecutor(workers) as pool:commands=list(pool.map(work,jobs))
    completed=all((root/j['contract']/j['label']/'SESSION_RECEIPT.json').exists() for j in jobs)
    result=signed(dict(protocol='FG_S3_E_QUALITY_EXAM_RUN_V1',status='EXAM_COMPLETE' if completed else 'EXAM_PAUSED',
        plan_hash=plan_hash(p),exam_mode=p['exam_mode'],jobs=commands,workers=workers,
        training_updates=0,teacher_used=False,test_prices_accessed=False))
    write_json(root/'EXAM_RECEIPT.json',result)
    if completed:
        _postprocess(root,root/'comparison')
        _report(root,root)
    return result


def _cases(root):
    from .fg_s3_e_quality_exam_compare import policy_jobs
    p=validate_plan(json.loads((root/'PLAN.json').read_text()))
    if _local_path(p['output_dir'])!=root:raise ValueError('EXAM_PLAN_LOCATION_MISMATCH')
    sources={c:source_for(p,c,existing=True) for c in p['contracts']};cases=[]
    for job in policy_jobs(p):
        directory=root/job['contract']/job['label'];receipt=read_manifest(directory/'SESSION_RECEIPT.json')
        if (receipt['contract']!=job['contract'] or receipt['policy_id']!=job['label']
                or receipt['plan_hash']!=plan_hash(p) or receipt['exam_mode']!=p['exam_mode']):
            raise ValueError('EXAM_CASE_PLAN_OR_ROLE_MISMATCH')
        cases.append((directory,sources[job['contract']]))
    return p,cases


def _postprocess(root,output):
    from .fg_s3_e_quality_exam_compare import compare_sessions
    from .fg_s3_e_quality_exam_evidence import assess
    from .fg_s3_e_quality_exam_verify import verify_policy
    from .fg_s3_e_quality_exam_compare import policy_jobs
    p,cases=_cases(root);_post_output(p,output)
    jobs=policy_jobs(p)
    checks=[verify_policy(p,job,directory,Path(output).parent/'verification_models')
        for job,(directory,_) in zip(jobs,cases)]
    report=compare_sessions(cases,output)
    write_json(Path(output)/'POLICY_CHECKS.json',signed(dict(checks=checks,training_updates=0)))
    evidence=assess(p,report);write_json(Path(output)/'EVIDENCE.json',evidence)
    return dict(comparison=report,evidence=evidence)


def verify(directory,output):
    root=_local_path(directory);out=_local_path(output)
    # Reconciliation and metrics are freshly recomputed in a new directory.
    result=_postprocess(root,out)
    return signed(dict(status='EXAM_NATIVE_VERIFICATION_PASS',plan_hash=plan_hash(json.loads((root/'PLAN.json').read_text())),
        account_count=result['comparison']['account_count'],independent_contract_days=result['comparison']['independent_contract_days'],
        evidence=result['evidence'],training_updates=0,test_prices_accessed=False))


def compare(directory,output):return verify(directory,output)


def _report(root,output):
    p=validate_plan(json.loads((root/'PLAN.json').read_text()));run=read_manifest(root/'EXAM_RECEIPT.json')
    comparison=read_manifest(root/'comparison/COMPARISONS.json');evidence=read_manifest(root/'comparison/EVIDENCE.json')
    out=Path(output)
    if out!=root:
        _post_output(p,out)
        if out.exists():raise ValueError('EXAM_REPORT_OUTPUT_MUST_BE_NEW')
        out.mkdir(parents=True)
    path=out/'EXAM_REPORT.md'
    if path.exists():raise ValueError('EXAM_REPORT_OUTPUT_MUST_BE_NEW')
    lines=['# Actor-only Exam report','',f"Mode: {p['exam_mode']}",f"Plan: `{plan_hash(p)}`",'',
        'Teacher / exploration / training updates / TEST: 0.',
        f"Independent contract-days: {comparison['independent_contract_days']}; accounts: {comparison['account_count']}.",'',
        '| Policy | Entries | Net PnL | TP<=5 | SL30 | Max drawdown |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    lines += [f"| {s['policy']} / {s['contract']} | {s['entries']} | {s['net_pnl']} | {s['tp_le_5']} | {s['sl30']} | {s['max_drawdown']} |" for s in comparison['summaries']]
    lines += ['',f"Real DEV strategy exam: {evidence['real_dev_strategy_exam']}.",
        'Synthetic engineering is not policy effectiveness. All-WAIT has no entry skill evidence; zero-trade TP rates are N/A.',
        'Quality remains INCONCLUSIVE without preregistered joint rules and independent market support.',
        'Source, full native traces, account reconciliation, MODEL_READONLY_RECEIPT and committed eval states remain in the exam directory.']
    path.write_text('\n'.join(lines)+'\n')
    return signed(dict(status='EXAM_REPORT_WRITTEN',report=str(path),sha256=digest(path),run=run['content_hash'],evidence=evidence))


def report(directory,output=None):
    root=_local_path(directory)
    return _report(root,_local_path(output) if output is not None else root/'report')


def _post_output(plan,output):
    from e_data_v1.io import independent_output
    out=_local_path(output)
    protected=[Path(a['run']) for a in plan['actors'].values()]
    protected += [Path(v) for k,v in plan['source'].items() if k!='kind' and v is not None]
    protected.append(Path(plan['output_dir'])/'sources')
    for source in protected:
        if out.is_relative_to(source) or source.is_relative_to(out):raise ValueError('EXAM_REPORT_SOURCE_OUTPUT_OVERLAP')
    independent_output(out,inputs=protected)

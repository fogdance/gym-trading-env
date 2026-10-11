#!/usr/bin/env python3
"""Prepare, user-start, resume and evaluate the single C1_M1_1M experiment."""
import argparse
import concurrent.futures
import fcntl
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))


def baseline_job(job):
    os.environ['CUDA_VISIBLE_DEVICES']=''
    os.environ['JAX_PLATFORMS']='cpu'
    os.environ['JAX_PLATFORM_NAME']='cpu'
    os.environ['OPENBLAS_NUM_THREADS']='1'
    os.environ['OMP_NUM_THREADS']='1'
    from gym_trading_env.research.fg_single3_evaluation import evaluate_strategy
    return evaluate_strategy(*job)


def baselines(prepared):
    from gym_trading_env.research.fg_single3_experiment import require_prepared
    require_prepared(prepared)
    jobs=[(str(prepared),name,str(Path(prepared)/'baselines'/name))
        for name in ('always_wait','always_add','random_add')]
    with concurrent.futures.ProcessPoolExecutor(max_workers=min(3,os.cpu_count()),
            mp_context=multiprocessing.get_context('spawn')) as pool:
        return list(pool.map(baseline_job,jobs))


def hardware_preflight(config):
    import psutil
    from gym_trading_env.research.fg_single3_experiment import DEFAULT_LOGDIR
    if Path(config.logdir).resolve()!=DEFAULT_LOGDIR:
        raise ValueError('Formal experiment must use the new allowed log directory')
    parent=DEFAULT_LOGDIR.parent
    while not parent.exists():
        parent=parent.parent
    if parent.stat().st_dev!=Path('/data').stat().st_dev:
        raise ValueError('Formal log directory is not on the allowed /data disk')
    available=psutil.virtual_memory().available
    if available<16*1024**3:
        raise MemoryError('Less than 16 GiB physical RAM available for the complete audited Replay and JAX')
    disk=shutil.disk_usage(parent).free
    if disk<60*1024**3:
        raise OSError('Less than 60 GiB free for retained full checkpoints and 20 GiB reserve')
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,memory.free,utilization.gpu',
        '--format=csv,noheader,nounits'],text=True).strip()
    return dict(physical_ram_available=available,disk_available=disk,gpu=gpu,time=time.time())


def start(prepared, resume=False):
    from gym_trading_env.research.fg_single3_experiment import require_prepared,DREAMER_ROOT,GYM_ROOT
    prepared=Path(prepared).resolve()
    receipt=require_prepared(prepared)
    if str(DREAMER_ROOT) not in sys.path:
        sys.path.insert(0,str(DREAMER_ROOT))
    import elements
    from dreamerv3.fg_single3_offline import validate
    from gym_trading_env.research.fg_replay import write_json,signed,digest
    from dreamerv3.configuration import read_config_yaml
    config=elements.Config(read_config_yaml(prepared/'config.yaml'))
    resources=hardware_preflight(config)
    logdir=Path(config.logdir)
    if resume:
        if not (logdir/'single3_identity.json').is_file() or not (logdir/'ckpt').is_dir():
            raise ValueError('Resume requires a matching complete experiment checkpoint')
    elif logdir.exists():
        raise FileExistsError('Experiment directory already exists; use resume explicitly')
    logdir.parent.mkdir(parents=True,exist_ok=True)
    # The lock is outside the result directory, so it cannot masquerade as a
    # partially initialized run during strict admission.
    with (logdir.parent/'.C1_M1_1M.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        validate(config,persist=True)
        attempt=uuid.uuid4().hex
        write_json(logdir/('resources_'+attempt+'.json'),signed(resources))
        # Training follows the official CLI and all its admission checks.
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',JAX_PLATFORMS='cuda',JAX_PLATFORM_NAME='gpu',
            PYTHONPATH=str(DREAMER_ROOT)+':'+str(GYM_ROOT/'src'),
            MPLCONFIGDIR=str(logdir/'matplotlib'))
        if not env.get('CUDA_VISIBLE_DEVICES'):
            env['CUDA_VISIBLE_DEVICES']='0'
        with (logdir/('stdout_'+attempt+'.log')).open('w') as log:
            process=subprocess.Popen(receipt['argv'],cwd=DREAMER_ROOT,env=env,
                stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
            for line in process.stdout:
                log.write(line);log.flush()
                print(line,end='',flush=True)
            code=process.wait()
        if code:
            write_json(logdir/('launch_failure_'+attempt+'.json'),signed(dict(status='FAILED',
                exit_code=code,stdout_sha256=digest(logdir/('stdout_'+attempt+'.log')))))
            raise subprocess.CalledProcessError(code,receipt['argv'])
    # Evaluation is independent; retry it without repeating a successful run.
    evaluate(prepared)


def evaluate(prepared):
    from gym_trading_env.research.fg_single3_experiment import require_prepared
    from gym_trading_env.research.fg_single3_evaluation import load_evaluation_agent,evaluate_strategy,require_training
    receipt=require_prepared(prepared)
    require_training(prepared)
    baselines(prepared)
    agent,training=load_evaluation_agent(prepared)
    evaluate_strategy(prepared,'agent',Path(receipt['logdir'])/'dev/agent',agent=agent)
    report(prepared)


def report(prepared):
    from gym_trading_env.research.fg_single3_experiment import DEFAULT_LOGDIR
    from gym_trading_env.research.fg_single3_evaluation import invalidate_report
    try:
        return _report(prepared)
    except BaseException:
        invalidate_report(DEFAULT_LOGDIR)
        raise


def _report(prepared):
    from gym_trading_env.research.fg_single3_experiment import require_prepared,MILESTONES,DEFAULT_LOGDIR
    from gym_trading_env.research.fg_single3_evaluation import comparison,require_dev,publish_report,require_training
    from gym_trading_env.research.fg_replay import read_manifest,check_files,digest
    prepared=Path(prepared)
    receipt=require_prepared(prepared)
    logdir=Path(receipt['logdir'])
    if logdir.resolve()!=DEFAULT_LOGDIR:
        raise ValueError('Report requires the formal C1_M1_1M directory')
    training,_=require_training(prepared)
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'dreamerv3'))
    from embodied.run.fg_checkpoint import validate_bundle
    expected=dict(protocol='FG_SINGLE3_OFFLINE_V1',identity_hash=receipt['identity']['identity_hash'])
    for update in MILESTONES:
        meta=validate_bundle(logdir/'ckpt'/f'step_{update:012d}',expected)
        if meta['step']!=update or not meta['protected']:
            raise ValueError('Required protected checkpoint is absent')
    results=[]
    external_files=dict(receipt['frozen_inputs'])
    external_files[str(prepared/'manifest.json')]=digest(prepared/'manifest.json')
    for strategy in ('always_wait','always_add','random_add','agent'):
        root=logdir/'dev/agent' if strategy=='agent' else prepared/'baselines'/strategy
        evidence=read_manifest(root/'evaluation_receipt.json')
        check_files({str(root/p):h for p,h in evidence['files'].items()})
        external_files[str(root/'evaluation_receipt.json')]=digest(root/'evaluation_receipt.json')
        external_files.update({str(root/p):h for p,h in evidence['files'].items()})
        if evidence['dev_manifest_sha256']!=digest(prepared/'dev_manifest.json'):
            raise ValueError('DEV evidence uses a different market manifest')
        if strategy=='agent' and evidence['model_checkpoint_sha256']!=training['checkpoint_manifest_sha256']:
            raise ValueError('DEV did not use the final training checkpoint')
        results.append(json.loads((root/'summary.json').read_text()))
    require_dev(prepared/'dev_manifest.json')
    result=comparison(results,training)
    lines=['# FG Single3 C1_M1_1M 训练与独立 DEV 报告','',
        f"工程：{result['engineering']}；训练：{result['training']}；择时：{result['timing']}。",'',
        f"实际梯度更新 {training['last_update']}；训练位置 {training['training_positions']}；"
        f"Gate 开启更新 {training['gate_open_updates']}；Actor 有效更新 {training['actor_effective_updates']}；"
        f"Critic 有效更新 {training['critic_effective_updates']}。",'',
        f"1. Actor 有效训练：{result['actor_effective_training']}。",
        f"2. 合法市场状态下 ADD/WAIT 变化证据：{result['legal_state_action_variation']}；择时结论 {result['timing']}。",
        f"3. DEV Reward/日与实际净PnL同时优于三个简单策略：{result['beats_simple_strategies_on_reward_and_net_pnl']}。",'',
        '|策略|开仓|主动WAIT|TP≤5 数量/比例|TP≤15 数量/比例|SL30 数量/比例|Reward|Reward/日|净PnL|最大权益回撤|',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in results:
        row=[str(r[k]) for k in ('strategy','entries','active_wait')]
        row += [str(r[k])+' / '+(f"{r[k+'_rate']:.2%}" if r[k+'_rate'] is not None else 'N/A')
            for k in ('tp5','tp15','sl30')]
        row += [str(r[k]) for k in ('total_reward','reward_per_day','net_pnl','max_equity_drawdown')]
        lines.append('|'+ '|'.join(row)+'|')
    lines+=['','TP/SL比例以全部真实开仓为分母，失败和期末未平仓保留；自然分钟计时。比例、分层、逐笔交易、权益与行动概率见各策略 CSV/summary.json。',
        '', 'Reward C1、Natural+P4、单仓3手、TP5/SL30、原网络、H333/imag15均保持冻结。',
        '潜状态由当前模型处理一帧因果前缀重建，停止上下文梯度；不等价于历史完整 RSSM 缓存。',
        '有效更新指 Gate 开启、梯度非零且参数实际变化；这一计数不直接证明优化充分或择时有效。',
        '评估没有 Oracle标签，封存TEST价格未读取；单模型seed、FG409 DEV不能证明长期泛化。',
        '', f"实验身份：`{expected['identity_hash']}`；源码版本与输入哈希见准备收据。"]
    report_path=publish_report(logdir,'\n'.join(lines)+'\n',result,dict(experiment='C1_M1_1M',
        identity_hash=expected['identity_hash'],conclusions={k:v for k,v in result.items() if k!='results'},
        prepared_manifest_sha256=digest(prepared/'manifest.json')),external_files)
    print(report_path,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare','baselines','start','resume','evaluate','report'])
    parser.add_argument('--prepared')
    parser.add_argument('--output')
    args=parser.parse_args()
    cpu=args.command in ('prepare','baselines','report')
    os.environ['PYTHONDONTWRITEBYTECODE']='1'
    os.environ['JAX_PLATFORMS']='cpu' if cpu else 'cuda'
    os.environ['JAX_PLATFORM_NAME']='cpu' if cpu else 'gpu'
    os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
    os.environ.setdefault('OMP_NUM_THREADS','1')
    if cpu:
        os.environ['CUDA_VISIBLE_DEVICES']=''
    if args.command=='prepare':
        from gym_trading_env.research.fg_single3_experiment import prepare,DEFAULT_OUTPUT
        result=prepare(args.output or DEFAULT_OUTPUT)
        print(result['status'],result['identity']['identity_hash'],flush=True)
    elif not args.prepared:
        parser.error('--prepared is required')
    elif args.command=='baselines':
        for result in baselines(args.prepared):
            print(json.dumps(result,allow_nan=False),flush=True)
    elif args.command in ('start','resume'):
        start(args.prepared,resume=args.command=='resume')
    elif args.command=='evaluate':
        evaluate(args.prepared)
    else:
        report(args.prepared)


if __name__=='__main__':
    main()

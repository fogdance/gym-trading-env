"""Finite scheduler subprocess fixture. Receipts are synthetic, never DATA."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

from gym_trading_env.research.fg_replay import signed,write_json,digest
from gym_trading_env.research.fg_s3_corpus_resources import phase_lease,pause_phase,resume_phase,rpc,run_policy_jobs
from gym_trading_env.research.fg_s3_corpus_production import batch_signed


def failing_account(job):
    if job['name']=='P0':raise ValueError('SYNTHETIC_STRATEGY_FAILURE')
    time.sleep(60)


def main(request_path):
    request=json.loads(Path(request_path).read_text())
    config,stage,batch,out=request['config'],request['stage'],request['batch']['id'],Path(request['out'])
    def event(kind):
        with Path(config['timeline']).open('a') as stream:
            stream.write(json.dumps(dict(batch=batch,stage=stage,event=kind,time=time.monotonic()))+'\n')
    with phase_lease():
        event('start')
        if stage=='generate' and config.get('fail_policy'):
            run_policy_jobs(failing_account,[dict(name=name) for name in request['policy_cpus']],request['policy_cpus'])
        if stage=='native_g3' and config.get('fail_once') and batch=='TRAIN01':
            flag=Path(config['fail_once'])
            if not flag.exists():
                flag.write_text('preserve failed attempt')
                return 9
        if stage in ('generate','native_g3'):
            pause_phase()
            def account(name):
                token=rpc('acquire',cpus=2,name=f'{batch}/{stage}/{name}',priority=0 if batch=='TRAIN01' else 10)
                try:
                    duration=4 if stage=='native_g3' and name in ('P0','P4') else .15
                    subprocess.run(['sleep',str(duration)],check=True)
                finally:rpc('release',token=token)
            with ThreadPoolExecutor(6) as executor:
                list(executor.map(account,('P0','P4','P1','P2_low','P2_mid','P3')))
            resume_phase()
        else:
            time.sleep(.03)
        if stage=='runtime_limits':
            result=dict(status='PASS_EXACT_RATIONAL_RUNTIME',python_int_max_str_digits=0)
        elif stage=='validate':
            source=signed(dict(fold='SYNTHETIC_NOT_TRAIN'))
            write_json(out/'VALIDATED_SOURCE_LOCK.json',source)
            result=dict(status='SOURCE_VALIDATED_ONLY',source=source)
        elif stage=='generate':
            corpus=out/'corpus';manifest=signed(dict(protocol='SYNTHETIC_NOT_TRAIN'))
            write_json(corpus/'corpus_manifest.json',manifest)
            result=dict(status='CORPUS_PRODUCED',corpus=str(corpus),corpus_identity=manifest['content_hash'])
            write_json(out/'generated.json',result)
        elif stage=='native_g3':
            identity=json.loads((Path(request['corpus'])/'corpus_manifest.json').read_text())['content_hash']
            native=dict(exit_code=0,result=dict(status='PASS',G3='GO',verification_state='NATIVE_G3_VERIFIED',
                corpus_identity=identity,native_replay=[dict(status='PASS',records=1,teacher_labels_verified=1)]*6))
            write_json(out/'native/native_verification.json',native)
            result=dict(status='NATIVE_G3_VERIFIED',corpus_identity=identity)
        elif stage=='reader':
            (out/'reader').mkdir();(out/'reader/join.sqlite').write_bytes(b'synthetic index')
            result=signed(dict(status='PASS_READER_ONLY',index_sha256=digest(out/'reader/join.sqlite')))
            write_json(out/'reader/WP03_COURSE_READER_COVERAGE.json',result)
        elif stage=='coverage':
            result=dict(status='OBSERVED_COVERAGE_ONLY')
            write_json(out/'quality_coverage.json',result)
        else:
            value=batch_signed(dict(status='NATIVE_G3_AND_INDEPENDENT_READER_COMPLETED',files={}))
            write_json(out/'BATCH_MANIFEST.json',value);write_json(out/'IDENTITY_LOCK.json',signed(dict(synthetic=True)))
            result={k:v for k,v in value.items() if k!='files'}
        write_json(out/'result.json',result)
        event('end')
    return 0


if __name__=='__main__':
    raise SystemExit(main(sys.argv[1]))

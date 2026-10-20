"""Recheck final graph binding against the already completed R6 baseline."""
import concurrent.futures,hashlib,json,os,pathlib,subprocess,sys,time,traceback
base=pathlib.Path(__file__).resolve().parents[2]
old=base/'docs/batch2_r6_r7_20261021/receipts/performance_final'
out=base/'docs/batch2_r6_r7_20261021/receipts/performance_final_v4';out.mkdir(parents=True,exist_ok=True)
artifact=base/'artifacts/batch2/performance_final_v4';artifact.mkdir(parents=True,exist_ok=True)
after=base.parent/'rounds/final-v4/gym-trading-env'
original=json.loads((old/'SOURCE_PAIRS.json').read_text());groups=original['groups']
before_results={(v['mode'],v['seen']):v for v in json.loads((old/'ALL_RESULTS.json').read_text())}
def git(root,*args):return subprocess.check_output(['git','-C',str(root),*args],text=True).strip()
pair={role:dict(path=str(root),sha=git(root,'rev-parse','HEAD'),status=git(root,'status','--porcelain'))
    for role,root in [('gym',after),('dreamer',after.parent/'dreamerv3')]}
assert all(not value['status'] for value in pair.values())
script_hash=hashlib.sha256((after/'artifacts/batch2/bench_final.py').read_bytes()).hexdigest()
assert script_hash==original['benchmark_script_sha256']
env=dict(PYTHONPATH='.:src:../dreamerv3:tests/entry_v1',JAX_PLATFORMS='cpu',CUDA_VISIBLE_DEVICES='',
    PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MPLCONFIGDIR='/tmp/batch2_mpl')
(out/'SOURCE_PAIRS.json').write_text(json.dumps(dict(before=original['pairs']['before'],after=pair,
    reused_before_results_sha256=hashlib.sha256((old/'ALL_RESULTS.json').read_bytes()).hexdigest(),
    benchmark_script_sha256=script_hash,groups=groups),indent=2)+'\n')
def run(name,args,cpus):
    argv=['taskset','-c',cpus,sys.executable,'-B','artifacts/batch2/bench_final.py',*args]
    t=time.monotonic()
    with (out/(name+'.log')).open('x') as log:
        r=subprocess.run(argv,cwd=after,env={**os.environ,**env},stdout=log,stderr=subprocess.STDOUT,timeout=600)
    record=dict(argv=argv,cwd=str(after),environment=env,source_pair=pair,
        benchmark_script_sha256=script_hash,exit_code=r.returncode,seconds=time.monotonic()-t)
    (out/(name+'_command.json')).write_text(json.dumps(record,indent=2)+'\n')
    if r.returncode:raise RuntimeError('FINAL_BENCHMARK_FAILED '+name+' exit '+str(r.returncode))
def group(cpus,cases):
    result=[]
    for mode,seen,n in cases:
        records=[]
        for rep in range(3):
            label=f'{mode}_{seen}_r{rep}_after';dest=artifact/label
            args=['--config',str(base/'artifacts/batch2/bench_v1/config.json'),'--out',str(dest),
                '--mode',mode,'--seen',str(seen),'--batches',str(n),'--versioned']
            run(label,args,cpus);run(label+'_cold_load',[*args,'--load-only'],cpus)
            value=json.loads((dest/'result.json').read_text());load=json.loads((dest/'result_load.json').read_text())
            assert value['state_digest']==load['state_digest'],label+' cold state differs'
            value['cold_load']=load
            baseline=before_results[(mode,seen)]['repeats'][rep]['before']
            assert baseline['golden']==value['golden'],label+' exposure or state differs'
            records.append(dict(before=baseline,after=value))
        summary=dict(mode=mode,seen=seen,timed_batches=n,cpus=cpus,repeats=records,
            baseline_reused=True,exact_golden_and_cold_states=True)
        (out/(f'{mode}_{seen}.json')).write_text(json.dumps(summary,indent=2)+'\n')
        result.append(summary)
        print(json.dumps(dict(case=f'{mode}_{seen}',three_repeats='PASS',golden_and_cold_parity=True)),flush=True)
    return result
try:
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        values=list(pool.map(lambda args:group(*args),groups))
    (out/'ALL_RESULTS.json').write_text(json.dumps([item for items in values for item in items],indent=2)+'\n')
except BaseException:
    (out/'FAILURE.txt').write_text(traceback.format_exc());raise

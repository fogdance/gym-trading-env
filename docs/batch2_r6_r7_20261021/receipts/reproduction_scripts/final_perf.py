import concurrent.futures,hashlib,json,os,pathlib,subprocess,sys,time,traceback
base=pathlib.Path(__file__).resolve().parents[2]
out=base/'docs/batch2_r6_r7_20261021/receipts/performance_final';out.mkdir(parents=True,exist_ok=True)
artifact=base/'artifacts/batch2/performance_final';artifact.mkdir(parents=True,exist_ok=True)
before=base.parent/'rounds/r6/gym-trading-env';after=base.parent/'rounds/final-v1/gym-trading-env'
groups=[('0-1',[('f1',1000,16),('f1',10000,8),('f1',100000,2),('f1',300000,2),('multi_f3',100000,2)]),
        ('2-3',[('f2',100000,2),('f3',100000,2),('multi',100000,2)]),
        ('4-7',[('f3_aux',1000,16),('f3_aux',10000,8),('f3_aux',100000,2),('f3_aux',300000,2)])]
def git(root,*args):return subprocess.check_output(['git','-C',str(root),*args],text=True).strip()
pair={name:{role:dict(path=str(root),sha=git(root,'rev-parse','HEAD'),status=git(root,'status','--porcelain'))
    for role,root in [('gym',path),('dreamer',path.parent/'dreamerv3')]} for name,path in [('before',before),('after',after)]}
assert all(not p['status'] for value in pair.values() for p in value.values())
script_hash=hashlib.sha256((after/'artifacts/batch2/bench_final.py').read_bytes()).hexdigest()
assert script_hash==hashlib.sha256((before/'artifacts/batch2/bench_final.py').read_bytes()).hexdigest()
env=dict(PYTHONPATH='.:src:../dreamerv3:tests/entry_v1',JAX_PLATFORMS='cpu',CUDA_VISIBLE_DEVICES='',
    PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MPLCONFIGDIR='/tmp/batch2_mpl')
(out/'SOURCE_PAIRS.json').write_text(json.dumps(dict(pairs=pair,benchmark_script_sha256=script_hash,groups=groups),indent=2)+'\n')
def run(name,root,args,cpus):
    argv=['taskset','-c',cpus,sys.executable,'-B','artifacts/batch2/bench_final.py',*args]
    t=time.monotonic()
    with (out/(name+'.log')).open('x') as log:
        r=subprocess.run(argv,cwd=root,env={**os.environ,**env},stdout=log,stderr=subprocess.STDOUT,timeout=600)
    record=dict(argv=argv,cwd=str(root),environment=env,source_pair=pair['before' if root==before else 'after'],
        benchmark_script_sha256=script_hash,exit_code=r.returncode,seconds=time.monotonic()-t)
    (out/(name+'_command.json')).write_text(json.dumps(record,indent=2)+'\n')
    if r.returncode:raise RuntimeError('BENCHMARK_FAILED '+name+' exit '+str(r.returncode))
def group(cpus,cases):
    result=[]
    for mode,seen,n in cases:
        records=[]
        for rep in range(3):
            versions=['before','after'] if rep%2==0 else ['after','before'];values={}
            for version in versions:
                label=f'{mode}_{seen}_r{rep}_{version}';dest=artifact/label;root=before if version=='before' else after
                args=['--config',str(base/'artifacts/batch2/bench_v1/config.json'),'--out',str(dest),
                    '--mode',mode,'--seen',str(seen),'--batches',str(n)]
                if version=='after':args.append('--versioned')
                run(label,root,args,cpus);run(label+'_cold_load',root,[*args,'--load-only'],cpus)
                value=json.loads((dest/'result.json').read_text());load=json.loads((dest/'result_load.json').read_text())
                assert value['state_digest']==load['state_digest'],label+' cold state differs'
                value['cold_load']=load;values[version]=value
            assert values['before']['golden']==values['after']['golden'],f'{mode} exposure or state differs'
            records.append(values)
        summary=dict(mode=mode,seen=seen,timed_batches=n,cpus=cpus,repeats=records,exact_golden_and_cold_states=True)
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

import argparse,hashlib,json,os,pathlib,subprocess,sys,time,xml.etree.ElementTree as ET
p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--cwd',default='artifacts/batch2/r7');p.add_argument('--cpus',default='0-23');p.add_argument('tests',nargs='+');a=p.parse_args()
base=pathlib.Path(__file__).resolve().parents[2];out=base/'docs/batch2_r6_r7_20261021/receipts';out.mkdir(parents=True,exist_ok=True)
cwd=(base/a.cwd).resolve();cwd.mkdir(parents=True,exist_ok=True)
env=dict(PYTHONPATH=':'.join(map(str,[base,base/'src',base.parent/'dreamerv3',base/'tests/entry_v1'])),JAX_PLATFORMS='cpu',CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
args=[str((base/t.split('::')[0]).resolve())+('::'+t.split('::',1)[1] if '::' in t else '') if not t.startswith('-') else t for t in a.tests]
argv=['taskset','-c',a.cpus,sys.executable,'-B','-m','pytest','-p','no:cacheprovider',*args,'--junitxml='+str(out/(a.name+'.xml'))]
sources={}
for name,root in [('gym',base),('dreamer',base.parent/'dreamerv3')]:
 def git(*args):return subprocess.check_output(['git','-C',str(root),*args],text=True).strip()
 sources[name]=dict(path=str(root),sha=git('rev-parse','HEAD'),status=git('status','--porcelain'))
record=dict(argv=argv,cwd=str(cwd),environment=env,sources=sources,scope='DEV_SYNTHETIC_CPU',formal_training='NOT_RUN',production_G3='NOT_RUN')
start=time.monotonic()
with (out/(a.name+'.log')).open('x') as log:r=subprocess.run(argv,cwd=cwd,env={**os.environ,**env},stdout=log,stderr=subprocess.STDOUT)
record.update(seconds=time.monotonic()-start,exit_code=r.returncode)
xml=out/(a.name+'.xml');cases=[]
if xml.exists():
 for c in ET.parse(xml).iter('testcase'):
  result='failed' if c.find('failure') is not None else 'error' if c.find('error') is not None else 'skipped' if c.find('skipped') is not None else 'passed'
  cases.append(dict(classname=c.get('classname'),name=c.get('name'),result=result))
record['cases']=cases;record['counts']={key:sum(c['result']==key for c in cases) for key in ['passed','failed','error','skipped']}
(out/(a.name+'.json')).write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps({key:record[key] for key in ['exit_code','seconds','counts']}),flush=True)
if r.returncode:print((out/(a.name+'.log')).read_text()[-6000:])
raise SystemExit(r.returncode)

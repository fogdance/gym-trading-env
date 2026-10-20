"""Read-only input/source audit; writes evidence only under this worktree."""
import ast,datetime,hashlib,json,pathlib,subprocess
base=pathlib.Path(__file__).resolve().parents[2]
out=base/'docs/batch2_r6_r7_20261021/receipts'
def sha(data):return hashlib.sha256(data).hexdigest()
def file_sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def git(root,*args):return subprocess.check_output(['git','-C',str(root),*args],text=True).strip()
initial=json.loads((out/'BASELINE.json').read_text())
assets={}
for root,expected in initial['assets'].items():
    root=pathlib.Path(root)
    actual={str(p.relative_to(root)):file_sha(p) for p in root.rglob('*') if p.is_file()}
    assert actual==expected,('PROTECTED_ASSET_CHANGED',str(root))
    assets[str(root)]=dict(files=len(actual),exact_file_set_and_bytes_unchanged=True,sha256=actual)
allowed={
 'gym':{'src/gym_trading_env/research/'+n+'.py' for n in (
     'fg_s3_e_quality_reader','fg_s3_e_quality_foundation','fg_s3_e_quality_heads',
     'fg_s3_e_quality_actor','fg_s3_e_quality_multi','fg_s3_e_quality_multi_actor')},
 'dreamer':{'embodied/core/streams.py','embodied/run/fg_s3_e_quality_foundation.py',
     'embodied/run/fg_s3_e_quality_heads.py','embodied/run/fg_s3_e_quality_actor.py',
     'embodied/run/fg_s3_e_quality_transfer.py'}}
runtime={}
for role,root in [('gym',base),('dreamer',base.parent/'dreamerv3')]:
    start=initial['repositories'][role]['head'];names=git(root,'ls-tree','-r','--name-only',start).splitlines()
    names=[n for n in names if n.endswith('.py') and n.startswith(('src/','tools/','dreamerv3/','embodied/'))
        and '/tests/' not in n]
    changed={};unchanged={}
    for name in names:
        old=subprocess.check_output(['git','-C',str(root),'show',start+':'+name]);p=root/name
        assert p.exists(),('RUNTIME_FILE_REMOVED',name)
        current=p.read_bytes()
        if current==old:unchanged[name]=sha(current)
        else:
            assert name in allowed[role],('OUT_OF_SCOPE_RUNTIME_CHANGE',name)
            changed[name]=dict(before=sha(old),after=sha(current))
    current_names=git(root,'ls-tree','-r','--name-only','HEAD').splitlines()
    added={n:file_sha(root/n) for n in current_names if n.startswith(('src/','embodied/','dreamerv3/','tools/'))
        and n.endswith('.py') and '/tests/' not in n and n not in names}
    runtime[role]=dict(baseline=start,final_code_sha=git(root,'rev-parse','HEAD'),
        unchanged_files=unchanged,changed_existing_runtime=changed,added_runtime=added)
frozen=base.parent.parent/'fg-s3-formal-corpus-optimized-20261020'
frozen_repos={}
for role in ['gym-trading-env','dreamerv3']:
    root=frozen/role
    frozen_repos[role]=dict(path=str(root),sha=git(root,'rev-parse','HEAD'),status=git(root,'status','--porcelain'))
    assert not frozen_repos[role]['status']
pin=frozen/'gym-trading-env/docs/formal_train_corpus_optimized_20261020/locks/PRODUCER_PIN.json'
assert file_sha(pin)=='d87082946aaa9db7408d68b2c2d147bf6cf1d28f27c1afb36d07c32b59239425'
task_docs=[pathlib.Path('/home/v/Downloads')/n for n in (
 'FG_S3_E_Quality_Batch2_R6_R7_Final_Codex_gy27_d327_20261021.md',
 'FG_S3_Batch1_to_Batch2_R6_R7_Interface_Review_gy27_d327_20261021.md',
 'FG_S3_E_Quality_Master_Roadmap_20261021_V16.md')]
record=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope='READ_ONLY_PROTECTED_ASSET_AND_SOURCE_AUDIT',
    assets=assets,runtime=runtime,frozen_producer=frozen_repos,producer_pin=dict(path=str(pin),sha256=file_sha(pin)),
    task_documents={str(p):file_sha(p) for p in task_docs},formal_training='NOT_RUN',production_G3='NOT_RUN',TEST='NOT_READ')
(out/'FINAL_PROTECTED_ASSET_AND_SOURCE_AUDIT.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(dict(protected_files=sum(v['files'] for v in assets.values()),
    runtime={k:dict(unchanged=len(v['unchanged_files']),changed=len(v['changed_existing_runtime']),added=len(v['added_runtime'])) for k,v in runtime.items()},
    frozen_producer=frozen_repos,exit_code=0)))

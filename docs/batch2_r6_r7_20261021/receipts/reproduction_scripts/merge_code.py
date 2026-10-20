"""Authorized ordinary merge of the tested code; never touches frozen Run trees."""
import datetime,json,pathlib,subprocess
base=pathlib.Path(__file__).resolve().parents[2];out=base/'docs/batch2_r6_r7_20261021/receipts'
integration=base.parent.parent/'fg-s3-next-integration';snapshot=base.parent/'rounds/final-v3'
selection=['f1','f2','f3','versions','shared','readers','handoff']
for name in selection:
    d=json.loads((snapshot/'gym-trading-env/docs/batch2_r6_r7_20261021/receipts'/('FINAL_V3_'+name+'.json')).read_text())
    assert d['exit_code']==0 and all(not v['status'] for v in d['sources'].values())
records=[];repos={}
def run(root,*args):
    argv=['git','-C',str(root),*args];r=subprocess.run(argv,text=True,capture_output=True)
    records.append(dict(argv=argv,exit_code=r.returncode,stdout=r.stdout,stderr=r.stderr))
    if r.returncode:raise RuntimeError('INTEGRATION_COMMAND_FAILED '+str(argv)+' '+r.stderr)
    return r.stdout.strip()
try:
    for name in ['dreamerv3','gym-trading-env']:
        target=integration/name;frozen=snapshot/name
        assert run(target,'branch','--show-current')=='integration/s3-next'
        assert not run(target,'status','--porcelain'),('INTEGRATION_DIRTY',name)
        run(target,'diff','--check');before=run(target,'rev-parse','HEAD')
        source=run(frozen,'rev-parse','HEAD')
        assert run(target,'rev-parse','feat/batch2-r6-r7-20261021')==source
        run(target,'merge','--no-ff','-m','Merge Batch 2 R7 exact Reader transactions and self-contained Prefetch recovery','feat/batch2-r6-r7-20261021')
        after=run(target,'rev-parse','HEAD');run(target,'merge-base','--is-ancestor',source,after)
        assert run(target,'rev-parse','HEAD^{tree}')==run(frozen,'rev-parse','HEAD^{tree}')
        assert not run(target,'status','--porcelain')
        repos[name]=dict(before=before,source=source,after=after,tree=run(target,'rev-parse','HEAD^{tree}'),
            exact_full_tree_equals_frozen_tested_source=True,branch='integration/s3-next',clean=True)
finally:
    record=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),repositories=repos,commands=records,
        ordinary_merge=True,reset_or_force=False,scope='CODE_MERGE_BEFORE_REPORT_DELIVERY')
    (out/'INTEGRATION_COMMANDS.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(repos))

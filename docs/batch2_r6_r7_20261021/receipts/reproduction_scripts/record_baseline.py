"""Record real ancestry without inspecting unrelated branches/worktrees."""
import datetime,json,pathlib,subprocess
b=pathlib.Path(__file__).resolve().parents[2];out=b/'docs/batch2_r6_r7_20261021/receipts'
base=json.loads((out/'BASELINE.json').read_text());records=[];repos={}
for role,root,first,r6merge in [
 ('gym',b,'20cdedf','710a1332d5f9ebc1993cd7c3a5002b0d732d3533'),
 ('dreamer',b.parent/'dreamerv3','4c4a366','a5631ffadd4815efcd6caf4c9026a274ea3d6ab7')]:
    def run(*args):
        argv=['git','-C',str(root),*args];r=subprocess.run(argv,text=True,capture_output=True)
        records.append(dict(argv=argv,exit_code=r.returncode,stdout=r.stdout,stderr=r.stderr))
        assert r.returncode==0,(args,r.stderr)
        return r.stdout.strip()
    original=base['repositories'][role]['head'];first_full=run('rev-parse',first)
    run('merge-base','--is-ancestor',first_full,original)
    run('merge-base','--is-ancestor',original,r6merge)
    run('merge-base','--is-ancestor',original,'HEAD')
    repos[role]=dict(initial_baseline=original,first_batch_feature=first_full,
        first_batch_already_in_initial_baseline=True,r6_integration_merge=r6merge,
        final_code=run('rev-parse','HEAD'),branch=run('branch','--show-current'))
record=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),repositories=repos,commands=records,
    no_duplicate_first_batch_merge=True,unrelated_branches_inspected=False)
(out/'BASELINE_ANCESTRY_AND_R6_INTEGRATION.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(repos))

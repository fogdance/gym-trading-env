"""Run existing regressions in an isolated cwd; preserve historical receipts."""
import json
import os
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path[:]=[str(Path(p or '.').resolve()) for p in sys.path]
os.environ['PYTHONPATH']=os.pathsep.join(str((ROOT/p).resolve()) for p in ('.','src','../dreamerv3','tests/entry_v1'))
name=sys.argv[1]
choices={
    'wp06_eval':['tests/entry_v1/test_wp06_evaluation.py::test_E03_original_driver_native_quality_reward_and_complete_account_replay',
                 'tests/entry_v1/test_wp06_evaluation.py::test_E04_evaluation_refuses_wrong_stage_and_invalid_budget'],
    'f1':['tests/entry_v1/test_wp04_wm.py::test_B01_reader_native_window_and_future_labels_are_separate',
          'tests/entry_v1/test_wp04_wm.py::test_B03_actual_original_optimizer_step_and_frozen_slots',
          'tests/entry_v1/test_wp04_wm.py::test_B05_only_T_targets_and_detached_K_history',
          'tests/entry_v1/test_wp04_wm.py::test_B06_each_window_ignores_previous_random_carry',
          'tests/entry_v1/test_wp04_checkpoint.py'],
    'f2':['tests/entry_v1/test_wp05_updates.py','tests/entry_v1/test_wp05_checkpoint.py'],
    'f3':['tests/entry_v1/test_wp06_updates.py','tests/entry_v1/test_wp06_checkpoint.py',
          'tests/entry_v1/test_wp06_run.py'],
}
for group,modules in dict(
    online_a=['contract'],online_b=['shadow'],online_c=['data'],online_d=['supervised'],
    online_e=['rl'],online_f=['gates'],online_g=['checkpoint','checkpoint_core'],
    online_h=['cli','history']).items():
    choices[group]=['tests/entry_v1/test_wp07_'+module+'.py' for module in modules]
if name not in choices:raise SystemExit('unknown regression group')
package=os.environ.get('FG_WP_REGRESSION_RECEIPTS','wp07')
if package not in ('wp07','wp08'):raise SystemExit('unknown receipt namespace')
out=ROOT/'artifacts/fg_s3_v2'/('e_quality_'+package)/sys.argv[2]
out.mkdir(parents=True,exist_ok=False)
for stage in ('wp04','wp05','wp06'):(out/'docs/receipts'/stage).mkdir(parents=True)
(out/'docs/receipts/wp07').mkdir(parents=True)
shutil.copyfile(ROOT/'docs/receipts/wp04/B_initial_layout.json',out/'docs/receipts/wp04/B_initial_layout.json')
os.chdir(out)
import pytest
args=['-p','no:cacheprovider','-o','addopts=','-q',
    '--junitxml='+str(ROOT/'docs/receipts'/package/('H_regression_'+name+'.xml')),
    *[str(ROOT/p) for p in choices[name]]]
status=pytest.main(args)
(out/'command.json').write_text(json.dumps(dict(argv=[sys.executable,'-B','-m','pytest',*args],
    cwd=str(out),exit_code=int(status)),indent=2)+'\n')
raise SystemExit(status)

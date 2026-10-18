"""Long exact witnesses through the user-operated, pinned DATA runtime."""
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT/'docs/formal_train_corpus_20261018_v2/RUN_TRAIN_CORPUS.sh'
FROZEN = Path('/data/s3_corpus/early_train_20261018_cpu20_v2/code')


def test_default_runtime_reproduces_both_decimal_conversion_failures():
    code = '''from fractions import Fraction
import sys
assert sys.get_int_max_str_digits() == 4300
for action in (lambda:str(Fraction(332,333)**2000),lambda:Fraction('1'+'0'*4999+'/3')):
    try: action()
    except ValueError as exc: assert '4300' in str(exc)
    else: raise AssertionError('Original long witness failure was not reproduced')
'''
    subprocess.run([sys.executable,'-B','-c',code],check=True,
                   env={**os.environ,'PYTHONINTMAXSTRDIGITS':'4300'})


def test_user_launcher_covers_parent_spawn_and_native_verifier_process():
    result = subprocess.run(['bash',str(SCRIPT),'runtime-check'],check=True,capture_output=True,
                            text=True,env={**os.environ,'PYTHONINTMAXSTRDIGITS':'4300'})
    report=json.loads(result.stdout)
    assert report['status']=='PASS_EXACT_RATIONAL_RUNTIME'
    assert report['python_int_max_str_digits']==0
    assert min(report['numerator_digits'],report['denominator_digits'])>80000
    assert report['exact_roundtrip'] and report['short_witness_unchanged']
    assert report['spawned_worker_matches'] and report['verifier_process_matches']
    assert report['optimizer_updates']==report['market_prices_opened']==0


def test_frozen_reader_catalog_preserves_long_exact_witness(tmp_path):
    output=tmp_path/'controlled_label.json'
    env={**os.environ,'PYTHONINTMAXSTRDIGITS':'0',
         'PYTHONPATH':os.pathsep.join(map(str,[FROZEN/'gym-trading-env',
             FROZEN/'gym-trading-env/src',FROZEN/'dreamerv3']))}
    code='''from fractions import Fraction
import json,sys
from gym_trading_env.research.fg_replay import signed
from gym_trading_env.research.fg_s3_quality_catalog_v2 import validate_corpus_label
v=Fraction(332,333)**32000
facts=signed(dict(protocol='FG_S3_QUALITY_TEACHER_V1',fold='TRAIN',
    score_policy='FG_S3_QUALITY_TP_MARKET_DAY_V1',training_action_label_allowed=False,
    state_id='controlled',account_snapshot_hash='controlled',score_valid=False,
    actual_legal_wait=True,actual_legal_add=True,event=[0]*7,event_valid=[False]*7,
    touch=[[0,0]]*4,touch_valid=[[False,False]]*4,adverse=[0]*4,adverse_valid=[False]*4))
strict=signed(dict(protocol='FG_S3_E_QUALITY_STRICT_SUFFIX_V2',teacher_valid=True,
    account_snapshot_hash='controlled',suffix_boundary_identity=dict(cursor=32000),
    value_rational_witness=[str(v),str(2*v)],strict_action=1,tie=False,
    action_value_wait=float(v),action_value_add=float(2*v),margin=float(v)))
label=signed(dict(protocol='FG_S3_E_QUALITY_TEACHER_V2',fold='TRAIN',
    state_id='controlled',account_snapshot_hash='controlled',quality_q_v2_valid=False,
    prediction_admitted=False,decision_admitted=False,L1=facts,L2=strict))
assert validate_corpus_label(label)==label
open(sys.argv[1],'w').write(json.dumps(label))
'''
    subprocess.run([sys.executable,'-B','-c',code,str(output)],env=env,check=True)
    parse='''import json,sys
from gym_trading_env.research.fg_s3_quality_catalog_v2 import validate_corpus_label
validate_corpus_label(json.load(open(sys.argv[1])))
'''
    failed=subprocess.run([sys.executable,'-B','-c',parse,str(output)],
                          env={**env,'PYTHONINTMAXSTRDIGITS':'4300'},capture_output=True,text=True)
    assert failed.returncode!=0 and 'Exceeds the limit (4300 digits)' in failed.stderr
    subprocess.run([sys.executable,'-B','-c',parse,str(output)],env=env,check=True)

"""Bounded baseline negatives; no model construction, market read or optimizer."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

from embodied.run.fg_s3_e_quality_online_gate import OnlineGates, NAMES
from embodied.run.fg_s3_e_quality_online_run import OnlineRun

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'docs/batch3_r2_r8_r3_20261021/receipts'
OUT.mkdir(parents=True, exist_ok=True)
command = [sys.executable,'-B',str(ROOT/'tools/fg_s3_v2_experiment.py'),
    'online-start','--prepared','/tmp/BATCH3_NONEXISTENT_PREPARED',
    '--producer-gym','/tmp/BATCH3_NONEXISTENT_VERIFIER',
    '--producer-lock','/tmp/BATCH3_NONEXISTENT_PIN.json','--producer-lock-sha256','0'*64]
r = subprocess.run(command,cwd=ROOT,capture_output=True,text=True)
assert r.returncode == 2 and 'only accepted by F1/F2/F3' in r.stderr
collector = SimpleNamespace(rows=[dict(action=0,legal_add=True)],teacher_enabled=False,
    policy=SimpleNamespace(beta=0),query_seconds=[],counts=dict(env_steps=16))
g = OnlineGates.__new__(OnlineGates); g.ready = dict.fromkeys(NAMES,False); g.evidence = {}
assert g.collection(collector,dict(status='PASS'))
collector.counts['env_steps'] = 48
collector.rows.append(dict(action=1,legal_add=False))
assert g.ready['COLLECTION_VALID']  # Actual old Gate stays true for the bad new tail.
run = OnlineRun.__new__(OnlineRun)
run.step=48;run.spec=dict(env_steps=48);run.phase=SimpleNamespace(state=dict(stage='IDLE'))
run.identity=dict(controlled_small=False,training_admitted=True)
run.mode='ONLINE_RL_CONTROLLED';run.counters=SimpleNamespace(save=lambda:dict(rl_updates=0))
run.gates=SimpleNamespace(report=lambda:dict(formal_rl_admitted=False))
run.phase.save=lambda:dict(stage='IDLE')
run.source_exposures=SimpleNamespace(save=lambda:{})
run.cp=SimpleNamespace(latest=lambda:'IN_MEMORY_ONLY')
run.agent=SimpleNamespace(n_updates=3)
result=run.result()
assert result['status']=='ONLINE_RUN_COMPLETE' and result['counters']['rl_updates']==0
evidence=dict(scope='CONTROLLED_BASELINE_NEGATIVES_NOT_FORMAL',
    cli=dict(argv=command,exit_code=r.returncode,stdout=r.stdout,stderr=r.stderr),
    collection=dict(first_verified_step=16,current_step=48,illegal_tail=True,
        old_collection_ready=g.ready['COLLECTION_VALID']),
    rl_result=result)
(OUT/'B3_00_BASELINE_NEGATIVES.json').write_text(json.dumps(evidence,indent=2)+'\n')

protected={}
workspace=ROOT.parent.parent
for relative in ('fg-s3-data-perf-phase2-20261020/artifacts/long_final_01/corpus',
    'fg-s3-data-perf-hotspots-20261020/artifacts/long_market_seed_01/corpus'):
    folder=workspace/relative
    if folder.is_dir():
        protected[str(folder)]={str(p.relative_to(folder)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(folder.rglob('*')) if p.is_file()}
frozen=workspace/'fg-s3-formal-corpus-optimized-20261020'
producer={}
for name in ('gym-trading-env','dreamerv3'):
    path=frozen/name
    if path.is_dir():
        producer[name]=dict(path=str(path),sha=subprocess.check_output(
            ['git','-C',str(path),'rev-parse','HEAD'],text=True).strip(),
            status=subprocess.check_output(['git','-C',str(path),'status','--porcelain=v1'],text=True))
pin=frozen/'gym-trading-env/docs/formal_train_corpus_optimized_20261020/locks/PRODUCER_PIN.json'
audit=dict(corpora=protected,frozen_producer=producer,
    producer_pin=None if not pin.is_file() else dict(path=str(pin),sha256=hashlib.sha256(pin.read_bytes()).hexdigest()),
    running_production_outputs='NOT_TOUCHED',TEST='NOT_READ')
(OUT/'B3_00_PROTECTED_ASSETS.json').write_text(json.dumps(audit,indent=2)+'\n')
print(json.dumps(dict(baseline_negatives=3,corpus_files=sum(map(len,protected.values())),
    cli_exit_code=r.returncode,formal_training='NOT_RUN')))

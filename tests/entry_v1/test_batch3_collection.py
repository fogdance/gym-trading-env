"""Real native online account reconciliation; bounded synthetic TRAIN only."""
import copy
import json
from pathlib import Path
import pickle

import pytest

from wp07_fixture import execution
from gym_trading_env.research.fg_replay import digest, read_manifest, signed, write_json


def shadow(root):
    from wp02_fixture import native_factory
    from embodied.run.fg_s3_e_quality_online_run import OnlineRun,engineering_permit
    factory,_,_=native_factory(root/'market',minutes=100,chunk=35)
    spec=execution(root/'run','SHADOW');spec.update(env_steps=48)
    return OnlineRun(spec,factory,dict(fold='TRAIN',scope='BUILTIN_SYNTHETIC_ONLY'),
        _probe=engineering_permit(spec))


def test_three_cycle_prefix_final_native_trace_and_idempotent_receipt(tmp_path):
    run=shadow(tmp_path)
    try:
        first=run.run_to(16)
        assert first['status']=='ONLINE_PARTIAL' and run.gates.ready['COLLECTION_VALID']
        assert run.gates.collection_through()==16
        run.run_to(32)
        assert not run.gates.ready['COLLECTION_VALID'] and run.gates.collection_through()==16
        result=run.run_to(48)
        assert result['status']=='SHADOW_COMPLETE_VERIFIED'
        assert result['counters']['optimizer_updates']==result['counters']['rl_updates']==0
        assert run.gates.collection_through()==48 and run.gates.ready['COLLECTION_VALID']
        evidence=result['account_verification']['final_evidence']
        assert evidence['receipt']['reconciliation']['account_restores']>0
        assert evidence['receipt']['collection_status']=='COLLECTION_FINAL_VERIFIED'
        node=run.cp.latest();manifest=json.loads((node/'manifest.json').read_text())
        saved=pickle.loads((node/'account_verification.pkl').read_bytes())
        assert manifest['protected'] and saved==result['account_verification']
        receipts=list((run.root/'completion').iterdir());assert len(receipts)==1
        receipt=read_manifest(receipts[0]);assert receipt['checkpoint_manifest_sha256']==digest(node/'manifest.json')
        before=receipts[0].read_bytes();assert run.run_to(48)==result
        assert receipts[0].read_bytes()==before and len(list((run.root/'completion').iterdir()))==1
        with pytest.raises(ValueError,match='ACTUAL_ACCOUNT_RECONCILIATION'):
            run.gates.collection(run.collector,dict(status='PASS'))
    finally:run.close()


@pytest.mark.parametrize('attack',['final_account','last_action_mask'])
def test_first_pass_cannot_certify_bad_third_cycle(tmp_path,monkeypatch,attack):
    run=shadow(tmp_path)
    try:
        run.run_to(16);assert run.gates.ready['COLLECTION_VALID']
        run.run_to(32)
        original=run.collector.snapshot_trace
        def changed(directory,identity):
            trace=original(directory,identity)
            if attack=='final_account':
                path=Path(directory)/'account_state.json'
                account=json.loads(path.read_text());account['BATCH3_BAD_FINAL_ACCOUNT']=True
                write_json(path,account)
            else:
                import gzip
                path=Path(directory)/'steps.jsonl.gz'
                with gzip.open(path,'rt') as f:rows=[json.loads(line) for line in f]
                rows[-1]['action']=1;rows[-1]['legal_add']=False
                with gzip.open(path,'wt') as f:
                    for row in rows:f.write(json.dumps(row)+'\n')
            m={k:v for k,v in read_manifest(Path(directory)/'manifest.json').items() if k!='content_hash'}
            m['files'][str(path.relative_to(directory))]=digest(path)
            write_json(Path(directory)/'manifest.json',signed(m))
            return trace
        monkeypatch.setattr(run.collector,'snapshot_trace',changed)
        with pytest.raises((ValueError,AssertionError)):run.run_to(48)
        result=run.result()
        assert result['status']=='COMPLETE_BUT_UNVERIFIED/NO_GO'
        assert result['account_verification']['status']=='NO_GO'
        assert not run.gates.ready['COLLECTION_VALID'] and run.gates.collection_through()==16
        assert not (run.root/'completion').exists()
        assert not json.loads((run.cp.latest()/'manifest.json').read_text())['protected']
    finally:run.close()


def test_final_commit_before_receipt_can_resume_publication(tmp_path,monkeypatch):
    run=shadow(tmp_path)
    try:
        publish=run._publish_completion
        def crash():raise RuntimeError('INJECTED_AFTER_FINAL_CHECKPOINT')
        monkeypatch.setattr(run,'_publish_completion',crash)
        with pytest.raises(RuntimeError,match='AFTER_FINAL_CHECKPOINT'):run.run_to(48)
        node=run.cp.latest()
        assert json.loads((node/'manifest.json').read_text())['protected']
        assert run.account_verification.state['status']=='VERIFIED'
        assert not (run.root/'completion').exists()
        monkeypatch.setattr(run,'_publish_completion',publish)
        result=run.run_to(48)
        assert result['status']=='SHADOW_COMPLETE_VERIFIED' and run.cp.latest()==node
        assert len(list((run.root/'completion').iterdir()))==1
    finally:run.close()

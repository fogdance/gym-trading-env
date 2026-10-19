"""Shared compatibility and independent review contracts, not FORMAL admission."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from wp07_fixture import f3_source,execution


def test_distinct_course_identities_use_original_native_compatibility(tmp_path):
    from data_multi_fixture import build_sources,descriptors
    from gym_trading_env.research.fg_s3_e_quality_multi_actor import MultiCorpusF3Stream
    from gym_trading_env.research.fg_s3_e_quality_online import validate_history_contract,_history
    from embodied.run.fg_s3_e_quality_online import inspect_f3
    config=json.loads(build_sources(tmp_path/'history').read_text());config['stage']='ACTOR_STRICT'
    source_descriptors=descriptors(config,'ACTOR_STRICT')
    proof,_=inspect_f3(execution(tmp_path/'unused'),allow_engineering=True)
    courses=[d['admission'].source.course_identity_hash for d in source_descriptors]
    assert len(set([proof['identity']['preparation']['course'],*courses]))==3
    files={p:p.read_bytes() for d in source_descriptors for p in Path(d['admission'].source.corpus_root).glob('*.json')}
    stream=MultiCorpusF3Stream.open(source_descriptors,settings=dict(batch_size=1,burnin_steps=4,target_steps=8),
        per_source_sampling_spec=config['per_source_sampling_spec'],source_mix_weights=config['source_mix_weights'],seed=202)
    try:
        validate_history_contract(stream,proof)
        assert {r['course_identity'] for r in stream.identity['roles'][0]['stream']['sources']}==set(courses)
        # The exact ABI remains mandatory; a mismatched header is refused.
        stream._identity['roles'][0]['stream']['compatibility']['model_contract']='0'*64
        with pytest.raises(ValueError,match='NATIVE_CONTRACT_MISMATCH'):validate_history_contract(stream,proof)
    finally:stream.close()
    assert all(p.read_bytes()==raw for p,raw in files.items())
    config_path=tmp_path/'history.json';config_path.write_text(json.dumps(config))
    # Synthetic Reader compatibility never upgrades unpublished sources to DATA.
    with pytest.raises(ValueError,match='PUBLISHED_BATCH_MANIFEST_REQUIRED'):
        _history(dict(history_config=str(config_path)),proof)


def test_review_verifier_binds_operator_digest_and_frozen_model_unit_only(tmp_path,monkeypatch):
    from embodied.run.fg_s3_e_quality_online_gate import OnlineGates
    from gym_trading_env.research.fg_s3_e_quality_online_checks import formal_mask_policy
    from gym_trading_env.research.fg_replay import digest
    from gym_trading_env.envs.entry_contract import fingerprint
    # Isolate ONLY the review document verifier. This stub is explicitly not a
    # native proof; the separate CLI negative exercises the real authority.
    preparation=SimpleNamespace(receipt=dict(scope='UNIT_ONLY_NOT_A_NATIVE_PROOF'))
    calls=[]
    def authority(p,*,authorized=False):
        assert p is preparation and authorized is True;calls.append(True)
    monkeypatch.setattr('gym_trading_env.research.fg_s3_e_quality_actor.require_training_authority',authority)
    g=OnlineGates('UNIT_ONLY',formal_mask_policy(f3_source()))
    g.ready=dict.fromkeys(g.ready,True);g.wm_identity='UNIT_WM';g.evidence=dict(numeric=dict(counter=1))
    body=dict(protocol='FG_S3_E_QUALITY_FORMAL_RL_REVIEW_V1',verdict='APPROVED_FOR_FORMAL_RL',
        gate_identity=g.identity,evidence_hash=fingerprint(g.evidence),wm_identity=g.wm_identity,
        f3_preparation_identity=fingerprint(preparation.receipt))
    path=tmp_path/'unit_review.json';path.write_text(json.dumps(body))
    with pytest.raises(ValueError,match='DIGEST_REQUIRED'):g.review_rl(preparation,path,'0'*64,authorized=True)
    with pytest.raises(ValueError,match='RESEARCH_REVIEW_REQUIRED'):g.require_rl(preparation,authorized=True,review=object())
    token=g.review_rl(preparation,path,digest(path),authorized=True)
    g.require_rl(preparation,authorized=True,review=token)
    g.evidence['numeric']['counter']+=1
    g.require_rl(preparation,authorized=True,review=token)
    g.wm_identity='CHANGED'
    with pytest.raises(ValueError,match='RESEARCH_REVIEW_REQUIRED'):g.require_rl(preparation,authorized=True,review=token)
    assert calls and not g.formal_admitted

"""Opt-in CPU preparation over real TRAIN data; never starts a GPU model.

The config must name a new preparation. Retain prepare_pool's original return
for the subsequent collection check, avoiding a second full Oracle load.
"""
import json
import os
from collections import Counter
from pathlib import Path
import time

import numpy as np
import pytest
import yaml


@pytest.mark.skipif(not os.environ.get('FG_S3_REAL_B2_PREPARE_CONFIG'),
                    reason='Explicit new real-data B2 preparation required')
def test_real_formal_preparation_pool_collection_and_recovery(monkeypatch):
    from gym_trading_env.research import fg_s3_v2_train_pool as pool_module
    from gym_trading_env.research.fg_s3_v2_experiment import prepare, read_settings, PROJECT
    from gym_trading_env.research.fg_s3_v2_collector import V2Collector
    from gym_trading_env.research.fg_replay import signed, write_json, read_manifest
    from embodied.run.fg_checkpoint import save_replay, load_replay, file_hash

    started = time.perf_counter()
    config = os.environ['FG_S3_REAL_B2_PREPARE_CONFIG']
    settings = read_settings(config)
    assert settings['mode'] == 'b2' and settings['scope'] == 'formal'
    original = pool_module.prepare_pool
    retained = {}

    def capture(*args, **kwargs):
        query = original(*args, **kwargs)
        retained['query'] = query
        return query

    with monkeypatch.context() as context:
        context.setattr(pool_module, 'prepare_pool', capture)
        preparation = prepare(config)
    assert preparation['status'] == 'PREPARED_NO_TRAINING'
    assert not preparation['diagnostic'] and not any(preparation['dirty'].values())
    query = retained['query']
    assert len(query.queries) == len(query.outcome_catalog.catalogs) == 18
    prepared = Path(settings['prepared_dir'])
    assert read_manifest(prepared/'outcome_catalog.json')['identity_hash'] == query.outcome_catalog.identity_hash
    root = PROJECT/'artifacts/fg_s3_v2/full_checks/b2_full_real_18_contract_cpu'
    root.mkdir(parents=True, exist_ok=False)
    # Only this diagnostic selector uses short slices. The formal snapshot
    # remains unchanged at 1000 bars, with the same authentic TRAIN manifest.
    selector = yaml.safe_load((prepared/'selector.yaml').read_text())
    assert selector['scheduler']['chunk_bars'] == settings['env_slice_bars'] == 1000
    selector['scheduler']['chunk_bars'] = 4
    (root/'selector.yaml').write_text(yaml.safe_dump(selector))
    print('Prepared 18 original TRAIN members; beginning CPU collection', flush=True)
    a = V2Collector(pool_module.ObservedTrainingPool(root/'selector.yaml'), query,
        root/'a', settings['dreamer_root'], beta=1., outcome_catalog=query.outcome_catalog)
    b = None
    try:
        a.advance(75)
        advanced = [r for r in a.records if r['execution_result']['advanced']]
        contracts = Counter(r['contract'] for r in advanced)
        assert set(contracts) == set(query.queries)
        valid = Counter(r['contract'] for r in advanced if r['outcome']['valid_event'])
        assert sum(valid.values()) > 0
        excluded = Counter(r['outcome']['reason'] for r in advanced if not r['outcome']['valid_event'])
        for row in a.records:
            assert query.accepts_identity(row['contract'], row['oracle_identity'])
            assert query.outcome_catalog.accepts_identity(row['contract'], row['outcome']['catalog_identity'])
            if not all(row['action_mask']):
                assert not row['outcome']['valid_event'] and not row['outcome']['valid_early']
                assert not row['teacher_valid']
        state, driver = a.save(), a.driver.save()
        replay = save_replay(a.replay, root/'snapshot')
        b = V2Collector(pool_module.ObservedTrainingPool(root/'selector.yaml'), query,
            root/'b', settings['dreamer_root'], beta=1., outcome_catalog=query.outcome_catalog)
        b.journal.write_bytes(a.journal.read_bytes())
        b.driver.load(driver); load_replay(b.replay, replay, root/'snapshot'); b.load(state)
        a.advance(80); b.advance(80)
        assert a.records == b.records and a.env.export_state() == b.env.export_state()
        x, y = a.sample(1), b.sample(1)
        for key in x:
            np.testing.assert_array_equal(x[key], y[key])
        assert x['outcome_event_valid'].any() and x['teacher_valid'].any()
        assert not any(k.startswith(('outcome', 'teacher')) for k in next(iter(a.replay.chunks.values())).data)
        receipt = signed(dict(protocol='FG_S3_B2_FULL_REAL_18_CONTRACT_CPU_V1', status='PASS',
            model_initialized=False, gpu_training='NOT_RUN', joint_updates='NOT_RUN',
            data='REAL_TRAIN_MARKET_MINUTE_V2', clock_protocol='S3_MARKET_MINUTE',
            preparation_sha256=file_hash(prepared/'manifest.json'),
            preparation_content_hash=preparation['content_hash'],
            diagnostic_slice_bars=4, formal_slice_bars=1000,
            config_sha256=settings['config_sha256'], oracle_identity=query.identity_hash,
            outcome_identity=query.outcome_catalog.identity_hash,
            contracts=dict(contracts), valid_outcomes=dict(valid), excluded_outcomes=dict(excluded),
            saved_env_steps=75, continued_env_steps=80, counts=a.counts,
            replay_sidecar_driver_restore='PASS', resumed_sample_equality='PASS',
            native_observation_has_teacher=False,
            sampled_event_valid=int(x['outcome_event_valid'].sum()),
            sampled_teacher_valid=int(x['teacher_valid'].sum()),
            elapsed_seconds=time.perf_counter()-started))
        write_json(root/'receipt.json', receipt)
        print(json.dumps(receipt, indent=2), flush=True)
    finally:
        a.close()
        if b:
            b.close()

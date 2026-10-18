import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest

from data_multi_fixture import build_sources, descriptors, open_stream, exact
from gym_trading_env.research.fg_s3_e_quality_multi import MultiCorpusCourseStream
from gym_trading_env.research.fg_s3_e_quality_reader import EQualityCourseReader, batch_pair_counts


@pytest.fixture(scope='module')
def corpus_pair(tmp_path_factory):
    path = build_sources(tmp_path_factory.mktemp('multi_native'))
    return json.loads(path.read_text()), path


@pytest.mark.parametrize('stage', ['WM_NATURAL', 'HEADS_VALID', 'ACTOR_STRICT', 'ACTOR_QUALITY_PAIR'])
def test_native_mixed_batch_and_provenance_all_stages(corpus_pair, stage):
    config, _ = corpus_pair
    stream = open_stream(config, stage=stage)
    try:
        data, audit = stream.sample(8, 8)
        assert {row['corpus_id'] for row in audit} == {'alpha', 'beta'}
        assert data['stepid'].shape == (8, 9, 20)
        assert 'corpus_id' not in data and not data['consec'].any()
        assert not data['is_first'][:, 1:].any()
        assert stream.audit_windows(data, audit) == audit
        for row in audit:
            assert len({(r['policy'], r['segment']) for r in row['rows']}) == 1
            assert [r['ordinal'] for r in row['rows']] == list(range(row['rows'][0]['ordinal'], row['rows'][0]['ordinal']+9))
        coverage = stream.coverage()
        assert coverage['delivered_counts'] == dict(batches=1, windows=8, effective_positions=64)
        assert coverage['peak_cached_bytes_upper_bound'] <= config['cache_bytes']
        assert coverage['actual_optimizer_exposures'] == 0 and not coverage['training_admitted']
    finally:
        stream.close()


@pytest.mark.parametrize('mode', ['NATURAL', 'STRATIFIED', 'MIXED'])
def test_exact_rng_restore_and_independent_source_rng(corpus_pair, mode):
    config, _ = corpus_pair
    spec = dict(mode=mode)
    if mode != 'NATURAL':
        spec['strata_weights'] = dict(STRICT_ADD=1, STRICT_WAIT=1)
    if mode == 'MIXED':
        spec['natural_weight'] = .5
    specs = {k: spec for k in ('alpha', 'beta')}
    a = open_stream(config, stage='ACTOR_STRICT', specs=specs)
    b = open_stream(config, stage='ACTOR_STRICT', specs=specs)
    baselines = {}
    try:
        # Compare each child directly to WP03 with that child's derived seed.
        for record in a.identity['sources']:
            name = record['corpus_id']; reader = a.readers[name]
            baselines[name] = EQualityCourseReader.open(reader.admission, stage=a.stage,
                sampling_spec=reader.spec, seed=record['reader_seed'], index_path=reader.index_path)
        raw, audit = a.sample(8, 8)
        for name, reader in baselines.items():
            positions = [i for i, row in enumerate(audit) if row['corpus_id'] == name]
            if positions:
                exact({k: v[positions] for k, v in raw.items()}, reader.sample(len(positions), 8))
            exact(a.readers[name].state_dict(), reader.state_dict())
        saved = json.loads(json.dumps(a.state_dict()))
        expected = [a.sample(8, 8) for _ in range(3)]
        b.sample(2, 8)
        b.load_state_dict(saved)
        for sample in expected:
            exact(sample, b.sample(8, 8))
        exact(a.state_dict(), b.state_dict())
    finally:
        a.close(); b.close()
        for reader in baselines.values(): reader.close()


def test_missing_support_and_invalid_restore_preserve_all_states(corpus_pair):
    config, _ = corpus_pair
    specs = copy.deepcopy(config['per_source_sampling_spec'])
    specs['beta'] = dict(mode='STRATIFIED', strata_weights={'DOES_NOT_EXIST': 1})
    stream = open_stream(config, specs=specs)
    try:
        before = stream.state_dict()
        with pytest.raises(ValueError, match='INSUFFICIENT_CURRICULUM_SUPPORT'):
            stream.sample(8, 8)
        exact(before, stream.state_dict())
    finally: stream.close()
    stream = open_stream(config)
    try:
        stream.sample(4, 8); before = stream.state_dict()
        bad = copy.deepcopy(before); bad['readers']['beta']['index'] = 'different'
        with pytest.raises(ValueError, match='RESTORE_IDENTITY'):
            stream.load_state_dict(bad)
        exact(before, stream.state_dict())
        bad = copy.deepcopy(before); bad['readers']['beta']['seen_market'] = [1]
        with pytest.raises(TypeError): stream.load_state_dict(bad)
        exact(before, stream.state_dict())
        bad = copy.deepcopy(before); bad['counts']['windows'] += 1
        with pytest.raises(ValueError, match='COUNTER_TOTAL'): stream.load_state_dict(bad)
        exact(before, stream.state_dict())
    finally: stream.close()


@pytest.mark.parametrize('change', ['order', 'weights', 'spec', 'source_list', 'stage'])
def test_aggregate_identity_changes_refuse_resume(corpus_pair, change):
    config, _ = corpus_pair
    a = open_stream(config); state = a.state_dict()
    sources = descriptors(config)
    kwargs = {}
    if change == 'order': sources.reverse()
    if change == 'weights': kwargs['weights'] = dict(alpha=2, beta=1)
    if change == 'spec':
        kwargs['specs'] = dict(alpha=dict(mode='NATURAL', policies=['P0']), beta=dict(mode='NATURAL'))
    if change == 'source_list':
        sources = sources[:1]; kwargs.update(weights=dict(alpha=1), specs=dict(alpha=dict(mode='NATURAL')))
    if change == 'stage':
        kwargs['stage'] = 'HEADS_VALID'; sources = descriptors(config, 'HEADS_VALID')
    b = open_stream(config, sources=sources, **kwargs)
    try:
        assert a.identity_hash != b.identity_hash
        with pytest.raises(ValueError, match='RESTORE_IDENTITY'): b.load_state_dict(state)
    finally: a.close(); b.close()


def test_cross_source_splice_and_owner_collision_never_create_pair(corpus_pair):
    config, _ = corpus_pair
    stream = open_stream(config, stage='ACTOR_QUALITY_PAIR')
    try:
        raw, audit = stream.sample(8, 8)
        other = next(i for i, row in enumerate(audit) if row['corpus_id'] != audit[0]['corpus_id'])
        attacked = copy.deepcopy(raw)
        for value in attacked.values(): value[0, 2] = value[other, 2]
        with pytest.raises(ValueError): stream.audit_windows(attacked, audit)
        # One source supplies only ADD, the other only WAIT; equal owners across
        # the batch dimension must not turn these into an artificial pair.
        pair = {k: v[[0, other]].copy() for k, v in raw.items()}
        pair['quality_owner'][...] = 7
        pair['teacher_valid'][...] = True
        pair['teacher_action'][0] = 1; pair['teacher_action'][1] = 0
        pair['quality_score_valid'][...] = True; pair['quality_outcome_valid'][...] = True
        pair['quality_score'][0] = 100; pair['quality_score'][1] = 0
        assert batch_pair_counts(pair) == dict(strict_pair=0, quality_pair=0)
    finally: stream.close()


def test_source_mutation_rejected_even_when_source_not_selected(corpus_pair, tmp_path):
    config, _ = corpus_pair
    changed = copy.deepcopy(config)
    source = tmp_path/'beta'; shutil.copytree(changed['sources'][1]['corpus_root'], source)
    changed['sources'][1]['corpus_root'] = str(source)
    stream = open_stream(changed, weights=dict(alpha=1, beta=1e-100))
    try:
        member = source/'teacher_sidecar/P0/steps.jsonl.gz'
        member.write_bytes(member.read_bytes()+b'corrupt')
        with pytest.raises(ValueError, match='CONTENT_CHANGED'): stream.sample(1, 8)
        with pytest.raises(ValueError, match='INVALID_REOPEN'): stream.sample(1, 8)
    finally: stream.close()


def test_hot_path_does_not_rehash_sources_and_adapter_reuses_f1_f2(corpus_pair, monkeypatch):
    from gym_trading_env.research.fg_s3_e_quality_foundation import F1WindowAdapter
    from gym_trading_env.research.fg_s3_e_quality_heads import F2WindowAdapter
    import gym_trading_env.research.fg_s3_e_quality_multi as module
    config, _ = corpus_pair
    for stage, cls in [('WM_NATURAL', F1WindowAdapter), ('HEADS_VALID', F2WindowAdapter)]:
        stream = open_stream(config, stage=stage)
        adapter = stream.bind(dict(batch_size=8, target_steps=8, burnin_steps=4), cls)
        try:
            def forbidden(*args, **kwargs): raise AssertionError('Full source rehash on batch hot path')
            with monkeypatch.context() as patch:
                patch.setattr(module, 'inspect_source', forbidden)
                data, audit = adapter.sample()
                saved = adapter.save(); expected = adapter.sample()
                adapter.load(saved); exact(expected, adapter.sample())
            assert 'corpus_id' not in data and 'teacher_action' not in data
            assert not any(k.startswith('quality_') for k in data) if stage == 'WM_NATURAL' else 'quality_event' in data
            assert len(audit) == 8
        finally: adapter.close()


@pytest.mark.parametrize('attack', ['duplicate_id', 'duplicate_content', 'zero_weight', 'underflow_weight'])
def test_invalid_source_contracts_fail_before_sampling(corpus_pair, attack):
    config, _ = corpus_pair
    sources = descriptors(config)
    weights = dict(alpha=1, beta=1)
    if attack == 'duplicate_id': sources[1]['corpus_id'] = 'alpha'
    if attack == 'duplicate_content': sources[1]['admission'] = sources[0]['admission']
    if attack == 'zero_weight': weights['beta'] = 0
    if attack == 'underflow_weight': weights = dict(alpha=1e-320, beta=1e300)
    with pytest.raises(ValueError): open_stream(config, sources=sources, weights=weights)


def test_adapter_failure_rolls_back_scheduler_and_every_reader(corpus_pair):
    from gym_trading_env.research.fg_s3_e_quality_foundation import F1WindowAdapter
    class BrokenAdapter(F1WindowAdapter):
        def adapt(self, raw):
            result = super().adapt(raw)
            if self.reader.root.name == 'beta': raise ValueError('controlled adapter failure')
            return result
    config, _ = corpus_pair
    stream = open_stream(config)
    try:
        bound = stream.bind(dict(batch_size=8, target_steps=8, burnin_steps=4), BrokenAdapter)
        before = stream.state_dict()
        with pytest.raises(ValueError, match='controlled adapter failure'): bound.sample()
        exact(before, stream.state_dict())
    finally: stream.close()


def test_two_corpus_original_checkpoint_cold_processes_and_cli(corpus_pair, tmp_path, record_property):
    config, path = corpus_pair
    output = tmp_path/'diagnostic'
    argv = [sys.executable, '-B', str(Path(__file__).resolve().parents[2]/'tools/fg_s3_e_quality_reader.py'),
            'multi-read', '--config', str(path), '--out', str(output), '--length', '12', '--batch-size', '4']
    run = subprocess.run(argv, capture_output=True, text=True)
    (tmp_path/'cli.log').write_text(run.stdout+run.stderr)
    assert run.returncode == 0, run.stderr[-5000:]+run.stdout[-3000:]
    report = json.loads(run.stdout.splitlines()[-1])
    assert report['status'] == 'DEV_PASS' and report['real_source_acceptance'] == 'NOT_RUN'
    assert report['optimizer_updates'] == 0 and report['training_admitted'] is False
    cold = json.loads((output/'CHECKPOINT_COLD_EVIDENCE.json').read_text())
    assert cold['status'] == 'PASS_EXACT_COLD_DATA_STATE' and len(cold['calls']) == 3
    assert cold['prefetched_batches'] == 2 and cold['copied_corpus_chunks'] == 0
    assert cold['all_native_array_bytes_and_provenance_equal']
    assert cold['runtime_and_pending_queue_equal'] and cold['scheduler_and_all_reader_states_equal']
    record_property('diagnostic_root', str(output))
    record_property('cli_argv', json.dumps(argv))


def test_real_receipts_required_and_reject_output_inside_either_corpus(corpus_pair, tmp_path):
    from gym_trading_env.research.fg_s3_e_quality_multi_checks import multi_evidence
    config, path = corpus_pair
    output = tmp_path/'real_unavailable'
    with pytest.raises(ValueError, match='PUBLISHED_BATCH_MANIFEST_REQUIRED'):
        multi_evidence(path, output, require_published=True)
    assert not output.exists()
    output = Path(config['sources'][1]['corpus_root'])/'forbidden_output'
    with pytest.raises(ValueError): multi_evidence(path, output)
    assert not output.exists()


@pytest.mark.parametrize('kind', ['index', 'receipt'])
def test_bound_index_and_receipt_mutation_refuse_restore(corpus_pair, tmp_path, kind):
    config, _ = corpus_pair
    changed = copy.deepcopy(config)
    index = tmp_path/'beta.sqlite'
    shutil.copyfile(config['sources'][1]['index_path'], index)
    shutil.copyfile(Path(config['sources'][1]['index_path']).with_suffix('.json'), index.with_suffix('.json'))
    changed['sources'][1]['index_path'] = str(index)
    stream = open_stream(changed)
    try:
        saved = stream.state_dict()
        target = index if kind == 'index' else index.with_suffix('.json')
        with target.open('ab') as output: output.write(b' ')
        with pytest.raises(ValueError, match='INDEX.*CHANGED'): stream.validate_snapshot(saved)
    finally: stream.close()


def test_incompatible_model_contract_and_changed_adapter_bytes_rejected(corpus_pair, tmp_path):
    from wp03_fixture import inventory
    from gym_trading_env.research.fg_replay import read_manifest, signed, write_json
    from gym_trading_env.research.fg_s3_e_quality_reader import build_join_index
    from gym_trading_env.research.fg_s3_e_quality_supervision import QualitySupervisionAdmission, SupervisionSource
    from gym_trading_env.research.fg_s3_e_quality_foundation import F1WindowAdapter
    config, _ = corpus_pair
    changed = copy.deepcopy(config)
    source = tmp_path/'different_model'
    shutil.copytree(config['sources'][1]['corpus_root'], source)
    model = read_manifest(source/'model_contract.json'); model.pop('content_hash')
    model['implementation_variant'] = 'controlled incompatible contract'
    model = signed(model); write_json(source/'model_contract.json', model)
    manifest = read_manifest(source/'corpus_manifest.json'); manifest['model_contract_hash'] = model['content_hash']
    inventory(source, manifest)
    admission = QualitySupervisionAdmission.preflight(SupervisionSource.from_corpus(source))
    index = tmp_path/'new.sqlite'; build_join_index(admission, index)
    changed['sources'][1].update(corpus_root=str(source), index_path=str(index))
    with pytest.raises(ValueError, match='INCOMPATIBLE_NATIVE_CONTRACTS'): open_stream(changed)
    class ChangingAdapter(F1WindowAdapter):
        def adapt(self, raw):
            data, audit = super().adapt(raw)
            data['reward'][...] = 123
            return data, audit
    stream = open_stream(config)
    try:
        before = stream.state_dict()
        adapter = stream.bind(dict(batch_size=8, target_steps=8, burnin_steps=4), ChangingAdapter)
        with pytest.raises(ValueError, match='ADAPTER_NATIVE_BYTES_CHANGED'): adapter.sample()
        exact(before, stream.state_dict())
    finally: stream.close()

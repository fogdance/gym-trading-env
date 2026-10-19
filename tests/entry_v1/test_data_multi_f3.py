import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest

from data_multi_fixture import build_sources, descriptors, exact
from gym_trading_env.research.fg_s3_e_quality_multi_actor import MultiCorpusF3Stream


@pytest.fixture(scope='module')
def corpus_pair(tmp_path_factory):
    path = build_sources(tmp_path_factory.mktemp('multi_f3_native'))
    config = json.loads(path.read_text())
    config['stage'] = 'ACTOR_STRICT'
    path.write_text(json.dumps(config, indent=2)+'\n')
    return config, path


def specs(mode):
    if mode is None:
        return None
    value = dict(mode=mode)
    if mode != 'NATURAL':
        value['strata_weights'] = dict(STRICT_ADD=1, STRICT_WAIT=1)
    if mode == 'MIXED':
        value['natural_weight'] = .5
    return {name: copy.deepcopy(value) for name in ('alpha', 'beta')}


def open_f3(config, mode=None, *, settings=None, auxiliary=None, weights=None):
    return MultiCorpusF3Stream.open(descriptors(config, 'ACTOR_STRICT'),
        settings=settings or dict(batch_size=8, target_steps=8, burnin_steps=4),
        per_source_sampling_spec=config['per_source_sampling_spec'],
        aux_per_source_sampling_spec=auxiliary if auxiliary is not None else specs(mode),
        source_mix_weights=weights or config['source_mix_weights'], seed=config['seed'],
        cache_bytes=config['cache_bytes'])


@pytest.mark.parametrize('mode', [None, 'NATURAL', 'STRATIFIED', 'MIXED'])
def test_f3_roles_native_bytes_provenance_and_exact_state(corpus_pair, mode):
    config, _ = corpus_pair
    a = open_f3(config, mode); b = open_f3(config, mode)
    try:
        data, audit = a.sample()
        size = 8 if mode is None else 16
        assert data['stepid'].shape == (size, 13, 20)
        assert data['f3_natural'].dtype == bool and data['f3_natural'][:8].all()
        assert not data['f3_natural'][8:].any()
        assert not {'corpus_id', 'role', 'quality_q_c1_valid', 'quality_soft_valid'} & data.keys()
        assert 'teacher_action' in data and 'quality_owner' in data
        for role, bound in a.role_streams.items():
            positions = [i for i, row in enumerate(audit) if row['role'] == role]
            assert {audit[i]['corpus_id'] for i in positions} == {'alpha', 'beta'}
            native_audit = [{k: v for k, v in audit[i].items() if k != 'role'} for i in positions]
            assert bound.stream.audit_windows({k: v[positions] for k, v in data.items()}, native_audit) == native_audit
            for row in native_audit:
                assert len({(r['policy'], r['segment']) for r in row['rows']}) == 1
        state = a.save(); expected = [a.sample() for _ in range(2)]
        b.sample(); b.load(json.loads(json.dumps(state)))
        for value in expected:
            exact(value, b.sample())
        exact(a.save(), b.save())
        coverage = a.coverage()
        assert coverage['peak_cached_bytes_upper_bound'] <= config['cache_bytes']
        assert coverage['actual_optimizer_exposures'] == 0 and not coverage['training_admitted']
        assert set(coverage['roles']) == set(a.role_streams)
    finally:
        a.close(); b.close()


def test_auxiliary_rng_and_objective_preserve_natural_stream(corpus_pair):
    from gym_trading_env.research.fg_s3_e_quality_multi_checks import f3_loss_evidence
    config, _ = corpus_pair
    natural = open_f3(config); both = open_f3(config, 'MIXED')
    try:
        base, _ = natural.sample(); mixed, mixed_audit = both.sample()
        exact(base, {k: v[:8] for k, v in mixed.items()})
        exact(natural.role_streams['NATURAL'].save(), both.role_streams['NATURAL'].save())
        roles = both.identity['roles']
        assert roles[0]['seed'] != roles[1]['seed']
        assert roles[0]['stream']['sources'][0]['reader_seed'] != roles[1]['stream']['sources'][0]['reader_seed']
        one = f3_loss_evidence(base, 4); two = f3_loss_evidence(mixed, 4)
        for key in ('natural_loss', 'natural_count', 'natural_add', 'natural_wait'):
            assert one['metrics']['f3/'+key] == two['metrics']['f3/'+key]
        assert one['metrics']['f3/auxiliary_count'] == 0
        assert two['metrics']['f3/auxiliary_count'] > 0
        # Force equal owners across two different source rows. No cross-row Pair.
        first = 0
        other = next(i for i, row in enumerate(mixed_audit[:8]) if row['corpus_id'] != mixed_audit[first]['corpus_id'])
        rows = {k: v[[first, other]].copy() for k, v in mixed.items()}
        rows['quality_owner'][...] = 7
        rows['teacher_valid'][...] = True
        rows['teacher_action'][0] = 1; rows['teacher_action'][1] = 0
        rows['quality_score_valid'][...] = True; rows['quality_outcome_valid'][...] = True
        rows['quality_score'][0] = 100; rows['quality_score'][1] = 0
        evidence = f3_loss_evidence(rows, 4)
        assert evidence['metrics']['f3/strict_pair_count'] == evidence['metrics']['f3/quality_pair_count'] == 0
        assert evidence['optimizer_updates'] == 0 and evidence['model_forward'] == 'NOT_RUN'
    finally:
        natural.close(); both.close()


def test_auxiliary_failure_and_bad_restore_roll_back_whole_group(corpus_pair, monkeypatch):
    config, _ = corpus_pair
    stream = open_f3(config, 'MIXED')
    try:
        stream.sample(); before = stream.save()
        auxiliary = stream.role_streams['AUXILIARY']; original = auxiliary.sample
        def fail_after_sampling():
            original()
            raise ValueError('controlled auxiliary failure')
        with monkeypatch.context() as patch:
            patch.setattr(auxiliary, 'sample', fail_after_sampling)
            with pytest.raises(ValueError, match='controlled auxiliary failure'):
                stream.sample()
        exact(before, stream.save())
        bad = copy.deepcopy(before)
        bad['group']['readers'][1]['stream']['readers']['beta']['seen_market'] = [1]
        with pytest.raises(TypeError):
            stream.load(bad)
        exact(before, stream.save())
        bad = copy.deepcopy(before); bad['last_batch_audit'][0]['role'] = 'AUXILIARY'
        with pytest.raises(ValueError, match='RESTORE_IDENTITY'):
            stream.load(bad)
        exact(before, stream.save())
    finally:
        stream.close()


@pytest.mark.parametrize('change', ['auxiliary', 'spec', 'geometry'])
def test_f3_role_identity_changes_refuse_restore(corpus_pair, change):
    config, _ = corpus_pair
    first = open_f3(config, 'MIXED')
    second = open_f3(config, None if change == 'auxiliary' else 'STRATIFIED' if change == 'spec' else 'MIXED',
        settings=dict(batch_size=8, target_steps=7, burnin_steps=5) if change == 'geometry' else None)
    try:
        assert first.identity_hash != second.identity_hash
        with pytest.raises(ValueError, match='RESTORE_IDENTITY'):
            second.load(first.save())
    finally:
        first.close(); second.close()


def test_f3_missing_support_and_canonical_label_attack(corpus_pair, monkeypatch):
    config, _ = corpus_pair
    auxiliary = specs('MIXED')
    auxiliary['beta'] = dict(mode='STRATIFIED', strata_weights={'NOT_A_CLASS': 1})
    with pytest.raises(ValueError, match='INSUFFICIENT_CURRICULUM_SUPPORT'):
        open_f3(config, auxiliary=auxiliary)
    stream = open_f3(config, 'MIXED')
    try:
        before = stream.save()
        reader = stream.role_streams['NATURAL'].stream.readers['beta']; original = reader.sample
        def altered(*args):
            raw = original(*args)
            raw['teacher_action'][raw['teacher_valid']] = 1-raw['teacher_action'][raw['teacher_valid']]
            return raw
        monkeypatch.setattr(reader, 'sample', altered)
        with pytest.raises(ValueError, match='POST_STEP_LABEL_MISMATCH'):
            stream.sample()
        exact(before, stream.save())
    finally:
        stream.close()


def test_f3_changed_external_source_invalidates_group(corpus_pair, tmp_path):
    config, _ = corpus_pair
    changed = copy.deepcopy(config)
    source = tmp_path/'beta'; shutil.copytree(config['sources'][1]['corpus_root'], source)
    changed['sources'][1]['corpus_root'] = str(source)
    stream = open_f3(changed, 'MIXED', weights=dict(alpha=1, beta=1e-100))
    try:
        member = source/'teacher_sidecar/P0/steps.jsonl.gz'
        member.write_bytes(member.read_bytes()+b'changed')
        with pytest.raises(ValueError, match='CONTENT_CHANGED'):
            stream.sample()
        with pytest.raises(ValueError, match='GROUP_INVALID_REOPEN'):
            stream.sample()
    finally:
        stream.close()


@pytest.mark.parametrize('mode', [None, 'MIXED'])
def test_f3_cli_and_original_checkpoint_cold_processes(corpus_pair, tmp_path, record_property, mode):
    config, _ = corpus_pair
    config = copy.deepcopy(config)
    if mode:
        config['f3_auxiliary_sampling_spec'] = specs(mode)
    path = tmp_path/'f3.json'; path.write_text(json.dumps(config, indent=2)+'\n')
    output = tmp_path/'diagnostic'
    argv = [sys.executable, '-B', str(Path(__file__).resolve().parents[2]/'tools/fg_s3_e_quality_reader.py'),
            'multi-read', '--config', str(path), '--out', str(output), '--adapter', 'f3',
            '--batch-size', '4', '--length', '12', '--burnin-steps', '4']
    result = subprocess.run(argv, capture_output=True, text=True)
    (tmp_path/'cli.log').write_text(result.stdout+result.stderr)
    assert result.returncode == 0, result.stdout[-3000:]+result.stderr[-5000:]
    report = json.loads(result.stdout.splitlines()[-1])
    assert report['adapter_mode'] == 'f3' and report['status'] == 'DEV_PASS'
    assert report['optimizer_updates'] == 0 and report['real_source_acceptance'] == 'NOT_RUN'
    cold = json.loads((output/'CHECKPOINT_COLD_EVIDENCE.json').read_text())
    assert cold['status'] == 'PASS_EXACT_COLD_DATA_STATE' and len(cold['calls']) == 3
    assert cold['runtime_and_pending_queue_equal'] and cold['scheduler_and_all_reader_states_equal']
    assert cold['prefetched_batches'] == 2 and cold['copied_corpus_chunks'] == 0
    loss = json.loads((output/'F3_READONLY_LOSS.json').read_text())
    assert loss['status'] == 'PASS_ORIGINAL_F3_OBJECTIVE' and loss['optimizer_updates'] == 0
    record_property('diagnostic_root', str(output))
    record_property('cli_argv', json.dumps(argv))


def test_f3_cli_requires_explicit_geometry_and_natural_specs(corpus_pair):
    from gym_trading_env.research.fg_s3_e_quality_multi_checks import _diagnostic_stream
    config, path = corpus_pair
    with pytest.raises(ValueError, match='EXPLICIT_BURNIN'):
        _diagnostic_stream(path, adapter='f3', batch_size=4, length=12, burnin_steps=None)
    changed = copy.deepcopy(config); changed['per_source_sampling_spec'] = specs('MIXED')
    with pytest.raises(ValueError, match='NATURAL_SPECS_REQUIRED'):
        open_f3(changed)


def test_f3_natural_statistics_preserve_source_and_global_denominators(corpus_pair):
    config, _ = corpus_pair
    stream = open_f3(config, 'MIXED')
    try:
        external = stream.identity
        external['settings']['target_steps'] = 999
        assert stream.identity['settings']['target_steps'] == 8
        before = stream.save()
        data, audit = stream.sample()
        report = stream.coverage()['natural_distribution']
        assert report['represents_true_market_natural_distribution'] is False
        assert report['source_selection']['basis'] == 'EXPLICIT_CONFIGURED_SOURCE_WEIGHTS'
        assert report['source_selection']['probabilities'] == dict(alpha=.5, beta=.5)
        assert report['loss_or_optimizer_exposure'] == 'NOT_MEASURED_BY_DATA_LAYER'
        global_counts = report['global_counts']
        assert global_counts['windows'] == 8 and global_counts['target_positions'] == 64
        assert sum(global_counts[k] for k in ('teacher_add', 'teacher_wait', 'teacher_invalid')) == 64
        for name, counts in report['per_source'].items():
            positions = [i for i, row in enumerate(audit) if row['role'] == 'NATURAL' and row['corpus_id'] == name]
            y = data['teacher_action'][positions, 5:]; valid = data['teacher_valid'][positions, 5:]
            assert counts['teacher_add'] == np.count_nonzero(valid & (y == 1))
            assert counts['teacher_wait'] == np.count_nonzero(valid & (y == 0))
        assert global_counts['teacher_add'] == sum(c['teacher_add'] for c in report['per_source'].values())
        expected = stream.save(); stream.load(before); stream.sample(); exact(expected, stream.save())
        bad = copy.deepcopy(expected); bad['statistics']['NATURAL']['alpha']['teacher_add'] += 1
        with pytest.raises(ValueError, match='STATISTICS_COUNTER'):
            stream.load(bad)
        exact(expected, stream.save())
    finally:
        stream.close()


def test_f3_pair_owner_keeps_teacher_suffix_domain(corpus_pair):
    from gym_trading_env.envs.entry_contract import fingerprint
    from gym_trading_env.research.fg_replay import trace_manifest, iter_sidecar, E_QUALITY_SCHEMA
    config, _ = corpus_pair
    stream = open_f3(config)
    try:
        data, audit = stream.sample()
        for b, row in enumerate(audit):
            reader = stream.role_streams['NATURAL'].stream.readers[row['corpus_id']]
            original = reader.root/'policies'/row['policy']
            trace = trace_manifest(original, schema=E_QUALITY_SCHEMA)
            sidecar = {native['ordinal']: native for native in iter_sidecar(original)}
            # The original Join binds Corpus, account trace, Teacher and suffix.
            for t, native in enumerate(row['rows'][1:], 1):
                owner = reader.db.execute('SELECT owner FROM eligibility WHERE policy=? AND ordinal=?',
                                          (native['policy'], native['ordinal'])).fetchone()[0]
                expected = fingerprint(dict(protocol='FG_S3_E_QUALITY_PAIR_OWNER_V1',
                    corpus=reader.admission.source.corpus_content_hash, policy=native['policy'],
                    trace=trace['replay_trace_hash'], segment=native['segment'],
                    teacher=reader.corpus['teacher_identity'],
                    suffix=sidecar[native['ordinal']]['teacher']['L2']['suffix_boundary_identity']))
                assert owner == expected
                canonical = json.loads(reader.db.execute('SELECT labels FROM rows WHERE policy=? AND ordinal=?',
                    (native['policy'], native['ordinal'])).fetchone()[0])
                projected = np.asarray(canonical['quality_owner'], np.uint32)
                np.testing.assert_array_equal(data['quality_owner'][b, t], projected)
                # Unsupported labels intentionally retain a zero owner.
                if projected.any():
                    np.testing.assert_array_equal(projected,
                        np.frombuffer(bytes.fromhex(owner), '>u4').astype(np.uint32))
        # Different valid suffix owners within one row cannot form a Pair.
        from gym_trading_env.research.fg_s3_e_quality_multi_checks import f3_loss_evidence
        rows = {k: v[:1].copy() for k, v in data.items()}
        rows['teacher_valid'][...] = True; rows['quality_score_valid'][...] = True
        rows['quality_outcome_valid'][...] = True
        rows['teacher_action'][...] = 0; rows['teacher_action'][:, 5] = 1
        rows['quality_score'][...] = 0; rows['quality_score'][:, 5] = 100
        first = np.frombuffer(bytes.fromhex(fingerprint(dict(suffix='a'))), '>u4').astype(np.uint32)
        second = np.frombuffer(bytes.fromhex(fingerprint(dict(suffix='b'))), '>u4').astype(np.uint32)
        # A controlled loss-only probe, never submitted as an admitted window.
        rows['action_mask'][...] = 1
        for key in ('is_first', 'is_last', 'is_terminal'):
            rows[key][...] = False
        rows['quality_owner'][...] = first
        positive = f3_loss_evidence(rows, 4)['metrics']
        assert positive['f3/strict_pair_count'] > 0 and positive['f3/quality_pair_count'] > 0
        rows['quality_owner'][...] = second; rows['quality_owner'][:, 5] = first
        metrics = f3_loss_evidence(rows, 4)['metrics']
        assert metrics['f3/strict_pair_count'] == metrics['f3/quality_pair_count'] == 0
    finally:
        stream.close()

"""Verify the actual GPU-produced checkpoints; this test runs no neural model."""
import json
from pathlib import Path
import pickle

import numpy as np
import pytest

from embodied.run.fg_checkpoint import file_hash, validate_bundle
from dreamerv3.fg_s3a_actor import parameter_digests
from gym_trading_env.research.fg_replay import read_manifest, signed, write_json


ROOT = Path(__file__).resolve().parents[2]/'artifacts/fg_s3_v2'


def load(node, name):
    with (node/(name+'.pkl')).open('rb') as stream:
        return pickle.load(stream)


def frozen(key):
    return key.startswith(('val/', 'slowval/', 'slowval_count/', 'retnorm/', 'valnorm/', 'advnorm/')) or (
        key.startswith('opt/state/') and '/val/' in key)


def test_actual_p0_gpu_recovery_and_bounded_teacher_snapshot():
    smoke=ROOT/'runs/A2_P0_GPU_SMOKE_1K_FAST_v1'
    train=ROOT/'runs/A2_P0_GPU_TRAIN_10K_FAST_v1'
    if not all((r/'training_receipt.json').is_file() for r in (smoke,train)):
        pytest.skip('Requires actual P0 GPU runs; never substitutes synthetic checkpoints')
    states=[load(smoke/'ckpt'/f'step_{n:012d}','agent') for n in (0,500,1000)]
    digests=[parameter_digests(s['params']) for s in states]
    assert [s['counters']['updates'] for s in states]==[21183,21203,21223]
    for a,b in zip(digests,digests[1:]):
        assert any(a[k]!=b[k] for k in a if k.startswith('pol/'))
        assert any(a[k]!=b[k] for k in a if k.startswith(('enc/','dyn/')))
        assert all(a[k]==b[k] for k in a if frozen(k))
    boundary=load(smoke/'ckpt/step_000000000500','collector')
    final=load(smoke/'ckpt/step_000000001000','collector')
    n=boundary['commit_seq']
    assert boundary['records']==final['records'][:n]
    for key in ('cursor','actual_observation_id','state_id','executed_action',
                'oracle_proof_hash','policy_model_update','beta_draw'):
        assert boundary['pending'][key]==final['records'][n][key]
    receipt=read_manifest(train/'training_receipt.json')
    node=Path(receipt['checkpoint'])
    teacher=load(node,'collector'); replay=load(node,'replay')
    native=set()
    for shard in replay['chunks']:
        with np.load(node/'replay'/shard['name'],allow_pickle=False) as data:
            assert not any(k.startswith(('teacher','oracle','q_')) for k in data.files)
            if shard['length']:
                native.update(x.tobytes().hex() for x in data['stepid'])
    assert native=={r['stepid'] for r in teacher['records']}
    assert len(native)==len(teacher['records'])<teacher['commit_seq']
    assert teacher['counts']['env_steps']==receipt['env_steps']==10000
    assert teacher['pending']['stepid'] is None and teacher['pending_native'] is not None
    assert len((train/'rollout/teacher_journal.jsonl').read_bytes().splitlines())==teacher['commit_seq']
    before=parameter_digests(load(train/'ckpt/step_000000000000','agent')['params'])
    trained=load(node,'agent'); after=parameter_digests(trained['params'])
    assert all(before[k]==after[k] for k in before if frozen(k))
    assert trained['counters']['updates']==21583
    assert load(node,'v2_counters')['all_updates']==receipt['counters']['all_updates']==400
    assert all(after[k]==h for k,h in parameter_digests(trained['policy_params']).items())
    assert trained['pending_sync'] is None
    transfer=read_manifest(train/'source_transfer.json')
    assert before==transfer['params_after'] and not transfer['independent_bc_optimizer_imported']
    spec=read_manifest(ROOT/'prepared/A2_P0_GPU_TRAIN_10K_FAST_v1/source_admission.json')
    source=Path(spec['origin']['path']); meta=json.loads((source/'manifest.json').read_text())
    assert file_hash(source/'manifest.json')==spec['origin']['manifest_sha256']
    assert file_hash(source/'agent.pkl')==meta['files']['agent.pkl']
    write_json(ROOT/'receipts/p0_actual_gpu_verification.json',signed(dict(
        status='PASS_ACTUAL_P0_GPU_ARTIFACTS',gpu_resume='500 -> 1000; 20 further real updates',
        actor_and_wm_continue_learning=True,frozen_critic_optimizer=True,
        retained_teacher_rows=len(native),committed_rows=teacher['commit_seq'],source_unchanged=True,
        uninterrupted_bitwise_equivalence='NOT_RUN',neural_computation='none; saved GPU artifacts only')))


def test_actual_gpu_checkpoints_inheritance_resume_and_learning():
    names = ['A2_GPU_FG201_1K_v2', 'A2_GPU_FG201_1K_TEACHER_v1', 'C2_GPU_FG201_500_v1']
    if any(not (ROOT/'runs'/n/'training_receipt.json').is_file() for n in names):
        pytest.skip('Requires the three actual GPU diagnostics; not replaced with synthetic artifacts')
    results = []
    for name in names:
        run = ROOT/'runs'/name
        identity = read_manifest(run/'run_identity.json')['identity']
        receipt = read_manifest(run/'training_receipt.json')
        transfer = read_manifest(run/'source_transfer.json')
        initial_node = run/'ckpt/step_000000000000'
        final_node = Path(receipt['checkpoint'])
        for node in (initial_node, final_node):
            validate_bundle(node, identity)
        before = parameter_digests(load(initial_node, 'agent')['params'])
        final = load(final_node, 'agent')
        after = parameter_digests(final['params'])
        assert before == transfer['params_after']
        assert transfer['status'] == 'PASS_REAL_MODEL_TRANSFER'
        assert not transfer['independent_bc_optimizer_imported']
        assert set(before) == set(after)
        assert all(before[k] == after[k] for k in before if frozen(k))
        assert any(before[k] != after[k] for k in before if k.startswith('pol/'))
        assert any(before[k] != after[k] for k in before if k.startswith(('enc/', 'dyn/')))
        assert all(after[k] == h for k, h in parameter_digests(final['policy_params']).items())
        assert final['pending_sync'] is None
        count = load(final_node, 'v2_counters')
        assert final['counters']['updates'] == identity['source_update_offset'] + count['all_updates']
        assert count == receipt['counters'] and count['rl_critic_updates'] == count['rl_actor_updates'] == 0
        records = load(final_node, 'collector')
        assert records['commit_seq'] == len(records['records']) == receipt['env_steps']
        assert records['pending']['stepid'] is None and records['pending_native'] is not None
        assert len({r['stepid'] for r in records['records']}) == len(records['records'])
        for row in records['records']:
            if row['teacher_valid']:
                assert row['oracle_valid'] and row['oracle_certified'] and all(row['action_mask'])
                assert row['execution_result']['advanced'] and row['teacher_action'] == row['oracle_action']
        gate = load(final_node, 'action_mask_warmup')
        assert not gate['actor_enabled'] and gate['reports'] == 0
        updates = [json.loads(line) for line in (run/'updates.jsonl').read_text().splitlines()]
        assert len(updates) == count['all_updates']
        assert [r['local_update'] for r in updates] == list(range(1, len(updates)+1))
        assert all(r['metrics']['opt/wm_grad_norm'] > 0 and
                   r['metrics']['opt/critic_grad_norm'] == 0 for r in updates)
        cycles = [json.loads(line) for line in (run/'cycles.jsonl').read_text().splitlines()]
        assert len(cycles) == count['cycles'] and all(r['wm_changed'] for r in cycles)
        if name == names[0]:
            boundary = run/'ckpt/step_000000000500'
            validate_bundle(boundary, identity)
            saved = load(boundary, 'collector')
            assert saved['records'] == records['records'][:500]
            queued = records['records'][500]
            for key in ('cursor', 'actual_observation_id', 'state_id', 'executed_action',
                        'oracle_proof_hash', 'policy_model_update', 'beta_draw'):
                assert saved['pending'][key] == queued[key]
            assert load(boundary, 'v2_counters')['all_updates'] == 4
            assert [r['env_steps'] for r in cycles] == [250,500,750,1000]
        if name == names[1]:
            assert count['all_updates'] == count['bc_valid_updates'] == count['bc_add_updates'] == 40
            assert all(r['actor_changed'] for r in cycles)
            assert all(r['metrics']['opt/actor_grad_norm'] > 0 for r in updates)
            assert len(receipt['actor_proposal_versions']) == 10
        spec = read_manifest(ROOT/'prepared'/name/'source_admission.json')
        source = Path(spec['origin']['path'])
        assert file_hash(source/'manifest.json') == spec['origin']['manifest_sha256']
        source_meta = json.loads((source/'manifest.json').read_text())
        assert file_hash(source/'agent.pkl') == source_meta['files']['agent.pkl']
        results.append(dict(run=name, env_steps=receipt['env_steps'], updates=count['all_updates'],
            bc_updates=count['bc_valid_updates'], wm_changed=True, actor_changed=True,
            frozen_critic_norm_optimizer=True, source_unchanged=True,
            checkpoint_manifest_sha256=file_hash(final_node/'manifest.json')))
        del final, records
    write_json(ROOT/'receipts/actual_checkpoint_verification.json', signed(dict(
        status='PASS_ACTUAL_GPU_CHECKPOINT_ARTIFACTS', neural_computation='none; verifies saved GPU results',
        runs=results, resume_scope='real 500-to-1000 continuation and preserved ownership/pending decision',
        uninterrupted_bitwise_comparison='NOT_RUN', production_gate_20_windows='NOT_RUN')))

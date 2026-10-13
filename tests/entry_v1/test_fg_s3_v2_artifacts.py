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


def test_actual_all_train_beta_gpu_rotation_and_resume():
    from gym_trading_env.research.fg_s3_v2_beta import BetaSchedule
    run=ROOT/'runs/A2_GPU_ALL_TRAIN_BETA_8K_v1'
    if not (run/'training_receipt.json').is_file():
        pytest.skip('Requires the real 18-contract GPU run; no synthetic substitute')
    receipt=read_manifest(run/'training_receipt.json')
    assert receipt['status']=='DIAGNOSTIC_COMPLETE' and receipt['env_steps']==8000
    identity=read_manifest(run/'run_identity.json')['identity']
    boundary=run/'ckpt/step_000000005000'
    final=Path(receipt['checkpoint'])
    for node in (boundary,final):
        validate_bundle(node,identity)
    saved=load(boundary,'collector'); state=load(final,'collector')
    schedule=BetaSchedule(identity['beta'],identity['beta_schedule'])
    assert saved['mixing']['schedule']['env_steps']==5000
    assert saved['mixing']['beta']==schedule.value(5000)<1.
    assert state['mixing']['schedule']==receipt['beta_schedule']
    assert state['mixing']['schedule']['env_steps']==8000 and state['mixing']['beta']==.90
    assert state['records'][:len(saved['records'])]==saved['records']
    queued=state['records'][saved['commit_seq']]
    for key in ('state_id','actual_observation_id','executed_action','oracle_proof_hash',
                'policy_model_update','beta','beta_draw','beta_env_steps'):
        assert queued[key]==saved['pending'][key]
    journal=[json.loads(line) for line in (run/'rollout/teacher_journal.jsonl').read_text().splitlines()]
    assert journal==state['records']
    members=read_manifest(ROOT/'prepared/A2_GPU_ALL_TRAIN_BETA_8K_v1/train_pool.json')['files']
    contracts={r['contract'] for r in members}
    advanced=[r for r in journal if r['execution_result']['advanced']]
    assert len(contracts)==18 and {r['contract'] for r in advanced}==contracts
    assert sum(load(final,'driver')['envs'][0]['env']['advanced'])==8000
    progress=0
    for row in journal:
        assert row['beta_env_steps']==progress and row['beta']==schedule.value(progress)
        progress+=int(row['execution_result']['advanced'])
        if row['teacher_valid']:
            assert row['oracle_valid'] and row['oracle_certified'] and all(row['action_mask'])
            assert row['execution_result']['advanced'] and row['teacher_action']==row['oracle_action']
        assert row['q_add'] is row['q_wait'] is row['q_margin'] is None
    assert progress==8000
    native={}
    for shard in load(final,'replay')['chunks']:
        with np.load(final/'replay'/shard['name'],allow_pickle=False) as data:
            assert not any(k.startswith(('teacher','oracle','q_','beta')) for k in data.files)
            native.update((k.tobytes().hex(),int(a)) for k,a in zip(data['stepid'],data['action']))
    assert set(native)=={r['stepid'] for r in journal}
    assert all(native[r['stepid']]==r['executed_action'] for r in journal)
    assert state['pending']['stepid'] is None and state['pending_native'] is not None
    before=load(boundary,'agent'); after=load(final,'agent')
    a=parameter_digests(before['params']); b=parameter_digests(after['params'])
    assert any(a[k]!=b[k] for k in a if k.startswith('pol/'))
    assert any(a[k]!=b[k] for k in a if k.startswith(('enc/','dyn/')))
    assert all(a[k]==b[k] for k in a if frozen(k))
    assert [s['counters']['updates'] for s in (before,after)]==[21383,21503]
    assert all(b[k]==h for k,h in parameter_digests(after['policy_params']).items())
    cycles=[json.loads(line) for line in (run/'cycles.jsonl').read_text().splitlines()]
    updates=[json.loads(line) for line in (run/'updates.jsonl').read_text().splitlines()]
    assert len(cycles)==80 and len(updates)==320
    assert any(r['actor_changed'] and r['wm_changed'] for r in cycles if r['env_steps']>5000)
    assert receipt['counters']['wm_updates']==320 and receipt['counters']['bc_valid_updates']>200
    assert receipt['formal_1m']=='NOT_RUN' and all('cuda' in d for d in receipt['devices'])
    def coverage(rows):
        legal=[r for r in rows if all(r['action_mask'])]
        return dict(env_steps=len(rows),legal_decisions=len(legal),
            certified_labels=sum(r['teacher_valid'] for r in rows),
            valid_ratio=sum(r['teacher_valid'] for r in rows)/max(len(rows),1),
            valid_legal_ratio=sum(r['teacher_valid'] for r in legal)/max(len(legal),1),
            actor_controlled_legal=sum(not r['oracle_selected'] for r in legal),
            autonomous_executed_add=sum(r['executed_action']==1 and not r['oracle_selected'] for r in rows),
            unproven_funds=sum(r['oracle_reason']=='FUNDS_OUTSIDE_ORIGINAL_CERTIFICATE' for r in rows))
    summary=signed(dict(status='PASS_REAL_GPU_ALL_18_BETA_AND_RESUME',run=str(run),
        source_update_offset=21183,model_updates=21503,updates=320,resumed_updates=120,
        actor_and_wm_continue_learning=True,critic_optimizer_frozen=True,
        beta_checkpoint_boundary=saved['mixing']['schedule'],beta_final=state['mixing']['schedule'],
        contracts=sorted(contracts),coverage=coverage(advanced),
        phases={str(left)+'_'+str(right):coverage([r for r in advanced if left<=r['beta_env_steps']<right])
                for left,right in ((0,3600),(3600,5200),(5200,8000))},
        per_contract={c:coverage([r for r in advanced if r['contract']==c]) for c in sorted(contracts)},
        real_contract_switches=sum(a['contract']!=b['contract'] for a,b in zip(advanced,advanced[1:])),
        replay_sidecar_owners=len(native),counts=receipt['counts'],
        core_env_steps_per_second=receipt['core_env_steps_per_second'],
        process_peak_rss_mib=receipt['process_peak_rss_mib'],
        formal_1m='NOT_RUN',formal_time_axis='NOT_RUN; accelerated beta and 200-bar slices',
        actor_only_evaluation='NOT_RUN; mixed TRAIN measurements only'))
    write_json(ROOT/'receipts/beta_all_train_summary.json',summary)


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

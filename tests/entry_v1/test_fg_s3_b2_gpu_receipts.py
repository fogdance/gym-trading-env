"""Read actual completed CUDA checkpoints; never synthesizes acceptance."""
import json
import os
from pathlib import Path
import pickle

import numpy as np
import pytest

from embodied.run.fg_checkpoint import validate_bundle, file_hash
from dreamerv3.fg_s3a_actor import parameter_digests
from gym_trading_env.research.fg_replay import read_manifest, write_json, signed

PROJECT = Path(__file__).resolve().parents[2]


def frozen(key):
    return key.startswith(('val/','slowval/','slowval_count/','retnorm/','valnorm/','advnorm/')) or (
        key.startswith('opt/state/') and '/val/' in key)


def test_actual_gpu_1k_model_and_full_checkpoint_resume():
    root = Path(os.environ.get('FG_S3_B2_RUN', PROJECT/'artifacts/fg_s3_v2/runs/B2_OUTCOME_GPU_TRAIN_10K_v3'))
    nodes = [root/'ckpt'/f'step_{n:012d}' for n in (0,500,1000)]
    if not all((n/'manifest.json').is_file() for n in nodes):
        pytest.skip('Needs actual B2 CUDA0/500/1000 checkpoints')
    identity = read_manifest(root/'run_identity.json')['identity']
    assert identity['mode']=='b2' and identity['scope']=='diagnostic' and identity['device']=='gpu'
    hashes, collectors, counters = [], [], []
    for updates,node in zip((0,20,40),nodes):
        validate_bundle(node,identity)
        state = pickle.loads((node/'agent.pkl').read_bytes())
        digests = parameter_digests(state['params'])
        assert state['counters']['updates']==updates
        assert int(state['params']['b2_schedule/updates/value'])==updates
        assert int(state['params']['b2_schedule/event_updates/value'])==updates
        if updates:
            policy = parameter_digests(state['policy_params'])
            assert all(policy[k]==digests[k] for k in policy)
            assert all(k in policy for k in digests if k.startswith('b2_'))
        hashes.append(digests)
        del state
        collectors.append(pickle.loads((node/'collector.pkl').read_bytes()))
        counters.append(pickle.loads((node/'v2_counters.pkl').read_bytes()))
    for before,after in zip(hashes,hashes[1:]):
        assert all(before[k]==after[k] for k in before if frozen(k))
        for prefix in ('pol/','enc/','dyn/','b2_event/','b2_early/','b2_adapter/'):
            assert any(before[k]!=after[k] for k in before if k.startswith(prefix)),prefix
    for updates,c in zip((0,20,40),counters):
        assert c['all_updates']==c['wm_updates']==c['outcome_event_updates']==updates
        assert c['rl_actor_updates']==c['rl_critic_updates']==0
    first,last = collectors[1:]
    assert first['records']==last['records'][:500]
    resumed = last['records'][500]
    for key in ('outcome','b2_policy','state_id','actual_observation_id','executed_action','policy_model_update'):
        assert first['pending'][key]==resumed[key]
    assert last['pending']['stepid'] is None and last['pending_native'] is not None
    replay = pickle.loads((nodes[-1]/'replay.pkl').read_bytes())
    native = {}
    for shard in replay['chunks']:
        with np.load(nodes[-1]/'replay'/shard['name'],allow_pickle=False) as data:
            assert not any(k.startswith(('teacher','outcome','b2/')) for k in data.files)
            native.update((s.tobytes().hex(),int(a)) for s,a in zip(data['stepid'],data['action']))
    assert set(native)=={r['stepid'] for r in last['records']}
    assert all(native[r['stepid']]==r['executed_action'] for r in last['records'])
    rows = [json.loads(x) for x in (root/'updates.jsonl').read_text().splitlines()][:40]
    assert [r['local_update'] for r in rows]==list(range(1,41))
    assert all(r['metrics']['opt/outcome_head_grad_norm']>0 and r['metrics']['opt/rssm_grad_norm']>0
        and r['metrics']['opt/actor_grad_norm']>0 and r['metrics']['opt/critic_grad_norm']==0 for r in rows)
    assert any(r['metrics']['b2/mix_value']>0 for r in rows)
    active = [r for r in last['records'] if all(r['action_mask']) and r['b2_policy']['b2/mix_value']>0]
    assert active and any(abs(r['b2_policy']['b2/delta_logit'])>0 for r in active)
    transfer = read_manifest(root/'source_transfer.json')
    assert transfer['status']=='PASS_REAL_PARAMETER_WARM_START' and transfer['devices']==['cuda:0']
    assert transfer['params_after']==hashes[0] and transfer['source_update_offset']==0
    write_json(root/'gpu_1k_verification.json',signed(dict(status='PASS_ACTUAL_GPU_B2_1K_AND_RESUME',
        run=str(root),versions=identity['versions'],env_steps=1000,joint_updates=40,resumed_updates=20,
        source_nn_exactly_inherited=True,critic_rl_and_moments_frozen=True,
        policy_heads_and_scheduler_synchronized=True,full_replay_sidecar_pending_consistent=True,
        mix_active_real_decisions=len(active),outcome_valid=last['counts']['outcome_event_valid'],
        checkpoint=str(nodes[-1]),checkpoint_manifest_sha256=file_hash(nodes[-1]/'manifest.json'),
        formal_1m='NOT_RUN',scope='actual CUDA original-model/original-TRAIN diagnostic; no trading-success claim')))

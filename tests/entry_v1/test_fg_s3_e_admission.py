"""Acceptance bounds are separate from production Composition capability."""
import copy

import pytest

from dreamerv3.fg_s3_composition import normalize_config
from gym_trading_env.research.fg_s3_e_admission import (
    PROTOCOL, validate_settings, _Admission, acceptance_model_config)
from gym_trading_env.research.fg_s3_v2_experiment import _validate_settings


def settings():
    return normalize_config(dict(schema='FG_S3_V2_COMPOSITION_CONFIG_V1',
        composition=dict(encoder='e_multiscale_1m15m',teaching='a2'),
        e_admission=PROTOCOL,scope='diagnostic',source_kind='direct_init',
        device='cpu',batch_size=1,model_init=dict(seed=0,presets=['size50m']),
        contracts=['CZCE.FG201','CZCE.FG301'],env_steps=2200,collect_every=1100,
        updates_per_collect=3,env_slice_bars=1024,replay_capacity=4096),executable=False)


def test_acceptance_bounds_cannot_enable_formal_or_combined_teaching():
    s=settings();validate_settings(s)
    for key,value in [('scope','formal'),('env_steps',1000000),('updates_per_collect',1001),
                      ('model_init',dict(seed=0,presets=['size1m'])),('device','gpu')]:
        bad={**s,key:value}
        with pytest.raises((ValueError,KeyError)):validate_settings(bad)
    bad=copy.deepcopy(s);bad['resolved_composition']['support_state']='waiting_e'
    with pytest.raises(ValueError,match='IDENTITY_MISMATCH'):validate_settings(bad)
    bad=copy.deepcopy(s)
    bad['resolved_composition']=normalize_config(dict(schema=s['schema'],composition=dict(
        encoder='e_multiscale_1m15m',teaching='tutor_cd_b2')),executable=False)['resolved_composition']
    with pytest.raises(ValueError,match='NOT_IMPLEMENTED'):_Admission.composition(bad)
    with pytest.raises(ValueError,match='bounded admit-e'):_validate_settings(s,None)


def test_gpu_acceptance_requires_completed_cpu_full_checkpoint(tmp_path):
    s=settings();s.update(device='gpu',env_steps=1000,collect_every=100,
        updates_per_collect=100,env_slice_bars=250,replay_capacity=2048,
        e_cpu_evidence=str(tmp_path/'absent.json'))
    with pytest.raises(FileNotFoundError):validate_settings(s)


def test_real_cpu_small_network_keeps_canonical_single3_safety_gate():
    from embodied.run.fg_s3_gate import SINGLE3, gate_policy
    cpu = acceptance_model_config(settings())
    gpu = acceptance_model_config({**settings(), 'device': 'gpu'})
    assert cpu['agent']['dyn']['rssm']['deter'] == 32
    assert gpu['agent']['dyn']['rssm']['deter'] == 4096
    assert cpu['action_mask_warmup'] == gpu['action_mask_warmup']
    policy = gate_policy(cpu, diagnostic=True, protocol=SINGLE3)
    assert policy == gate_policy(gpu, diagnostic=True, protocol=SINGLE3)
    assert policy['config']['enabled'] and policy['config']['add_recall_enabled']
    assert cpu['agent']['avail_actor_enabled'] is False


def test_next_update_probe_never_rewinds_a_started_prefetch(tmp_path):
    from types import SimpleNamespace
    import numpy as np
    from gym_trading_env.research.fg_s3_e_admission import _EAdmissionRun
    (tmp_path/'manifest.json').write_text('immutable checkpoint bytes')
    # No live load API: the actual next update is deliberately uncommitted,
    # and a separate cold process must restore the immutable bundle later.
    runner = SimpleNamespace(cp=SimpleNamespace(latest=lambda:tmp_path),
        update=lambda:dict(metrics={'opt/loss':1.}),
        last_batch={'seed':np.array([1],np.uint32)},
        agent=SimpleNamespace(params={'enc/x':np.ones(1,np.float32)},
            save_exact=lambda:dict(counters={'updates':4})))
    value = _EAdmissionRun.probe(runner)
    assert value['checkpoint'] == str(tmp_path)
    assert value['agent_counters']['updates'] == 4
    assert value['metrics']['opt/loss'] == 1.
    assert (tmp_path/'manifest.json').read_text() == 'immutable checkpoint bytes'

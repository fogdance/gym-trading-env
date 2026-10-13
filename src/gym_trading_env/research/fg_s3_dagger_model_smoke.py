"""Actual CPU FullCheckpoint transfer, mixed policy and one original WM update."""
from pathlib import Path
import copy
from functools import partial
import time

import numpy as np

from .fg_replay import check_files, digest, signed, write_json


def model_smoke(config, query, env, output):
    from gym_trading_env.research.fg_s3_teacher_catalog import load_catalog
    from embodied.run.fg_s3_source_admission import inspect_source
    from embodied.run.fg_s3_source_transfer import make_agent, apply_and_audit_transfer
    from dreamerv3.fg_s3a_actor import parameter_digests
    from .fg_s3_dagger_d0 import D0Collector
    from .fg_s3_dagger_bridge import prove_bridge
    started = time.perf_counter()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    original = {str(Path(config['source_checkpoint'])/'manifest.json'):
                digest(Path(config['source_checkpoint'])/'manifest.json'),
                str(Path(config['source_checkpoint'])/'agent.pkl'):
                digest(Path(config['source_checkpoint'])/'agent.pkl')}
    catalog = load_catalog(config['catalog'])
    first = inspect_source('direct_full', catalog=catalog, path=config['source_checkpoint'])
    target = copy.deepcopy(first['target_config'])
    target['agent']['avail_threshold'] = .30
    spec = inspect_source('direct_full', catalog=catalog, path=config['source_checkpoint'],
                          config=target, gate_protocol='FG_SINGLE3_MASK_GATE_V1')
    runtime = dict(spec['target_config'], batch_size=1)
    agent = make_agent(runtime, output/'agent', device='cpu')
    transfer = apply_and_audit_transfer(agent, spec)
    write_json(output/'source_admission.json', spec)
    write_json(output/'transfer.json', transfer)
    before = parameter_digests(agent.params)
    old_updates = int(agent.n_updates)
    old_actions = int(agent.n_actions)
    actor = lambda carry, obs: agent.policy(carry, obs, mode='eval')
    collector = D0Collector(env, query, output/'rollout', config['dreamer_root'],
        mode='mixed', seed=config['seed'], beta=1., actor=actor, agent=agent,
        source_checkpoint_identity=spec['target_model_identity_hash'])
    try:
        collector.advance(100)
        rollout = collector.finish()
        bridge = prove_bridge(collector, catalog)
        if parameter_digests(agent.params) != before or int(agent.n_updates) != old_updates:
            raise RuntimeError('Policy smoke changed training parameters or updates')
        import embodied
        source = embodied.streams.Consec(embodied.streams.Stateless(
            partial(collector.replay.sample, 1)), length=64, consec=1, prefix=1, strict=True)
        stream = agent.stream(source)
        stream.enable_deterministic()
        batch = next(iter(stream))
        with stream.pause():
            _, _, metrics = agent.train(agent.init_train(1), batch)
        after = parameter_digests(agent.params)
        changed = [k for k in before if before[k] != after[k]]
        if int(agent.n_updates) != old_updates+1:
            raise RuntimeError('CPU WM interface did not perform exactly one original update')
        if not any(k.startswith('enc/') for k in changed) or not any(k.startswith('dyn/') for k in changed):
            raise RuntimeError('Encoder/RSSM did not learn on the new trajectory')
        gate = bool(np.asarray(agent.params['avail_actor_gate/value']))
        if gate:
            raise RuntimeError('D0 WM interface unexpectedly activated imagination RL')
        check_files(original)
        result = signed(dict(status='PASS_REAL_CPU_MODEL_INTERFACES', seconds=time.perf_counter()-started,
            source_identity=spec['target_model_identity_hash'], source_files_unchanged=True,
            rollout_env_steps=rollout['env_steps'], source_updates=old_updates,
            updates_after=int(agent.n_updates), policy_calls=int(agent.n_actions)-old_actions,
            runtime_batch_size=1, threshold=.30, gate_ready=gate,
            changed_keys=changed, actor_keys_changed=[k for k in changed if k.startswith('pol/')],
            critic_keys_changed=[k for k in changed if k.startswith('val/')],
            metrics={k:float(v) for k,v in metrics.items() if np.asarray(v).ndim==0},
            source_actor_success_required=False, bc_gradient_updates=0,
            replay_bridge=bridge['status'],
            full_v2_training='NOT_RUN', gpu_training='NOT_RUN', dev='NOT_RUN', test='NOT_RUN'))
        write_json(output/'receipt.json', result)
        return result
    finally:
        collector.close()
        for array in catalog.arrays:
            mapping = getattr(array, '_mmap', None)
            if mapping is not None:
                mapping.close()

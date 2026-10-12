#!/usr/bin/env python3
"""Read-only CPU availability audit of frozen Single3 checkpoints on TRAIN Replay."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['JAX_PLATFORMS'] = 'cpu'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '24')

import argparse
import concurrent.futures
import csv
import gc
import hashlib
import json
import pickle
from pathlib import Path
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
DREAMER = ROOT.parent / 'dreamerv3'
sys.path[:0] = [str(ROOT / 'src'), str(DREAMER)]


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def checkpoint(run, explicit=None):
    from embodied.run.fg_checkpoint import file_hash
    directory = Path(run).resolve() / 'ckpt'
    path = (Path(explicit).resolve() if explicit else
        directory / (directory / 'latest').read_text().strip())
    if path.parent != directory:
        raise ValueError('Checkpoint must belong to the configured experiment stage')
    if not (path / 'done').is_file():
        raise ValueError(f'Incomplete checkpoint: {path}')
    meta = json.loads((path / 'manifest.json').read_text())
    if meta.get('protocol') != 'FG_FULL_V1':
        raise ValueError('Expected a full FG checkpoint')
    # Diagnostics need the model/configuration, not restoration of the financial account.
    for name in ('agent.pkl', 'config.yaml', 'step.pkl'):
        assert file_hash(path / name) == meta['files'][name], name
    assert pickle.loads((path / 'step.pkl').read_bytes()) == meta['step']
    return path, meta


def load_replay(cp, meta, workers):
    import numpy as np
    from embodied.run.fg_checkpoint import file_hash
    assert file_hash(cp / 'replay.pkl') == meta['files']['replay.pkl']
    saved = pickle.loads((cp / 'replay.pkl').read_bytes())
    chunks = saved['chunks']
    assert len(saved['current']) == 1, 'This audit requires one chronological stream'
    assert all(a['succ'] == b['uuid'] for a, b in zip(chunks, chunks[1:]))
    assert chunks[-1]['uuid'] == saved['current'][0][0]
    def read(info):
        name = 'replay/' + info['name']
        assert file_hash(cp / name) == meta['files'][name], name
        with np.load(cp / name, allow_pickle=False) as z:
            return {key: z[key][:info['length']] for key in z.files}
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        parts = list(pool.map(read, chunks))
    keys = set(parts[0])
    assert all(set(part) == keys for part in parts)
    data = {key: np.concatenate([part[key] for part in parts]) for key in keys}
    assert all(np.isfinite(value).all() for value in data.values())
    return data


def model_config(config):
    import elements
    return elements.Config(**config['agent'], **{key: config[key] for key in (
        'logdir', 'seed', 'jax', 'batch_size', 'batch_length', 'replay_context',
        'report_length', 'replica', 'replicas')})


def make_predict(config, data):
    import elements
    import jax
    import jax.numpy as jnp
    import ninjax as nj
    import numpy as np
    from dreamerv3.agent import Agent
    from embodied.jax import internal
    assert config['replay_context'] == 1
    assert config['agent']['replay_context_mode'] == 'recompute_prefix'
    internal.setup(platform='cpu', compute_dtype=config['jax']['compute_dtype'],
        transfer_guard=False, gpuflags=False, prealloc=False)
    assert all(device.platform == 'cpu' for device in jax.devices())
    # Instantiate the existing inner neural model; never create an Env or optimizer update.
    obs_space = {key: elements.Space(data[key].dtype, data[key].shape[1:]) for key in (
        'market_seq', 'position_state', 'action_mask', 'reward',
        'is_first', 'is_last', 'is_terminal')}
    act_space = {'action': elements.Space(np.int32, (), 0, 2)}
    model = object.__new__(Agent)
    Agent.__init__(model, obs_space, act_space, model_config(config))
    def predict(batch):
        # Reuse the exact causal-prefix reconstruction and availability heads from training.
        carry, obs, prevact, _ = model._apply_replay_context(
            model.init_train(batch['action'].shape[0]), batch)
        _, _, tokens = model.enc(carry[0], obs, obs['is_first'], training=False)
        _, _, _, post, prior, _ = model.dyn.loss(
            carry[1], tokens, prevact, obs['is_first'], training=False)
        probs = {}
        for name, feat in (('post', post), ('prior', prior)):
            output = model.avail(model.feat2tensor(feat), 2)
            probs[name] = jax.nn.sigmoid(model._binary_logits(output))
        return probs
    pure = nj.pure(predict)
    return jax.jit(lambda params, batch, seed: pure(params, batch, seed=seed)[1])


def infer(cp, meta, data, starts, config, predict, output, label):
    import embodied
    import jax
    import numpy as np
    from embodied.run.fg_checkpoint import file_hash
    state = pickle.loads((cp / 'agent.pkl').read_bytes())
    selected = {k: v for k, v in state['params'].items()
        if k.startswith(('enc/', 'dyn/', 'avail/'))}
    shapes = {k: list(v.shape) for k, v in selected.items()}
    counters = state['counters']
    assert float(state['params']['avail_actor_gate/value']) == 0
    params = jax.device_put(selected)
    del state, selected
    gc.collect()
    bsize, length = config['batch_size'], config['batch_length']
    collected = {'post': [], 'prior': []}
    t0 = time.monotonic()
    for offset in range(0, len(starts), bsize):
        real = starts[offset:offset + bsize]
        padded = np.pad(real, (0, bsize - len(real)), mode='edge')
        batch = {key: np.stack([value[start:start + length + 1] for start in padded])
            for key, value in data.items()}
        batch = embodied.Replay._annotate_batch(None, batch, None, True)
        assert batch['is_first'][:, 0].all()
        batch['consec'] = np.zeros((bsize, length + 1), np.int32)
        seed = np.random.default_rng([202, offset // bsize]).integers(
            0, np.iinfo(np.uint32).max, 2, dtype=np.uint32)
        result = jax.device_get(predict(params, jax.device_put(batch), jax.device_put(seed)))
        for name in collected:
            assert result[name].shape == (bsize, length, 2)
            assert np.isfinite(result[name]).all()
            collected[name].append(result[name][:len(real)].reshape(-1, 2))
        if offset == 0 or (offset // bsize + 1) % 20 == 0:
            print(label, 'windows', min(offset + bsize, len(starts)), '/', len(starts),
                'seconds', round(time.monotonic() - t0, 1), flush=True)
    result = {key: np.concatenate(value) for key, value in collected.items()}
    np.savez_compressed(output / f'{label}_probabilities.npz', **result)
    # Read-only inference does not mutate the source checkpoint or its counters.
    assert file_hash(cp / 'agent.pkl') == meta['files']['agent.pkl']
    return result, dict(checkpoint=str(cp), step=meta['step'], counters=counters,
        parameter_shapes=shapes, inference_seconds=time.monotonic() - t0)


def metrics(prob, target, threshold):
    import numpy as np
    pred = prob >= threshold
    add = target[:, 1]
    tp = int((pred[:, 1] & add).sum())
    fp = int((pred[:, 1] & ~add).sum())
    positives, negatives = int(add.sum()), int((~target).sum())
    return dict(threshold=float(threshold), add_recall=tp / positives if positives else None,
        fpr=fp / negatives if negatives else None, accuracy=float((pred == target).mean()),
        exact_accuracy=float((pred == target).all(-1).mean()),
        precision=tp / (tp + fp) if tp + fp else None,
        true_positives=tp, false_positives=fp, false_negatives=int(add.sum()) - tp)


def frontier(prob, target, cap, output):
    import numpy as np
    order = np.argsort(prob[:, 1], kind='stable')[::-1]
    scores = prob[order, 1]
    truth = target[order, 1]
    if not truth.any() or truth.all():
        return dict(max_recall_under_fpr_cap=None, minimum_fpr_at_80pct_recall=None,
            jointly_feasible=None, assessable=False)
    last = np.r_[np.flatnonzero(scores[1:] != scores[:-1]), len(scores) - 1]
    tp = np.cumsum(truth)[last]
    fp = np.cumsum(~truth)[last]
    rows = [dict(threshold=float(scores[i]), add_recall=float(t / truth.sum()),
        fpr=float(f / (~truth).sum()), true_positives=int(t), false_positives=int(f))
        for i, t, f in zip(last, tp, fp)]
    with output.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    eligible = [row for row in rows if row['fpr'] <= cap]
    best = max(eligible, key=lambda row: (row['add_recall'], -row['fpr'])) if eligible else None
    recall80 = [row for row in rows if row['add_recall'] >= .8]
    best80 = min(recall80, key=lambda row: row['fpr'])
    return dict(assessable=True, max_recall_under_fpr_cap=best, minimum_fpr_at_80pct_recall=best80,
        jointly_feasible=bool(best and best['add_recall'] >= .8))


def main(args):
    import jax
    import numpy as np
    from dreamerv3.configuration import read_config_yaml
    from embodied.run.fg_checkpoint import file_hash
    from embodied.run.action_mask_warmup import ActionMaskWarmup
    import elements
    started = time.monotonic()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'status.json', dict(status='RUNNING'))
    settings = read_config_yaml(args.config)
    offline_cp, offline_meta = checkpoint(settings['offline']['logdir'], args.offline_checkpoint)
    online_cp, online_meta = checkpoint(settings['online']['logdir'], args.online_checkpoint)
    configs = [read_config_yaml(path / 'config.yaml') for path in (offline_cp, online_cp)]
    assert model_config(configs[0]).update(logdir='audit') == model_config(configs[1]).update(logdir='audit')
    assert configs[0]['batch_size'] == 16 and configs[0]['batch_length'] == 64
    train_path = Path(configs[1]['fg_entry']['manifest'])
    train_manifest = json.loads(train_path.read_text())
    assert all(row['role'] == 'train' for row in train_manifest['files'])
    sources = [Path(__file__), Path(args.config), train_path]
    sources += [DREAMER / name for name in ('dreamerv3/agent.py', 'dreamerv3/rssm.py',
        'dreamerv3/configuration.py', 'embodied/jax/heads.py', 'embodied/jax/nets.py',
        'embodied/jax/outs.py', 'embodied/run/action_mask_warmup.py')]
    source_hashes = {str(path): file_hash(path) for path in sources}
    before = {str(cp): file_hash(cp / 'manifest.json') for cp in (offline_cp, online_cp)}
    data = load_replay(online_cp, online_meta, args.workers)
    length = configs[0]['batch_length']
    starts = np.arange(0, len(data['action']) - length, length, dtype=np.int64)
    if args.limit_batches:
        starts = starts[:args.limit_batches * configs[0]['batch_size']]
    indices = (starts[:, None] + np.arange(1, length + 1)[None]).reshape(-1)
    assert len(indices) == len(np.unique(indices))
    target = data['action_mask'][indices] > .5
    assert target[:, 0].all(), 'FG WAIT must always be legal'
    prev = indices - 1
    previous_held = data['position_state'][prev, 1] > .5
    reset = data['is_first'][indices]
    groups = {'sustained_flat_wait': ~reset & ~previous_held & (data['action'][prev] == 0),
        'held_then_released': ~reset & previous_held,
        'flat_add_then_exit_or_reject': ~reset & ~previous_held & (data['action'][prev] == 1),
        'reset': reset}
    assert np.array_equal(sum(g.astype(int) for g in groups.values()), np.ones(len(indices), int))
    np.savez_compressed(output / 'samples.npz', starts=starts, indices=indices, target=target,
        **groups)
    print('TRAIN rows', len(data['action']), 'audit unique positions', len(indices),
        'ADD positives', int(target[:, 1].sum()), flush=True)
    predict = make_predict(configs[0], data)
    gate = ActionMaskWarmup.for_single3_1m(elements.Config(configs[1]['action_mask_warmup']))
    summary = dict(mode='READ_ONLY_CPU', source_checkpoint_hashes=before,
        sources=source_hashes, actual_gate_opening_thresholds=gate.opening_thresholds,
        gate_required_passes=gate.required_passes, batch_shape=[16, 65], seed=202,
        replay_rows=len(data['action']), audited_unique_positions=len(indices),
        add_positives=int(target[:, 1].sum()), add_prevalence=float(target[:, 1].mean()),
        sample_policy='nonoverlapping 64-position windows with one causal prefix frame',
        native_replay_annotations=True,
        formal_gate_pass_not_certified=True, training_modified=False,
        test_prices_accessed=False, models={})
    for label, cp, meta in zip(('offline', 'online'),
            (offline_cp, online_cp), (offline_meta, online_meta)):
        probs, provenance = infer(cp, meta, data, starts, configs[0], predict, output, label)
        result = dict(provenance=provenance,
            current_threshold={key: metrics(value, target, .5) for key, value in probs.items()},
            frontier=frontier(probs['prior'], target, gate.opening_thresholds['prior_fpr'],
                output / f'{label}_frontier.csv'), cohorts={})
        best = result['frontier']['max_recall_under_fpr_cap']
        if best:
            result['best_under_fpr_cap'] = {
                key: metrics(value, target, best['threshold']) for key, value in probs.items()}
        for name, group in groups.items():
            positive = group & target[:, 1]
            if positive.any():
                scores = probs['prior'][positive, 1]
                result['cohorts'][name] = dict(positives=int(positive.sum()),
                    recall_at_05=float((scores >= .5).mean()),
                    mean_probability=float(scores.mean()),
                    probability_quantiles=np.quantile(scores, [0, .1, .5, .9, 1]).tolist())
        summary['models'][label] = result
        write_json(output / 'analysis.json', summary)
        print(label, json.dumps(result['current_threshold']['prior']),
            'jointly_feasible', result['frontier']['jointly_feasible'], flush=True)
    assert all(file_hash(Path(path)) == value for path, value in source_hashes.items())
    assert all(file_hash(Path(cp) / 'manifest.json') == value for cp, value in before.items())
    summary['devices'] = [str(device) for device in jax.devices()]
    summary['elapsed_seconds'] = time.monotonic() - started
    summary['peak_rss_bytes'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    summary['status'] = 'READ_ONLY_PROBABILITY_AUDIT_COMPLETE'
    write_json(output / 'analysis.json', summary)
    write_json(output / 'status.json', dict(status=summary['status']))
    write_json(output / 'manifest.json', dict(status=summary['status'],
        files={path.name: file_hash(path) for path in output.iterdir() if path.is_file()}))
    print('AUDIT_COMPLETE', round(summary['elapsed_seconds'], 1), 'seconds', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--offline-checkpoint')
    parser.add_argument('--online-checkpoint')
    parser.add_argument('--workers', type=int, default=12)
    parser.add_argument('--limit-batches', type=int, default=0)
    args = parser.parse_args()
    existed = Path(args.output).exists()
    try:
        main(args)
    except Exception as exc:
        if not existed and Path(args.output).is_dir():
            write_json(Path(args.output) / 'status.json', dict(
                status='FAILED', error_type=type(exc).__name__, error=str(exc)))
        raise

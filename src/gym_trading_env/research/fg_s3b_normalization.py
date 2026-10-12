"""V2 frozen population means over unchanged V1 TRAIN weights."""
import copy
from pathlib import Path

import numpy as np

from .fg_replay import check_files, digest, read_manifest, signed, write_json
from .fg_s3b_weights import (audit_weights as audit_v1, bind_feature_cache as bind_v1,
                             code_identity as v1_code_identity)

PROTOCOL = 'FG_S3B_GLOBAL_MEAN_WEIGHTS_V2'
LOSS_VERSION = 'global_train_mean_v2'


def code_identity():
    root = Path(__file__).resolve().parents[3]
    return dict(v1_code_identity(), **{str(Path(__file__).resolve().relative_to(root)): digest(__file__)})


def population_stats(weights):
    w = np.asarray(weights, np.float64)
    if w.ndim != 1 or not len(w) or not np.isfinite(w).all() or (w < 0).any():
        raise ValueError('Invalid frozen training weight population')
    total = float(w.sum(dtype=np.float64))
    mean = total / len(w)
    if not np.isfinite(total) or not np.isfinite(mean) or mean <= 0:
        raise ValueError('Frozen training mean must be finite and positive')
    return dict(population_rows=len(w), sum_weights=total,
                global_mean_weight=mean, global_mean_weight_hex=mean.hex())


def _normalization(base, manifest):
    weights = np.concatenate([np.load(base / s['directory'] / 'weights.npy', allow_pickle=False)
                              for s in manifest['shards']])
    full = population_stats(weights)
    matched = population_stats(weights[weights > 0])
    if full['population_rows'] != manifest['rows'] or matched['population_rows'] != manifest['matched_rows']:
        raise ValueError('Frozen normalization population count differs from weight index')
    interfaces = {}
    for mode, raw, population in [('S3A_FULL', full, 'TRAIN_FULL'),
                                  ('S3A_MATCHED', matched, 'TRAIN_MATCHED'),
                                  ('S3B_CERTIFIED', matched, 'TRAIN_MATCHED')]:
        stats = (raw if mode == 'S3B_CERTIFIED' else
                 dict(population_rows=raw['population_rows'], sum_weights=float(raw['population_rows']),
                      global_mean_weight=1., global_mean_weight_hex=float(1).hex()))
        interfaces[mode] = dict(stats, population=population, loss_version=LOSS_VERSION,
                               sampling='uniform_permutation_of_declared_population')
    return dict(loss_version=LOSS_VERSION, objective='sum_population(w*CE)/sum_population(w)',
                batch_loss='sum_batch((w/global_mean_weight)*CE)/batch_rows',
                raw_weight_populations={'TRAIN_FULL': full, 'TRAIN_MATCHED': matched},
                interfaces=interfaces)


def build_normalization(weights, output):
    """Publish metadata only; do not regenerate weights, labels or Oracle."""
    base, root = Path(weights).resolve(), Path(output).resolve()
    old = audit_v1(base)
    lm = read_manifest(Path(old['labels']) / 'manifest.json')
    for p in (base, Path(old['labels']), Path(old['values']),
              Path(lm['source']['dataset']), Path(lm['source']['oracle_receipt'])):
        if root.is_relative_to(p) or p.is_relative_to(root):
            raise ValueError('Normalization output overlaps immutable inputs')
    if root.exists():
        raise FileExistsError('Frozen normalization output already exists')
    normalization = _normalization(base, old)
    root.mkdir(parents=True)
    write_json(root / 'normalization.json', signed(normalization))
    m = copy.deepcopy(old)
    m.pop('content_hash')
    m.update(protocol=PROTOCOL, base_weights=str(base), base_weights_manifest_sha256=digest(base / 'manifest.json'),
             normalization=normalization, code_sha256=code_identity(),
             files={'normalization.json': digest(root / 'normalization.json')})
    for mode, fields in normalization['interfaces'].items():
        m['interfaces'][mode].update(fields)
    write_json(root / 'manifest.json', signed(m))
    (root / 'done').write_text('')
    return audit_weights(root)


def audit_weights(directory):
    root = Path(directory).resolve()
    m = read_manifest(root / 'manifest.json')
    if (m['protocol'] != PROTOCOL or m['fold'] != 'TRAIN' or m['status'] != 'PASS'
            or not (root / 'done').is_file() or m['code_sha256'] != code_identity()
            or set(m['files']) != {'normalization.json'}):
        raise ValueError('S3-B V2 fixed-mean weight identity required; legacy V1 is not admitted')
    check_files({str(root / p): h for p, h in m['files'].items()})
    base = Path(m['base_weights'])
    if digest(base / 'manifest.json') != m['base_weights_manifest_sha256']:
        raise ValueError('Frozen base weights manifest changed')
    old = audit_v1(base)
    expected = _normalization(base, old)
    if read_manifest(root / 'normalization.json') != signed(expected) or m['normalization'] != expected:
        raise ValueError('Frozen TRAIN population normalization differs')
    inherited = {k: v for k, v in old.items() if k not in ('content_hash', 'protocol', 'files', 'code_sha256', 'interfaces')}
    if any(m.get(k) != v for k, v in inherited.items()):
        raise ValueError('V2 weights differ from immutable V1 provenance')
    for mode, interface in old['interfaces'].items():
        if m['interfaces'][mode] != dict(interface, **expected['interfaces'][mode]):
            raise ValueError('Frozen comparison population/mean identity differs')
    return m


def bind_feature_cache(weights, cache, *, diagnostic=False):
    m = audit_weights(weights)
    _, raw = bind_v1(m['base_weights'], cache, diagnostic=diagnostic)
    return m, raw


def training_normalization(manifest, mode, raw_weights, indices, *, diagnostic=False):
    frozen = manifest['normalization']['interfaces'][mode]
    n = len(indices)
    if n == frozen['population_rows']:
        return dict(frozen)
    if not diagnostic:
        raise ValueError('Training population differs from frozen TRAIN normalization')
    # A partial CPU diagnostic has its own fixed population, never a changing
    # batch denominator and never a claim about complete TRAIN sampling.
    w = raw_weights[indices] if mode == 'S3B_CERTIFIED' else np.ones(n, np.float64)
    return dict(population_stats(w), population='DIAGNOSTIC_FEATURE_SUBSET',
                loss_version=LOSS_VERSION, sampling=frozen['sampling'])

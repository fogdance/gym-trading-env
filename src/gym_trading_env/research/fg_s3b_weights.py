"""One exact S3-A index, frozen TRAIN calibration, three fair BC interfaces."""
from collections import Counter
import json
import math
from pathlib import Path

import numpy as np

from .fg_replay import check_files, digest, read_manifest, signed, write_json
from .fg_s3a_labels import load_labels
from .fg_s3b_advantage import audit_values, code_identity as proof_code_identity

PROTOCOL = 'FG_S3B_WEIGHTS_V1'
MODES = ('S3A_FULL', 'S3A_MATCHED', 'S3B_CERTIFIED')
DEFAULT_WEIGHTS = dict(w_max=3., percentile=90., min_total_ess=2560., min_class_ess=32.,
                       max_contract_contribution=.5)


def code_identity():
    return dict(proof_code_identity(), **{str(Path(__file__).resolve().relative_to(Path(__file__).resolve().parents[3])):
                                         digest(__file__)})


def join_s3a_labels(values, labels, *, diagnostic_partial=False):
    root = Path(values).resolve()
    vm = audit_values(root)
    lm = load_labels(labels, verify_sources=False)
    if (digest(Path(labels)/'manifest.json') != vm['identity']['s3a_labels_sha256']
            or (not vm['complete'] and not diagnostic_partial)):
        raise ValueError('Exact S3-A label identity/full supervision required')
    by_contract = {r['contract']:r for r in lm['traces']}
    result = []
    for s in vm['shards']:
        a = np.load(root/s['directory']/'index.npy', mmap_mode='r', allow_pickle=False)
        v = np.load(root/s['directory']/'values.npy', mmap_mode='r', allow_pickle=False)
        rec = by_contract[s['contract']]
        expected = np.load(Path(labels)/rec['labels_file'], mmap_mode='r', allow_pickle=False)
        if not np.array_equal(a, expected[:len(a)]) or len(a) != len(v):
            raise ValueError('Contract/trace/ordinal/cursor/stepid/label/mask join failed')
        selected = np.isin(v['certificate'], ['ADD_BETTER','WAIT_BETTER']) & (v['margin_lower'] > 0)
        if ((selected & ((v['certificate'] == 'ADD_BETTER') != (a['label'] == 1))).any()
                or (v['margin_lower'][~selected] != 0).any()):
            raise ValueError('Certificate direction conflicts with original teacher')
        result.append((s['contract'], a, v, selected))
    return vm, lm, result


def fit_weight_scale(margins, selected, *, percentile=90.):
    margins = np.asarray(margins, np.float64)
    selected = np.asarray(selected, bool)
    if (margins.ndim != 1 or selected.shape != margins.shape or not np.isfinite(margins).all()
            or (margins < 0).any() or not 0 < percentile <= 100):
        raise ValueError('Invalid TRAIN margin calibration')
    known = margins[selected & (margins > 0)]
    if not len(known):
        raise ValueError('INSUFFICIENT_SUPERVISION_COVERAGE: no certified positive TRAIN margins')
    scale = float(np.percentile(known, percentile, method='linear'))
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError('Nonfinite/zero TRAIN weight scale')
    return dict(protocol='FG_S3B_TRAIN_SCALE_V1', fold='TRAIN', percentile=percentile,
                method='numpy_percentile_linear', scale=scale, positive_rows=len(known),
                per_contract_normalization=False, class_reweighting=False)


def scaled_weights(margins, scale, cap):
    if not math.isfinite(scale) or scale <= 0 or not math.isfinite(cap) or cap <= 0:
        raise ValueError('Invalid frozen weight scale/cap')
    m = np.asarray(margins,np.float64)
    if not np.isfinite(m).all() or (m < 0).any():
        raise ValueError('Invalid margins')
    # Compare before division; avoid overflow when finite margin/scale is huge.
    threshold = scale*cap
    clipped = m >= threshold
    result = np.full(m.shape,cap,np.float64)
    result[~clipped] = m[~clipped]/scale
    if not np.isfinite(result).all() or ((m > 0) & (result == 0)).any():
        raise ValueError('Nonfinite/underflowed positive supervision weights')
    return result


def weight_statistics(weights):
    w = np.asarray(weights, np.float64)
    if w.ndim != 1 or not np.isfinite(w).all() or (w < 0).any():
        raise ValueError('Invalid supervision weights')
    total = float(w.sum())
    squares = float(np.square(w).sum())
    return dict(rows=len(w), positive=int((w > 0).sum()), zero=int((w == 0).sum()),
                sum_w=total, sum_w_squared=squares, ess=total*total/squares if squares else 0.,
                quantiles={k:float(np.percentile(w,p)) for k,p in
                    [('P10',10),('P50',50),('P90',90),('P99',99)]} if len(w) else {},
                maximum=float(w.max()) if len(w) else 0.)


def _summary(joined, weights, cfg):
    y = np.concatenate([a['label'] for _,a,_,_ in joined])
    margin = np.concatenate([v['margin_lower'] for _,_,v,_ in joined])
    cert = np.concatenate([v['certificate'] for _,_,v,_ in joined])
    all_stats = weight_statistics(weights)
    classes = {name:weight_statistics(weights[y == value]) for name,value in [('ADD',1),('WAIT',0)]}
    strata = {}
    offset = 0
    for contract, a, v, _ in joined:
        local = weights[offset:offset+len(a)]
        groups = [('contract/'+contract, np.ones(len(a),bool))]
        # A global year/direction breakdown is appended below; here keep the
        # contract-specific subgroups to diagnose discount-scale concentration.
        for name, field in [('year','year'),('direction','direction'),('session','session')]:
            for value in np.unique(a[field]):
                groups.append((contract+'/'+name+'/'+str(value), a[field] == value))
        for name, select in groups:
            m = v['margin_lower'][select]
            strata[name] = dict(weight_statistics(local[select]),
                certificates=dict(Counter(v['certificate'][select].tolist())),
                margin_quantiles={k:float(np.percentile(m,p)) for k,p in
                    [('P10',10),('P50',50),('P90',90),('P99',99)]},
                uncertified_fraction=float((v['certificate'][select] == 'UNCERTIFIED').mean()),
                zero_margin_fraction=float((m == 0).mean()),
                capped_fraction=float((local[select] == cfg['w_max']).mean()))
        offset += len(a)
    index = np.concatenate([a for _,a,_,_ in joined])
    for field in ('year','direction','session'):
        for value in np.unique(index[field]):
            select = index[field] == value
            m = margin[select]
            strata[field+'/'+str(value)] = dict(weight_statistics(weights[select]),
                certificates=dict(Counter(cert[select].tolist())),
                margin_quantiles={k:float(np.percentile(m,p)) for k,p in
                    [('P10',10),('P50',50),('P90',90),('P99',99)]},
                zero_margin_fraction=float((m == 0).mean()))
    contributions = {name.split('/',1)[1]:stats['sum_w']/all_stats['sum_w']
                     for name,stats in strata.items() if name.startswith('contract/')}
    admission = 'PASS'
    if (all_stats['ess'] < cfg['min_total_ess'] or any(c['positive'] == 0 or c['ess'] < cfg['min_class_ess']
                                                   for c in classes.values())):
        admission = 'INSUFFICIENT_SUPERVISION_COVERAGE'
    flags = ['WEIGHT_CONCENTRATION'] if max(contributions.values()) > cfg['max_contract_contribution'] else []
    return dict(all_stats, classes=classes, certificates=dict(Counter(cert.tolist())),
        certified_coverage=all_stats['positive']/len(weights), strata=strata,
        contract_contributions=contributions, maximum_contract_contribution=max(contributions.values()),
        capped_fraction=float((weights == cfg['w_max']).mean()), training_admission=admission, flags=flags,
        teacher_ADD_fraction=float(y.mean()), no_resampling=True, class_reweighting=False)


def build_weight_index(values, labels, output, *, seed=0, weight_config=None,
                       diagnostic_partial=False, config_sha256=None):
    vm, lm, joined = join_s3a_labels(values, labels, diagnostic_partial=diagnostic_partial)
    cfg = dict(DEFAULT_WEIGHTS, **(weight_config or {}))
    if (set(cfg) != set(DEFAULT_WEIGHTS) or not all(math.isfinite(float(x)) for x in cfg.values())
            or cfg['w_max'] <= 0 or cfg['min_total_ess'] < 0 or cfg['min_class_ess'] < 0
            or not 0 < cfg['max_contract_contribution'] <= 1):
        raise ValueError('Invalid/unknown weight configuration')
    margins = np.concatenate([v['margin_lower'] for _,_,v,_ in joined])
    eligible = np.concatenate([s for _,_,_,s in joined])
    scale = fit_weight_scale(margins, eligible, percentile=cfg['percentile'])
    scale.update(values_manifest_sha256=digest(Path(values)/'manifest.json'),
                 label_manifest_sha256=digest(Path(labels)/'manifest.json'), w_max=cfg['w_max'])
    scale = signed(scale)
    # Division could overflow for extreme but finite values. Compare against
    # the clipping threshold before dividing, preserving the same definition.
    weights = np.zeros(len(margins), np.float64)
    ratio = scaled_weights(margins[eligible], scale['scale'], cfg['w_max'])
    if not np.isfinite(ratio).all() or (ratio <= 0).any():
        raise ValueError('Nonfinite/underflowed positive supervision weights')
    weights[eligible] = ratio
    summary = _summary(joined, weights, cfg)
    root = Path(output).resolve()
    for source in (values, labels, lm['source']['dataset'], lm['source']['oracle_receipt']):
        p = Path(source).resolve()
        if root.is_relative_to(p) or p.is_relative_to(root):
            raise ValueError('Weights output overlaps immutable sources')
    root.mkdir(parents=True, exist_ok=False)
    write_json(root/'scale.json', scale)
    write_json(root/'audit.json', summary)
    shards, offset = [], 0
    for contract,a,v,selected in joined:
        directory = root/'contracts'/contract
        directory.mkdir(parents=True)
        np.save(directory/'index.npy', a, allow_pickle=False)
        np.save(directory/'weights.npy', weights[offset:offset+len(a)], allow_pickle=False)
        np.save(directory/'matched.npy', selected, allow_pickle=False)
        shards.append(dict(contract=contract,directory='contracts/'+contract,rows=len(a),
                           matched_rows=int(selected.sum()), index_sha256=digest(directory/'index.npy')))
        offset += len(a)
    matched = np.flatnonzero(eligible).astype(np.int64)
    np.save(root/'matched_global.npy', matched, allow_pickle=False)
    manifest = signed(dict(protocol=PROTOCOL, status='PASS', fold='TRAIN', seed=seed,
        complete=vm['complete'], same_supervision_index_as_s3a=vm['complete'],
        diagnostic_partial=diagnostic_partial, values=str(Path(values).resolve()),
        values_manifest_sha256=digest(Path(values)/'manifest.json'), labels=str(Path(labels).resolve()),
        labels_manifest_sha256=digest(Path(labels)/'manifest.json'), config=cfg, config_sha256=config_sha256,
        scale_sha256=digest(root/'scale.json'), rows=len(weights), matched_rows=len(matched),
        matched_index_sha256=digest(root/'matched_global.npy'), shards=shards,
        interfaces={
            'S3A_FULL':dict(rows=len(weights), mask='all_exact_S3A_legal_rows', weight='ones'),
            'S3A_MATCHED':dict(rows=len(matched),mask_sha256=digest(root/'matched_global.npy'),weight='ones'),
            'S3B_CERTIFIED':dict(rows=len(matched),mask_sha256=digest(root/'matched_global.npy'),
                                 weight='min(w_max,margin_lower/global_TRAIN_P90)')},
        feature_binding='REQUIRED_AT_TRAIN_ADMISSION', code_sha256=code_identity(), summary=summary,
        files={str(p.relative_to(root)):digest(p) for p in root.rglob('*') if p.is_file()}))
    write_json(root/'manifest.json', manifest)
    (root/'done').write_text('')
    return audit_weights(root)


def audit_weights(directory, *, verify_sources=True):
    root = Path(directory).resolve()
    m = read_manifest(root/'manifest.json')
    if (m['protocol'] != PROTOCOL or m['status'] != 'PASS' or m['fold'] != 'TRAIN'
            or not (root/'done').is_file() or m['code_sha256'] != code_identity()
            or digest(Path(m['labels'])/'manifest.json') != m['labels_manifest_sha256']
            or digest(Path(m['values'])/'manifest.json') != m['values_manifest_sha256']):
        raise ValueError('Weight/Oracle/S3-A/code identity mismatch')
    check_files({str(root/p):h for p,h in m['files'].items()})
    vm, lm, joined = join_s3a_labels(m['values'], m['labels'], diagnostic_partial=m['diagnostic_partial'])
    scale = read_manifest(root/'scale.json')
    expected = fit_weight_scale(np.concatenate([v['margin_lower'] for _,_,v,_ in joined]),
                                np.concatenate([s for _,_,_,s in joined]), percentile=m['config']['percentile'])
    if (scale['scale'] != expected['scale'] or scale['positive_rows'] != expected['positive_rows']
            or scale['values_manifest_sha256'] != m['values_manifest_sha256']
            or scale['label_manifest_sha256'] != m['labels_manifest_sha256']
            or digest(root/'scale.json') != m['scale_sha256']):
        raise ValueError('Frozen global TRAIN calibration differs')
    weights = []
    if len(m['shards']) != len(joined):
        raise ValueError('Weight shard set differs from value proof')
    for shard, (contract,a,v,selected) in zip(m['shards'], joined):
        p = root/shard['directory']
        saved = np.load(p/'index.npy', allow_pickle=False)
        w = np.load(p/'weights.npy', allow_pickle=False)
        mask = np.load(p/'matched.npy', allow_pickle=False)
        expected_w = np.zeros(len(a),np.float64)
        expected_w[selected] = scaled_weights(v['margin_lower'][selected],scale['scale'],m['config']['w_max'])
        if (shard['contract'] != contract or not np.array_equal(saved,a) or w.dtype != np.float64
                or w.shape != a.shape or not np.array_equal(mask,selected) or not np.array_equal(w,expected_w)):
            raise ValueError('Weight exact index/mask/formula mismatch')
        weights.append(w)
    all_w = np.concatenate(weights)
    matched = np.load(root/'matched_global.npy', allow_pickle=False)
    if (len(all_w) != m['rows'] or not np.array_equal(matched,np.flatnonzero(all_w > 0))
            or len(matched) != m['matched_rows'] or _summary(joined,all_w,m['config']) != m['summary']
            or m['summary'] != json.loads((root/'audit.json').read_text())):
        raise ValueError('Weight coverage/ESS/matched interface audit differs')
    if (m['complete'] != vm['complete'] or m['same_supervision_index_as_s3a'] != vm['complete']
            or m['interfaces']['S3A_FULL']['rows'] != len(all_w)
            or any(m['interfaces'][mode]['rows'] != len(matched)
                   or m['interfaces'][mode]['mask_sha256'] != m['matched_index_sha256']
                   for mode in ('S3A_MATCHED','S3B_CERTIFIED'))):
        raise ValueError('Fair comparison interface identity differs')
    return m


def bind_feature_cache(weights, cache, *, diagnostic=False):
    """Final causal feature identity join; proof fields NEVER enter model input."""
    root = Path(weights).resolve()
    m = audit_weights(root)
    if (cache.manifest['labels_manifest_sha256'] != m['labels_manifest_sha256']
            or (not diagnostic and (not cache.manifest['complete'] or not m['complete']))):
        raise ValueError('Frozen feature/complete S3-A index identity mismatch')
    records = {s['contract']:s for s in m['shards']}
    positions, result = Counter(), []
    previous = None
    for shard, a in zip(cache.manifest['shards'], cache.index):
        contract = shard['contract']
        if contract not in records or (previous is not None and contract < previous):
            raise ValueError('Feature contract escaped ordered TRAIN index')
        p = root/records[contract]['directory']
        expected = np.load(p/'index.npy', mmap_mode='r', allow_pickle=False)
        w = np.load(p/'weights.npy', mmap_mode='r', allow_pickle=False)
        offset = positions[contract]
        if not np.array_equal(a,expected[offset:offset+len(a)]):
            raise ValueError('Feature stepid/ordinal/cursor/label/mask join differs')
        result.append(w[offset:offset+len(a)])
        positions[contract] += len(a)
        previous = contract
    flat = np.concatenate(result)
    if len(flat) != cache.rows or (not diagnostic and cache.rows != m['rows']):
        raise ValueError('Feature/weight full row count mismatch')
    return m, flat

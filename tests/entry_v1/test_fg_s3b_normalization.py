"""Fixed population means bind existing arrays; no new Oracle computation."""
import shutil

import numpy as np
import pytest

from test_fg_s3b_weights import packages
from gym_trading_env.research.fg_replay import digest,read_manifest,signed,write_json
from gym_trading_env.research.fg_s3b_normalization import (
    audit_weights,build_normalization,population_stats,training_normalization)


def test_full_matched_population_means_and_frozen_identity(packages,tmp_path):
    base = packages['weights']; before = digest(base/'manifest.json')
    root = tmp_path/'fixed_mean'
    m = build_normalization(base,root)
    raw = np.concatenate([np.load(base/s['directory']/'weights.npy',allow_pickle=False)
                          for s in packages['wm']['shards']])
    full = m['normalization']['raw_weight_populations']['TRAIN_FULL']
    matched = m['normalization']['raw_weight_populations']['TRAIN_MATCHED']
    assert full == population_stats(raw) and matched == population_stats(raw[raw > 0])
    assert m['interfaces']['S3A_FULL']['global_mean_weight'] == 1.
    assert m['interfaces']['S3A_MATCHED']['global_mean_weight'] == 1.
    assert m['interfaces']['S3B_CERTIFIED']['global_mean_weight'] == matched['global_mean_weight']
    assert digest(base/'manifest.json') == before
    with pytest.raises(FileExistsError): build_normalization(base,root)
    with pytest.raises(ValueError,match='V2 fixed-mean'):
        audit_weights(base)
    with pytest.raises(ValueError,match='population differs'):
        training_normalization(m,'S3B_CERTIFIED',raw,np.array([0]))
    positive = np.flatnonzero(raw > 0)[:1]
    local = training_normalization(m,'S3B_CERTIFIED',raw,positive,diagnostic=True)
    if len(positive) != matched['population_rows']:
        assert local['population'] == 'DIAGNOSTIC_FEATURE_SUBSET'
        assert local['global_mean_weight'] == raw[positive].mean()
    changed = tmp_path/'changed'; shutil.copytree(root,changed)
    bad = read_manifest(changed/'manifest.json'); bad.pop('content_hash')
    bad['normalization']['interfaces']['S3B_CERTIFIED']['global_mean_weight'] *= 2
    write_json(changed/'normalization.json',signed(bad['normalization']))
    bad['files']['normalization.json'] = digest(changed/'normalization.json')
    write_json(changed/'manifest.json',signed(bad))
    with pytest.raises(ValueError,match='normalization differs'):
        audit_weights(changed)

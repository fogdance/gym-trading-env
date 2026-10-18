"""Two independently collected native synthetic Corpora, never real TRAIN."""
import json
from pathlib import Path

from gym_trading_env.research.fg_replay import digest
from gym_trading_env.research.fg_s3_e_quality_reader import build_join_index
from gym_trading_env.research.fg_s3_e_quality_supervision import (
    SupervisionSource, QualitySupervisionAdmission)
from wp02_fixture import native_factory
from wp03_fixture import native_corpus


def build_sources(root):
    import pandas as pd
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    sources = []
    for number, name in enumerate(('alpha', 'beta')):
        market = root/(name+'_market')
        bundle = native_factory(market, minutes=40, chunk=45)
        if number:
            path = market/'market.csv'
            frame = pd.read_csv(path)
            frame[['Open', 'High', 'Low', 'Close']] += 17
            frame.to_csv(path, index=False)
            bundle[2].source_receipt['sources']['CZCE.FG2701']['market']['sha256'] = digest(path)
        corpus = native_corpus(root/name, factory_bundle=bundle)
        admission = QualitySupervisionAdmission.preflight(SupervisionSource.from_corpus(corpus))
        index = root/(name+'.sqlite')
        build_join_index(admission, index)
        sources.append(dict(corpus_id=name, corpus_root=str(corpus), index_path=str(index)))
    config = dict(schema='FG_S3_E_MULTICORPUS_CONFIG_V1', sources=sources, stage='WM_NATURAL',
        per_source_sampling_spec={s['corpus_id']: dict(mode='NATURAL') for s in sources},
        source_mix_weights={s['corpus_id']: 1. for s in sources}, seed=202, cache_bytes=1 << 20)
    path = root/'sources.json'
    path.write_text(json.dumps(config, indent=2)+'\n')
    return path


def descriptors(config, stage='WM_NATURAL'):
    from gym_trading_env.research.fg_s3_e_quality_multi import STAGES
    return [dict(corpus_id=s['corpus_id'], index_path=s['index_path'],
        admission=QualitySupervisionAdmission.preflight(
            SupervisionSource.from_corpus(s['corpus_root'], stage=STAGES[stage])))
        for s in config['sources']]


def open_stream(config, *, stage='WM_NATURAL', specs=None, weights=None, seed=202, sources=None):
    from gym_trading_env.research.fg_s3_e_quality_multi import MultiCorpusCourseStream
    return MultiCorpusCourseStream.open(sources or descriptors(config, stage), stage=stage,
        per_source_sampling_spec=specs or config['per_source_sampling_spec'],
        source_mix_weights=weights or config['source_mix_weights'], seed=seed,
        cache_bytes=config['cache_bytes'])


def exact(a, b):
    import numpy as np
    if isinstance(a, dict):
        assert set(a) == set(b)
        for key in a:
            exact(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for left, right in zip(a, b):
            exact(left, right)
    elif isinstance(a, np.ndarray):
        assert a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()
    else:
        assert a == b

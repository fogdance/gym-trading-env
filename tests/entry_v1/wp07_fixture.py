"""Persistent, explicitly synthetic F3 evidence; never a FORMAL permit."""
import copy
import json
import os
from functools import lru_cache
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT/'artifacts/fg_s3_v2/e_quality_wp07'


@lru_cache(maxsize=1)
def f3_source():
    root = Path(os.environ.get('FG_WP07_SOURCE_ROOT', ARTIFACTS/'synthetic_source_01'))
    saved = root/'f3_settings.json'
    if saved.exists():
        return json.loads(saved.read_text())
    from wp04_fixture import synthetic_foundation
    from test_wp06_contract import settings
    from gym_trading_env.research.fg_s3_e_quality_actor import preflight
    from gym_trading_env.research.fg_s3_e_quality_reader import build_join_index
    from embodied.run.fg_s3_e_quality_actor import ActorRun, engineering_permit
    f1 = synthetic_foundation(root/'f1', multiscale=True)
    s = settings(Path(f1['corpus_root']))
    s.update(source_run=f1['run_dir'], source_checkpoint=str(Path(f1['run_dir'])/'ckpt/step_000000000004'),
        prepared_dir=str(root/'f3_prepared'), run_dir=str(root/'f3'), target_steps=8, burnin_steps=4)
    p = preflight(s); index = root/'f3_index.sqlite'; build_join_index(p.source,index)
    run = ActorRun(p,index,root/'f3',_probe=engineering_permit(p))
    try:
        run.run()
    finally:
        run.source.close()
    saved.write_text(json.dumps(s,indent=2)+'\n')
    return s


def execution(root, mode='SHADOW'):
    from dreamerv3.fg_s3_e_quality_online import PROTOCOL
    s = f3_source()
    return dict(protocol=PROTOCOL, source_stage='ACTOR_SUPERVISED', execution_mode=mode,
        fold='TRAIN', source_run=s['run_dir'], source_checkpoint=str(Path(s['run_dir'])/'ckpt/step_000000000004'),
        run_dir=str(root),model_seed=303,env_seed=202,replay_seed=404,env_steps=128,
        collect_every=16,updates_per_collect=0 if mode=='SHADOW' else 2,
        max_consumed_batches=0 if mode=='SHADOW' else 32,save_every=32,replay_capacity=512,
        exploration_rate=0.,history_ratio=.5)


def exact(a,b):
    import numpy as np
    if isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:exact(a[k],b[k])
    elif isinstance(a,(tuple,list)):
        assert type(a)==type(b) and len(a)==len(b)
        for x,y in zip(a,b):exact(x,y)
    elif isinstance(a,np.ndarray):
        assert a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
    else:assert a==b

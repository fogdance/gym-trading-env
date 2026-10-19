"""Explicitly synthetic, native accounts and frozen models; no DEV claims."""
from pathlib import Path
import json
import os

RELEASE=Path('/home/v/Documents/work/research/fg-s3-e-quality-wp07-release-20261019/gym-trading-env/artifacts/fg_s3_v2/e_quality_wp07')
ARTIFACTS=Path(__file__).resolve().parents[2]/'artifacts/fg_s3_v2/e_quality_wp08'


def f3_reference():
    path=Path(os.environ.get('FG_WP08_F3_SETTINGS',RELEASE/'synthetic_source_01/f3_settings.json'))
    return json.loads(path.read_text())


def controlled_actor(output,*,add=True):
    from gym_trading_env.research.fg_s3_e_quality_actor import create_agent
    from gym_trading_env.research.fg_s3_e_quality_exam_actor import ExamFrozenActor
    import jax.numpy as jnp
    agent=create_agent(f3_reference(),output,controlled_small=True)
    for k,v in agent.params.items():
        if k.startswith('quality_actor/') and k.endswith('/kernel') and v.shape[-1]==2:
            agent.params[k]=jnp.zeros_like(v)
        if k.startswith('quality_actor/') and k.endswith('/bias') and v.shape==(2,):
            agent.params[k]=jnp.array([0.,2.] if add else [2.,0.],dtype=v.dtype)
    return ExamFrozenActor(agent,dict(scope='BUILTIN_SYNTHETIC_FIXED_ACTOR_NOT_TRAINED',kind='SYNTHETIC'))

from gym_trading_env.research.fg_s3_e_quality_exam_synthetic import native_source

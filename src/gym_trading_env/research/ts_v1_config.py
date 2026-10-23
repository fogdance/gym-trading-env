"""Thinking Student shared configuration locations."""
from pathlib import Path


SHARED_ROOT = Path('/data/logdir/fg_single3/shared')
TRAIN_POOL = SHARED_ROOT / 'pools/fg_train18_quality_tp_v1/train_pool.json'
ENV_CONFIGS = TRAIN_POOL.parent / 'envs'

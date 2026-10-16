"""Bounded CPU parallelism for independent native-account teacher queries.

The Driver still advances each financial account in order. Only posterior
queries run concurrently; workers reconstruct every history row and invoke
the original Env restore, source guards, L1 and L2. No training entrypoint.
"""
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

import numpy as np


def _initialize(row, profile, initial, source, strict, teacher_identity):
    from .fg_s3_e_quality_corpus_cli import _factory
    from .fg_s3_quality_teacher_v2 import NativeQualityTeacher
    from .fg_s3_e_quality_corpus import l1_labeler
    global _env, _teacher, _label, _history
    factory = _factory(row, profile)
    _env = factory()
    _env.import_state(initial)
    _teacher = NativeQualityTeacher(_env, factory, fold='TRAIN', source=source)
    if _teacher.identity_hash != teacher_identity:
        raise ValueError('Parallel teacher differs from the native starting account identity')
    _label = l1_labeler(_teacher, strict)
    _history = initial['equity_history']


def _query(state, suffix, mask):
    # Transport deduplication only: the original complete financial state is
    # restored, including the full native WAIT history before the sample.
    state['equity_history'] = _history + suffix
    _env.import_state(state)
    if not np.array_equal(mask, _env._published_mask):
        raise ValueError('Parallel teacher mask differs from native snapshot')
    return _label(_env, None, None)


class NativeLabelPool:
    def __init__(self, row, profile, initial, source, strict, teacher_identity, workers):
        if type(workers) is not int or workers < 1:
            raise ValueError('Positive native label worker count required')
        self._history = initial['equity_history']
        self.queue_depth = 2*workers
        self._pool = ProcessPoolExecutor(max_workers=workers,
            mp_context=multiprocessing.get_context('spawn'), initializer=_initialize,
            initargs=(row, profile, initial, source, strict, teacher_identity))

    def __call__(self, env, step, row):
        state = env.export_state()
        history = state.pop('equity_history')
        count = len(self._history)
        if history[:count] != self._history:
            raise ValueError('Parallel label transport changed complete starting financial history')
        return self._pool.submit(_query, state, history[count:], env._published_mask.copy())

    def close(self):
        self._pool.shutdown(wait=True, cancel_futures=True)

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
    global _env, _teacher, _label, _history, _initial_history, _strict_value_store
    factory = _factory(row, profile)
    _env = factory()
    _env.import_state(initial)
    _teacher = NativeQualityTeacher(_env, factory, fold='TRAIN', source=source)
    if _teacher.identity_hash != teacher_identity:
        raise ValueError('Parallel teacher differs from the native starting account identity')
    _label = l1_labeler(_teacher, strict)
    _strict_value_store = getattr(strict,'_value_store',None)
    from gym_trading_env.utils.native_account import NativeHistory,native_owner,native_source
    _history=NativeHistory.from_public(initial['equity_history'],owner=native_owner(_env),source=native_source(_env))
    _initial_history=_history.version()
    _env.equity_history=_history


def _begin_stream():
    global _history
    _teacher._check_source_stats()
    from gym_trading_env.utils.native_account import NativeHistory
    _history=NativeHistory.from_version(_initial_history)


def _finish_stream():
    _teacher.close()


def _statistics():
    import resource
    usage=resource.getrusage(resource.RUSAGE_SELF)
    market=getattr(_teacher.clone,'_e_market',None)
    result=dict(_teacher.statistics,user_cpu_seconds=usage.ru_utime,
        system_cpu_seconds=usage.ru_stime,peak_rss_bytes=usage.ru_maxrss*1024,
        macd_cached_rows=len(getattr(market,'_macd_cache',{})),
        macd_sparse_checkpoints=len(getattr(market,'_macd_checkpoints',{})))
    if _strict_value_store is not None:
        result['strict_pair_cache']=_strict_value_store._pair_cached.cache_info()._asdict()
    version=_history.version()
    result['private_history']=dict(rows=version.count,sealed_blocks=len(version.blocks),
        tail_rows=0 if version.tail is None else len(version.tail.rows),
        commitment=version.commitment,public_hash='original_sorted_JSON_SHA256')
    return result


def _query(state, delta, mask, expected_snapshot_hash, base_count):
    # Each executor owns one ordered account-history replica. Only newly
    # appended samples travel over IPC; restore still receives ALL samples.
    if len(_history) != base_count:
        raise ValueError('Parallel account history delta is missing or out of order')
    _history.extend(delta)
    from gym_trading_env.utils.native_account import NativeAccountSnapshotView
    view=NativeAccountSnapshotView.capture(_env,state,_history.version())
    _env._restore_private_state(view,source_env=_env)
    if not np.array_equal(mask, _env._published_mask):
        raise ValueError('Parallel teacher mask differs from native snapshot')
    label = _label(_env, None, None)
    # Bind reconstruction to the complete snapshot independently emitted by
    # the Driver/reconciler, including every old history row. A modified or
    # omitted prefix cannot silently reuse this worker's initial history.
    if label['account_snapshot_hash'] != expected_snapshot_hash:
        raise ValueError('Parallel transport changed the complete native account snapshot')
    return label


class NativeLabelPool:
    def __init__(self, row, profile, initial, source, strict, teacher_identity, workers):
        if type(workers) is not int or workers < 1:
            raise ValueError('Positive native label worker count required')
        self._history = initial['equity_history']
        self.queue_depth = 2*workers
        self._pools = [ProcessPoolExecutor(max_workers=1,
            mp_context=multiprocessing.get_context('spawn'), initializer=_initialize,
            initargs=(row, profile, initial, source, strict, teacher_identity)) for _ in range(workers)]
        self._sent = [len(self._history)]*workers
        self._closed = False
        self.statistics = dict(queries=0,delta_history_items=0,full_history_items=0,
                               history_replicas=workers)

    def report_statistics(self):
        # One aggregate per worker after all ordered queries; no per-minute
        # diagnostics, extra financial replay, or enlarged query payloads.
        futures = [pool.submit(_statistics) for pool in self._pools]
        self.statistics['native_workers'] = [future.result() for future in futures]
        return self.statistics

    def __call__(self, env, step, row):
        # Copy only hot state and the delta; the Driver's independently emitted
        # complete JSON SHA below still authenticates the entire old prefix.
        import json
        state=env._snapshot_fields()
        history=state.pop('equity_history')
        state=json.loads(json.dumps(state,default=str))
        worker = self.statistics['queries'] % len(self._pools)
        count = self._sent[worker]
        if len(history) < count:
            raise ValueError('Parallel account history cannot rewind within one stream')
        from gym_trading_env.utils.native_account import NativeHistory
        delta = history.public_slice(count) if isinstance(history,NativeHistory) else history[count:]
        from gym_trading_env.utils.account_fingerprint import account_fingerprint
        expected = row['account_snapshot_hash'] if row is not None else account_fingerprint(env)
        future = self._pools[worker].submit(_query,state,delta,env._published_mask.copy(),expected,count)
        self._sent[worker] = len(history)
        self.statistics['queries'] += 1
        self.statistics['delta_history_items'] += len(delta)
        self.statistics['full_history_items'] += len(history)
        return future

    def close(self):
        if self._closed:return
        self._closed = True
        try:
            for future in [pool.submit(_finish_stream) for pool in self._pools]:future.result()
        finally:
            for pool in self._pools:pool.shutdown(wait=True, cancel_futures=True)

    def begin_stream(self):
        """Explicitly rewind replicas for a new independent account replay."""
        for future in [pool.submit(_begin_stream) for pool in self._pools]:future.result()
        self._sent = [len(self._history)]*len(self._pools)

"""Real native Consec/Prefetch and mixed historical/new teacher contract proof."""
from functools import partial
from pathlib import Path

import numpy as np

from .fg_replay import check_files, signed, write_json
from .fg_s3_dagger_d0 import join_d0_teacher


def prove_bridge(collector, catalog):
    import embodied
    source = embodied.streams.Consec(embodied.streams.Stateless(
        partial(collector.replay.sample, 1)), length=64, consec=1, prefix=1, strict=True)
    prefetch = embodied.streams.Prefetch(source, amount=1)
    prefetch.enable_deterministic()
    current = next(iter(prefetch))
    state = prefetch.save_full()
    assert current['stepid'].shape == (1,65,20)
    # This D0 process exits after verification; the original Prefetch daemon
    # blocks on its bounded semaphore, never launches a training loop.
    path = next(Path(p) for p in sorted(catalog.manifest['frozen_inputs'])
                if '/C1/P4/' in p and p.endswith('.npz'))
    check_files({str(path):catalog.manifest['frozen_inputs'][str(path)]})
    with np.load(path, allow_pickle=False) as data:
        historical = {k:data[k][64:129][None].copy() for k in data.files}
    historical['is_first'][:,0] = True
    historical['consec'] = np.zeros((1,65),np.int32)
    assert set(current)==set(historical)
    mixed = {k:np.concatenate([current[k],historical[k]],axis=0) for k in current}
    joined = join_d0_teacher(mixed, collector.records, catalog=catalog)
    np.savez_compressed(collector.root/'mixed_p4_dagger_train_batch.npz', **joined)
    result = signed(dict(status='PASS_NATIVE_REPLAY_CONSEC_PREFETCH_AND_P4_DAGGER_JOIN',
        batch_shape=list(joined['teacher_action'].shape), prefix=1, effective_length=64,
        new_dagger_valid=int(joined['teacher_valid'][0,1:].sum()),
        historical_p4_valid=int(joined['teacher_valid'][1,1:].sum()),
        prefetch_pending=len(state['pending']),
        old_catalog_sha256=catalog.manifest_sha256,
        oracle_label_need_not_equal_executed_action=True,
        native_replay_contains_teacher=False, full_v2_training='NOT_RUN'))
    write_json(collector.root/'bridge_receipt.json', result)
    return result

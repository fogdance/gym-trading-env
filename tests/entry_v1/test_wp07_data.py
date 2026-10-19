import copy
import json
import os
from pathlib import Path

import numpy as np
import pytest

from wp07_fixture import ARTIFACTS,f3_source,exact


@pytest.fixture(scope='module')
def live():
    import jax.numpy as jnp
    from wp02_fixture import native_factory
    from gym_trading_env.research.fg_s3_e_quality_actor import create_agent
    from gym_trading_env.research.fg_s3_e_quality_online_collector import OnlineCollector
    from gym_trading_env.research.fg_s3_e_quality_online_data import LabelPublisher,OnlineWindowReader
    root=Path(os.environ.get('FG_WP07_C_ROOT',ARTIFACTS/'C_data_01'))
    factory,_,_=native_factory(root/'market',minutes=80,chunk=85)
    a=create_agent(f3_source(),root/'model',controlled_small=True)
    for k in a.params:
        if k.startswith('quality_actor/') and k.endswith('/kernel') and a.params[k].shape[-1]==2:a.params[k]=jnp.zeros_like(a.params[k])
        if k.startswith('quality_actor/') and k.endswith('/bias') and a.params[k].shape==(2,):a.params[k]=jnp.array([2.,0.])
    a.sync_policy_from_train()
    source=dict(fold='TRAIN',scope='BUILTIN_SYNTHETIC_ONLY')
    c=OnlineCollector(factory(),source,root/'rollout',Path(__file__).resolve().parents[3]/'dreamerv3',
        agent=a,seed=202,budget=64,replay_length=13,capacity=256,chunksize=32)
    c.advance(48)
    p=LabelPublisher(c,factory,source)
    r=OnlineWindowReader(c,p,dict(batch_size=4,burnin_steps=4,target_steps=8),seed=202)
    yield root,c,p,r
    p.close();c.close()


def test_pending_is_invalid_not_wait_then_native_post_step_publication(live):
    root,c,p,r=live
    p.scan();assert p.ledger and all(x['status']=='PENDING' for x in p.ledger.values())
    state=r.save();data,_=r.sample()
    assert not data['teacher_valid'].any() and not data['quality_owner'].any()
    assert p.publish(limit=7)==7
    assert p.publish()>0
    seq=p.sequence;assert p.publish()==0 and p.sequence==seq
    r.load(state);data,audit=r.sample()
    assert data['teacher_valid'][:,5:].any()
    labels=[x['label']['L2'] for x in p.ledger.values() if x['status']=='CERTIFIED']
    assert {x['strict_action'] for x in labels}=={0,1}
    assert all(x['action_value_add'] is not None and x['action_value_wait'] is not None for x in labels)
    assert all(not row['action'] for row in c.rows)  # Executed WAIT never manufactures teacher ADD.
    assert all(x['label']['L1']['decision_cursor']==c.index[sid]['cursor']
               for sid,x in p.ledger.items() if x['label'])
    assert any(x['label']['L2']['strict_action']!=c.index[sid]['action']
               for sid,x in p.ledger.items() if x['status']=='CERTIFIED')
    assert not p.consumed  # Prefetch/sampling is not optimizer consumption.
    Path('docs/receipts/wp07/C_publication.json').write_text(json.dumps(dict(
        published=p.sequence,pending=sum(x['status']=='PENDING' for x in p.ledger.values()),
        teacher_add=sum(x['strict_action']==1 for x in labels),teacher_wait=sum(x['strict_action']==0 for x in labels),
        actor_actions='ALL_WAIT_SYNTHETIC_FIXTURE',no_action_derived_labels=True,coverage=r.coverage()),indent=2)+'\n')


def test_live_owner_label_attacks_and_rng_restore(live):
    _,c,p,r=live
    state=r.save();expected=r.sample();r.load(state);exact(expected,r.sample())
    data,_=expected;bad={k:v.copy() for k,v in data.items()};bad['teacher_action'][0,5]=1-bad['teacher_action'][0,5]
    with pytest.raises(ValueError,match='NONCANONICAL'):r.audit_windows(bad,supervision_fields=('teacher_action',))
    row=c.index[data['stepid'][0,5].tobytes().hex()];owner=row['account_owner'];row['account_owner']='f'*64
    try:
        with pytest.raises(ValueError,match='ACCOUNT_OWNER'):r.sample()
    finally:row['account_owner']=owner
    changed=copy.deepcopy(state);changed['identity']='f'*64
    with pytest.raises(ValueError,match='IDENTITY'):r.load(changed)
    saved=p.save();p.load(saved);exact(saved,p.save())


def test_shared_multi_history_and_online_sampler_exact_all_states(live):
    from data_multi_fixture import build_sources,descriptors
    from gym_trading_env.research.fg_s3_e_quality_multi_actor import MultiCorpusF3Stream
    from embodied.run.fg_s3_e_quality_online_stream import OnlineHistoryStream
    root,c,p,r=live
    path=build_sources(root/'history');cfg=json.loads(path.read_text())
    history=MultiCorpusF3Stream.open(descriptors(cfg,'ACTOR_STRICT'),settings=r.settings,
        per_source_sampling_spec=cfg['per_source_sampling_spec'],source_mix_weights=cfg['source_mix_weights'],seed=202)
    mixed=OnlineHistoryStream(history,r,history_ratio=.5,seed=303)
    try:
        before=mixed.save();batches=[next(mixed) for _ in range(8)];after=mixed.save()
        assert all(mixed.counts.values())
        mixed.load(before)
        for expected in batches:exact(expected,next(mixed))
        exact(after,mixed.save())
        assert all(len({x['owner'] for x in row['rows']})==1 for row in r.last_batch_audit)
        Path('docs/receipts/wp07/C_mixed.json').write_text(json.dumps(mixed.coverage(),indent=2)+'\n')
    finally:mixed.close()

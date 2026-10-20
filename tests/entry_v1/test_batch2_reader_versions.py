"""The real shared Reader graph, pending leases and self-contained cold state."""
import copy
import json
import pickle
import pytest
import embodied

from data_multi_fixture import build_sources,exact
from batch2_reader_benchmark import open_case,wait_pending
from gym_trading_env.research.fg_s3_reader_versions import VersionedStream,_hash


@pytest.fixture(scope='module')
def config(tmp_path_factory):
    return json.loads(build_sources(tmp_path_factory.mktemp('batch2_versions')).read_text())


@pytest.mark.parametrize('mode',['f1','f2','f3','f3_aux','multi','multi_f3'])
def test_all_source_topologies_deliver_identical_batches_and_exact_coverage(config,mode):
    a=open_case(config,mode);b=open_case(config,mode);versioned=VersionedStream(b)
    try:
        for _ in range(8):
            exact(next(a),next(versioned));exact(a.save(),b.save())
        assert not versioned.nodes
    finally:a.close();b.close()


@pytest.mark.parametrize('amount',[1,2,4])
@pytest.mark.parametrize('mode',['f1','f3_aux','multi_f3'])
def test_prefetch_full_snapshot_restores_each_pending_and_rng(config,amount,mode):
    a=VersionedStream(open_case(config,mode));b=VersionedStream(open_case(config,mode))
    pa=embodied.streams.Prefetch(a,amount=amount);pb=embodied.streams.Prefetch(b,amount=amount)
    try:
        pa.enable_deterministic();iter(pa)
        for _ in range(4):next(pa)
        wait_pending(pa)
        state=pickle.loads(pickle.dumps(pa.save_full()))
        pb.load_full(state);pb.enable_deterministic();iter(pb)
        for _ in range(8):exact(next(pa),next(pb))
        wait_pending(pa);wait_pending(pb)
        with pa.pause(),pb.pause():
            exact(a.source.save(),b.source.save());exact(pa.save_full(),pb.save_full())
            assert len(a.nodes)<=amount and len(b.nodes)<=amount
    finally:a.close();b.close()


def test_unstarted_prefetch_zero_pending_is_self_contained(config):
    a=VersionedStream(open_case(config,'multi_f3'));b=VersionedStream(open_case(config,'multi_f3'))
    pa=embodied.streams.Prefetch(a,amount=4);pb=embodied.streams.Prefetch(b,amount=4)
    try:
        state=pa.save_full();assert state['pending']==[]
        pb.load_full(pickle.loads(pickle.dumps(state)))
        exact(a.save_full(),b.save_full())
    finally:a.close();b.close()


def test_one_reader_cannot_belong_to_two_active_version_graphs(config):
    source=open_case(config,'f1');owner=VersionedStream(source)
    try:
        with pytest.raises(ValueError,match='EXCLUSIVE_READER_OWNERSHIP'):VersionedStream(source)
    finally:owner.close()


def test_hot_restore_cannot_rewind_a_prefetch_with_restored_pending_batches(config):
    a=VersionedStream(open_case(config,'f1'));b=VersionedStream(open_case(config,'f1'))
    pa=embodied.streams.Prefetch(a,amount=2);pb=embodied.streams.Prefetch(b,amount=2)
    try:
        iter(pa);wait_pending(pa);pb.load_full(pa.save_full())
        before=pb.save_full()
        with pytest.raises(ValueError,match='pending'):
            pb.load(pb.state)
        with pytest.raises(ValueError,match='pending'):
            pb.load_full(before)
        exact(before,pb.save_full())
    finally:a.close();b.close()


@pytest.mark.parametrize('attack',['missing_delta','digest','duplicate_version','wrong_source','wrong_role','missing_parent','pending_order','wrong_capacity','foreign_ref','batch_bytes'])
def test_graph_and_pending_corruption_reject_before_source_mutation(config,attack):
    a=VersionedStream(open_case(config,'multi_f3'));b=VersionedStream(open_case(config,'multi_f3'))
    pa=embodied.streams.Prefetch(a,amount=4);pb=embodied.streams.Prefetch(b,amount=4)
    try:
        iter(pa);wait_pending(pa);state=pa.save_full();bad=copy.deepcopy(state)
        nodes=bad['source']['nodes']
        if attack=='missing_delta':nodes[-1]['delta'].pop()
        elif attack=='digest':nodes[-1]['digest']='f'*64
        elif attack=='duplicate_version':nodes.append(copy.deepcopy(nodes[-1]))
        elif attack=='wrong_source':bad['source']['binding']='f'*64
        elif attack=='wrong_role':nodes[-1]['small']['group']['roles'].reverse()
        elif attack=='missing_parent':nodes[-1]['parent']=10000000
        elif attack=='pending_order':bad['pending'].reverse()
        elif attack=='wrong_capacity':bad['amount']=1
        elif attack=='foreign_ref':bad['pending'][0][1]['session']='f'*32
        else:bad['pending'][0][0]['teacher_action'][...]^=1
        # Rehash the outer graph to reach structural/link validation.
        bad['source']['graph_sha256']=_hash({k:v for k,v in bad['source'].items() if k!='graph_sha256'})
        before=b.source.save()
        with pytest.raises(ValueError):pb.load_full(bad)
        exact(before,b.source.save())
    finally:a.close();b.close()

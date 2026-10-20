"""S3 lifecycle snapshots: local version leases and self-contained full state.

Sampling remains in the existing WP03 Reader and stage/source adapters.
"""
from collections import Counter
import copy
import hashlib
import json
import uuid
import weakref

import numpy as np

from .fg_s3_e_quality_reader import READER_SCHEMA

STATE_PROTOCOL='READER_STREAM_STATE_V2'
PREFETCH_PROTOCOL='PREFETCH_PENDING_STATE_V2'
REF_PROTOCOL='S3_READER_VERSION_REF_V2'
CHECKPOINT_REF_PROTOCOL='S3_READER_CHECKPOINT_REF_V2'


def _hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def _batch_hash(batch):
    h=hashlib.sha256()
    for key in sorted(set(batch)-{'seed'}):
        value=np.asarray(batch[key]);h.update(key.encode());h.update(str((value.shape,value.dtype)).encode())
        h.update(value.tobytes())
    return h.hexdigest()


def _leaves(state):
    if isinstance(state,dict):
        if state.get('protocol')==READER_SCHEMA:
            yield state
        else:
            for value in state.values():yield from _leaves(value)
    elif isinstance(state,(list,tuple)):
        for value in state:yield from _leaves(value)


def _stable(state):
    changing={'rng','scheduler_rng','counts','source_counts','last_batch_audit','statistics'}
    if isinstance(state,dict):return {k:_stable(v) for k,v in state.items() if k not in changing}
    if isinstance(state,(list,tuple)):return [_stable(v) for v in state]
    return state


def _keyset(values):
    if not isinstance(values,list) or any(not isinstance(x,(list,tuple)) or len(x)!=2
            or not isinstance(x[0],str) or type(x[1]) is not int or x[1]<0 for x in values):
        raise ValueError('READER_VERSION_COVERAGE_SCHEMA')
    result={tuple(x) for x in values}
    if len(result)!=len(values):raise ValueError('READER_VERSION_DUPLICATE_COVERAGE')
    return result


class VersionedStream:
    """One version graph for an existing complete S3 source topology."""
    prefetch_state_protocol=PREFETCH_PROTOCOL

    def __init__(self,source):
        self.source=source
        values=source.readers
        self.readers=list(values.values()) if isinstance(values,dict) else list(values)
        if not self.readers or len({id(r) for r in self.readers})!=len(self.readers):
            raise ValueError('READER_VERSION_EXCLUSIVE_READER_OWNERSHIP')
        if any(getattr(r,'_version_owner',lambda:None)() is not None for r in self.readers):
            raise ValueError('READER_VERSION_EXCLUSIVE_READER_OWNERSHIP')
        self.signature=_stable(source.save_small())
        adapter=getattr(source,'adapter',None)
        self.binding=_hash(dict(source_type=type(source).__module__+'.'+type(source).__qualname__,
            topology=self.signature,adapter=type(adapter).__qualname__ if adapter else None,
            settings=getattr(adapter,'settings',None)))
        self.session=uuid.uuid4().hex
        self.base_id=self.current=0;self.next_id=1;self.nodes={};self.pins=Counter()
        self.base_small=source.save_small()
        self.base_seen=[dict(market=set(r.seen_market),account=set(r.seen_account)) for r in self.readers]
        self.base_hash=_hash(dict(binding=self.binding,session=self.session,initial=self.base_small))
        self._expected_checkpoint_ref=None
        self._checkpoint_cache=None
        for reader in self.readers:reader._version_owner=weakref.ref(self)

    def __iter__(self):return self

    def __next__(self):
        self._checkpoint_cache=None
        if any(r._transactions.stack for r in self.readers):
            raise ValueError('READER_VERSION_EXTERNAL_TRANSACTION_ACTIVE')
        mark=self.source.begin_batch()
        try:
            batch=next(self.source)
            small=self.source.save_small()
            delta=[dict(market=[list(x) for x in sorted(r._transactions.stack[-1].market)],
                        account=[list(x) for x in sorted(r._transactions.stack[-1].account)]) for r in self.readers]
            node=dict(version=self.next_id,parent=self.current,small=small,delta=delta,batch_sha256=_batch_hash(batch))
            node['digest']=_hash(dict(parent_digest=self._digest(self.current),**node))
        except BaseException:
            self.source.rollback_batch(mark)
            raise
        self.source.commit_batch(mark)
        self.nodes[self.next_id]=node;self.current=self.next_id;self.next_id+=1
        if not self.pins:self._compact()
        return batch

    def _digest(self,version):return self.base_hash if version==self.base_id else self.nodes[version]['digest']
    def _small(self,version):return self.base_small if version==self.base_id else self.nodes[version]['small']

    def checkpoint_ref(self):return self._ref(self.current)

    def _ref(self,version):
        ref=dict(protocol=REF_PROTOCOL,binding=self.binding,session=self.session,version=version,digest=self._digest(version))
        if 'last_batch_audit' in self._small(version):
            ref['last_batch_audit']=copy.deepcopy(self._small(version)['last_batch_audit'])
        return ref

    def _check_ref(self,ref):
        if (not isinstance(ref,dict) or ref.get('protocol')!=REF_PROTOCOL or ref.get('binding')!=self.binding
                or ref.get('session')!=self.session or type(ref.get('version')) is not int
                or ref['version'] not in {self.base_id,*self.nodes}
                or ref!=self._ref(ref['version'])):
            raise ValueError('READER_VERSION_REFERENCE_SCOPE_OR_DIGEST')

    def save(self):
        ref=self.checkpoint_ref();self.pins[self.current]+=1
        return ref

    def retain_snapshot(self,ref):
        self._check_ref(ref);self.pins[ref['version']]+=1

    def release_snapshot(self,ref):
        self._check_ref(ref);version=ref['version']
        if self.pins[version]<=0:raise ValueError('READER_VERSION_LEASE_NOT_OWNED')
        self.pins[version]-=1
        if not self.pins[version]:del self.pins[version]
        self._compact()

    def _path(self,version):
        path=[]
        while version!=self.base_id:
            path.append(version);version=self.nodes[version]['parent']
        return [self.base_id,*reversed(path)]

    def _compact(self):
        self._checkpoint_cache=None
        paths=[self._path(v) for v in {self.current,*self.pins}]
        common=set(paths[0])
        for path in paths[1:]:common.intersection_update(path)
        anchor=max(common)
        if anchor!=self.base_id:
            for version in self._path(anchor)[1:]:
                for seen,delta in zip(self.base_seen,self.nodes[version]['delta']):
                    for kind in ('market','account'):seen[kind].update(map(tuple,delta[kind]))
            self.base_small=copy.deepcopy(self._small(anchor));self.base_hash=self._digest(anchor);self.base_id=anchor
        keep=set().union(*(set(path[path.index(anchor)+1:]) for path in paths))
        self.nodes={v:node for v,node in self.nodes.items() if v in keep}

    def _seen(self,version):
        seen=[{k:set(v) for k,v in item.items()} for item in self.base_seen]
        for v in self._path(version)[1:]:
            for item,delta in zip(seen,self.nodes[v]['delta']):
                for kind in ('market','account'):
                    keys=_keyset(delta[kind])
                    if item[kind]&keys:raise ValueError('READER_VERSION_DELTA_NOT_NEW')
                    item[kind].update(keys)
        return seen

    def _full_state(self,version):
        state=copy.deepcopy(self._small(version));leaves=list(_leaves(state));seen=self._seen(version)
        if len(leaves)!=len(seen):raise ValueError('READER_VERSION_TOPOLOGY')
        for leaf,item in zip(leaves,seen):
            leaf.update(seen_market=[list(x) for x in sorted(item['market'])],
                        seen_account=[list(x) for x in sorted(item['account'])])
        return state

    def load(self,ref):
        self._check_ref(ref)
        self._checkpoint_cache=None
        self.source.load(self._full_state(ref['version']));self.current=ref['version']

    def save_full(self):
        if self._checkpoint_cache is not None:
            state=self._checkpoint_cache;self._checkpoint_cache=None
            return state
        return self._export_full()

    def _export_full(self):
        if any(r._transactions.stack for r in self.readers):raise ValueError('READER_VERSION_TRANSACTION_OPEN')
        state=dict(protocol=STATE_PROTOCOL,binding=self.binding,session=self.session,next_id=self.next_id,
            base=dict(version=self.base_id,digest=self.base_hash,small=copy.deepcopy(self.base_small),
                seen=[{k:[list(x) for x in sorted(v)] for k,v in item.items()} for item in self.base_seen]),
            nodes=copy.deepcopy([self.nodes[v] for v in sorted(self.nodes)]),current=self.checkpoint_ref())
        return dict(**state,graph_sha256=_hash(state))

    def _parse_full(self,state):
        try:
            if (not isinstance(state,dict) or state.get('protocol')!=STATE_PROTOCOL or state.get('binding')!=self.binding
                    or set(state)!={'protocol','binding','session','next_id','base','nodes','current','graph_sha256'}
                    or _hash({k:v for k,v in state.items() if k!='graph_sha256'})!=state['graph_sha256']
                    or not isinstance(state['session'],str) or len(state['session'])!=32):
                raise ValueError('READER_VERSION_GRAPH_IDENTITY_OR_DIGEST')
            parsed=copy.copy(self);base=state['base']
            parsed.session=state['session'];parsed.base_id=base['version'];parsed.base_hash=base['digest']
            parsed.base_small=copy.deepcopy(base['small']);parsed.nodes={};parsed.pins=Counter()
            if type(parsed.base_id) is not int or parsed.base_id<0:raise ValueError('READER_VERSION_BASE')
            parsed.base_seen=[{k:_keyset(item[k]) for k in ('market','account')} for item in base['seen']]
            if len(parsed.base_seen)!=len(self.readers):raise ValueError('READER_VERSION_SOURCE_COUNT')
            previous=parsed.base_id
            for node in state['nodes']:
                v=node['version']
                if (type(v) is not int or type(node['parent']) is not int or v<=previous
                        or node['parent'] not in {parsed.base_id,*parsed.nodes}
                        or len(node['delta'])!=len(self.readers)):
                    raise ValueError('READER_VERSION_LINK_OR_DUPLICATE')
                raw={k:value for k,value in node.items() if k!='digest'}
                if node['digest']!=_hash(dict(parent_digest=parsed._digest(node['parent']),**raw)):
                    raise ValueError('READER_VERSION_NODE_DIGEST')
                parsed.nodes[v]=copy.deepcopy(node);previous=v
            for v in [parsed.base_id,*parsed.nodes]:
                small=parsed._small(v)
                if _stable(small)!=self.signature:raise ValueError('READER_VERSION_ROLE_OR_SOURCE')
                for leaf in _leaves(small):
                    rng=np.random.default_rng();rng.bit_generator.state=leaf['rng']
                    if any(type(n) is not int or n<0 for n in leaf['counts'].values()):
                        raise ValueError('READER_VERSION_COUNTER')
                self.source.validate_snapshot(parsed._full_state(v))
            parsed._check_ref(state['current']);parsed.current=state['current']['version']
            parsed.next_id=state['next_id']
            if type(parsed.next_id) is not int or parsed.next_id<=max([parsed.base_id,*parsed.nodes]):
                raise ValueError('READER_VERSION_COUNTER_REUSES_ALLOCATED_ID')
            return parsed
        except (KeyError,TypeError,IndexError,AttributeError) as error:
            raise ValueError('READER_VERSION_GRAPH_SCHEMA') from error

    def load_full(self,state):
        parsed=self._parse_full(state)
        self.source.load(parsed._full_state(parsed.current))
        self._checkpoint_cache=None
        for key in ('session','base_id','base_hash','base_small','base_seen','nodes','pins','current','next_id'):
            setattr(self,key,getattr(parsed,key))

    def validate_snapshot(self,state=None):
        self.source.validate_snapshot()
        if state is not None:self._parse_full(state)

    def validate_prefetch(self,state,*,amount=None):
        if (not isinstance(state,dict) or state.get('protocol')!=PREFETCH_PROTOCOL
                or type(state.get('amount')) is not int or state['amount']<1
                or amount is not None and state['amount']!=amount
                or not isinstance(state.get('pending'),list) or len(state['pending'])>state['amount']):
            raise ValueError('PREFETCH_VERSION_SCHEMA_OR_CAPACITY')
        parsed=self._parse_full(state['source']);parsed._check_ref(state['state'])
        last=state['state']['version']
        for item in state['pending']:
            if not isinstance(item,(tuple,list)) or len(item)!=2:raise ValueError('PREFETCH_VERSION_PENDING_SCHEMA')
            batch,ref=item;parsed._check_ref(ref);v=ref['version']
            if v==parsed.base_id or parsed.nodes[v]['parent']!=last:
                raise ValueError('PREFETCH_VERSION_PENDING_ORDER_OR_LINK')
            if not isinstance(batch,dict) or 'stepid' not in batch:raise ValueError('PREFETCH_VERSION_BATCH')
            if _batch_hash(batch)!=parsed.nodes[v]['batch_sha256']:
                raise ValueError('PREFETCH_VERSION_BATCH_BYTES_MISMATCH')
            last=v
        if last!=parsed.current:raise ValueError('PREFETCH_VERSION_SOURCE_FRONTIER')

    def coverage(self):return self.source.coverage()
    def close(self):
        for reader in self.readers:
            if getattr(reader,'_version_owner',lambda:None)() is self:reader._version_owner=lambda:None
        self.source.close()


class CheckpointSource:
    """The other CP member is a bound ref; the runtime owns the sole full graph."""
    def __init__(self,stream):self.stream=stream
    def save(self):
        # FullCheckpoint holds Prefetch.pause across both member saves. Export
        # once here; runtime consumes precisely this same immutable graph.
        graph=self.stream._export_full();self.stream._checkpoint_cache=graph
        return dict(protocol=CHECKPOINT_REF_PROTOCOL,reader_ref=copy.deepcopy(graph['current']),
                    graph_sha256=graph['graph_sha256'])
    def validate_snapshot(self,state=None):
        self.stream.source.validate_snapshot()
        if state is not None:
            if (not isinstance(state,dict) or set(state)!={'protocol','reader_ref','graph_sha256'}
                    or state.get('protocol')!=CHECKPOINT_REF_PROTOCOL or not isinstance(state.get('reader_ref'),dict)
                    or state['reader_ref'].get('protocol')!=REF_PROTOCOL
                    or state['reader_ref'].get('binding')!=self.stream.binding
                    or not isinstance(state.get('graph_sha256'),str) or len(state['graph_sha256'])!=64):
                raise ValueError('READER_CHECKPOINT_REFERENCE_IDENTITY')
            self.stream._expected_checkpoint_ref=copy.deepcopy(state)
    def load(self,state):
        self.validate_snapshot(state)  # Graph installation belongs to runtime.load.


def validate_runtime_snapshot(stream,state,*,amount=None):
    if not isinstance(state,dict) or len(state.get('streams',[]))!=1:
        raise ValueError('READER_CHECKPOINT_RUNTIME_TOPOLOGY')
    stream.validate_prefetch(state['streams'][0],amount=amount)
    if stream._expected_checkpoint_ref is None:
        raise ValueError('READER_CHECKPOINT_REFERENCE_VALIDATION_REQUIRED')
    expected=stream._expected_checkpoint_ref;graph=state['streams'][0]['source']
    if expected['reader_ref']!=graph['current'] or expected['graph_sha256']!=graph['graph_sha256']:
        raise ValueError('READER_CHECKPOINT_CROSS_MEMBER_REFERENCE')


def materialize_prefetch(stream,state):
    """Diagnostic logical state; compares independent Runs without session IDs."""
    stream.validate_prefetch(state)
    parsed=stream._parse_full(state['source'])
    return dict(source=parsed._full_state(parsed.current),state=parsed._full_state(state['state']['version']),
        pending=[(batch,parsed._full_state(ref['version'])) for batch,ref in state['pending']])


# Keep general RuntimeState behavior; add the missing S3 pre-mutation validator.
from embodied.run.fg_checkpoint import RuntimeState


class VersionedRuntimeState(RuntimeState):
    @classmethod
    def from_runtime(cls,runtime,source):
        result=cls.__new__(cls);result.__dict__.update(runtime.__dict__);result.versioned_source=source
        return result
    def validate_snapshot(self,state=None):
        if state is not None:
            validate_runtime_snapshot(self.versioned_source,state,amount=self.streams[0].amount)


def validate_checkpoint_stream(source,checkpoint):
    """Called after strict bundle selection, before constructing the Agent."""
    import pickle
    CheckpointSource(source).validate_snapshot(pickle.loads((checkpoint/'course_reader.pkl').read_bytes()))
    validate_runtime_snapshot(source,pickle.loads((checkpoint/'runtime.pkl').read_bytes()),amount=1)

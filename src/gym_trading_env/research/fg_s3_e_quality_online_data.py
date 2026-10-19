"""Delayed posterior publication and thin windows over the ORIGINAL live Replay."""
from __future__ import annotations

import copy
from types import SimpleNamespace

import numpy as np

from gym_trading_env.envs.entry_contract import fingerprint
from .fg_replay import signed
from .fg_s3_e_quality_reader import audit_physical_row, _join_row
from .fg_s3_e_quality_supervision import project_label


class LabelPublisher:
    """Private original Env/teachers; never callable from the policy path."""
    protocol='FG_S3_E_QUALITY_ONLINE_PUBLICATION_V1'

    def __init__(self,collector,factory,source):
        from .fg_s3_quality_teacher_v2 import NativeQualityTeacher
        from .fg_s3_e_quality_strict import QualityStrictTeacher
        if source.get('fold')!='TRAIN':raise ValueError('ONLINE_TEACHER_TRAIN_ONLY')
        self.collector=collector;self.env=factory();self.source=copy.deepcopy(source)
        if self.env.e_observation_identity!=collector.env.e_observation_identity:
            raise ValueError('ONLINE_TEACHER_SOURCE_OBSERVATION_MISMATCH')
        self.teacher=NativeQualityTeacher(self.env,factory,fold='TRAIN',source=source)
        self.blockers=[]
        try:self.strict=QualityStrictTeacher(self.env,source=source)
        except ValueError as exc:
            self.strict=None;self.blockers.append(str(exc))
        self.identity=fingerprint(dict(protocol=self.protocol,collector=collector.identity_hash,
            teacher=self.teacher.identity_hash,strict=None if self.strict is None else self.strict.identity_hash))
        self.ledger={};self.sequence=0;self.consumed={}

    def scan(self):
        for r in self.collector.rows:
            self.ledger.setdefault(r['stepid'],dict(ordinal=r['ordinal'],status='PENDING',
                publication_seq=None,label=None,account_snapshot_hash=r['account_snapshot_hash']))

    def publish(self,limit=None):
        from .fg_s3_e_quality_corpus import l1_labeler
        self.scan()
        if self.strict is None:return 0
        published=0
        with self.collector.commit_lock:
            for r in self.collector.rows[:-1]:
                entry=self.ledger[r['stepid']]
                if entry['status']!='PENDING':continue
                if limit is not None and published>=limit:break
                snapshot=self.collector.snapshots[r['stepid']]
                if fingerprint(snapshot)!=r['account_snapshot_hash']:
                    raise ValueError('ONLINE_TEACHER_POST_STEP_SNAPSHOT_MISMATCH')
                self.env.import_state(snapshot)
                label=l1_labeler(self.teacher,self.strict)(self.env,None,r)
                self.sequence+=1
                entry.update(status='CERTIFIED' if label['L2']['teacher_valid'] else 'INVALID',
                    publication_seq=self.sequence,label=label)
                published+=1
        return published

    def projection(self,row,step):
        from .fg_s3_quality_catalog_v2 import FIELDS
        snapshot=self.collector.snapshots[row['stepid']]
        expected=fingerprint(dict(run=self.collector.identity_hash,contract=snapshot['contract'],
                                  account_sequence=snapshot['account_sequence']))
        if row['account_owner']!=expected:raise ValueError('ONLINE_ACCOUNT_OWNER_MISMATCH')
        entry=self.ledger.get(row['stepid'])
        if not entry or entry['status']=='PENDING':
            return {**{k:np.zeros(shape,dtype) for k,(dtype,shape) in FIELDS.items()},
                    'teacher_action':np.array(0,np.int32),'teacher_valid':np.array(False)}
        if (entry['ordinal']!=row['ordinal'] or entry['account_snapshot_hash']!=row['account_snapshot_hash']
                or fingerprint(self.collector.snapshots[row['stepid']])!=row['account_snapshot_hash']):
            raise ValueError('ONLINE_PUBLICATION_OWNER_OR_SNAPSHOT_MISMATCH')
        label=entry['label']
        trace=dict(contract=self.env.contract,data_identity=self.env._e_profile.identity,
            identity=dict(teacher_identity=self.teacher.identity_hash,source=self.source))
        _join_row({**row,'teacher':label},step,row['ordinal'],trace,dict(teacher_identity=self.strict.identity_hash))
        segment=sum(r['is_first'] for r in self.collector.rows[:row['ordinal']+1])
        owner=fingerprint(dict(protocol=self.protocol,source=self.identity,account=row['account_owner'],
            segment=segment,suffix=self.strict.identity['suffix_boundary_identity']))
        return project_label(label,row,step,owner)

    def save(self):
        return copy.deepcopy(dict(protocol=self.protocol,identity=self.identity,sequence=self.sequence,
            ledger=self.ledger,consumed=self.consumed))
    def validate_snapshot(self,state=None):
        if state is None:state=self.save()
        if state.get('protocol')!=self.protocol or state.get('identity')!=self.identity:
            raise ValueError('ONLINE_PUBLICATION_IDENTITY_MISMATCH')
        seq=[r['publication_seq'] for r in state['ledger'].values() if r['status']!='PENDING']
        if sorted(seq)!=list(range(1,state['sequence']+1)):
            raise ValueError('ONLINE_PUBLICATION_SEQUENCE_MISMATCH')
        for row in state['ledger'].values():
            if row['status'] not in ('PENDING','INVALID','CERTIFIED'):
                raise ValueError('ONLINE_PUBLICATION_STATUS_INVALID')
            if row['status']=='PENDING' and (row['label'] is not None or row['publication_seq'] is not None):
                raise ValueError('ONLINE_PENDING_LABEL_FORGED')
    def load(self,state):
        self.validate_snapshot(state)
        self.sequence=state['sequence'];self.ledger=copy.deepcopy(state['ledger']);self.consumed=copy.deepcopy(state['consumed'])
    def close(self):
        self.teacher.close();self.env.close()


class OnlineWindowReader:
    """Use native Replay item pointers and original F3 projection/adapter only."""
    stage='ACTOR_STRICT'
    protocol='FG_S3_E_QUALITY_ONLINE_READER_V1'

    def __init__(self,collector,publisher,settings,seed):
        from .fg_s3_e_quality_actor import F3WindowAdapter
        self.collector=collector;self.publisher=publisher;self.settings=copy.deepcopy(settings)
        self.length=1+settings['burnin_steps']+settings['target_steps']
        if collector.replay.length!=self.length:raise ValueError('ONLINE_REPLAY_WINDOW_LENGTH_MISMATCH')
        self.admission=SimpleNamespace(source=SimpleNamespace(stage='ACTOR_SUPERVISED'))
        self.adapter=F3WindowAdapter(self,settings);self.rng=np.random.default_rng(seed)
        self.identity=fingerprint(dict(protocol=self.protocol,collector=collector.identity_hash,
            publication=publisher.identity,settings=settings,seed=seed))
        self.batches=0;self.last_batch_audit=[];self._candidate_key=None;self._candidates=[]

    @property
    def readers(self):return [self]

    def _eligible(self):
        replay=self.collector.replay
        key=(replay.itemid,len(self.collector.rows))
        if key!=self._candidate_key:
            candidates=[]
            for item,pointer in replay.items.items():
                ids=replay._getseq(*pointer,keys=['stepid'])['stepid']
                rows=[self.collector.index[x.tobytes().hex()] for x in ids]
                if (len({r['account_owner'] for r in rows})==1 and not any(r['is_first'] for r in rows[1:])
                        and not any(r['is_last'] for r in rows[:-1])
                        and all(b['ordinal']==a['ordinal']+1 for a,b in zip(rows,rows[1:]))):
                    candidates.append(item)
            self._candidate_key=key;self._candidates=candidates
        return self._candidates

    def sample(self):
        from .fg_s3_quality_catalog_v2 import FIELDS
        with self.collector.commit_lock:
            eligible=self._eligible()
            if not eligible:raise ValueError('INSUFFICIENT_CURRICULUM_SUPPORT: ONLINE_EPISODE_WINDOW')
            before=copy.deepcopy(self.rng.bit_generator.state)
            try:
                windows=[]
                for _ in range(self.settings['batch_size']):
                    item=eligible[int(self.rng.integers(len(eligible)))]
                    raw=self.collector.replay._getseq(*self.collector.replay.items[item])
                    windows.append({k:v.copy() for k,v in raw.items()})
                raw={k:np.stack([w[k] for w in windows]) for k in windows[0]}
                shape=raw['is_first'].shape;raw['consec']=np.zeros(shape,np.int32)
                for k,(dtype,dims) in FIELDS.items():raw[k]=np.zeros((*shape,*dims),dtype)
                raw.update(teacher_action=np.zeros(shape,np.int32),teacher_valid=np.zeros(shape,bool))
                for b in range(shape[0]):
                    for t in range(shape[1]):
                        step={k:v[t] for k,v in windows[b].items()};row=self.collector.index[step['stepid'].tobytes().hex()]
                        projected=self.publisher.projection(row,step)
                        if t:
                            for k,v in projected.items():raw[k][b,t]=v
                data,audit=self.adapter.adapt(raw)
                self.last_batch_audit=audit;self.batches+=1
                return data,audit
            except BaseException:
                self.rng.bit_generator.state=before
                raise

    def audit_windows(self,batch,*,supervision_fields=()):
        result=[]
        for b in range(len(batch['stepid'])):
            rows=[]
            for t,stepid in enumerate(batch['stepid'][b]):
                sid=stepid.tobytes().hex()
                if sid not in self.collector.index:raise ValueError('ONLINE_FOREIGN_REPLAY_OWNER')
                row=self.collector.index[sid];step={k:v[b,t] for k,v in batch.items()}
                audit_physical_row(row,step,row['ordinal'])
                projection=self.publisher.projection(row,step)
                for k in supervision_fields:
                    expected=projection[k] if t else np.zeros_like(projection[k])
                    if not np.array_equal(batch[k][b,t],expected):raise ValueError('ONLINE_NONCANONICAL_LABEL: '+k)
                rows.append(dict(stepid=sid,cursor=row['cursor'],ordinal=row['ordinal'],owner=row['account_owner']))
            if (len({r['owner'] for r in rows})!=1 or any(b['ordinal']!=a['ordinal']+1 for a,b in zip(rows,rows[1:]))
                    or batch['is_first'][b,1:].any() or batch['is_last'][b,:-1].any()):
                raise ValueError('ONLINE_ACCOUNT_OR_EPISODE_BOUNDARY_CROSSED')
            result.append(dict(corpus_id='ONLINE_LIVE',source_identity=self.identity,role='NATURAL',rows=rows))
        return result

    def __iter__(self):return self
    def __next__(self):return self.sample()[0]
    def save(self):return dict(protocol=self.protocol,identity=self.identity,rng=copy.deepcopy(self.rng.bit_generator.state),
        batches=self.batches,last_batch_audit=copy.deepcopy(self.last_batch_audit))
    save_full=save
    def validate_snapshot(self,state=None):
        if state is not None and (state.get('protocol')!=self.protocol or state.get('identity')!=self.identity):
            raise ValueError('ONLINE_READER_IDENTITY_MISMATCH')
    def load(self,state):
        self.validate_snapshot(state);self.rng.bit_generator.state=copy.deepcopy(state['rng'])
        self.batches=state['batches'];self.last_batch_audit=copy.deepcopy(state['last_batch_audit']);self._candidate_key=None
    load_full=load
    def coverage(self):return dict(protocol=self.protocol,batches=self.batches,eligible_windows=len(self._eligible()),
        pending=sum(r['status']=='PENDING' for r in self.publisher.ledger.values()),training_admitted=False)
    def close(self):pass

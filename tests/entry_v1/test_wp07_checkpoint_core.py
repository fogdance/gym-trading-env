import copy
import pickle

import elements
import numpy as np
import pytest

from embodied.run.fg_checkpoint import FullCheckpoint
from embodied.run.fg_s3_e_quality_online_state import ChunkIdentity


class State:
    def __init__(self,value):self.params={'value':np.array(value)}
    def save_exact(self):return copy.deepcopy(self.params)
    def load_exact(self,value):self.params=copy.deepcopy(value)
    save=save_exact
    load=load_exact


def make(root):
    import embodied
    replay=embodied.Replay(length=3,capacity=16,chunksize=4,chunk_identity=ChunkIdentity('stable'),seed=202)
    cp=FullCheckpoint(root,elements.Counter(),dict(protocol='CORE_TEST_ONLY'),(),reserve_gib=0,keep_all=True,strict_latest=True)
    cp.replay=replay;cp.agent=State(123);cp.step=cp._step
    return cp,replay


def test_seeded_native_chunks_and_sampler_continue_exactly_across_cold_restore(tmp_path):
    from wp07_fixture import exact
    a,ra=make(tmp_path/'cp')
    for i in range(7):ra.add(dict(value=np.array(i,np.int32)))
    a._step.increment();a.save()
    b,rb=make(tmp_path/'cp');b.load()
    for i in range(7,14):
        ra.add(dict(value=np.array(i,np.int32)));rb.add(dict(value=np.array(i,np.int32)))
    assert list(ra.chunks)==list(rb.chunks)
    for key in ra.chunks:
        x,y=ra.chunks[key],rb.chunks[key]
        assert x.time==y.time and x.succ==y.succ and x.length==y.length
        exact({k:v[:x.length] for k,v in x.data.items()},{k:v[:y.length] for k,v in y.data.items()})
    exact(ra.sample(4),rb.sample(4));exact(ra.chunk_identity.save(),rb.chunk_identity.save())


def test_corrupt_latest_and_external_identity_refuse_before_model_mutation(tmp_path):
    class Guard:
        def __init__(self):self.identity='original'
        def save(self):return self.identity
        def validate_snapshot(self,value=None):
            if value is not None and value!=self.identity:raise ValueError('EXTERNAL_IDENTITY_CHANGED')
        def load(self,value):self.validate_snapshot(value)
    cp,replay=make(tmp_path/'cp');guard=Guard();cp.course_reader=guard
    cp._step.increment();cp.save()
    guard.identity='changed';cp.agent.params['value']=np.array(999)
    with pytest.raises(ValueError,match='EXTERNAL_IDENTITY_CHANGED'):cp.load()
    assert cp.agent.params['value']==999
    guard.identity='original';cp._step.increment();cp.save()
    (cp.latest()/'agent.pkl').write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='corrupt'):cp.latest()


def test_native_uuid_lookup_survives_different_process_hash_seeds(tmp_path):
    import os,subprocess,sys
    code='''
import sys
from pathlib import Path
import elements, embodied, numpy as np
import embodied.run.fg_checkpoint as cpmod
from embodied.run.fg_s3_e_quality_online_state import ChunkIdentity
class State:
 def __init__(self):self.params={'value':np.array(123)}
 def save(self):return self.params
 def load(self,x):self.params=x
 save_exact=save
 load_exact=load
r=embodied.Replay(length=3,capacity=16,chunksize=4,chunk_identity=ChunkIdentity('stable'),seed=202)
c=cpmod.FullCheckpoint(Path(sys.argv[1]),elements.Counter(),dict(scope='UUID_COLD_CORE_ONLY'),(),reserve_gib=0,keep_all=True,strict_latest=True)
c.replay=r;c.agent=State();c.step=c._step
if sys.argv[2]=='save':
 for i in range(6):r.add(dict(value=np.array(i,np.int32)))
 c._step.increment();c.save()
else:
 if sys.argv[2]=='mutant':cpmod.rehydrate_native_uuids=lambda value:value
 c.load()
 for key in r.chunks:
  fresh=elements.UUID(key.value)
  assert fresh in r.chunks and fresh in r.refs,'FRESH_NATIVE_UUID_LOOKUP_FAILED'
print('PASS')
'''
    records=[]
    for action,seed in [('save','11'),('load','22'),('mutant','22')]:
        result=subprocess.run([sys.executable,'-B','-c',code,str(tmp_path/'cp'),action],
            env={**os.environ,'PYTHONHASHSEED':seed},capture_output=True,text=True)
        records.append(dict(action=action,hash_seed=seed,exit_code=result.returncode,stdout=result.stdout,stderr=result.stderr))
        if action=='mutant':assert result.returncode!=0 and 'FRESH_NATIVE_UUID_LOOKUP_FAILED' in result.stderr
        else:assert result.returncode==0,result.stderr
    import json
    from pathlib import Path
    Path('docs/receipts/wp07/G_uuid_hash_seed.json').write_text(json.dumps(records,indent=2)+'\n')

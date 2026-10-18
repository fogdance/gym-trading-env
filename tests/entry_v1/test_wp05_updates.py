import copy,json
from pathlib import Path
import jax,jax.numpy as jnp,ninjax as nj
import numpy as np
import pytest
from wp05_fixture import engineering_settings
from gym_trading_env.research.fg_s3_e_quality_heads import create_agent,preflight,F2WindowAdapter
from gym_trading_env.research.fg_s3_e_quality_reader import EQualityCourseReader,build_join_index
from gym_trading_env.research.fg_s3_v2_gpu_check import gpu_batch
from dreamerv3.fg_s3_e_quality_heads import GROUPS,TARGETS,LABELS,features,objective,register_state_schema
from dreamerv3.fg_s3a_actor import parameter_digests

@pytest.fixture(scope='module')
def case(tmp_path_factory):
 root=tmp_path_factory.mktemp('wp05_update');s=engineering_settings(root);p=preflight(s)
 index=root/'join.sqlite';build_join_index(p.source,index)
 r=EQualityCourseReader.open(p.source,stage='HEADS_VALID',sampling_spec=s['sampling_spec'],seed=202,index_path=index)
 a=create_agent(s,root/'model',controlled_small=True);data,_=F2WindowAdapter(r,s).sample()
 yield a,a.save_exact(),data,r,s
 r.close()

def labels_for(data,names=TARGETS):
 d=copy.deepcopy(data)
 for n in TARGETS:
  d['quality_'+n+'_valid'][...]=n in names
  d['quality_'+n][...]=2 if n=='event' else 90 if n=='score' else 1
 return d

def train(a,d,seed=202):return a.train(a.init_train(1),gpu_batch(a,copy.deepcopy(d),seed))

def forward(a,d):
 def f(raw):
  feat,carry,ids=features(a.model,raw);value,(pred,m)=objective(a.model,feat,{k:raw[k][:,5:] for k in LABELS})
  return feat,ids,value,pred
 return jax.tree.map(np.asarray,nj.pure(f)(a.params,{k:jnp.asarray(v) for k,v in d.items()},seed=202,create=False,modify=True)[1])

@pytest.mark.parametrize('name',TARGETS)
def test_C01_each_head_updates_only_its_params_and_slots(case,name):
 a,initial,data,_,_=case;a.load_exact(initial);d=labels_for(data,(name,));before=parameter_digests(a.params)
 train(a,d);_,out,m=train(a,d,203);after=parameter_digests(a.params);prefix='quality_'+name+'/'
 assert int(a.n_updates)==2 and m['f2/updated'] and m['f2/finite']
 assert any(before[k]!=after[k] for k in before if k.startswith(prefix))
 assert any(before[k]!=after[k] for k,o in a.model.f2_state_owner.items() if o==prefix and k.startswith('opt/'))
 assert all(before[k]==after[k] for k,o in a.model.f2_state_owner.items() if o not in (prefix,'CLOCK'))
 np.testing.assert_array_equal(out['effective_stepid'],d['stepid'][:,5:])
 evidence=dict(before=before,after=after,state_owner=a.model.f2_state_owner,valid_head=name,updates=2)
 Path('docs/receipts/wp05/C_'+name+'_delta.json').write_text(json.dumps(evidence,indent=2))

def test_C02_all_invalid_skips_with_fresh_or_inherited_momentum(case):
 a,initial,data,_,_=case;a.load_exact(initial);train(a,labels_for(data));d=labels_for(data,())
 for n in TARGETS:
  if np.issubdtype(d['quality_'+n].dtype,np.floating):d['quality_'+n][...]=np.nan
  else:d['quality_'+n][...]=999
 before=parameter_digests(a.params);count=int(a.n_updates)
 _,_,m=train(a,d)
 assert m['f2/skip_no_valid'] and not m['f2/updated'] and int(a.n_updates)==count
 assert parameter_digests(a.params)==before

def test_C03_teacher_changes_loss_only_and_burnin_is_not_supervised(case):
 a,initial,data,_,_=case;a.load_exact(initial);d=labels_for(data);f,ids,value,p=forward(a,d)
 changed=copy.deepcopy(d);changed['quality_score'][...]=0;changed['quality_event'][...]=0
 ff,ii,vv,pp=forward(a,changed)
 assert f.tobytes()==ff.tobytes() and ids.tobytes()==ii.tobytes() and value!=vv
 assert all(p[k].tobytes()==pp[k].tobytes() for k in p)
 changed=copy.deepcopy(d)
 for k in LABELS:changed[k][:,:5]=0
 assert forward(a,changed)[2].tobytes()==value.tobytes()

def test_C04_frozen_nonzero_momentum_negative_zero(case):
 a,initial,data,_,_=case;a.load_exact(initial)
 for k,owner in a.model.f2_state_owner.items():
  if owner is None:a.params[k]=jnp.full_like(a.params[k],.125 if k.startswith('opt/') else -0.)
 before=parameter_digests(a.params);train(a,labels_for(data));after=parameter_digests(a.params)
 assert all(before[k]==after[k] for k,o in a.model.f2_state_owner.items() if o is None)

def test_C05_nonfinite_rejected_and_all_state_rolled_back(case):
 a,initial,data,_,_=case;a.load_exact(initial)
 for k in a.params:
  if k.startswith('quality_event/'):a.params[k]=jnp.full_like(a.params[k],1e38)
 before=parameter_digests(a.params)
 with pytest.raises(FloatingPointError):train(a,labels_for(data))
 assert parameter_digests(a.params)==before and int(a.n_updates)==0

def test_C06_unknown_leaf_or_slot_rejected(case):
 from types import SimpleNamespace
 a,initial,_,_,_=case
 for key in ('alien/kernel','opt/state/1/1/alien/kernel'):
  with pytest.raises(ValueError):register_state_schema(SimpleNamespace(config=a.model.config),{**initial['params'],key:np.zeros(2,np.float32)})

def test_C07_invalid_placeholders_safe_no_q_computation():
 from dreamerv3.fg_s3_quality_v2 import prediction_terms
 from dreamerv3.fg_s3_e_quality_heads import model_extensions
 shapes={'event':(1,2,7,4),'touch':(1,2,4,2),'adverse':(1,2,4),'score':(1,2)}
 pred={k:jnp.zeros(s) for k,s in shapes.items()};labels={}
 for n in TARGETS:
  shape=shapes[n][:-1] if n=='event' else shapes[n]
  labels['quality_'+n]=jnp.full(shape,999,jnp.int32) if n=='event' else jnp.full(shape,jnp.nan);labels['quality_'+n+'_valid']=jnp.zeros(shape,bool)
 value,grad=jax.value_and_grad(lambda p:prediction_terms(p,labels,model_extensions(8,4))[0])(pred)
 assert value==0 and all(np.all(np.asarray(x)==0) for x in grad.values())
 # No Q entries exist at all. Official-day endpoints alone cannot enable mono.
 labels['quality_event_valid']=jnp.array([[[False]*5+[True,True]]*2])
 labels['quality_event']=jnp.zeros((1,2,7),jnp.int32)
 _,m=prediction_terms(pred,labels,model_extensions(8,4))
 assert m['quality/mono_loss']==0


def test_C08_disabled_weights_do_not_activate_heads():
 from dreamerv3.fg_s3_e_quality_heads import active_heads,model_extensions
 labels={'quality_'+n+'_valid':jnp.ones((1,2,7) if n=='event' else (1,2),bool) for n in TARGETS}
 cfg=model_extensions(8,4,{'quality_score_lambda':0.,'quality_outcome_lambda':0.})
 active=active_heads(labels,cfg)
 assert active['quality_event/'] and not active['quality_score/'] and not active['quality_touch/'] and not active['quality_adverse/']
 labels['quality_event_valid']=jnp.zeros((1,2,7),bool).at[...,5:].set(True)
 assert not any(bool(x) for x in active_heads(labels,cfg).values())

def test_C09_positive_allowlist_rejects_injected_wm_rule():
 from types import SimpleNamespace
 import optax
 from dreamerv3.fg_s3_v2_agent import masked_joint_optimizer
 from dreamerv3.fg_s3_e_quality_heads import STAGE
 opt=SimpleNamespace(opt=optax.sgd(.1));params={'enc/kernel':jnp.ones(2),'quality_event/kernel':jnp.ones(2)}
 state=opt.opt.init(params)
 with masked_joint_optimizer(opt,False,False,{**dict.fromkeys(GROUPS,True),'enc/':True},allowed_prefixes=GROUPS,stage=STAGE):
  with pytest.raises(ValueError,match='OUTSIDE_ALLOWLIST'):opt.opt.update(params,state,params)
 with masked_joint_optimizer(opt,True,True,dict.fromkeys(GROUPS,True),allowed_prefixes=GROUPS,stage=STAGE):
  updates,_=opt.opt.update(params,state,params)
 assert not np.any(updates['enc/kernel']) and np.any(updates['quality_event/kernel'])

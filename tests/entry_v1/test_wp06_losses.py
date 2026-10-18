import copy,json
from pathlib import Path
import jax,jax.numpy as jnp
import numpy as np
import embodied
from dreamerv3.fg_s3_e_quality_actor import actor_objective,model_extensions

def example(y=(1,0,0,1),score=(90,60,10,0)):
    shape=(1,len(y))
    labels=dict(teacher_action=jnp.array([y],jnp.int32),teacher_valid=jnp.ones(shape,bool),
        quality_owner=jnp.ones((*shape,4),jnp.uint32),quality_score=jnp.array([score],jnp.float32),
        quality_score_valid=jnp.ones(shape,bool),quality_outcome_valid=jnp.ones(shape,bool),f3_natural=jnp.ones(shape,bool))
    return labels,jnp.ones((*shape,2)),jnp.zeros(shape,bool)

def objective(logits,labels,mask,first,last=None,terminal=None,loss=None):
    no=jnp.zeros_like(first)
    return actor_objective(embodied.jax.outs.MaskedCategorical(logits,mask>.5,unimix=.01,allow_fallback=False),
        labels,mask,first,no if last is None else last,no if terminal is None else terminal,model_extensions(8,4,loss))

def test_D01_natural_balanced_denominators_and_single_class_fallback():
    labels,mask,first=example((1,0,0,0));logits=jnp.array([[[0.,-1.]]*4])
    _,m=objective(logits,labels,mask,first,loss={'balanced':1})
    assert m['f3/natural_count']==4 and m['f3/natural_add']==1 and m['f3/natural_wait']==3
    assert m['f3/balanced_count']==4 and m['f3/auxiliary_count']==0 and m['f3/natural_loss']!=m['f3/balanced_loss']
    labels['teacher_action']=jnp.zeros((1,4),jnp.int32)
    _,m=objective(logits,labels,mask,first)
    assert m['f3/natural_loss']==m['f3/balanced_loss'] and m['f3/no_pair']

def test_D02_l1_conflicts_never_reverse_l2_and_default_off():
    labels,mask,first=example((0,1,0,1),(100,80,10,0));logits=jnp.array([[[0.,1.],[0.,2.],[0.,-1.],[0.,0.]]])
    base,m=objective(logits,labels,mask,first)
    assert m['f3/quality_pair_count']==1 and m['f3/quality_pair_candidates']==5
    assert m['f3/quality_pair_conflicts']==4
    assert m['f3/quality_pair_same_action_filtered']==2 and m['f3/quality_pair_reversed_filtered']==2
    assert base==m['f3/natural_loss']
    value,extra=objective(logits,labels,mask,first,loss={'quality_pair':1.})
    np.testing.assert_allclose(value,base+extra['f3/quality_pair_loss'])
    Path('docs/receipts/wp06/D_l1_conflicts.json').write_text(json.dumps({k:np.asarray(v).tolist() for k,v in extra.items()},indent=2))

def test_D03_no_pairs_across_owner_segment_batch_or_zero_owner():
    labels,mask,first=example((1,0),(90,0));logits=jnp.zeros((1,2,2))
    for attack in ('owner','reset','zero','batch'):
        d=copy.deepcopy(labels);f=first;m=mask;x=logits
        if attack=='owner':d['quality_owner']=d['quality_owner'].at[:,1].set(2)
        elif attack=='reset':f=f.at[:,1].set(True)
        elif attack=='zero':d['quality_owner']=jnp.zeros_like(d['quality_owner'])
        else:
            d={k:v.reshape((2,1,*v.shape[2:])) for k,v in d.items()};f=f.reshape(2,1);m=m.reshape(2,1,2);x=x.reshape(2,1,2)
        _,metrics=objective(x,d,m,f,loss={'strict_pair':1,'quality_pair':1})
        assert metrics['f3/strict_pair_count']==metrics['f3/quality_pair_count']==0

def test_D04_l2_flip_changes_gradient_and_invalid_placeholders_are_zero():
    labels,mask,first=example();logits=jnp.zeros((1,4,2))
    gradient=lambda d:jax.grad(lambda x:objective(x,d,mask,first)[0])(logits)
    one=gradient(labels);changed=copy.deepcopy(labels);changed['teacher_action']=1-changed['teacher_action']
    np.testing.assert_allclose(one,-gradient(changed))
    changed['teacher_valid']=jnp.zeros((1,4),bool);changed['teacher_action']=jnp.full((1,4),999,jnp.int32)
    changed['quality_score_valid']=jnp.zeros((1,4),bool);changed['quality_score']=jnp.full((1,4),jnp.nan)
    value,m=objective(logits,changed,mask,first,loss={'balanced':1,'strict_pair':1,'quality_pair':1})
    assert value==0 and not m['f3/active'] and np.all(np.asarray(gradient(changed))==0)

def test_D05_actual_dual_mask_and_reset_terminal_define_valid_l2():
    labels,mask,first=example();logits=jnp.zeros((1,4,2))
    mask=mask.at[:,0,1].set(0);first=first.at[:,1].set(True)
    last=jnp.array([[False,False,True,False]]);terminal=jnp.array([[False,False,False,True]])
    _,m=objective(logits,labels,mask,first,last,terminal)
    assert m['f3/valid_count']==0 and m['f3/strict_pair_count']==0

def test_D06_auxiliary_never_changes_natural_prior_or_pairs_across_streams():
    labels,mask,first=example((1,0,0,0));logits=jnp.zeros((1,4,2))
    base,_=objective(logits,labels,mask,first)
    both={k:jnp.concatenate([v,v],0) for k,v in labels.items()}
    both['f3_natural']=both['f3_natural'].at[1].set(False)
    both['teacher_action']=both['teacher_action'].at[1].set(1)
    x=jnp.concatenate([logits,jnp.array([[[0.,3.]]*4])],0)
    value,m=objective(x,both,jnp.concatenate([mask,mask],0),jnp.concatenate([first,first],0))
    assert value==base and m['f3/natural_count']==4 and m['f3/natural_add']==1
    assert m['f3/auxiliary_count']==4 and m['f3/auxiliary_add']==4 and m['f3/strict_pair_count']==0

def test_D07_pair_temperature_and_explicit_weight_ablation():
    labels,mask,first=example();logits=jnp.array([[[0.,1.],[0.,-1.],[0.,0.],[0.,2.]]])
    loss=dict(natural=1,balanced=.2,strict_pair=.3,quality_pair=.4,pair_tau=2.)
    value,m=objective(logits,labels,mask,first,loss=loss)
    np.testing.assert_allclose(value,m['f3/natural_loss']+.2*m['f3/balanced_loss']+.3*m['f3/strict_pair_loss']+.4*m['f3/quality_pair_loss'])

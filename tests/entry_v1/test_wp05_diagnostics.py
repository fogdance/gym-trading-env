import copy,json
from pathlib import Path
import numpy as np
import pytest
from wp05_fixture import engineering_settings
from gym_trading_env.research.fg_s3_e_quality_heads import preflight,create_agent,F2WindowAdapter,validate_settings
from gym_trading_env.research.fg_s3_e_quality_reader import EQualityCourseReader,build_join_index
from gym_trading_env.research.fg_s3_e_quality_heads_checks import forward,score_batch,index_coverage,ablations,evaluate_batch
from dreamerv3.fg_s3a_actor import parameter_digests

@pytest.fixture(scope='module')
def indexed(tmp_path_factory):
 root=tmp_path_factory.mktemp('wp05_diag');s=engineering_settings(root);p=preflight(s)
 index=root/'join.sqlite';build_join_index(p.source,index)
 return root,s,p,index

def reader(case,spec):
 _,s,p,index=case
 return EQualityCourseReader.open(p.source,stage='HEADS_VALID',sampling_spec=spec,seed=202,index_path=index)

@pytest.mark.parametrize('mode',['NATURAL','MIXED','STRATIFIED'])
def test_D01_unique_reader_modes_rng_and_native_owner(indexed,mode):
 spec=dict(mode=mode)
 if mode!='NATURAL':spec['strata_weights']={'POLICY:P0':1.}
 if mode=='MIXED':spec['natural_weight']=.5
 s=validate_settings({**indexed[1],'sampling_spec':spec});r=reader(indexed,spec)
 try:
  r.sample(1,12);state=r.state_dict();a=r.sample(2,12);r.load_state_dict(state);b=r.sample(2,12)
  assert all(a[k].tobytes()==b[k].tobytes() for k in a)
  for meta in r.audit_windows(a):assert len({x['policy'] for x in meta['rows']})==1
  if mode!='NATURAL':
   with pytest.raises(ValueError,match='NATURAL_EVALUATION_REQUIRED'):evaluate_batch(r,None,None)
 finally:r.close()

def test_D02_four_head_read_only_metrics_and_static_ablations(indexed):
 root,s,_,_=indexed;r=reader(indexed,dict(mode='NATURAL'))
 try:
  a=create_agent(s,root/'model',controlled_small=True);data,_=F2WindowAdapter(r,s).sample();before=parameter_digests(a.params)
  out=forward(a,data);m=score_batch(data,out,5);coverage=index_coverage(r)
  assert m['prediction_finite'] and len(m['horizons'])==7 and all(c['valid']>0 for c in coverage['counts'].values())
  assert 'decision_components' not in m
  ab=ablations(a,data,out);assert set(ab)=={'1m','15m','partial5m_macd'}
  assert parameter_digests(a.params)==before and int(a.n_updates)==0
  changed=copy.deepcopy(data)
  for name in ('event','touch','adverse','score'):changed['quality_'+name+'_valid'][...]=False
  empty=score_batch(changed,out,5)
  assert empty['score_mae'] is None and empty['horizons']['0']['nll'] is None
  assert all(x['status']=='UNSUPPORTED' and x['auc'] is None for x in empty['binary_discrimination'].values())
  Path('docs/receipts/wp05/D_diagnostic_evidence.json').write_text(json.dumps(dict(metrics=m,coverage=coverage,ablations=ab,updates=0),indent=2))
 finally:r.close()

@pytest.mark.parametrize('attack',['owner','shift','q','soft','valid_score','missing_support'])
def test_D03_window_and_projection_negatives(indexed,attack):
 s=indexed[1];spec=dict(mode='NATURAL')
 if attack=='missing_support':spec=dict(mode='STRATIFIED',strata_weights={'NO_SUCH_STRATUM':1.})
 r=reader(indexed,spec)
 try:
  adapter=F2WindowAdapter(r,s)
  if attack=='missing_support':
   before=r.state_dict()
   with pytest.raises(ValueError,match='INSUFFICIENT_CURRICULUM_SUPPORT'):adapter.sample()
   assert r.state_dict()==before;return
  raw=r.sample(1,12)
  if attack=='owner':
   other=r.sample(1,12)
   while r.audit_windows(other)[0]['policy']==r.audit_windows(raw)[0]['policy']:other=r.sample(1,12)
   for k in raw:raw[k][:,2]=other[k][:,2]
  elif attack=='shift':raw['stepid']=np.roll(raw['stepid'],1,axis=1)
  elif attack=='q':raw['quality_q_v2_valid'][...]=True
  elif attack=='soft':raw['quality_soft_valid'][...]=True
  else:raw['quality_score_valid'][...]=True;raw['quality_score'][...]=np.nan
  with pytest.raises(ValueError):adapter.adapt(raw)
 finally:r.close()

def test_D04_visible_native_macd_and_context_reach_F2(indexed,tmp_path):
 from wp05_fixture import BASE
 # Same preserved WP04 run; this source includes complete 320-minute prehistory.
 source=BASE.parents[1]/'G0_cli_probe'
 s={**indexed[1],'target_steps':64,'burnin_steps':32,'corpus_root':str(source/'source/corpus'),'source_f1_run':str(source/'run'),
    'source_f1_checkpoint':str(source/'run/ckpt/step_000000000004')}
 p=preflight(s);index=tmp_path/'join.sqlite';build_join_index(p.source,index)
 r=EQualityCourseReader.open(p.source,stage='HEADS_VALID',sampling_spec={'mode':'NATURAL'},seed=202,index_path=index)
 try:
  a=create_agent(s,tmp_path/'model',controlled_small=True);data,_=F2WindowAdapter(r,s).sample();before=parameter_digests(a.params)
  assert data['market_macd_5m'][...,-1].any() and data['market_context_15m'][...,-1].any()
  report=ablations(a,data)
  print('MULTISCALE_REPORT',json.dumps(report))
  Path('docs/receipts/wp05/D_visible_multiscale.json').write_text(json.dumps(report,indent=2))
  assert all(v['valid_visible'] and v['encoder_max_abs_delta']>0 and v['posterior_logits_max_abs_delta']>0
             and max(v['max_abs_logit_delta'].values())>0 for v in report.values())
  assert parameter_digests(a.params)==before
  Path('docs/receipts/wp05/D_visible_multiscale.json').write_text(json.dumps(report,indent=2))
 finally:r.close()


def test_D05_L2_and_invalid_Q_values_never_enter_F2_inputs(indexed):
 r=reader(indexed,{'mode':'NATURAL'})
 try:
  raw=r.sample(1,12);adapter=F2WindowAdapter(r,indexed[1]);before,_=adapter.adapt(raw)
  raw['teacher_action'][...]=1-raw['teacher_action'];raw['teacher_valid'][...]=~raw['teacher_valid']
  raw['quality_q_c1'][...]=np.nan;raw['quality_q_v2'][...]=np.nan
  after,_=adapter.adapt(raw)
  assert all(before[k].tobytes()==after[k].tobytes() for k in before)
  assert not any(k.startswith('teacher_') or k.startswith('quality_q') for k in after)
 finally:r.close()

def test_D06_burnin_only_labels_rejected_without_resampling_or_rng_change(indexed):
 s={**indexed[1],'burnin_steps':90,'target_steps':8,'sampling_spec':{'mode':'NATURAL','policies':['P1']}}
 r=reader(indexed,s['sampling_spec'])
 try:
  # Current native positive baseline supports normal K4/T8 in this same view.
  assert F2WindowAdapter(r,{**s,'burnin_steps':4}).validate_support()['components']['NATURAL']
  before=r.state_dict();raw=r.sample(1,98)
  assert r._groups(98) and any(raw['quality_'+n+'_valid'].any() for n in ('event','touch','adverse','score'))
  assert not any(raw['quality_'+n+'_valid'][:,91:].any() for n in ('event','touch','adverse','score'))
  r.load_state_dict(before)
  with pytest.raises(ValueError,match='INSUFFICIENT_CURRICULUM_SUPPORT: F2_TARGET_LOSS'):F2WindowAdapter(r,s).sample()
  assert r.state_dict()==before
 finally:r.close()

def test_D07_target_support_guard_does_not_change_original_sampling(indexed):
 r=reader(indexed,{'mode':'NATURAL'})
 try:
  state=r.state_dict();expected=r.sample(2,12);r.load_state_dict(state)
  adapter=F2WindowAdapter(r,{**indexed[1],'batch_size':2});adapter.validate_support()
  assert r.state_dict()==state
  actual,_=adapter.sample()
  from dreamerv3.fg_s3_e_quality_heads import LABELS
  assert all(actual[k].tobytes()==expected[k].tobytes() for k in actual)
 finally:r.close()

@pytest.mark.parametrize('mode',['MIXED','STRATIFIED'])
def test_D08_target_support_checks_each_explicit_sampling_component(indexed,mode):
 spec={'mode':mode,'strata_weights':{'POLICY:P0':1.}}
 if mode=='MIXED':spec['natural_weight']=.5
 r=reader(indexed,spec)
 try:
  state=r.state_dict();report=F2WindowAdapter(r,{**indexed[1],'sampling_spec':spec}).validate_support()
  assert report['components']['POLICY:P0'] and r.state_dict()==state
 finally:r.close()

@pytest.mark.parametrize('weights',[
 {'quality_outcome_lambda':0.,'quality_mono_lambda':0.},
 {'quality_score_lambda':0.,'quality_mono_lambda':0.},
 {'quality_outcome_lambda':0.,'quality_score_lambda':0.}])
def test_D09_target_support_matches_enabled_score_outcome_or_mono(indexed,weights):
 r=reader(indexed,{'mode':'NATURAL'})
 try:assert F2WindowAdapter(r,{**indexed[1],'prediction_weights':weights}).validate_support()['components']['NATURAL']
 finally:r.close()

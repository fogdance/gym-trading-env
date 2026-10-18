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
  Path('docs/receipts/wp05/D_diagnostics.json').write_text(json.dumps(dict(metrics=m,coverage=coverage,ablations=ab,updates=0),indent=2))
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

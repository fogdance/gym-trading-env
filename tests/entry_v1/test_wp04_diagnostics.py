import copy
import json
from pathlib import Path
import jax
import jax.numpy as jnp
import ninjax as nj
import numpy as np
import pytest
from dreamerv3.fg_s3a_actor import parameter_digests
from gym_trading_env.research.fg_s3_e_quality_foundation import create_agent,F1WindowAdapter
from gym_trading_env.research.fg_s3_e_quality_foundation_checks import forward,summarize,support
from wp04_fixture import multiscale_reader
from test_wp04_wm import train_batch

@pytest.fixture(scope='module')
def case(tmp_path_factory):
    root=tmp_path_factory.mktemp('wp04_multiscale');s,r=multiscale_reader(root)
    a=create_agent(s,root/'model',controlled_small=True);adapter=F1WindowAdapter(r,s)
    data,meta=adapter.sample();before=parameter_digests(a.params)
    report,raw=summarize(a,data,meta)
    assert before==parameter_digests(a.params) and int(a.n_updates)==0
    try:yield s,r,a,data,meta,report,raw
    finally:r.close()

def test_C01_native_multiscale_gradients_and_actual_learning(case):
    s,r,a,data,meta,report,_=case
    assert np.any(data['market_context_15m'][...,6]) and np.any(data['market_context_15m'][...,:6])
    assert np.any(data['market_macd_5m'][...,3]) and np.any(data['market_macd_5m'][...,:3])
    assert report['finite']
    for p,norm in report['gradient_norms'].items():assert norm>0,(p,norm)
    before=parameter_digests(a.params)
    for i in range(4):train_batch(a,data,202+i)
    after=parameter_digests(a.params)
    for p in report['gradient_norms']:assert any(before[k]!=after[k] for k in before if k.startswith(p)),p
    assert all(before[k]==after[k] for k,v in a.model.f1_mutable_keys.items() if not v)
    evidence=dict(report=report,before=before,after=after,updates=4,scope='SYNTHETIC_CPU_ONLY',
        reader_coverage=r.coverage(),optimizer_target_exposures=256,optimizer_burnin_exposures=128,
        distinct_target_market_cursors=len(set(row['cursor'] for row in meta[0]['rows'][33:])))
    path=Path('docs/receipts/wp04/C_multiscale_learning.json')
    if path.exists():path=path.with_name(path.stem+'_'+__import__('uuid').uuid4().hex+'.json')
    path.write_text(json.dumps(evidence,indent=2)+'\n')

def encoder(a,data):
    def fn(obs):return a.model.enc({},obs,obs['is_first'],False)[2]
    obs={k:jnp.asarray(data[k]) for k in a.model.obs_space}
    return np.asarray(nj.pure(fn)(a.params,obs,seed=202,create=False,modify=False)[1])

def test_C02_valid_branch_ablation_and_invalid_values(case):
    _,_,a,data,_,_,_=case
    z=encoder(a,data)
    for name,valid in [('market_macd_5m',3),('market_context_15m',6)]:
        changed=copy.deepcopy(data);changed[name][...,:valid]=0
        assert not np.array_equal(z,encoder(a,changed))
        invalid=copy.deepcopy(data);invalid[name][...,valid]=0
        baseline=encoder(a,invalid);invalid[name][...,:valid]=1e10
        np.testing.assert_array_equal(baseline,encoder(a,invalid))

def test_C03_future_labels_and_diagnostic_switch_do_not_change_loss_or_gradients(case):
    _,_,a,data,_,_,_=case
    baseline=forward(a,data,gradients=True,diagnostics=False)
    labels=copy.deepcopy(data);labels['teacher_action']=np.ones_like(data['reward'],np.int32)
    labels['quality_q_v2_valid']=np.ones_like(data['is_first'])
    # F1 API's actual observed fields are selected before enc; posterior extras
    # cannot enter its loss. Production Adapter drops these fields entirely.
    other=forward(a,labels,gradients=True,diagnostics=True)
    assert float(baseline[0])==float(other[0])
    for key in baseline[1]:np.testing.assert_array_equal(baseline[1][key],other[1][key])

def test_C04_native_loss_events_clock_and_coverage(case):
    s,r,a,data,meta,report,(out,mets,grad)=case
    assert set(report['losses'])==set(a.model.scales)-{'policy','value','repval'}
    assert report['target_positions']==64 and report['burnin_positions']==32
    assert report['reward_events']['nonzero']['count']>0
    assert report['availability']['post']['legal_add_count']>0
    expected=(~data['is_terminal'][:,33:])*(1-1/a.model.config.horizon)**data['market_advance'][:,33:]
    np.testing.assert_allclose(mets['f1_raw/continue_target'],expected,rtol=1e-7)
    assert not {'market_macd_5m','market_context_15m'} & out['losses'].keys()

@pytest.mark.parametrize('case_name',['zero_reward','only_wait','all_censored'])
def test_C05_pathologies_never_market_go(case_name):
    reward=np.ones((1,4));mask=np.ones((1,4,2));censored=0
    if case_name=='zero_reward':reward*=0
    if case_name=='only_wait':mask[...,1]=0
    if case_name=='all_censored':censored=4
    result=support(reward,mask,candidate_count=4,right_censored_count=censored)
    assert result['status']=='NO_SUPPORT' and not result['model_research_go'] and not result['training_admitted']

def test_C06_reset_episode_full_forward_and_finite_K_comparison(case):
    _,_,a,data,_,_,_=case
    # Controlled short episode reference: same visible sequence and actions,
    # explicit reset at start; never submitted as authenticated Reader data.
    raw={k:jnp.asarray(v) for k,v in data.items()};raw['is_first']=raw['is_first'].at[:,0].set(True)
    def reference(d):
        initial=a.model.init_train(1);carry=initial[:3]
        obs={k:d[k] for k in a.model.obs_space};obs['market_advance']=d['market_advance']
        act={k:jnp.concatenate([initial[3][k][:,None],d[k][:,:-1]],1) for k in a.model.act_space}
        full,(_,_,out,mets)=a.model.wm_only(carry,obs,act,False,diagnostics=True)
        prefix,tail,prev,_=__import__('dreamerv3.fg_s3_e_quality_foundation',fromlist=['context']).context(a.model,d)
        finite,(_,_,short,sm)=a.model.wm_only(prefix,tail,prev,False,diagnostics=True)
        _,nobs,nact,_=a.model._apply_replay_context(initial,d)
        _,(_,_,none,nm)=a.model.wm_only(carry,nobs,nact,False,diagnostics=True)
        return full,finite,out['repfeat'],short['repfeat'],none['repfeat'],mets['f1_raw/encoder_tokens'],sm['f1_raw/encoder_tokens']
    _,result=nj.pure(reference)(a.params,raw,seed=202,create=False,modify=True)
    full,finite,allfeat,kfeat,nofeat,tokens,ktokens=jax.tree.map(np.asarray,result)
    np.testing.assert_allclose(tokens[:,33:],ktokens,rtol=0,atol=1e-7)
    delta=lambda lhs,rhs:float(np.linalg.norm(lhs['deter']-rhs['deter']))
    fulltail={k:v[:,33:] for k,v in allfeat.items()};notail={k:v[:,32:] for k,v in nofeat.items()}
    assert delta(kfeat,notail)>0 and np.isfinite(full) and np.isfinite(finite)
    def warmed(d):
        from dreamerv3.fg_s3_e_quality_foundation import context
        return context(a.model,d)[0][1]['deter']
    changed=dict(raw)
    changed['market_seq_e_shared']=raw['market_seq_e_shared'].at[:,1:33].set(0)
    def state(d):return np.asarray(nj.pure(warmed)(a.params,d,seed=202,create=False,modify=True)[1])
    same_seed_history_effect=float(np.linalg.norm(state(raw)-state(changed)))
    assert same_seed_history_effect>0
    path=Path('docs/receipts/wp04/C_burnin_reference.json')
    evidence=dict(full_reset_forward_finite=True,encoder_token_max_abs_delta=float(np.max(np.abs(tokens[:,33:]-ktokens))),
        encoder_token_atol=1e-7,
        same_PRNG_partition_past_input_effect=same_seed_history_effect,
        K32_vs_one_frame_deter_norm=delta(kfeat,notail),K32_vs_full_reset_deter_norm=delta(kfeat,fulltail),
        interpretation='Finite K approximation. Full scan and split scans consume different original RSSM PRNG partitions; stochastic states are compared, not asserted bitwise equal.',
        target_steps=64,burnin_steps=32,real_TRAIN_updates=0)
    if path.exists():path=path.with_name(path.stem+'_'+__import__('uuid').uuid4().hex+'.json')
    path.write_text(json.dumps(evidence,indent=2)+'\n')

def test_C07_readonly_diagnostics_do_not_change_next_update(case):
    _,_,a,data,meta,_,_=case;sealed=a.save_exact()
    train_batch(a,data,710);expected=parameter_digests(a.params)
    a.load_exact(sealed);forward(a,data,diagnostics=True)
    train_batch(a,data,710)
    assert expected==parameter_digests(a.params)

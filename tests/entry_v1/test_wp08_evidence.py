import copy

from test_wp08_contract import plan
from gym_trading_env.research.fg_s3_e_quality_exam_evidence import assess


def comparison(n=0):
    row=dict(policy='ACTOR',contract='CZCE.FG2701',entries=n,eligible_add_states=100,
        account_reconciled=True,reward_reconciled=True,illegal_add=0,mask_conflicts=0,
        tp_le_5=n,tp_le_5_rate=1. if n else None,sl30=0,sl_rate=0. if n else None,
        mae_15_mean=1. if n else None,mae_15_valid_count=n,
        net_pnl=100. if n else 0.,max_drawdown=0.)
    return dict(summaries=[row],blocks=[dict(policy='ACTOR',contract='CZCE.FG2701',official_day='2026-01-08')])


def test_all_wait_and_too_few_trades_never_become_quality_go(tmp_path):
    p=plan(tmp_path/'out')
    wait=assess(p,comparison())['results'][0]
    assert wait['entry_evidence']=='NO_ENTRY_SKILL_EVIDENCE'
    assert wait['entry_reason']=='ALL_WAIT_WHEN_ADD_LEGAL' and wait['tp_le_5_rate'] is None
    assert not wait['strategy_quality_go'] and wait['support']=='INSUFFICIENT_TRADE_SUPPORT'
    p['evidence_policy']=dict(min_trades=20,min_contracts=2,min_days=3,min_tp5_rate=.4,max_sl_rate=.2,
        max_mae15=10.,min_net_pnl=0.,max_drawdown=10000.)
    few=assess(p,comparison(2))['results'][0]
    assert few['support']=='UNDERPOWERED' and not few['strategy_quality_go']


def test_unregistered_and_synthetic_outcomes_not_dev_or_generalization(tmp_path):
    p=plan(tmp_path/'out');report=assess(p,comparison(1000))
    r=report['results'][0]
    assert report['real_dev_strategy_exam']=='NOT_RUN' and r['valid_dev_evidence']=='NOT_RUN'
    assert r['quality_signal']=='INCONCLUSIVE' and not r['strategy_quality_go']
    assert not report['sealed_test_ready'] and not report['research_policy_registered']


def test_registered_joint_checks_and_market_support_are_both_required(tmp_path):
    p=plan(tmp_path/'out');p['exam_mode']='DEV_EXAM'
    p['source']=dict(kind='VALIDATION',manifest='/tmp/metadata.json',env_receipt='/tmp/native')
    p['evidence_policy']=dict(min_trades=20,min_contracts=2,min_days=3,min_tp5_rate=.4,max_sl_rate=.2,
        max_mae15=10.,min_net_pnl=0.,max_drawdown=10000.)
    c=comparison(100)
    assert assess(p,c)['results'][0]['support']=='UNDERPOWERED'
    c['summaries'].append({**c['summaries'][0],'contract':'CZCE.FG2705','net_pnl':-100000.})
    c['blocks'] += [dict(policy='ACTOR',contract='CZCE.FG2705',official_day=d) for d in ('2026-01-08','2026-01-09')]
    r=assess(p,c)['results'][0]
    assert r['support']=='DECLARED_SUPPORT_MET' and not r['registered_checks']['costs_net_pnl']
    assert not r['strategy_quality_go']  # Fast TP alone cannot qualify.


def test_native_comparison_has_engineering_evidence_without_research_claim(tmp_path):
    import json
    from wp08_fixture import ARTIFACTS
    p=json.loads((ARTIFACTS/'E_compare_02/plan.json').read_text())
    c=json.loads((ARTIFACTS/'E_compare_02/comparison/COMPARISONS.json').read_text())
    report=assess(p,c)
    assert report['engineering_code_go']
    assert all(not r['strategy_quality_go'] for r in report['results'])
    assert report['real_dev_strategy_exam']=='NOT_RUN' and not report['training_admitted']

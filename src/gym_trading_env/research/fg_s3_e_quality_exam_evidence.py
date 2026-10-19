"""Engineering, validation and research evidence are separate decisions."""
from .fg_s3_e_quality_exam_contract import validate_plan
from .fg_replay import signed


def assess(plan,comparison):
    plan=validate_plan(plan);policy=plan['evidence_policy'];results=[]
    grouped={}
    for row in comparison['summaries']:grouped.setdefault(row['policy'],[]).append(row)
    blocks=comparison['blocks']
    for name,rows in grouped.items():
        n=sum(r['entries'] for r in rows);eligible=sum(r['eligible_add_states'] for r in rows)
        contracts=len({r['contract'] for r in rows})
        days=len({(r['contract'],r['official_day']) for r in blocks if r['policy']==name})
        proofs=[e['receipt'] for e in comparison.get('evidence',[]) if e['receipt']['policy_id']==name]
        engineering=(len(proofs)==len(rows) and all(r['account_reconciled'] and r['reward_reconciled']
            and r['illegal_add']==r['mask_conflicts']==0 for r in rows)
            and all(r.get('teacher_used') is False and r.get('training_updates')==0 and r.get('test_prices_accessed') is False
                and r.get('reconciliation',{}).get('status')=='PASS' and r.get('frozen',{}).get('training_updates')==0
                and (name not in plan['actors'] or all(r['frozen'].get(k) is True for k in
                    ('all_parameter_bytes_equal','policy_bytes_equal','training_counters_equal','source_files_unchanged')))
                for r in proofs))
        real=plan['exam_mode']=='DEV_EXAM'
        if n==0:
            entry='NO_ENTRY_SKILL_EVIDENCE';support='INSUFFICIENT_TRADE_SUPPORT'
            reason='ALL_WAIT_WHEN_ADD_LEGAL' if eligible else 'NO_ELIGIBLE_ADD_STATES'
        elif policy is None:
            entry='OBSERVED_REAL_ENTRY_FILLS';support='UNREGISTERED_SUPPORT_POLICY';reason='QUALITY_RULES_NOT_PREREGISTERED'
        elif n<policy['min_trades']:
            entry='OBSERVED_REAL_ENTRY_FILLS';support='UNDERPOWERED';reason='INSUFFICIENT_TRADE_SUPPORT'
        elif contracts<policy['min_contracts'] or days<policy['min_days']:
            entry='OBSERVED_REAL_ENTRY_FILLS';support='UNDERPOWERED';reason='INSUFFICIENT_INDEPENDENT_MARKET_BLOCKS'
        else:entry='ENTRY_SKILL_EVIDENCE';support='DECLARED_SUPPORT_MET';reason=None
        tp5=None if n==0 else sum(r['tp_le_5'] for r in rows)/n
        sl=None if n==0 else sum(r['sl30'] for r in rows)/n
        # Mean risk is weighted by its explicit valid-trade count, never by
        # policy seed count or an unweighted average of contract averages.
        risk_count=sum(r.get('mae_15_valid_count',0) for r in rows)
        mae=None if not risk_count else sum(r['mae_15_mean']*r.get('mae_15_valid_count',0)
            for r in rows if r['mae_15_mean'] is not None)/risk_count
        net=sum(r['net_pnl'] for r in rows);dd=max((r['max_drawdown'] for r in rows),default=0.)
        tested=None if policy is None or not n or mae is None else dict(
            tp_speed=tp5>=policy['min_tp5_rate'],sl=sl<=policy['max_sl_rate'],mae15=mae<=policy['max_mae15'],
            costs_net_pnl=net>=policy['min_net_pnl'],drawdown=dd<=policy['max_drawdown'])
        per_contract=None if policy is None else {r['contract']:bool(r['entries'] and r['mae_15_mean'] is not None
            and r['tp_le_5_rate']>=policy['min_tp5_rate'] and r['sl_rate']<=policy['max_sl_rate']
            and r['mae_15_mean']<=policy['max_mae15'] and r['net_pnl']>=policy['min_net_pnl']
            and r['max_drawdown']<=policy['max_drawdown']) for r in rows}
        quality=(name in plan['actors'] and real and engineering and support=='DECLARED_SUPPORT_MET'
            and tested is not None and all(tested.values()) and all(per_contract.values()))
        results.append(dict(policy=name,engineering='CODE_GO' if engineering else 'NO_GO',
            valid_dev_evidence='VALID_DEV_EVIDENCE' if real and engineering else 'NOT_RUN',
            entry_evidence=entry,entry_reason=reason,support=support,
            quality_signal='QUALITY_SIGNAL' if quality else 'INCONCLUSIVE',
            generalization_support='GENERALIZATION_SUPPORT' if quality and contracts>=2 else 'UNDERPOWERED',
            strategy_quality_go=quality,entries=n,eligible_add_states=eligible,contracts=contracts,
            independent_contract_days=days,tp_le_5_rate=tp5,sl_rate=sl,mae_15_mean=mae,
            mae_15_valid_count=risk_count,net_pnl=net,max_contract_drawdown=dd,registered_checks=tested,
            registered_contract_checks=per_contract))
    return signed(dict(protocol='FG_S3_E_QUALITY_EXAM_EVIDENCE_V1',exam_mode=plan['exam_mode'],
        results=results,engineering_code_go=bool(results) and all(r['engineering']=='CODE_GO' for r in results),
        real_dev_strategy_exam='RUN' if plan['exam_mode']=='DEV_EXAM' else 'NOT_RUN',
        research_policy_registered=policy is not None,sealed_test_ready=False,
        training_admitted=False,research_authorization=False,
        note='Synthetic outcomes and repeated accounts never establish real DEV strategy quality.'))

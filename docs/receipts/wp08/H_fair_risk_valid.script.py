"""Re-audit immutable engineering traces with the final risk aggregation."""
from pathlib import Path
import json


def main():
    from gym_trading_env.research.fg_s3_e_quality_exam_compare import compare_sessions,policy_jobs
    from gym_trading_env.research.fg_s3_e_quality_exam_evidence import assess
    from gym_trading_env.research.fg_s3_e_quality_exam_synthetic import native_source
    from gym_trading_env.research.fg_replay import digest,write_json
    root=Path.cwd();base=root/'artifacts/fg_s3_v2/e_quality_wp08/Final_E_01'
    p=json.loads((base/'plan.json').read_text())
    source=native_source(root/'artifacts/fg_s3_v2/e_quality_wp08/C_native_04/source')
    cases=[(base/j['contract']/j['label'],source) for j in policy_jobs(p)]
    out=base/'comparison_risk_valid';r=compare_sessions(cases,out)
    e=assess(p,r);write_json(out/'EVIDENCE.json',e)
    assert r['account_count']==10 and r['independent_contract_days']==4
    assert e['engineering_code_go'] and not any(x['strategy_quality_go'] for x in e['results'])
    assert len([x for x in r['paired_deltas'] if x['policy']=='F4_RL' and x['reference']=='F3_FROZEN'])==4
    receipt=dict(status='PASS_NATIVE_FINANCIAL_AND_CURRENT_METRICS',accounts=10,independent_contract_days=4,
        original_generation='H_final_fair.json',current_comparison=str(out),
        comparison_sha256=digest(out/'COMPARISONS.json'),evidence_sha256=digest(out/'EVIDENCE.json'),
        script_sha256=digest(__file__),training_updates=0,real_dev_strategy_exam='NOT_RUN')
    write_json(root/'docs/receipts/wp08/H_current_comparison.json',receipt)
    print(json.dumps(receipt))


if __name__=='__main__':main()

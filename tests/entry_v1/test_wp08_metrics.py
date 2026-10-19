import json
import os
from pathlib import Path

import pytest

from wp08_fixture import native_source,ARTIFACTS


def test_native_trade_denominators_risk_fees_and_daily_accounts():
    from gym_trading_env.research.fg_s3_e_quality_exam_metrics import analyze_trace
    root=ARTIFACTS/'C_native_04'
    source=native_source(root/'source')
    output=Path(os.environ.get('FG_WP08_D_ROOT',ARTIFACTS/'D_metrics_01')).absolute()
    report=analyze_trace(root/'session/trace',source,output)
    summary=report['summary'];trades=report['trades']
    account=json.loads((root/'session/trace/account_state.json').read_text())
    entries=[f for f in account['fills'] if f['reason']=='ENTRY' and f['committed']]
    assert summary['entries']==len(entries)==len(trades)>0
    assert summary['tp'] and summary['sl30'] and summary['account_reconciled']
    assert all(d['ledger_reconciled'] for d in report['daily'])
    assert summary['reward']==pytest.approx(sum(t['reward'] for t in trades))
    assert summary['fees']==pytest.approx(sum(t['open_fee']+t['close_fee'] for t in trades))
    assert summary['net_pnl']==pytest.approx(sum(d['net_pnl'] for d in report['daily']))
    assert any(t['adverse10_le_5_then_sl30'] for t in trades)
    assert all(t['mae_5']<=t['mae_15']<=t['mae_full'] for t in trades)
    assert all(t['held_market_minutes']>=1 for t in trades)
    for h in (5,15,30,60):
        assert summary['tp_le_'+str(h)+'_rate']==sum(t['tp_le_'+str(h)] for t in trades)/len(entries)
    assert sum(c['valid_minutes'] for c in report['coverage'])==1024
    Path('docs/receipts/wp08/D_accounting.json').write_text(json.dumps(dict(
        summary=summary,daily=report['daily'],coverage=report['coverage'],metrics_path=str(output)),indent=2)+'\n')
    with pytest.raises(ValueError,match='NEW'):analyze_trace(root/'session/trace',source,output)

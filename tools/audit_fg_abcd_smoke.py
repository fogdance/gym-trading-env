"""Audit real complete smoke checkpoints; no training or model inference."""
import argparse
import json
from pathlib import Path
import pickle
import numpy as np
from gym_trading_env.research.fg_entry_v1 import write_json


def checkpoint(root):
    directory=Path(root)/'ckpt'
    latest=directory/(directory/'latest').read_text().strip()
    if not (latest/'done').is_file():
        raise ValueError('Incomplete checkpoint')
    return {name:pickle.loads((latest/f'{name}.pkl').read_bytes())
            for name in ('agent','driver','step','action_mask_warmup')}


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--logdir',default='/data/logdir/fg_abcd_v1/v2_sequential/smoke')
    parser.add_argument('--before')
    parser.add_argument('--output',default='artifacts/fg_abcd_v1/v2_sequential/smoke_audit.json')
    args=parser.parse_args()
    source=checkpoint(Path(args.logdir)/'pretrain')
    target=checkpoint(Path(args.logdir)/'recent')
    before=json.loads((Path(args.before) if args.before else Path(args.logdir)/'before_resume.json').read_text())
    old=before['driver']['envs'][0]['env']
    current=source['driver']['envs'][0]['env']
    recent=target['driver']['envs'][0]['env']
    assert source['step']==1120 and target['step']==920
    assert source['agent']['counters']['updates']>0
    assert source['agent']['params'].keys()==target['agent']['params'].keys()
    assert all(np.array_equal(v,target['agent']['params'][k]) for k,v in source['agent']['params'].items())
    assert source['agent']['counters']['updates']==target['agent']['counters']['updates']
    assert source['action_mask_warmup']==target['action_mask_warmup']
    assert old['index']==current['index']
    assert old['schema']==current['schema']==recent['schema']
    old_accounts=({int(k):v for k,v in old['accounts'].items()} if old['schema']==2
                  else {old['index']:old['account']})
    current_accounts=current['accounts'] if current['schema']==2 else {current['index']:current['account']}
    for i,account in old_accounts.items():
        restored=current_accounts[i]
        if i!=old['index']:
            assert account==restored
        else:
            assert restored['current_step']-account['current_step']==source['step']-before['step']
            assert restored['account_sequence']==account['account_sequence']
            assert restored['fills'][:len(account['fills'])]==account['fills']
            assert restored['ledger_entries'][:len(account['ledger_entries'])]==account['ledger_entries']
    recent_advanced=sum(recent['advanced']) if recent['schema']==2 else recent['account']['current_step']
    assert recent_advanced==919
    assert recent['manifest_hash']!=current['manifest_hash']
    assert recent['completed']==[]
    # Recent is below Replay's learning threshold: exact optimizer/model equality proves transfer.
    receipt=json.loads((Path(args.logdir)/'pretrain/c_transfer_receipt.json').read_text())
    assert receipt['replay_protocol']=='EMPTY_RECENT_ONLY'
    result=dict(status='PASS',total_interactions=source['step']+target['step'],
        gpu_updates=source['agent']['counters']['updates'],full_agent_optimizer_exact_match=True,
        optimizer_variables=sum('opt/' in k for k in source['agent']['params']),
        inactive_accounts_preserved=True,current_account_resume_forward_only=True,
        warmup_gate_preserved=True,new_recent_accounts=True,recent_advanced_bars=recent_advanced,
        policy_learning_claim=False)
    write_json(args.output,result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()

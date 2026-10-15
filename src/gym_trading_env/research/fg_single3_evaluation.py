"""Causal Single3 DEV evaluation using the original Driver and reconciler."""
from collections import Counter
from pathlib import Path
import json
import pickle
import sys
import uuid

import numpy as np
import pandas as pd

from .fg_replay import (SINGLE3_SCHEMA, check_files, collect_trace, digest, iter_sidecar,
                        read_manifest, reconcile_trace, signed, trace_manifest, write_json)
from .fg_single3 import load_env
from .fg_single3_replay import Single3BehaviorPolicy


def embodied_runtime(root):
    """Do not override the backend of an already initialized neural Agent."""
    if str(root) not in sys.path:
        sys.path.insert(0,str(root))
    import embodied
    if Path(embodied.__file__).resolve().parent != Path(root) / 'embodied':
        raise ValueError('Foreign Embodied runtime')
    return embodied


class ActorPolicy:
    """Only Env observations reach the Actor; diagnostics stay in the sidecar."""
    waited = 0
    waiting = False

    def __init__(self, agent):
        self.agent = agent
        self.last_diagnostics = {}

    def observe_row(self, row, *, legal_after):
        pass

    def __call__(self, carry, obs):
        if carry is None:
            carry = self.agent.init_policy(1)
        carry, action, out = self.agent.policy(carry,obs,mode='eval')
        p_add = float(out['p_add'][0])
        if not np.isfinite(p_add):
            raise FloatingPointError('Non-finite Actor probability')
        self.last_diagnostics = dict(p_add=p_add,
            past_market_features=[float(x) for x in obs['market_seq'][0,-1]],
            position_state=[float(x) for x in obs['position_state'][0]])
        if self.agent.config.get('s3_v2_mode') == 'b2':
            self.last_diagnostics['b2'] = {k: np.asarray(v[0]).tolist() for k, v in out.items()
                                           if k.startswith('b2/')}
        if self.agent.config.get('s3_v2_mode') == 'tutor':
            self.last_diagnostics['tutor'] = {k: np.asarray(v[0]).tolist() for k,v in out.items()
                                             if k.startswith('tutor/')}
            if not self.last_diagnostics['tutor']['tutor/logit_margin_defined']:
                self.last_diagnostics['tutor']['tutor/final_logit_add_minus_wait'] = None
        if self.agent.config.get('s3_v2_mode') == 'quality_v2':
            self.last_diagnostics['quality']={k:np.asarray(v[0]).tolist() for k,v in out.items() if k.startswith('quality/')}
        # Do not add model caches or diagnostics to model observations/Replay.
        return carry, action, {}


def require_dev(path):
    manifest = read_manifest(path)
    if (manifest['role'] != 'DEV' or manifest['oracle_labels_used']
            or manifest['test_prices_accessed'] or len(manifest['rows']) != 1
            or max(manifest['train_days']) >= min(manifest['dev_days'])
            or pd.Timestamp(manifest['train_last_timestamp']) >= pd.Timestamp(manifest['dev_first_timestamp'])):
        raise ValueError('Invalid independent causal DEV manifest')
    row = manifest['rows'][0]
    if row['role'] != 'validation' or row['eligible_for_training'] or row['contract'] != 'CZCE.FG409':
        raise ValueError('Foreign DEV price member')
    check_files(manifest['frozen_inputs'])
    return manifest


def analyze_trace(directory, dev, strategy):
    directory = Path(directory)
    manifest = trace_manifest(directory,schema=SINGLE3_SCHEMA)
    state = json.loads((directory / 'account_state.json').read_text())
    events = {r['entry_id']:r for r in state['attributions']}
    reward_by_entry = Counter()
    for event in state['reward_events']:
        reward_by_entry[event['entry_id']] += float(event['reward'])
    total_reward = sum(reward_by_entry.values())
    if not np.isclose(total_reward,float(state['total_reward']),rtol=0,atol=1e-8):
        raise ValueError('DEV event Reward does not reconcile')
    trades = []
    for fill in state['fills']:
        if fill['reason'] != 'ENTRY' or not fill['committed']:
            continue
        event = events.get(fill['entry_id'])
        duration = event['held_market_bars'] if event else None
        reason = event['reason'] if event else 'RIGHT_CENSORED'
        trades.append(dict(contract=manifest['contract'],direction=fill['side'],entry_id=fill['entry_id'],
            entry_timestamp=fill['time'],entry_price=fill['price'],
            exit_timestamp=event['exit_time'] if event else None,
            exit_price=event['exit_price'] if event else None,exit_reason=reason,
            held_market_bars=duration,tp5=bool(reason=='TAKE_PROFIT' and duration<=5),
            tp15=bool(reason=='TAKE_PROFIT' and duration<=15),sl30=reason=='STOP_LOSS',
            reward=reward_by_entry[fill['entry_id']],
            net_trade_pnl=float(event['net_trade_pnl']) if event else None))
    actions = []
    for row in iter_sidecar(directory):
        if not row['decision_executed']:
            continue
        diagnostic = row.get('policy_diagnostics',{})
        actions.append(dict(cursor=row['cursor'],contract=manifest['contract'],
            signal_timestamp=row['time'],direction=row['direction'],action=row['action'],legal_add=row['legal_add'],
            active_wait=row['action']==0 and row['legal_add'],
            forced_wait=row['action']==0 and not row['legal_add'],
            p_add=diagnostic.get('p_add'),past_market_features=diagnostic.get('past_market_features')))
    equity = np.asarray([r['equity'] for r in state['equity_history']],np.float64)
    if not equity.size or not np.isfinite(equity).all():
        raise ValueError('Missing or non-finite DEV equity history')
    net_pnl = float(equity[-1]-equity[0])
    closed_pnl = sum(t['net_trade_pnl'] for t in trades if t['net_trade_pnl'] is not None)
    open_fees = sum(float(item['open_fee']) for item in state['entries'].values())
    reconciled_pnl = closed_pnl + float(state['unrealized_pnl']) - open_fees
    if not np.isclose(net_pnl,reconciled_pnl,rtol=0,atol=1e-6):
        raise ValueError(f'DEV net PnL does not reconcile: {net_pnl} vs {reconciled_pnl}')
    n = len(trades)
    legal = [r for r in actions if r['legal_add']]
    probabilities = [r['p_add'] for r in legal if r['p_add'] is not None]
    summary = dict(strategy=strategy,contract=manifest['contract'],entries=n,
        active_wait=sum(r['active_wait'] for r in actions),forced_wait=sum(r['forced_wait'] for r in actions),
        legal_add_decisions=sum(r['action']==1 for r in legal),
        illegal_add=sum(r['action']==1 and not r['legal_add'] for r in actions),
        tp5=sum(t['tp5'] for t in trades),tp15=sum(t['tp15'] for t in trades),
        tp=sum(t['exit_reason']=='TAKE_PROFIT' for t in trades),sl30=sum(t['sl30'] for t in trades),
        right_censored=sum(t['exit_reason']=='RIGHT_CENSORED' for t in trades),
        total_reward=total_reward,reward_per_day=total_reward/len(dev['dev_days']),
        eligible_trading_days=len(dev['dev_days']),net_pnl=net_pnl,closed_net_pnl=closed_pnl,
        unrealized_pnl=float(state['unrealized_pnl']),open_fees=open_fees,
        max_equity_drawdown=float((np.maximum.accumulate(equity)-equity).max()),
        p_add_min=min(probabilities) if probabilities else None,
        p_add_max=max(probabilities) if probabilities else None,
        p_add_std=float(np.std(probabilities)) if probabilities else None,
        denominator='all actual entries, including failures and right-censored positions',
        timing_status='NOT_ESTABLISHED',clock_protocol='S3_MARKET_MINUTE',simulated_fills=True)
    for key in ('tp5','tp15','sl30'):
        summary[key+'_rate'] = summary[key]/n if n else None
    if summary['illegal_add']:
        raise ValueError('DEV policy issued an illegal ADD')
    strata = []
    for direction in ('LONG','SHORT'):
        group = [t for t in trades if t['direction']==direction]
        strata.append(dict(contract=manifest['contract'],direction=direction,entries=len(group),
            tp5=sum(t['tp5'] for t in group),tp15=sum(t['tp15'] for t in group),
            sl30=sum(t['sl30'] for t in group),reward=sum(t['reward'] for t in group),
            closed_net_pnl=sum(t['net_trade_pnl'] or 0 for t in group)))
    pd.DataFrame(trades,columns=['contract','direction','entry_id','entry_timestamp','entry_price',
        'exit_timestamp','exit_price','exit_reason','held_market_bars','tp5','tp15','sl30','reward','net_trade_pnl']).to_csv(directory/'trades.csv',index=False)
    pd.DataFrame(actions).to_csv(directory/'actions.csv',index=False)
    pd.DataFrame(state['equity_history']).to_csv(directory/'equity.csv',index=False)
    pd.DataFrame(strata).to_csv(directory/'strata.csv',index=False)
    write_json(directory/'summary.json',summary)
    return summary


def evaluate_strategy(prepared, strategy, output, *, agent=None):
    from .fg_single3_experiment import DREAMER_ROOT,EXPERIMENT
    prepared = Path(prepared)
    dev = require_dev(prepared/'dev_manifest.json')
    output = Path(output)
    model_hash = getattr(agent,'evaluation_checkpoint_sha256',None) if strategy=='agent' else None
    if strategy=='agent' and not model_hash:
        raise ValueError('Actor evaluation must identify the verified trained checkpoint')
    if output.exists() and not (output/'evaluation_receipt.json').exists():
        # Preserve a failed evaluation; retrying never restarts training.
        output.rename(output.with_name(output.name+'.incomplete_'+uuid.uuid4().hex))
    if output.exists():
        receipt = read_manifest(output/'evaluation_receipt.json')
        if (receipt['strategy'] != strategy or receipt['dev_manifest_sha256'] != digest(prepared/'dev_manifest.json')
                or receipt.get('model_checkpoint_sha256') != model_hash):
            raise ValueError('Foreign evaluation result')
        check_files({str(output/p):h for p,h in receipt['files'].items()})
        return json.loads((output/'summary.json').read_text())
    row = dev['rows'][0]
    factory = lambda:load_env(row,dev['frozen_inputs'])
    kinds = {'always_wait':'always_wait','always_add':'always_add','random_add':'probabilistic'}
    if strategy=='agent':
        if agent is None:
            raise ValueError('Actual trained neural Agent is required')
        policy_factory = lambda _:ActorPolicy(agent)
    elif strategy in kinds:
        policy_factory = Single3BehaviorPolicy
    else:
        raise ValueError('Only the four agreed causal strategies are allowed')
    spec = dict(id=strategy,kind=kinds.get(strategy,'neural_actor'),seed_components=[0],probability=.5)
    identity = dict(role='DEV',experiment=EXPERIMENT,dev_hash=dev['content_hash'])
    if model_hash:
        identity['model_checkpoint_sha256'] = model_hash
    collect_trace(row['env_config'],spec,output,identity,str(DREAMER_ROOT),env_factory=factory,
        protocol='fg_single3_v1',policy_factory=policy_factory,runtime_factory=embodied_runtime)
    reconciliation = reconcile_trace(output,str(DREAMER_ROOT),env_factory=factory,
        schema=SINGLE3_SCHEMA,runtime_factory=embodied_runtime)
    if reconciliation['status']!='PASS':
        raise ValueError('DEV original-Env action replay failed')
    write_json(output/'reconciliation.json',signed(reconciliation))
    result = analyze_trace(output,dev,strategy)
    check_files(dev['frozen_inputs'])
    files = {str(p.relative_to(output)):digest(p) for p in output.rglob('*') if p.is_file()}
    check_files({str(output/p):h for p,h in files.items()})
    write_json(output/'evaluation_receipt.json',signed(dict(status='PASS',strategy=strategy,
        dev_manifest_sha256=digest(prepared/'dev_manifest.json'),files=files,
        model_checkpoint_sha256=model_hash,
        oracle_labels_used=False,test_prices_accessed=False)))
    return result


def require_training(prepared, *, load_agent_state=False):
    """Bind reported counters and logs to the authoritative full checkpoint."""
    from .fg_single3_experiment import require_prepared,UPDATES,DREAMER_ROOT,EXPERIMENT
    receipt = require_prepared(prepared)
    embodied_runtime(DREAMER_ROOT)
    from embodied.run.fg_checkpoint import validate_bundle
    logdir = Path(receipt['logdir'])
    summary = json.loads((logdir/'training_summary.json').read_text())
    expected = dict(protocol='FG_SINGLE3_OFFLINE_V1',identity_hash=receipt['identity']['identity_hash'])
    checkpoint = logdir/'ckpt'/f'step_{UPDATES:012d}'
    meta = validate_bundle(checkpoint,expected)
    if (summary['identity_hash'] != expected['identity_hash'] or summary['last_update'] != UPDATES
            or summary['status']!='TRAINING_COMPLETE' or summary['experiment']!=EXPERIMENT
            or summary['training_positions']!=UPDATES*16*64
            or Path(summary['checkpoint']).resolve()!=checkpoint.resolve()
            or meta['step'] != UPDATES or not meta['protected']
            or digest(checkpoint/'manifest.json') != summary['checkpoint_manifest_sha256']):
        raise ValueError('Final trained checkpoint/summary mismatch')
    state=pickle.loads((checkpoint/'training_audit.pkl').read_bytes())
    if not isinstance(state,dict) or any(summary.get(key)!=value for key,value in state.items()):
        raise ValueError('Training summary differs from checkpointed audit counters')
    check_files({str(logdir/'train_updates.jsonl'):summary['log_sha256']})
    data = pickle.loads((checkpoint/'agent.pkl').read_bytes())
    if data['counters']['updates'] != UPDATES:
        raise ValueError('Neural Agent update budget was not completed')
    return summary,data if load_agent_state else None


def load_evaluation_agent(prepared):
    from .fg_single3_experiment import DREAMER_ROOT
    embodied_runtime(DREAMER_ROOT)
    import elements
    from dreamerv3.fg_single3_offline import validate
    from dreamerv3.main import make_agent
    from dreamerv3.configuration import read_config_yaml
    config = elements.Config(read_config_yaml(Path(prepared)/'config.yaml'))
    validate(config,persist=False)
    summary,data=require_training(prepared,load_agent_state=True)
    # Full bundle is verified first. This is inference, not partial training resume.
    agent = make_agent(config.update({'jax.precompile':False,'jax.profiler':False}))
    agent.load(data,restore_batches=True)
    agent.sync_policy_from_train(reset_counter=True)
    agent.evaluation_checkpoint_sha256 = summary['checkpoint_manifest_sha256']
    for key in agent.policy_keys:
        np.testing.assert_array_equal(np.asarray(agent.policy_params[key]),np.asarray(agent.params[key]))
    return agent,summary


def comparison(results, training):
    by_name = {r['strategy']:r for r in results}
    if set(by_name) != {'always_wait','always_add','random_add','agent'}:
        raise ValueError('Missing agreed evaluation strategy')
    actor = by_name['agent']
    better = all(actor['reward_per_day']>r['reward_per_day'] and actor['net_pnl']>r['net_pnl']
        for name,r in by_name.items() if name!='agent')
    effective = training['actor_effective_updates']>0
    variation = (actor['active_wait']>0 and actor['legal_add_decisions']>0
        and actor['p_add_std'] is not None and actor['p_add_std']>1e-6)
    return dict(engineering='PASS',training='COMPLETE',actor_effective_training=effective,
        legal_state_action_variation=variation,beats_simple_strategies_on_reward_and_net_pnl=better,
        timing='EVIDENCE_SUPPORTED_IN_THIS_DEV' if effective and variation and better else 'NOT_ESTABLISHED',
        generalization='NOT_ESTABLISHED',results=results)


def invalidate_report(logdir):
    """Remove authoritative names while retaining every failed artifact."""
    from .fg_single3_experiment import EXPERIMENT
    logdir=Path(logdir)
    suffix=uuid.uuid4().hex
    for name,failed in (('completion_manifest.json','completion_manifest.failed_'+suffix+'.json'),
            (EXPERIMENT+'_Report.md',EXPERIMENT+'_Report.failed_'+suffix+'.md')):
        if (logdir/name).exists():
            (logdir/name).rename(logdir/failed)


def publish_report(logdir, body, result, metadata, external_files):
    """Publish completion last; preserve failed and corrupt attempts visibly."""
    from .fg_single3_experiment import EXPERIMENT
    logdir=Path(logdir)
    report_path=logdir/(EXPERIMENT+'_Report.md')
    marker=logdir/'completion_manifest.json'
    if marker.exists():
        try:
            existing=read_manifest(marker)
            if existing['identity_hash']!=metadata['identity_hash']:
                raise ValueError('Foreign completed report identity')
            check_files(existing['external_files'])
            check_files({str(logdir/p):h for p,h in existing['files'].items()})
            return report_path
        except BaseException:
            invalidate_report(logdir)
            raise
    try:
        check_files(external_files)
        report_path.write_text(body)
        write_json(logdir/'comparison.json',result)
        files={str(p.relative_to(logdir)):digest(p) for p in logdir.rglob('*') if p.is_file()
            and p.name!='completion_manifest.json'}
        check_files({str(logdir/p):h for p,h in files.items()})
        check_files(external_files)
        write_json(marker,signed(dict(metadata,status='COMPLETE',files=files,external_files=external_files)))
    except BaseException:
        invalidate_report(logdir)
        raise
    return report_path

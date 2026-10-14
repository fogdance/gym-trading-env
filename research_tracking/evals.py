"""Post-process immutable Actor-only receipts; never guide the Actor."""
from decimal import Decimal
from pathlib import Path

from .eval_schema import evaluation_metrics, unique_events
from .io import TrackingError, digest, fingerprint, read_json
from .metrics import number
from .composition import validate as validate_composition


def entry_count(state):
    if state.get('fills') is not None:
        return sum(f.get('reason')=='ENTRY' for f in state['fills'])
    return len(unique_events(state.get('attributions',[])))+len(state.get('entries',{}))


def parse_eval(source, receipt_path):
    path = Path(receipt_path).resolve()
    receipt = read_json(path, signed=True, limit=source.limit)
    composition = validate_composition(source.settings, receipt)
    if (receipt.get('status') != 'ACTOR_ONLY_DEV_DIAGNOSTIC_COMPLETE'
            or receipt.get('fold') != 'DEV' or receipt.get('teacher_guidance') is not False
            or receipt.get('oracle_files_read') is not False or receipt.get('beta') != 0
            or receipt.get('policy_mode') != 'eval_argmax'):
        raise TrackingError('Evaluation is not an authenticated Actor-only DEV receipt')
    node = Path(receipt['checkpoint']).resolve()
    if not node.is_relative_to((source.root/'ckpt').resolve()):
        raise TrackingError('Evaluation references a different training Run')
    available = node.is_dir() and (node/'done').is_file() and (node/'manifest.json').is_file()
    if available:
        meta = read_json(node/'manifest.json', limit=source.limit)
        if (meta.get('identity') != source.identity
                or digest(node/'manifest.json') != receipt['checkpoint_manifest_sha256']):
            raise TrackingError('Evaluation Checkpoint identity/hash differs')
        checkpoint_step = meta['step']
    else:
        checkpoint_step = int(node.name.removeprefix('step_')) if node.name.startswith('step_') else None
    if source.clock and receipt.get('clock_protocol') and source.clock != receipt['clock_protocol']:
        raise TrackingError('Evaluation and training clock identities differ')
    receipt_sha = digest(path)
    eval_id = source.publishing_id('eval', str(path), receipt_sha, receipt['checkpoint_manifest_sha256'])
    metrics = {'eval/checkpoint_env_steps': checkpoint_step}
    for key in ('env_steps', 'legal_decisions', 'actor_add_proposed', 'add_executed', 'entries',
                'tp_count', 'tp_target_points', 'sl30', 'reward', 'tp_market_minutes_p50',
                'tp_market_minutes_p90', 'right_censored', 'all_wait'):
        if receipt.get(key) is not None:
            metrics['eval/'+key] = receipt[key]
    for key in ('legal_argmax_changed', 'mix_active_decisions', 'p_add_base_mean', 'p_add_final_mean', 'delta_logit_mean'):
        value = receipt.get('b2', {}).get(key)
        if number(value):
            metrics['eval/b2/'+key] = value
    if receipt.get('early_mae10') is not None:
        metrics['eval/adverse10_any_hold'] = receipt['early_mae10']
    report = dict(schema='FG_S3_TRACKER_EVAL_DERIVED_V2', receipt_sha256=receipt_sha,
        receipt_content_hash=receipt['content_hash'], metric_schema='S3_EVAL_METRIC_V2',
        checkpoint_manifest_sha256=receipt['checkpoint_manifest_sha256'],
        parent_training_run_id=source.ids['train'], local_run_id=source.run_id,
        eval_run_id=eval_id, clock_protocol=receipt.get('clock_protocol'), contract=receipt.get('contract'),
        teacher_guidance=False, oracle_files_read=False, beta=0., policy_mode='eval_argmax',
        local_artifact_available=available, derivation='READ_ONLY_OFFLINE_RECALCULATION',
        metrics=metrics, trades=[], missing=[])
    account = path.parent/'account_state.json'
    state = None
    if not account.exists():
        report['missing'].append('account_state: no trade PnL or adverse-window reconstruction')
    else:
        try:
            state = read_json(account, limit=source.limit)
        except TrackingError:
            report['missing'].append('account_state exceeds metadata budget or is invalid')
        else:
            attrs = unique_events(state.get('attributions', []))
            entries = entry_count(state)
            if entries != receipt['entries']:
                raise TrackingError('Evaluation opening count differs from account evidence')
            metrics['eval/entries_recomputed'] = entries
            report['account_state_sha256'] = digest(account)
            if (state.get('trade_history') is not None and not state.get('entries')
                    and all(a.get('net_trade_pnl') is not None for a in attrs)):
                net = sum((Decimal(str(a['net_trade_pnl'])) for a in attrs), Decimal(0))
                history_net = sum((Decimal(str(t.get('pnl') or 0))-Decimal(str(t.get('fee') or 0))
                                   for t in state['trade_history']),Decimal(0))
                if history_net != net:
                    raise TrackingError('Evaluation trade history PnL/fees differ from attributions')
                metrics['eval/history_net_pnl'] = float(history_net)
            adverse = {e['entry_id']: e['effective_bar'] for e in state.get('reward_events', [])
                       if e.get('reason') == 'FIRST_ADVERSE10' and type(e.get('effective_bar')) is int}
            # Explicit small allowlist; no account IDs, prices, raw orders or timestamps.
            for i, attr in enumerate(attrs, 1):
                report['trades'].append(dict(trade_index=i, exit_reason=attr['reason'],
                    held_market_bars=attr.get('held_market_bars'),
                    net_pnl=float(attr['net_trade_pnl']) if attr.get('net_trade_pnl') is not None else None,
                    adverse_first_market_minute=adverse.get(attr['entry_id'])))
    try:
        derived = evaluation_metrics(state, receipt)
    except ValueError as exc:
        raise TrackingError(str(exc)) from exc
    metrics.update({'eval/'+k: v for k, v in derived.items() if k != 'metric_schema'})
    if not derived['adverse10_windows_complete']:
        report['missing'].append('complete certified market-minute adverse-window events')
    report['report_sha256'] = fingerprint(report)
    if composition is not None:
        report['resolved_composition'] = composition
        report['report_sha256'] = fingerprint({k:v for k,v in report.items() if k != 'report_sha256'})
    return report

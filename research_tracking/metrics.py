"""Preserve training semantics; undefined observations never become zeros."""
import math
from urllib.parse import quote

from .redaction import SECRET


def number(value):
    return type(value) in (int, float) and math.isfinite(value)


def metric_key(key):
    if key.startswith('bc/'):
        return 'train/' + key
    if key.startswith(('tutor/', 'b2/')):
        return 'train/' + key
    exact = {'opt/loss': 'train/optimizer/loss', 'v2/wm_loss': 'train/wm/loss',
             'opt/actor_grad_norm': 'train/actor/grad_norm',
             'opt/wm_grad_norm': 'train/wm/grad_norm',
             'opt/critic_grad_norm': 'train/critic/grad_norm'}
    if key in exact:
        return exact[key]
    return 'raw_train/' + quote(key, safe='/_.-')


def aggregate_updates(rows):
    last = rows[-1]
    result = {'progress/env_steps': last['env_steps'], 'progress/local_update': last['local_update'],
              'progress/agent_update': last['agent_update'], 'train/aggregation_updates': len(rows),
              'train/teacher_valid': sum(r['valid_count'] for r in rows),
              'train/teacher_add': sum(r['add_count'] for r in rows)}
    keys = set().union(*(r['metrics'] for r in rows))
    counts = ('bc/tp', 'bc/fp', 'bc/fn', 'bc/tn')
    weighted = {'bc/nll': 'bc/valid_count', 'tutor/natural_ce': 'bc/valid_count',
                'tutor/pair_accuracy': 'tutor/pair_count',
                'tutor/p_add_pos_mean': 'tutor/eligible_pos',
                'tutor/p_add_neg_mean': 'tutor/eligible_neg',
                'tutor/difficulty_weight_pos_mean':'tutor/eligible_pos',
                'tutor/difficulty_weight_neg_mean':'tutor/eligible_neg',
                'tutor/difficulty_weight_mean':'bc/valid_count',
                'tutor/difficulty_boosted_fraction':'bc/valid_count',
                'tutor/fn_rate': 'tutor/eligible_pos', 'tutor/fp_rate': 'tutor/eligible_neg'}
    for key in sorted(keys):
        if SECRET.search(key):
            continue
        pairs = [(r['metrics'][key], r['metrics']) for r in rows if number(r['metrics'].get(key))]
        if not pairs:
            continue
        target = metric_key(key)
        if (key in counts or key in ('tutor/eligible_pos', 'tutor/eligible_neg')
                or key.startswith('b2/count_') or key.endswith(('/valid_count', '/add_count', '/wait_count', '/pair_count',
                    '_valid_count', '_add_count', '_wait_count', '_pair_count', '_add_positives', '_add_true_positives'))):
            result[target] = sum(value for value, _ in pairs)
        elif key in ('opt/updates', 'train/update_index','b2/mix_stage','b2/mix_next',
                     'b2/schedule_updates_before','b2/schedule_event_updates_before','b2/event_preheat_ready'):
            result[target] = pairs[-1][0]
        elif key.startswith('tutor/difficulty_weight') and key.endswith(('_max','_min')):
            result[target] = (max if key.endswith('_max') else min)(value for value,_ in pairs)
        elif key in weighted or key.startswith('b2/recall_'):
            weight = weighted.get(key, 'b2/count_'+key.removeprefix('b2/recall_'))
            denominator = sum(m.get(weight, 0) for _, m in pairs)
            if denominator:
                result[target] = sum(v*m.get(weight, 0) for v, m in pairs)/denominator
        else:
            result[target] = sum(value for value, _ in pairs)/len(pairs)
        result['observations/' + target] = len(pairs)
    confusion = {k: sum(r['metrics'].get('bc/'+k, 0) for r in rows) for k in ('tp', 'fp', 'fn')}
    measured = [r for r in rows if number(r.get('oracle_add_recall'))]
    tp = sum(r['metrics'].get('bc/tp', 0) for r in measured)
    support = sum(r['metrics'].get('bc/tp', 0)+r['metrics'].get('bc/fn', 0) for r in measured)
    if support:
        result['train/oracle_add_recall'] = tp/support
        result['train/oracle_add_recall_support'] = support
    if confusion['tp'] + confusion['fp']:
        result['train/oracle_add_precision'] = confusion['tp']/(confusion['tp']+confusion['fp'])
    for key in ('tutor_pair_update_count', 'tutor_valid_update_count', 'tutor_pairable_batch_rate'):
        if number(last.get(key)):
            result['train/tutor/' + key.removeprefix('tutor_')] = last[key]
    return result


def cycle_metrics(row, *, provisional=False, segment=0, checkpoint_env_steps=0):
    result = {'progress/env_steps': row['env_steps'], 'progress/agent_update': row['agent_updates']}
    for key in ('collect_seconds', 'update_seconds', 'replay_items', 'retained_teacher_rows'):
        if number(row.get(key)):
            result['train/cycle/' + key] = row[key]
    seconds = (row['collect_seconds'] + row['update_seconds'] if
               all(number(row.get(k)) for k in ('collect_seconds', 'update_seconds')) else 0)
    # Use an interval only when its actual step count is supplied by the importer.
    if seconds > 0 and row.get('interval_env_steps') is not None:
        result['train/core_env_steps_per_second'] = row['interval_env_steps']/seconds
    for group in ('counts', 'gate'):
        for key, value in row.get(group, {}).items():
            if number(value):
                result[f'train/{group}/{key}'] = value
    if number(row.get('beta_schedule', {}).get('beta')):
        result['train/beta'] = row['beta_schedule']['beta']
    count = row.get('counts', {})
    if count.get('env_steps', 0) and number(count.get('certified_labels')):
        result['train/teacher_coverage'] = count['certified_labels']/count['env_steps']
    if provisional:
        result = {'provisional/' + k: v for k, v in result.items()}
        result.update({'provisional/is_provisional': True, 'provisional/segment': segment,
                       'provisional/latest_checkpoint_env_steps': checkpoint_env_steps})
    return result


def provisional_updates(rows, segment, checkpoint_env_steps):
    result = {'provisional/'+k: v for k, v in aggregate_updates(rows).items()}
    result.update({'provisional/is_provisional': True, 'provisional/segment': segment,
                   'provisional/latest_checkpoint_env_steps': checkpoint_env_steps})
    return result

"""DEV-only posthoc candidate scoring; sealed causal predictions come first.

No model, optimizer, Replay, or TRAIN teacher sidecar is used by this module.
Labels have an explicit evaluation-only role and cannot be joined as teachers.
"""
from collections import Counter
from decimal import Decimal
import json
from pathlib import Path
import time
import subprocess

import embodied
from embodied.envs.from_gymnasium import FromGymnasium
import numpy as np

from gym_trading_env.envs.entry_contract import fingerprint
from .fg_replay import check_files, digest, read_manifest, signed, write_json
from .fg_single3 import load_env
from .fg_single3_evaluation import require_dev
from .fg_single3_oracle import CandidateIndex, verify_candidates, optimize
from .fg_s3_b2_outcome_catalog import EVENTS, outcome_label
from .fg_s3_oracle_state_query import decision_state, rules_identity, outcome_account_reason
from .fg_s3_eval_metrics import evaluation_metrics, quantile

SCHEMA = 'S3_DEV_PREDICTION_EVAL_V1'
ROLE = 'DEV_EVALUATION_ONLY'


def probability(row):
    d = row.get('b2')
    if d is None:
        return None
    p, e = np.asarray(d['b2/outcome_probs'], float), float(d['b2/early_prob'])
    if (p.shape != (6,) or not np.isfinite(p).all() or (p < 0).any() or (p > 1).any()
            or not np.isclose(p.sum(), 1, atol=2e-5) or not np.isfinite(e) or not 0 <= e <= 1):
        raise ValueError('Invalid sealed causal outcome probabilities')
    return p, e


def require_prediction_mode(records, mode):
    if mode not in ('b2','tutor'):
        raise ValueError('Posthoc scorer supports the authenticated Tutor/B2 model modes')
    is_b2 = mode == 'b2'
    for row in records:
        if (probability(row) is not None) != is_b2:
            raise ValueError('Causal prediction diagnostics differ from Checkpoint model mode')
    return is_b2


def calibration(prob, correct):
    prob, correct = np.asarray(prob, float), np.asarray(correct, float)
    rows = []
    for i in range(10):
        valid = (prob >= i/10) & ((prob < (i+1)/10) if i < 9 else (prob <= 1))
        count = int(valid.sum())
        rows.append(dict(lower=i/10, upper=(i+1)/10, count=count,
            prediction_mean=float(prob[valid].mean()) if count else None,
            target_mean=float(correct[valid].mean()) if count else None))
    ece = sum(r['count']*abs(r['prediction_mean']-r['target_mean'])
              for r in rows if r['count'])/len(prob) if len(prob) else None
    return dict(bins=rows, ece=ece)


def outcome_scores(rows):
    """Matched-mask metrics; absent support stays null, not perfect accuracy."""
    event_rows = [r for r in rows if r.get('prediction') and r['label']['valid_event']]
    early_rows = [r for r in rows if r.get('prediction') and r['label']['valid_early']]
    out = dict(event_samples=len(event_rows), early_samples=len(early_rows),
        event_nll=None, event_brier=None, early_nll=None, early_brier=None,
        event_constant_prior_nll=None, event_constant_prior_brier=None,
        early_constant_prior_nll=None, early_constant_prior_brier=None,
        baseline_scope='empirical constant prior on the same matched DEV samples; descriptive, not TRAIN prior',
        event_calibration=calibration([], []), early_calibration=calibration([], []), classes={})
    if event_rows:
        p = np.asarray([r['prediction']['event'] for r in event_rows])
        y = np.asarray([r['label']['event'] for r in event_rows], int)
        target = np.eye(6)[y]
        pred = p.argmax(-1)
        prior = target.mean(0)
        out.update(event_nll=float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1)).mean()),
            event_brier=float(np.square(p-target).sum(-1).mean()),
            event_constant_prior_nll=float(-np.log(np.clip(prior[y], 1e-12, 1)).mean()),
            event_constant_prior_brier=float(np.square(prior-target).sum(-1).mean()),
            event_calibration=calibration(p.max(-1), pred == y))
        for i, name in enumerate(EVENTS):
            n, proposed, tp = int((y == i).sum()), int((pred == i).sum()), int(((y == i)&(pred == i)).sum())
            out['classes'][name] = dict(support=n, predictions=proposed,
                recall=tp/n if n else None, precision=tp/proposed if proposed else None,
                calibration=calibration(p[:,i], y == i))
    else:
        out['classes'] = {name:dict(support=0,predictions=0,recall=None,precision=None) for name in EVENTS}
    if early_rows:
        p = np.asarray([r['prediction']['early'] for r in early_rows])
        y = np.asarray([r['label']['early10'] for r in early_rows])
        prior = y.mean()
        def nll(prob):
            return float(-np.log(np.clip(np.where(y,prob,1-prob),1e-12,1)).mean())
        out.update(early_nll=nll(p), early_brier=float(np.square(p-y).mean()),
            early_constant_prior_nll=nll(prior), early_constant_prior_brier=float(np.square(prior-y).mean()),
            early_target_mean=float(prior), early_calibration=calibration(p,y))
    return out


def decision_scores(rows):
    keys = ('p_add', 'b2/p_add_base', 'b2/p_add_final', 'b2/delta_logit',
            'b2/adapter_unclipped_delta_logit', 'b2/adapter_clipped_delta_logit',
            'b2/base_logit_add_minus_wait', 'b2/final_logit_add_minus_wait',
            'tutor/base_logit_add_minus_wait', 'tutor/final_logit_add_minus_wait')
    result = dict(legal_samples=len(rows), argmax_changed=sum(bool(r.get('b2',{}).get('b2/argmax_changed')) for r in rows))
    for key in keys:
        values = [r['p_add'] if key == 'p_add' else r.get('b2',r.get('tutor',{})).get(key) for r in rows]
        values = [float(v) for v in values if v is not None]
        result[key] = dict(count=len(values), mean=float(np.mean(values)) if values else None,
            quantiles={str(q):quantile(values,q) for q in (0.,.5,.9,.99,1.)})
    known = [r['b2'] for r in rows if 'b2/clip_saturation' in r.get('b2',{})]
    result['clip_saturation_fraction'] = float(np.mean([r['b2/clip_saturation'] for r in known])) if known else None
    result['legacy_raw_delta_logit_semantics'] = 'clipped adapter delta; not an unclipped logit'
    return result


def score_predictions(eval_dir, output_dir):
    started = time.perf_counter()
    project = Path(__file__).resolve().parents[3]
    code = {str(Path(__file__).with_name(name)):digest(Path(__file__).with_name(name))
            for name in ('fg_s3_prediction_evaluation.py','fg_s3_oracle_state_query.py',
                         'fg_s3_b2_outcome_catalog.py','fg_single3_oracle.py','fg_s3_eval_metrics.py')}
    versions = {str(repo):dict(
            sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
            dirty=subprocess.check_output(['git','status','--porcelain'],cwd=repo,text=True).splitlines())
            for repo in (project,project.parent/'dreamerv3')}
    source, output = Path(eval_dir).resolve(), Path(output_dir).resolve()
    receipt = read_manifest(source/'receipt.json')
    if (receipt['status'] != 'ACTOR_ONLY_DEV_DIAGNOSTIC_COMPLETE' or receipt['fold'] != 'DEV'
            or receipt['teacher_guidance'] or receipt['oracle_files_read'] or receipt['beta'] != 0
            or receipt['policy_mode'] != 'eval_argmax' or receipt['clock_protocol'] != 'S3_MARKET_MINUTE'):
        raise ValueError('Posthoc scoring requires an authenticated causal Actor-only market-minute DEV run')
    checkpoint = Path(receipt['checkpoint']).resolve()
    meta = json.loads((checkpoint/'manifest.json').read_text())
    identity = read_manifest(checkpoint.parent.parent/'run_identity.json')['identity']
    if (not (checkpoint/'done').is_file() or meta['identity'] != identity
            or digest(checkpoint/'manifest.json') != receipt['checkpoint_manifest_sha256']):
        raise ValueError('Prediction source Checkpoint identity/hash differs')
    if output == source or output.is_relative_to(source):
        raise ValueError('Prediction scoring must use a new independent output directory')
    if not receipt.get('causal_prediction_files') and output.is_relative_to(checkpoint.parent.parent):
        raise ValueError('Historical run artifacts are read-only; select an external output directory')
    dev = require_dev(source/'dev_manifest.json')
    files = {str(source/name):digest(source/name) for name in
             ('receipt.json','actions.jsonl','account_state.json','dev_manifest.json','dev_env.yaml')}
    if receipt.get('causal_prediction_files'):
        check_files({str(source/name):sha for name,sha in receipt['causal_prediction_files'].items()})
    records = [json.loads(line) for line in (source/'actions.jsonl').read_text().splitlines()]
    if not records:
        raise ValueError('No sealed causal predictions')
    is_b2 = require_prediction_mode(records,identity['mode'])
    for r in records:
        p = probability(r)
        if not np.isfinite(r['p_add']) or not 0 <= r['p_add'] <= 1:
            raise ValueError('Invalid Actor score')
        if p is not None and r['p_add'] != r['b2']['b2/p_add_final']:
            raise ValueError('Stored outcome decision and Actor final score disagree')
    output.mkdir(parents=True, exist_ok=False)
    seal = signed(dict(schema=SCHEMA, role=ROLE, phase='SEALED_BEFORE_FUTURE_LABELS',
        source_files=files, source_receipt_content_hash=receipt['content_hash'],
        checkpoint_manifest_sha256=receipt['checkpoint_manifest_sha256'],
        source_run_id=identity['run_id'], predictions=len(records), future_labels_generated=False))
    seal = signed({**{k:v for k,v in seal.items() if k != 'content_hash'},
                   'posthoc_code_files':code,'posthoc_versions':versions})
    write_json(output/'prediction_seal.json', seal)
    env, index_env = None, None
    driver = None
    try:
        env = load_env(dev['rows'][0], dev['frozen_inputs'])
        if not env.single3 or env.reward_config.candidate != 'C1':
            raise ValueError('Prediction evaluation requires the original Single3 C1 rule')
        rules, market = rules_identity(env), env._raw.copy(deep=True)
        # The certificate is the same conservative mathematical envelope as
        # TRAIN; it does not turn DEV labels into training-certified labels.
        index_env = load_env(dev['rows'][0], dev['frozen_inputs'])
        index = CandidateIndex(index_env)
        floor = Decimal(index.proof['retained_paths_cash_lower_bound'])
        wrapped = embodied.wrappers.CheckSpaces(embodied.wrappers.UnifyDtypes(FromGymnasium(env, reset_seed=0)))
        driver = embodied.Driver([lambda:wrapped], parallel=False)
        rows, cursor = [], env.current_step
        def recorded_policy(carry, obs):
            r = records[len(rows)]
            return carry, dict(action=np.asarray([r['action']], np.int32)), {}
        def capture(step, worker):
            nonlocal cursor
            i = len(rows)
            r = records[i]
            delta = env.current_step-cursor
            cursor = env.current_step
            if (delta not in (0,1) or bool(delta) != r['advanced'] or cursor != r['cursor']
                    or bool(step['is_first']) != r['is_first'] or bool(step['is_last']) != r['is_last']
                    or int(step['action']) != r['action'] or not np.array_equal(step['action_mask'],r['mask'])
                    or not np.isclose(float(step['reward']),r['reward_from_previous_action'],atol=1e-7,rtol=0)):
                raise ValueError('Recorded Driver chronology/account diverged at row '+str(i))
            snapshot = decision_state(env)
            state_id = fingerprint(snapshot)
            if r.get('decision_state_id') and r['decision_state_id'] != state_id:
                raise ValueError('Causal prediction account state fingerprint differs')
            reason = outcome_account_reason(env,r['mask'],snapshot,rules=rules,market=market,
                floor=floor,covered=cursor < index.flat_end)
            rows.append(dict(row_index=i,cursor=cursor,decision_state_id=state_id,reason=reason,
                             executed=(i+1 < len(records) and records[i+1]['advanced'] and not r['is_last'])))
        driver.on_step(capture)
        driver(recorded_policy, steps=len(records))
        original = json.loads((source/'account_state.json').read_text())
        if fingerprint(env.export_state()) != fingerprint(original):
            raise ValueError('Replayed actual final account differs from sealed evaluation')
        write_json(output/'account_proof.json', signed(dict(schema=SCHEMA,role=ROLE,
            phase='ACTUAL_ACTIONS_REPLAYED_BEFORE_LABEL_QUERY',rule_identity=rules,
            funds_certificate=index.proof,funding_certificate_usage_role=ROLE,
            prediction_seal_hash=seal['content_hash'], decisions=rows)))
        eligible = [r for r in rows if r['reason'] is None and r['executed']]
        cursors = sorted({r['cursor'] for r in eligible}) if is_b2 else range(index.flat_end)
        candidates = {t:index.candidate(t) for t in cursors}
        check = verify_candidates(index,list(candidates.values()),count=8) if candidates else dict(status='NO_CERTIFIABLE_SAMPLES')
        oracle_first = None
        if not is_b2:
            plan = optimize(list(candidates.values()),index.flat_end,env.reward_config.candidate,
                clock=index.clock,return_suffix=True)
            oracle_first = plan['suffix']['first']
        labeled = []
        for r in eligible:
            record = records[r['row_index']]
            c = candidates[r['cursor']]
            label = outcome_label(c)
            p = probability(record)
            labeled.append(dict(**r,role=ROLE,candidate_identity=fingerprint(c),label=label,
                prediction=dict(event=p[0].tolist(),early=p[1]) if p is not None else None,
                actor_action=record['action'],p_add=record['p_add'],
                oracle_action=int(oracle_first[r['cursor']] == r['cursor']) if oracle_first is not None else None))
        with (output/'labeled_predictions.jsonl').open('x') as stream:
            for row in labeled:stream.write(json.dumps(row,allow_nan=False)+'\n')
        legal = [records[r['row_index']] for r in rows if r['executed'] and all(records[r['row_index']]['mask'])]
        reason_counts = Counter(r['reason'] or 'CERTIFIED_ACCOUNT' for r in rows if r['executed'])
        result = dict(schema=SCHEMA,status='DEV_PREDICTION_EVALUATION_COMPLETE',role=ROLE,
            evaluation_kind='outcome_prediction' if is_b2 else 'tutor_scoring',fold='DEV',
            oracle_files_read_in_policy=False,future_data_read_posthoc=True,eligible_for_training=False,
            policy_observations_modified=False,clock_protocol='S3_MARKET_MINUTE',
            source_run_id=identity['run_id'],source_eval_receipt_sha256=files[str(source/'receipt.json')],
            source_eval_receipt_content_hash=receipt['content_hash'],
            source_eval_dir=str(source),checkpoint_manifest_sha256=receipt['checkpoint_manifest_sha256'],
            checkpoint_env_steps=meta['step'],prediction_seal_hash=seal['content_hash'],
            source_files=files,source_data_identity=dev['rows'][0],rule_identity=rules,
            agent_extension=receipt.get('agent_extension',identity.get('agent_extension')),
            original_env_candidate_verification=check,account_replay_exact=True,
            certified_accounts=len(eligible),executed_decisions=sum(r['executed'] for r in rows),
            effective_supervision_coverage=len(eligible)/max(sum(r['executed'] for r in rows),1),
            account_rejection_counts=dict(reason_counts),outcome=outcome_scores(labeled),
            decision=decision_scores(legal),actor_only=evaluation_metrics(original,receipt),
            research_success='NOT_ESTABLISHED',elapsed_seconds=time.perf_counter()-started)
        result['posthoc_code_files'],result['posthoc_versions'] = code,versions
        if not is_b2:
            groups = {'oracle_add':[r for r in labeled if r['oracle_action']==1],
                      'oracle_wait':[r for r in labeled if r['oracle_action']==0]}
            result['tutor'] = {k:dict(count=len(v),p_add_mean=float(np.mean([r['p_add'] for r in v])) if v else None,
                p_add_quantiles={str(q):quantile([r['p_add'] for r in v],q) for q in (0.,.5,.9,.99,1.)})
                                for k,v in groups.items()}
            tp = sum(r['actor_action']==r['oracle_action']==1 for r in labeled)
            pred = sum(r['actor_action']==1 for r in labeled)
            positives = len(groups['oracle_add'])
            result['tutor'].update(add_recall=tp/positives if positives else None,
                add_precision=tp/pred if pred else None,false_positive_count=pred-tp)
            for name, members in dict(actor_add=[r for r in labeled if r['actor_action']==1],
                    false_positive=[r for r in labeled if r['actor_action']==1 and r['oracle_action']==0]).items():
                result['tutor'][name+'_certified_candidate_outcomes'] = dict(
                    samples=len(members),event_counts=dict(Counter(EVENTS[r['label']['event']]
                        for r in members if r['label']['valid_event'])),
                    event_censored=sum(not r['label']['valid_event'] for r in members),
                    early10_count=sum(r['label']['valid_early'] and r['label']['early10'] for r in members))
        check_files(files)
        check_files(code)
        check_files(dev['frozen_inputs'])
        result['files'] = {p.name:digest(p) for p in output.iterdir() if p.is_file()}
        result = signed(result)
        write_json(output/'receipt.json',result)
        return result
    finally:
        if driver:driver.close()
        elif env:env.close()
        if index_env:index_env.close()

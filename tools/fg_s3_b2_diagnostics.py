#!/usr/bin/env python3
"""Read-only B2 checkpoint ablation and DEV-only posthoc diagnosis.

The neural runtime is imported from the original frozen producer pair. This
tool never trains, writes to the producer, or publishes evaluation to W&B.
"""
import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


@contextmanager
def adapter_off(schedule_class, zeros_like):
    """Read the original variables, then zero only the effective mix.

    Reading still registers the original Ninjax variables at initialization.
    Counter values, NN keys/weights and masks are not rewritten.
    """
    original = schedule_class.read
    def read(instance):
        return zeros_like(original(instance))
    schedule_class.read = read
    try:
        yield
    finally:
        schedule_class.read = original


def bootstrap(gym_root, gpu):
    gym_root = Path(gym_root).resolve()
    dreamer = gym_root.parent/'dreamerv3'
    os.environ['JAX_PLATFORMS'] = 'cuda' if gpu else 'cpu'
    os.environ.pop('JAX_PLATFORM_NAME', None)
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    sys.dont_write_bytecode = True
    os.environ['WANDB_MODE'] = 'disabled'
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/s3_b2_diag_mpl')
    sys.path[:0] = [str(gym_root/'src'), str(dreamer)]
    import gym_trading_env, dreamerv3, embodied
    for module, directory in ((gym_trading_env,gym_root/'src/gym_trading_env'),
                              (dreamerv3,dreamer/'dreamerv3'),(embodied,dreamer/'embodied')):
        if Path(module.__file__).resolve().parent != directory:
            raise ValueError('Diagnostic must use the frozen producer runtime')
    return gym_root


def evaluate_off(gym_root, prepared, source_run, output, steps):
    import jax.numpy as jnp
    import numpy as np
    from dreamerv3.fg_s3_b2_outcome import MixSchedule
    from dreamerv3.fg_s3a_actor import parameter_digests
    from gym_trading_env.research import fg_s3_v2_evaluation as evaluation
    from gym_trading_env.research.fg_replay import read_manifest, signed, write_json
    source_run, output = Path(source_run).resolve(), Path(output).resolve()
    name = (source_run/'ckpt/latest').read_text().strip()
    if Path(name).name != name:
        raise ValueError('Invalid checkpoint pointer')
    checkpoint = source_run/'ckpt'/name
    identity = read_manifest(source_run/'run_identity.json')['identity']
    if identity['mode'] != 'b2' or not (checkpoint/'done').is_file():
        raise ValueError('A completed original B2 checkpoint is required')
    source_files = {str(p):digest(p) for p in (
        checkpoint/'manifest.json', source_run/'run_identity.json')}
    output.mkdir(parents=True, exist_ok=False)
    (output/'ckpt').mkdir()
    (output/'run_identity.json').symlink_to(source_run/'run_identity.json')
    (output/'ckpt'/name).symlink_to(checkpoint, target_is_directory=True)
    (output/'ckpt/latest').write_text(name+'\n')
    runtime = {str(p):dict(sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=p,text=True).strip(),
        dirty=subprocess.check_output(['git','status','--porcelain'],cwd=p,text=True).splitlines())
        for p in (gym_root,gym_root.parent/'dreamerv3')}
    if any(v['dirty'] for v in runtime.values()):
        raise ValueError('Frozen producer worktrees must remain clean')
    audit, models = {}, []
    original_factory = evaluation.make_agent
    def audited_factory(*args, **kwargs):
        model = original_factory(*args, **kwargs)
        original_load = model.load_exact
        def load(data, *a, **kw):
            expected = parameter_digests(data['params'])
            result = original_load(data, *a, **kw)
            actual = parameter_digests(model.params)
            if actual != expected:
                raise ValueError('Diagnostic changed restored checkpoint parameters')
            audit['loaded_param_digests'] = actual
            audit['original_counters'] = data['counters']
            return result
        model.load_exact = load
        models.append(model)
        return model
    evaluation.make_agent = audited_factory
    try:
        with adapter_off(MixSchedule, jnp.zeros_like):
            receipt = evaluation.evaluate(prepared, output, steps)
    finally:
        evaluation.make_agent = original_factory
    if parameter_digests(models[0].params) != audit['loaded_param_digests']:
        raise ValueError('Evaluation modified model/optimizer/normalizer parameters')
    actions = [json.loads(r) for r in (Path(receipt['output'])/'actions.jsonl').read_text().splitlines()]
    if not all(r['b2']['b2/mix_value'] == 0 and r['b2']['b2/delta_logit'] == 0
               and np.isclose(r['b2']['b2/p_add_base'],r['b2']['b2/p_add_final'],atol=1e-7)
               for r in actions):
        raise ValueError('Adapter OFF did not equal the base masked Actor')
    if any(digest(p) != h for p,h in source_files.items()):
        raise ValueError('Original source artifact changed')
    metadata = dict(diagnostic_only=True, adapter_mode='OFF',
        intervention='process-local MixSchedule.read returns zero after reading original variables',
        checkpoint_parameters_unchanged=True, policy_observations_modified=False,
        source_run=str(source_run), runtime_versions=runtime, source_files=source_files,
        diagnostic_tool_sha256=digest(__file__),eligible_for_training=False,wandb_uploaded=False)
    updated = signed({**{k:v for k,v in receipt.items() if k!='content_hash'},**metadata})
    write_json(Path(receipt['output'])/'receipt.json',updated)
    write_json(output/'diagnostic_model_audit.json',signed(dict(**metadata,**audit,
        restored_schema_and_all_param_equality=True, post_evaluation_all_param_equality=True)))
    return updated


def posthoc(on_dir, scored, output):
    """Four independent ADD intervals plus exact suffix Oracle, no execution policy."""
    from gym_trading_env.research.fg_replay import read_manifest, check_files, signed, write_json
    from gym_trading_env.research.fg_single3_evaluation import require_dev
    from gym_trading_env.research.fg_single3 import load_env
    from gym_trading_env.research.fg_single3_oracle import CandidateIndex, optimize, verify_candidates, GAMMA
    from gym_trading_env.research.fg_s3_b2_outcome_catalog import EVENTS, outcome_label
    from gym_trading_env.envs.entry_contract import fingerprint
    on_dir, scored, output = Path(on_dir).resolve(), Path(scored).resolve(), Path(output).resolve()
    receipt = read_manifest(on_dir/'receipt.json')
    scores = read_manifest(scored/'receipt.json')
    if (scores['source_eval_receipt_sha256'] != digest(on_dir/'receipt.json')
            or not scores['account_replay_exact'] or scores['eligible_for_training']):
        raise ValueError('Posthoc diagnosis requires the exact original account replay proof')
    check_files({str(on_dir/n):h for n,h in receipt['causal_prediction_files'].items()})
    check_files({str(scored/n):h for n,h in scores['files'].items()})
    records = [json.loads(line) for line in (on_dir/'actions.jsonl').read_text().splitlines()]
    cases = [r for r in records if all(r['mask']) and r['b2']['b2/argmax_changed']]
    if not cases or any(r['action'] != 0 for r in cases):
        raise ValueError('Expected ADD-to-WAIT vetoes')
    labels = {r['cursor']:r for r in (json.loads(l) for l in
        (scored/'labeled_predictions.jsonl').read_text().splitlines())}
    if any(r['cursor'] not in labels for r in cases):
        raise ValueError('A veto lies outside the certified actual-account domain')
    dev = require_dev(on_dir/'dev_manifest.json')
    output.mkdir(parents=True,exist_ok=False)
    env = load_env(dev['rows'][0], dev['frozen_inputs'])
    try:
        index = CandidateIndex(env)
        cursors = {r['cursor'] for r in cases}
        selected, candidates = {}, []
        with (output/'dev_candidates.jsonl').open('x') as stream:
            for t in range(index.flat_end):
                candidate = index.candidate(t)
                candidates.append(candidate)
                stream.write(json.dumps(candidate)+'\n')
                if t in cursors:
                    selected[t] = candidate
                if t and t % 5000 == 0:
                    print('Posthoc candidates',t,'/',index.flat_end,flush=True)
        verification = verify_candidates(index,list(selected.values()),count=len(selected))
        plan = optimize(candidates,index.flat_end,env.reward_config.candidate,
                        clock=index.clock,return_suffix=True)
        first = plan.pop('suffix')['first']
        rows=[]
        for r in cases:
            t=r['cursor'];c=selected[t];label=outcome_label(c)
            sealed=labels[t]
            if label != sealed['label'] or fingerprint(c) != sealed['candidate_identity']:
                raise ValueError('Candidate differs from sealed prediction scoring')
            ex=c.get('exit');b=r['b2']
            rows.append(dict(cursor=t,signal_time=str(env._raw.index[t]),
                actual_decision_state_id=r['decision_state_id'],
                base_add_logit_margin=b['b2/base_logit_add_minus_wait'],
                final_add_logit_margin=b['b2/final_logit_add_minus_wait'],
                base_p_add=b['b2/p_add_base'],final_p_add=b['b2/p_add_final'],
                adapter_delta=b['b2/delta_logit'],adapter_unclipped=b['b2/adapter_unclipped_delta_logit'],
                clip_saturated=b['b2/clip_saturation'],outcome_prediction=dict(zip(EVENTS,b['b2/outcome_probs'])),
                early10_prediction=b['b2/early_prob'],adapter_parts=dict(zip(EVENTS,b['b2/adapter_parts'])),
                adapter_early_part=b['b2/adapter_early'],
                outcome_at_h60=EVENTS[label['event']] if label['valid_event'] else None,
                early10_within_first5=label['early10'] if label['valid_early'] else None,
                first_adverse_market_bar=c['first_adverse_effective_bar'],
                independent_entry_cursor=c['entry_cursor'],entry_time=c['entry_time'],
                exit_reason=ex['reason'] if ex else None,exit_time=ex['exit_time'] if ex else None,
                held_market_minutes=c['held_market_bars'],tp_market_minutes=c['tp_market_minutes'],
                net_trade_pnl=float(ex['net_trade_pnl']) if ex else None,
                c1_reward_sum=sum(v for _,v in c['rewards']['C1']),
                c1_discounted_reward=sum(float(GAMMA)**index.clock.reward_exponent(t,s)*v for s,v in c['rewards']['C1']),
                oracle_add_now=bool(first[t]==t),oracle_first_optimal_add_cursor=first[t],
                right_censored=c['right_censored'],role='DEV_EVALUATION_ONLY',eligible_for_training=False))
        result=signed(dict(status='B2_VETO_POSTHOC_DIAGNOSIS_COMPLETE',role='DEV_EVALUATION_ONLY',
            checkpoint_manifest_sha256=receipt['checkpoint_manifest_sha256'],
            future_used_posthoc_only=True,oracle_used_for_execution=False,eligible_for_training=False,
            source_eval_receipt_sha256=digest(on_dir/'receipt.json'),
            source_prediction_receipt_sha256=digest(scored/'receipt.json'),
            original_env_reference_verification=verification,funding_certificate=index.proof,
            oracle_plan=plan,independent_counterfactual_cases=rows,
            caveat='Independent flat ADD intervals are not a sequential Adapter-OFF trading trajectory.',
            files={'dev_candidates.jsonl':digest(output/'dev_candidates.jsonl')},tool_sha256=digest(__file__)))
        write_json(output/'receipt.json',result)
        return result
    finally:
        env.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('off','posthoc'))
    p.add_argument('--source-gym',required=True)
    p.add_argument('--prepared');p.add_argument('--source-run')
    p.add_argument('--on-eval');p.add_argument('--scores');p.add_argument('--output',required=True)
    p.add_argument('--steps',type=int,default=5000)
    args=p.parse_args()
    if args.command=='off' and os.environ.get('LD_LIBRARY_PATH'):
        env=dict(os.environ);env.pop('LD_LIBRARY_PATH')
        os.execve(sys.executable,[sys.executable,*sys.argv],env)
    root=bootstrap(args.source_gym,args.command=='off')
    if args.command=='off':
        if not args.prepared or not args.source_run:p.error('off requires prepared and source-run')
        result=evaluate_off(root,args.prepared,args.source_run,args.output,args.steps)
    else:
        if not args.on_eval or not args.scores:p.error('posthoc requires sealed on-eval and scores')
        result=posthoc(args.on_eval,args.scores,args.output)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()

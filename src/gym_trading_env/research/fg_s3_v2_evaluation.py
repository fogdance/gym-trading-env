"""Independent, bounded Actor-only evaluation with no Oracle file access."""
import copy
import json
from pathlib import Path
import pickle
import uuid

import embodied
import jax
import numpy as np
import pandas as pd
import yaml

from embodied.envs.from_gymnasium import FromGymnasium
from embodied.run.fg_checkpoint import file_hash, validate_bundle
from embodied.run.fg_s3_source_transfer import make_agent
from embodied.run.fg_s3_v2_train import execution_code
from dreamerv3.fg_s3_v2_agent import S3V2Agent
from dreamerv3.fg_s3_tutor_loss import agent_extensions
from .fg_single3 import load_env
from .fg_single3_evaluation import ActorPolicy, require_dev
from .fg_replay import read_manifest, check_files, signed, write_json
from .fg_s3_eval_metrics import evaluation_metrics


def prepare_dev(s, directory):
    """Reuse the existing signed FG409 DEV member and frozen C1 TRAIN rules."""
    source = read_manifest(s['dev_source_manifest'])
    roles = source['roles']
    rows = [r for r in source['files'] if r['role'] == 'validation']
    if (len(rows) != 1 or rows[0]['contract'] != 'CZCE.FG409'
            or rows[0]['eligible_for_training'] or max(roles['train']) >= min(roles['validation'])):
        raise ValueError('Expected the existing independent FG409 DEV member')
    row = copy.deepcopy(rows[0])
    inputs = read_manifest(Path(s['env_receipt'])/'inputs/inputs.json')
    train_env = yaml.safe_load(Path(inputs['views']['C1']['rows'][0]['env_config']).read_text())
    master = Path(train_env['entry']['strategy_config'])
    # The rule snapshot is a data artifact. Never read the external producer's
    # source repository or its mutable config tree as a code baseline.
    if str(master) not in inputs['frozen_files']:
        raise ValueError('DEV rule snapshot is not an authenticated TRAIN input')
    check_files({str(master): inputs['frozen_files'][str(master)]})
    rules = yaml.safe_load(master.read_text())
    env = yaml.safe_load(Path(row['env_config']).read_text())
    env['entry'].update(protocol='fg_single3_v1', strategy_config=str(master),
                        contract_config=row['contract_config'], chunk_steps=1024)
    env['trading'].update(rules['environment']['trading'])
    config = directory/'dev_env.yaml'
    config.write_text(yaml.safe_dump(env, sort_keys=False))
    frozen = {s['dev_source_manifest']: file_hash(s['dev_source_manifest']),
        str(master): file_hash(master), row['path']: row['sha256'],
        row['env_config']: row['env_sha256'], row['contract_config']: row['contract_sha256'],
        str(config): file_hash(config)}
    check_files(frozen)
    row.update(env_config=str(config), env_sha256=file_hash(config))
    manifest = signed(dict(role='DEV', contract='CZCE.FG409', rows=[row], frozen_inputs=frozen,
        train_days=roles['train'], dev_days=roles['validation'],
        train_last_timestamp=max(r['last_timestamp'] for r in source['files'] if r['role'] == 'train'),
        dev_first_timestamp=row['first_timestamp'], test_prices_accessed=False,
        oracle_labels_used=False, strategies=['agent']))
    write_json(directory/'dev_manifest.json', manifest)
    return require_dev(directory/'dev_manifest.json')


def evaluate(prepared, run_dir, steps=1000):
    if type(steps) is not int or not 1 <= steps <= 10000:
        raise ValueError('Actor-only diagnosis is bounded to 1..10000 actual advances')
    prepared, root = Path(prepared).resolve(), Path(run_dir).resolve()
    preparation = read_manifest(prepared/'manifest.json')
    check_files({str(prepared/k): v for k, v in preparation['files'].items()})
    identity = read_manifest(root/'run_identity.json')['identity']
    if identity['code'] != execution_code() or identity['preparation_sha256'] != file_hash(prepared/'manifest.json'):
        raise ValueError('Evaluation code/preparation differs from the trained run')
    name = (root/'ckpt/latest').read_text().strip()
    if Path(name).name != name:
        raise ValueError('Checkpoint pointer escaped run')
    node = root/'ckpt'/name
    validate_bundle(node, identity)
    s = read_manifest(prepared/'settings.json')['settings']
    frozen_model = read_manifest(prepared/'model_config.json')
    cfg = frozen_model['config']
    extensions = agent_extensions(s)
    if s['mode'] in ('tutor', 'b2') and frozen_model['agent_extension'] != extensions:
        raise ValueError('Frozen extended model configuration differs from settings')
    output = root/'evaluations'/('actor_only_'+uuid.uuid4().hex[:10])
    output.mkdir(parents=True, exist_ok=False)
    dev = prepare_dev(s, output)
    model = make_agent(cfg, output/'model', agent_cls=S3V2Agent, device='gpu',
        bc_lambda=s['bc_lambda'], extra_agent_config=extensions)
    model.load_exact(pickle.loads((node/'agent.pkl').read_bytes()))
    model.sync_policy_from_train()
    model.evaluation_checkpoint_sha256 = file_hash(node/'manifest.json')
    policy = ActorPolicy(model)
    env = load_env(dev['rows'][0], dev['frozen_inputs'])
    wrapped = embodied.wrappers.CheckSpaces(embodied.wrappers.UnifyDtypes(FromGymnasium(env, reset_seed=0)))
    driver = embodied.Driver([lambda: wrapped], parallel=False)
    driver.reset(model.init_policy)
    rows = []
    advanced, cursor = 0, env.current_step
    def capture(step, worker):
        nonlocal advanced, cursor
        delta = env.current_step - cursor
        if delta not in (0, 1):
            raise ValueError('Evaluation account chronology changed')
        advanced += delta
        cursor = env.current_step
        rows.append(dict(cursor=cursor, action=int(step['action']),
            mask=np.asarray(step['action_mask']).tolist(), is_first=bool(step['is_first']),
            is_last=bool(step['is_last']), reward_from_previous_action=float(step['reward']),
            p_add=policy.last_diagnostics['p_add'], advanced=bool(delta)))
        if s['mode'] == 'b2':
            rows[-1]['b2'] = policy.last_diagnostics['b2']
    driver.on_step(capture)
    try:
        while advanced < steps:
            driver(policy, steps=1)
            if env.terminated or env.status in ('DATA_END', 'DATA_GAP'):
                break
        exits = env.attributions
        tp = [e for e in exits if e['reason'] == 'TAKE_PROFIT']
        minutes = [e['tp_market_minutes'] for e in tp]
        legal = [r for r in rows if r['mask'][1] and not r['is_last']]
        # The final proposal is not executed at the budget boundary.
        executed = rows[:-1]
        entries = [f for f in env.fills if f['reason'] == 'ENTRY']
        with (output/'actions.jsonl').open('x') as f:
            for r in rows:
                f.write(json.dumps(r)+'\n')
        write_json(output/'account_state.json', env.export_state())
        result = signed(dict(status='ACTOR_ONLY_DEV_DIAGNOSTIC_COMPLETE', output=str(output),
            fold='DEV', contract=env.contract, env_steps=advanced,
            checkpoint=str(node), checkpoint_manifest_sha256=file_hash(node/'manifest.json'),
            devices=[str(d) for d in jax.devices()], oracle_files_read=False, teacher_guidance=False,
            beta=0., policy_mode='eval_argmax', legal_decisions=len(legal),
            actor_add_proposed=sum(r['action'] == 1 for r in legal),
            add_executed=sum(r['action'] == 1 for r in executed), entries=len(entries),
            tp_target_points=5, tp_count=len(tp), sl30=sum(e['reason'] == 'STOP_LOSS' for e in exits),
            clock_protocol='S3_MARKET_MINUTE',
            tp_market_minutes_p50=float(np.quantile(minutes,.5)) if minutes else None,
            tp_market_minutes_p90=float(np.quantile(minutes,.9)) if minutes else None,
            early_mae10=sum(e['reason'] == 'FIRST_ADVERSE10' for e in env.reward_events),
            reward=float(env.total_reward), right_censored=len(env.entries),
            all_wait=bool(legal and not any(r['action'] for r in legal)),
            research_success='NOT_ESTABLISHED', test='NOT_RUN'))
        check_files(dev['frozen_inputs'])
        if s['mode'] == 'b2':
            result = {k:v for k,v in result.items() if k != 'content_hash'}
            decision = [r['b2'] for r in legal]
            result['b2'] = dict(agent_extension=extensions,
                legal_argmax_changed=sum(d['b2/argmax_changed'] for d in decision),
                mix_active_decisions=sum(d['b2/mix_value'] > 0 for d in decision),
                p_add_base_mean=float(np.mean([d['b2/p_add_base'] for d in decision])) if decision else None,
                p_add_final_mean=float(np.mean([d['b2/p_add_final'] for d in decision])) if decision else None,
                delta_logit_mean=float(np.mean([d['b2/delta_logit'] for d in decision])) if decision else None,
                outcome_probabilities_source='causal RSSM predictions; no labels/Oracle queries')
            result = signed(result)
        result = {k:v for k,v in result.items() if k != 'content_hash'}
        result.update(evaluation_metrics(env.export_state(), result))
        result = signed(result)
        write_json(output/'receipt.json', result)
        return result
    finally:
        driver.close()

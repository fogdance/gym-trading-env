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
from dreamerv3.fg_s3_composition import identity_fields, settings_composition, validate_records
from .fg_single3 import load_env
from .fg_single3_evaluation import ActorPolicy, require_dev
from .fg_replay import read_manifest, check_files, signed, write_json
from .fg_s3_eval_metrics import evaluation_metrics
from .fg_s3_oracle_state_query import decision_state
from gym_trading_env.envs.entry_contract import fingerprint


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
    original_master=master
    if s['mode']=='quality_v2':
        rules=copy.deepcopy(rules);rules['reward']=s['reward']
        master=directory/'dev_single3.yaml'
        master.write_text(yaml.safe_dump(rules,sort_keys=False))
    env = yaml.safe_load(Path(row['env_config']).read_text())
    env['entry'].pop('state_path',None)
    env['entry'].update(protocol='fg_single3_v1', strategy_config=str(master),
                        contract_config=row['contract_config'], chunk_steps=1024)
    env['trading'].update(rules['environment']['trading'])
    config = directory/'dev_env.yaml'
    config.write_text(yaml.safe_dump(env, sort_keys=False))
    frozen = {s['dev_source_manifest']: file_hash(s['dev_source_manifest']),
        str(original_master):file_hash(original_master),
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


def require_quality_evaluation_snapshot(path, *, identity=None):
    """Authenticate DEV or a disjoint holdout of the new quality supervision."""
    path=Path(path)
    manifest=read_manifest(path)
    if manifest.get('role')=='DEV': return require_dev(path)
    if (manifest.get('protocol')!='FG_S3_QUALITY_TRAIN_HOLDOUT_V1' or
            manifest.get('role')!='TRAIN_HOLDOUT' or not manifest.get('evaluation_only') or
            manifest.get('train_join_allowed') is not False or len(manifest['rows'])!=1):
        raise ValueError('Expected an authenticated evaluation-only TRAIN holdout')
    check_files(manifest['frozen_inputs'])
    inputs=read_manifest(manifest['source_inputs'])
    contract=manifest['rows'][0]['contract']
    whitelist={r['contract']:r for r in inputs['views']['C1']['rows']
               if r['role']=='train' and r['eligible_for_training']}
    trained=set(manifest['trained_contracts'])
    if (inputs['original_content_hash']!=manifest['source_training_hash'] or
            contract not in whitelist or contract in trained or
            manifest['rows'][0]['sha256']!=whitelist[contract]['sha256'] or
            manifest['rows'][0]['path']!=whitelist[contract]['path']):
        raise ValueError('TRAIN holdout/source whitelist is not disjoint and exact')
    if identity is not None and (trained!=set(identity['quality_training_contracts']) or
            manifest['source_training_hash']!=identity['training_identity']['training_hash']):
        raise ValueError('TRAIN holdout differs from the trained Checkpoint source')
    return manifest


def prepare_holdout(settings,directory,contract,identity):
    """Reuse the signed TRAIN derivation, without building a TRAIN teacher."""
    if settings['mode']!='quality_v2': raise ValueError('Holdout evaluation is Quality-only')
    trained=identity['quality_training_contracts']
    if contract in trained: raise ValueError('A trained quality contract cannot be its holdout')
    if settings.get('quality_fixture_env'):
        raise ValueError('A single diagnostic fixture is not a disjoint TRAIN holdout')
    from .fg_s3_quality_experiment_v2 import _derive_rows
    selected={**settings,'contracts':[contract],'contract':contract}
    rows,frozen,training_hash,_=_derive_rows(selected,directory)
    inputs_path=str(Path(settings['env_receipt'])/'inputs/inputs.json')
    frozen[inputs_path]=file_hash(inputs_path)
    manifest=signed(dict(protocol='FG_S3_QUALITY_TRAIN_HOLDOUT_V1',role='TRAIN_HOLDOUT',
        evaluation_only=True,train_join_allowed=False,trained_contracts=trained,
        source_training_hash=training_hash,source_inputs=inputs_path,
        rows=rows,frozen_inputs=frozen,test_prices_accessed=False,oracle_labels_used=False))
    path=directory/'holdout_manifest.json';write_json(path,manifest)
    return require_quality_evaluation_snapshot(path,identity=identity)


def evaluate(prepared, run_dir, steps=1000,output_dir=None,holdout_contract=None):
    if type(steps) is not int or not 1 <= steps <= 10000:
        raise ValueError('Actor-only diagnosis is bounded to 1..10000 actual advances')
    prepared, root = Path(prepared).resolve(), Path(run_dir).resolve()
    preparation = read_manifest(prepared/'manifest.json')
    s = read_manifest(prepared/'settings.json')['settings']
    settings_composition(s)
    check_files({str(prepared/k): v for k, v in preparation['files'].items()})
    identity = read_manifest(root/'run_identity.json')['identity']
    frozen_model = read_manifest(prepared/'model_config.json')
    spec = read_manifest(prepared/'source_admission.json')
    documents = [preparation, frozen_model, spec, identity]
    if 'resolved_composition' in s:
        documents.append(spec['target_model_identity'])
    composition = validate_records(s, *documents)
    if identity['code'] != execution_code() or identity['preparation_sha256'] != file_hash(prepared/'manifest.json'):
        raise ValueError('Evaluation code/preparation differs from the trained run')
    name = (root/'ckpt/latest').read_text().strip()
    if Path(name).name != name:
        raise ValueError('Checkpoint pointer escaped run')
    node = root/'ckpt'/name
    validate_bundle(node, identity)
    cfg = frozen_model['config']
    extensions = agent_extensions(s)
    if s['mode'] in ('tutor', 'b2','quality_v2') and frozen_model['agent_extension'] != extensions:
        raise ValueError('Frozen extended model configuration differs from settings')
    output = Path(output_dir).resolve() if output_dir else root/'evaluations'/('actor_only_'+uuid.uuid4().hex[:10])
    return _run_actor_evaluation(s,node,cfg,extensions,composition,output,steps,
                                 holdout_contract=holdout_contract)


def evaluate_source(prepared,checkpoint,output_dir,steps=5000,device='cpu'):
    """Explicit baseline evaluation in current reviewed code, outside old runs.

    Authenticate the original complete source, its producer bytes and prepared
    configuration. This is a parameter-exact reference evaluation, not a claim
    to resume its old training or reproduce another device's arithmetic.
    """
    if type(steps) is not int or not 1<=steps<=10000 or device not in ('cpu','gpu'):
        raise ValueError('Invalid source Actor-only evaluation budget/device')
    node,prepared,output=Path(checkpoint).resolve(),Path(prepared).resolve(),Path(output_dir).resolve()
    if output.is_relative_to(node.parent.parent) or output.is_relative_to(prepared):
        raise ValueError('Historical source/prepared artifacts are read-only')
    meta=validate_bundle(node)
    identity=meta['identity']
    receipt=read_manifest(node.parent.parent/'training_receipt.json')
    preparation=read_manifest(prepared/'manifest.json')
    if (receipt['status'] not in ('FORMAL_COMPLETE','DIAGNOSTIC_COMPLETE') or
            receipt['checkpoint_manifest_sha256']!=file_hash(node/'manifest.json') or
            identity['preparation_sha256']!=file_hash(prepared/'manifest.json') or
            receipt['identity']!=identity or identity['code']!=preparation['code']):
        raise ValueError('Source completion/preparation/Checkpoint identities differ')
    check_files(identity['code'])
    check_files({str(prepared/n):h for n,h in preparation['files'].items()})
    s=read_manifest(prepared/'settings.json')['settings']
    if s['mode'] not in ('tutor','b2'): raise ValueError('Source baseline supports frozen Tutor/B2')
    frozen=read_manifest(prepared/'model_config.json')
    extensions=agent_extensions(s)
    if extensions!=frozen['agent_extension']: raise ValueError('Source Actor extension differs')
    s={**s,'device':device}
    result=_run_actor_evaluation(s,node,frozen['config'],extensions,settings_composition(s),output,steps)
    result={k:v for k,v in result.items() if k!='content_hash'}
    result['reference_source']=dict(parameter_exact_checkpoint=True,source_prepared=str(prepared),
        source_producer_code=identity['code'],evaluation_code=execution_code(),
        original_checkpoint_modified=False,device_arithmetic_parity='NOT_CLAIMED',
        role='historical reference; not a matched-budget quality ablation')
    result=signed(result);write_json(output/'receipt.json',result)
    return result


def _run_actor_evaluation(s,node,cfg,extensions,composition,output,steps, *,holdout_contract=None):
    root=node.parent.parent
    output.mkdir(parents=True, exist_ok=False)
    if s['mode']=='quality_v2':
        prior=read_manifest(root/'train_prediction_prior.json')
        if prior['checkpoint_manifest_sha256']!=file_hash(node/'manifest.json'):
            raise ValueError('TRAIN prior is not bound to the evaluated Checkpoint')
        write_json(output/'train_prediction_prior.json',prior)
    identity=read_manifest(root/'run_identity.json')['identity']
    dev = (prepare_holdout(s,output,holdout_contract,identity) if holdout_contract is not None
           else prepare_dev(s,output))
    fold='TRAIN_HOLDOUT' if holdout_contract is not None else 'DEV'
    manifest_name='holdout_manifest.json' if holdout_contract is not None else 'dev_manifest.json'
    model = make_agent(cfg, output/'model', agent_cls=S3V2Agent, device=s['device'],
        bc_lambda=s['bc_lambda'], extra_agent_config=extensions, composition=composition)
    model.load_exact(pickle.loads((node/'agent.pkl').read_bytes()))
    # New evaluation episodes use the frozen config seed at counter zero.
    # Training/resume RNG remains untouched; this is a loaded read-only copy.
    model.sync_policy_from_train(reset_counter=True)
    model.evaluation_checkpoint_sha256 = file_hash(node/'manifest.json')
    policy = ActorPolicy(model)
    env = load_env(dev['rows'][0], dev['frozen_inputs'])
    shadow=None;shadow_counts=dict(fills=0,exits=0)
    if s['mode']=='quality_v2':
        from gym_trading_env.rewards.single3_reward import Single3Reward,Single3RewardConfig
        shadow=Single3Reward(Single3RewardConfig.for_candidate('C1'))
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
        if shadow is not None and delta:
            bar=env._raw.iloc[cursor]
            shadow.step(dict(bar_open=str(env.bobs[cursor]),bar_end=str(env._raw.index[cursor]),
                **{k.lower():float(bar[k]) for k in ('Open','High','Low','Close')},
                valid_bar=bool(env.valid_fill_clock[cursor]),eligible_market_minute=False,
                entries=[f for f in env.fills[shadow_counts['fills']:] if f['reason']=='ENTRY'],
                exits=env.attributions[shadow_counts['exits']:],true_terminal=env.terminated),env.terms.multiplier)
            shadow_counts.update(fills=len(env.fills),exits=len(env.attributions))
        rows.append(dict(cursor=cursor, action=int(step['action']),
            mask=np.asarray(step['action_mask']).tolist(), is_first=bool(step['is_first']),
            is_last=bool(step['is_last']), reward_from_previous_action=float(step['reward']),
            p_add=policy.last_diagnostics['p_add'], advanced=bool(delta)))
        rows[-1]['decision_state_id'] = fingerprint(decision_state(env))
        rows[-1]['prediction_phase'] = 'CAUSAL_ACTOR_ONLY'
        if s['mode'] == 'b2':
            rows[-1]['b2'] = policy.last_diagnostics['b2']
        if s['mode'] == 'tutor':
            rows[-1]['tutor'] = policy.last_diagnostics['tutor']
        if s['mode']=='quality_v2': rows[-1]['quality']=policy.last_diagnostics['quality']
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
        result = signed(dict(status='ACTOR_ONLY_'+fold+'_DIAGNOSTIC_COMPLETE', output=str(output),
            fold=fold,evaluation_manifest=manifest_name,contract=env.contract,env_steps=advanced,
            checkpoint=str(node), checkpoint_manifest_sha256=file_hash(node/'manifest.json'),
            devices=[str(d) for d in jax.devices()], oracle_files_read=False, teacher_guidance=False,
            beta=0., policy_mode='eval_argmax', legal_decisions=len(legal),
            policy_rng='frozen_config_seed_counter_zero',
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
        if s['mode']=='quality_v2':
            from .fg_s3_quality_teacher_v2 import trade_facts
            actual_quality=[]
            for event in env.attributions:
                fill=next(f for f in env.fills if f['reason']=='ENTRY' and f['entry_id']==event['entry_id'])
                end=int(env._raw.index.get_indexer([pd.Timestamp(event['exit_time'])])[0])
                actual_quality.append(trade_facts(env,fill,event,end,day_complete=bool(dev['rows'][0]['calendar_complete'])))
            result['quality']=dict(run_q_admitted=s['quality_q_admitted'],
                configured_reward=env.total_reward,original_c1_shadow_reward=shadow.total,
                actual_executed_trade_scores=[f['score'] for f in actual_quality],
                causal_prediction_source='current causal RSSM; no future labels',calibration_status='NOT_CALIBRATED')
        names=['actions.jsonl','account_state.json',manifest_name]
        names.extend(str(Path(p).relative_to(output)) for p in dev['frozen_inputs']
                     if Path(p).is_relative_to(output))
        result['causal_prediction_files'] = {name:file_hash(output/name) for name in names}
        if s['mode']=='quality_v2':
            for name in ('train_prediction_prior.json',):
                result['causal_prediction_files'][name]=file_hash(output/name)
        result['agent_extension'] = extensions
        result.update(identity_fields(s))
        result = signed(result)
        write_json(output/'receipt.json', result)
        return result
    finally:
        driver.close()

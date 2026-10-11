"""Read-only CUDA inference of existing FG pilot models; never calls train()."""
import argparse
import gc
import json
import pickle
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from gym_trading_env.research.entry_pipeline import file_hash
from gym_trading_env.research.fg_abcd_evaluation import replay_policy, validate_validation_inputs
from gym_trading_env.research.fg_entry_v1 import write_json
from gym_trading_env.research.fg_learning_diagnostics import (
    add_confusion, completed_returns, distribution, imagined_return)

OBS_KEYS = ('market_seq', 'position_state', 'action_mask', 'reward',
            'is_first', 'is_last', 'is_terminal')


def replay_chunks(checkpoint):
    """Saved successor links, never UUID sorting or reward-conditioned order."""
    state = pickle.loads((checkpoint/'replay.pkl').read_bytes())
    chunks = {str(x['uuid']): x for x in state['chunks']}
    successors = {str(x['succ']) for x in chunks.values()}
    starts = [key for key in chunks if key not in successors]
    if len(starts) != 1:
        raise ValueError('Expected one complete single-worker Replay chain')
    result = []; key = starts[0]; visited = set()
    while key in chunks:
        if key in visited:
            raise ValueError('Replay successor cycle')
        visited.add(key); meta = chunks[key]
        with np.load(checkpoint/'replay'/meta['name'], allow_pickle=False) as data:
            result.append({k: data[k].copy() for k in (*OBS_KEYS, 'action')})
        if len(result[-1]['action']) != meta['length']:
            raise ValueError('Replay chunk length mismatch')
        key = str(meta['succ'])
    if len(visited) != len(chunks):
        raise ValueError('Disconnected Replay chunks')
    return result


def prepare(root, output):
    from embodied.run.fg_checkpoint import validate_bundle
    manifest = root/'experiment_manifest.json'
    experiment = json.loads(manifest.read_text())
    fold = next(x for x in experiment['folds'] if x['validation'] == 'CZCE.FG409')
    inputs = validate_validation_inputs(manifest, fold['validation_env'])
    pilot = json.loads((root/'pilot_comparison.json').read_text())
    models = {}
    for row in pilot['results']:
        checkpoint = Path(row['stages'][-1]['checkpoint'])
        meta = validate_bundle(checkpoint)
        original = root/'pilot_evaluation/CZCE.FG409'/row['method']/str(row['seed'])
        receipt = json.loads((original/'evaluation_receipt.json').read_text())
        if (receipt['agent_hash'] != meta['files']['agent.pkl'] or
                receipt['checkpoint'] != str(checkpoint) or
                receipt['experiment_hash'] != experiment['content_hash']):
            raise ValueError('Pilot model/evaluation identity mismatch')
        models[row['method']] = dict(checkpoint=str(checkpoint), original=str(original),
            agent_sha256=meta['files']['agent.pkl'], manifest_sha256=file_hash(checkpoint/'manifest.json'),
            config_sha256=meta['files']['config.yaml'])
    # Different models observe identical real account trajectories. C is merged
    # with B only after checking the whole timestamp/action series.
    sources = ['A', 'B']
    b = pd.read_csv(Path(models['B']['original'])/'dreamerv3/policy_actions.csv')
    c = pd.read_csv(Path(models['C']['original'])/'dreamerv3/policy_actions.csv')
    identical = b[['signal_timestamp','action']].equals(c[['signal_timestamp','action']])
    if not identical:
        sources.append('C')
    references = {}
    for method in sources:
        old = Path(models[method]['original'])/'dreamerv3'
        archived = pd.read_csv(old/'policy_actions.csv')
        rows = []; index = [0]
        def recorded(obs, first, previous_reward):
            i = index[0]; index[0] += 1
            rows.append({**{k: np.asarray(v).copy() for k,v in obs.items()},
                         'reward': np.float32(previous_reward), 'is_first': bool(first),
                         'is_last': False, 'is_terminal': False,
                         'action': np.int32(archived.iloc[i].action)})
            return int(archived.iloc[i].action)
        new = output/f'reference_{method}'
        summary = replay_policy(fold['validation_env'], new, policy=recorded, seed=0)
        actual = pd.read_csv(new/'policy_actions.csv')
        pd.testing.assert_frame_equal(actual, archived)
        old_summary = json.loads((old/'summary.json').read_text())
        for key in ('total_reward','ADD_request','ADD_fill','completed_batches',
                    'max_floating_loss','net_equity_change','full_capacity_time_fraction'):
            if summary[key] != old_summary[key]:
                raise ValueError(f'Reference {method} reconciliation failed: {key}')
        path = output/f'reference_{method}.npz'
        np.savez_compressed(path, **{k:np.stack([r[k] for r in rows]) for k in rows[0]})
        references[method] = dict(path=str(path.resolve()), sha256=file_hash(path),
            rows=len(rows), original_actions_sha256=file_hash(old/'policy_actions.csv'),
            original_summary_sha256=file_hash(old/'summary.json'), reconciled=True)
        print(f'Reference {method}: {len(rows)} decisions reconciled', flush=True)
    write_json(output/'inputs.json', dict(models=models, references=references,
        c_reference='B' if identical else 'C', validation_env=fold['validation_env'],
        frozen_inputs=inputs, experiment_sha256=file_hash(manifest),
        tool_sha256=file_hash(__file__), helper_sha256=file_hash(
            Path(__file__).resolve().parents[1]/'src/gym_trading_env/research/fg_learning_diagnostics.py')))


class ReadOnlyNetwork:
    def __init__(self, checkpoint, env_config):
        import elements
        import embodied.jax.internal as internal
        from embodied.jax.agent import Options
        from dreamerv3.agent import Agent
        from dreamerv3.main import make_env
        from ruamel import yaml
        import jax
        import ninjax as nj
        cfg = elements.Config(yaml.YAML(typ='safe').load((checkpoint/'config.yaml').read_text()))
        cfg = cfg.update({'task':'gymnasium_FixedDirectionEntryEnv-v0',
                          'env.gymnasium.config_path':env_config})
        internal.setup(**{k:v for k,v in cfg.jax.items() if k not in Options.__dataclass_fields__})
        if not jax.devices() or any(d.platform != 'gpu' for d in jax.devices()):
            raise RuntimeError('FG diagnostics require GPU; CPU inference forbidden')
        print('Diagnostic devices:', jax.devices(), flush=True)
        env = make_env(cfg, 0)
        obs_space = {k:v for k,v in env.obs_space.items() if not k.startswith('log/')}
        act_space = {k:v for k,v in env.act_space.items() if k != 'reset'}
        env.close()
        # Construct only the model definition: no wrapper initialization, loss,
        # optimizer, parameter creation, train, report, load/save, or counters.
        model = object.__new__(Agent)
        Agent.__init__(model, obs_space, act_space, elements.Config(**cfg.agent,
            logdir=cfg.logdir, seed=cfg.seed, jax=cfg.jax, batch_size=cfg.batch_size,
            batch_length=cfg.batch_length, replay_context=cfg.replay_context,
            report_length=cfg.report_length, replica=cfg.replica, replicas=cfg.replicas))
        state = pickle.loads((checkpoint/'agent.pkl').read_bytes())
        self.action_counter = state['counters']['actions']
        self.config = cfg; self.model = model
        self.params = {k:jax.device_put(v) for k,v in state['params'].items()
                       if k.startswith(('enc/','dyn/','pol/','avail/','rew/','con/',
                                        'val/','valnorm/','avail_actor_gate/'))}
        del state; gc.collect()
        def readonly(fun):
            pure = nj.pure(fun)
            def run(params, *args):
                # This Ninjax version's scan requires modify=True for access
                # discovery. Reject any attempted parameter write explicitly.
                _, out, _, modified, created = pure(params, *args[:-1],
                    seed=args[-1], create=False, modify=True, track=True)
                if modified or created:
                    raise RuntimeError('Read-only diagnosis attempted state modification')
                return out
            return jax.jit(run)
        # Initial batch size is a static shape; keep this separate from the
        # shape-polymorphic inference arguments.
        self.initial = jax.jit(lambda params, seed:nj.pure(model.init_policy)(
            params, 1, seed=seed, create=False, modify=False)[1])
        step = nj.pure(self._step)
        def observe(params, carry, obs, actions, seeds):
            def one(carry, data):
                ob, actual, seed = data
                return step(params, carry, ob, actual, seed=seed,
                            create=False, modify=False)[1]
            carry, result = jax.lax.scan(one, carry, (obs, actions, seeds))
            return carry, jax.tree.map(lambda v:v[:,0],result)
        self.observe = jax.jit(observe)
        self.extract = jax.jit(lambda result,indices:
            {k:result[k][indices] for k in ('deter','stoch')})
        self.free = readonly(self._free)
        self.branch_wait = readonly(lambda starts:self._branch(starts, 0))
        self.branch_add = readonly(lambda starts:self._branch(starts, 1))
        self.open_loop = readonly(self._open_loop)

    def _heads(self, feat):
        import jax
        import jax.numpy as jnp
        from dreamerv3.agent import hard_nonempty_mask
        m = self.model; inp = m.feat2tensor(feat)
        policy = m.pol(inp, 1)
        prob = jax.nn.sigmoid(m._binary_logits(m.avail(inp, 1)))
        mask, fallback = hard_nonempty_mask(prob, m.config.avail_threshold,
                                           m.config.avail_always_available_index)
        offset, scale = m.valnorm.stats()
        return dict(add_probability=prob[...,1], wait_pred=prob[...,0]>=.5,
                    add_pred=prob[...,1]>=.5, allowed_add=mask[...,1], fallback=fallback,
                    raw_add=jax.nn.softmax(policy[m.action_key].logits,-1)[...,1],
                    reward=m.rew(inp,1).pred(), value=m.val(inp,1).pred()*scale+offset,
                    continuation=m.con(inp,1).prob(1))

    def _step(self, carry, obs, actual):
        import jax
        import jax.numpy as jnp
        import ninjax as nj
        import embodied.jax.nets as nn
        m = self.model
        carry, chosen, _ = m.policy(carry, obs, mode='eval')
        feat = carry[1]
        post = self._heads(feat)
        prior = dict(deter=feat['deter'], stoch=nn.cast(m.dyn._dist(
            m.dyn._prior(feat['deter'])).sample(nj.seed())))
        pre = self._heads(prior)
        masked = m._masked_policy(m.pol(m.feat2tensor(feat),1),
                                  obs['action_mask'],allow_fallback=False)
        result = dict(raw_add=post['raw_add'],
            masked_add=jax.nn.softmax(masked[m.action_key].logits,-1)[...,1],
            choice=chosen[m.action_key], value=post['value'],
            reward_post=post['reward'], reward_prior=pre['reward'],
            deter=feat['deter'], stoch=feat['stoch'])
        for name, heads in [('post',post),('prior',pre)]:
            result.update({f'{name}_{k}':heads[k] for k in
                ('add_probability','wait_pred','add_pred','allowed_add','fallback')})
        # RSSM sees the real reference action on the next observation, even if
        # this model would choose a different action on the shared trajectory.
        carry = (*carry[:-1], {m.action_key:actual})
        return carry, result

    def _free(self, starts):
        m = self.model
        _, _, actions = m.dyn.imagine(starts,
            lambda f:m._sample_imag_action(f,1), m.config.imag_length, training=False)
        return actions

    def _branch(self, starts, action):
        import jax
        import jax.numpy as jnp
        m = self.model; n = len(starts['deter'])
        carry, (first, _) = m.dyn.imagine(starts,
            {m.action_key:jnp.full((n,),action,jnp.int32)}, 1, training=False,single=True)
        _, rest, _ = m.dyn.imagine(carry, lambda f:m._sample_imag_action(f,1),
                                  m.config.imag_length-1,training=False)
        feat = jax.tree.map(lambda a,b:jnp.concatenate([a[:,None],b],1),first,rest)
        inp = m.feat2tensor(feat); offset, scale = m.valnorm.stats()
        return dict(reward=m.rew(inp,2).pred(), continuation=m.con(inp,2).prob(1),
                    bootstrap=m.val(inp,2).pred()[:,-1]*scale+offset)

    def _open_loop(self, starts, actions):
        m = self.model
        _, feat, _ = m.dyn.imagine(starts, {m.action_key:actions},
                                 actions.shape[1], training=False)
        return m.rew(m.feat2tensor(feat),2).pred()[:,-1]


def summarize(data, results):
    target = data['action_mask'][:,1] > .5
    eligible = ~(data['is_first'] | data['is_last'] | data['is_terminal'])
    lots = np.rint(data['position_state'][:,1]*6).astype(int)
    groups = dict(all=np.ones(len(target),bool), effective=eligible,
                  legal_add=eligible & target)
    groups.update({f'legal_add_{n}_lots':eligible & target & (lots==n) for n in (0,3,6)})
    answer = dict(rows=len(target), groups={})
    for name, select in groups.items():
        row = dict(rows=int(select.sum()),
            raw_actor_add=distribution(results.raw_add[select]),
            masked_actor_add=distribution(results.masked_add[select]),
            argmax_add_count=int((results.choice[select]==1).sum()))
        for head in ('post','prior'):
            predicted = results[f'{head}_add_pred'].to_numpy()
            row[head] = add_confusion(target[select],predicted[select])
            row[head]['add_probability'] = distribution(results[f'{head}_add_probability'][select])
            row[head]['overall_mask_accuracy'] = float(np.mean(np.stack([
                results[f'{head}_wait_pred'].to_numpy()[select],predicted[select]],-1)
                == data['action_mask'][select])) if select.any() else None
            row[head]['safe_mask'] = add_confusion(target[select],
                                                 results[f'{head}_allowed_add'][select])
        answer['groups'][name] = row
    reward = data['reward']
    answer['reward_predictions'] = {}
    for name, select in [('positive',reward>0),('zero',reward==0)]:
        answer['reward_predictions'][name] = dict(count=int(select.sum()),target=distribution(reward[select]))
        for head in ('post','prior'):
            prediction = results[f'reward_{head}'].to_numpy()[select]
            error = prediction-reward[select]
            answer['reward_predictions'][name][head] = dict(prediction=distribution(prediction),
                MAE=float(np.abs(error).mean()) if len(error) else None,
                bias=float(error.mean()) if len(error) else None)
    answer['action_match_count'] = int((results.choice.to_numpy()==data['action']).sum())
    return answer


def diagnose_source(network, data, output, source, draws):
    import jax
    n = len(data['action']); legal = (data['action_mask'][:,1]>.5) & ~(
        data['is_first'] | data['is_last'] | data['is_terminal'])
    indices = np.flatnonzero(legal)
    selected = indices[np.linspace(0,len(indices)-1,min(128,len(indices)),dtype=int)] if len(indices) else indices
    cases = {}
    if source == 'replay':
        positive = np.flatnonzero(data['reward']>0)
        for lag in (1,15,64,333):
            eligible = [int(t) for t in positive if t>=lag and
                        not data['is_terminal'][t-lag:t].any() and not data['is_terminal'][t-lag]]
            picked = [eligible[i] for i in np.linspace(0,len(eligible)-1,min(16,len(eligible)),dtype=int)] if eligible else []
            cases[lag] = [(t-lag,t) for t in picked]
    wanted = set(map(int,selected)) | {start for rows in cases.values() for start,_ in rows}
    collected = {}; frames = []
    put = jax.device_put
    carry = network.initial(network.params,put(np.array([0,0],np.uint32)))
    counter = network.action_counter if source != 'replay' else 0
    for start in range(0,n,64):
        end = min(start+64,n)
        # Each observation has the same seed construction as embodied.Agent.
        seeds = np.stack([np.random.default_rng([int(network.config.seed),counter+i]).integers(
            0,np.iinfo(np.uint32).max,(2,),np.uint32) for i in range(start,end)])
        obs = {k:put(data[k][start:end,None]) for k in OBS_KEYS}
        carry, result = network.observe(network.params,carry,obs,
                                        put(data['action'][start:end,None]),put(seeds))
        indices_here = sorted(i for i in wanted if start<=i<end)
        if indices_here:
            local = put(np.array([i-start for i in indices_here],np.int32))
            features = jax.device_get(network.extract(result,local))
            for j,i in enumerate(indices_here):
                collected[i] = {k:v[j] for k,v in features.items()}
        flat = jax.device_get({k:v for k,v in result.items() if k not in ('deter','stoch')})
        frames.append(pd.DataFrame(flat))
        if start % 8192 == 0:
            print(f'{source}: {end}/{n} re-observed',flush=True)
    results = pd.concat(frames,ignore_index=True)
    results.insert(0,'observation_index',np.arange(n))
    results['legal_add'] = data['action_mask'][:,1]>.5
    results['lots'] = np.rint(data['position_state'][:,1]*6)
    results['actual_action'] = data['action']; results['actual_reward'] = data['reward']
    results.to_csv(output/f'{source}_predictions.csv',index=False)
    answer = summarize(data,results)
    answer['basis'] = 'checkpoint re-observation with actual action history; per-step inference seeds'
    answer['selection'] = dict(rule='chronologically_equally_spaced_legal_nonboundary',
                              available=int(legal.sum()),selected=list(map(int,selected)),draws=draws)
    if len(selected):
        starts = {k:put(np.stack([collected[int(i)][k] for i in selected])) for k in ('deter','stoch')}
        imagination = []; differences = []
        for draw in range(draws):
            seed = put(np.array([20261011,draw],np.uint32))
            free = jax.device_get(network.free(network.params,starts,seed))
            imagination.append(dict(add_available=free['_action_mask'][...,1],
                                    add_action=free['action']==1,fallback=free['_action_mask_fallback']))
            wait = jax.device_get(network.branch_wait(network.params,starts,seed))
            add = jax.device_get(network.branch_add(network.params,starts,seed))
            differences.append(imagined_return(**dict(reward=add['reward'],continuation=add['continuation'],
                final_value=add['bootstrap']))-imagined_return(wait['reward'],wait['continuation'],wait['bootstrap']))
        by_step = pd.DataFrame({k:np.stack([x[k] for x in imagination]).mean((0,1)) for k in imagination[0]})
        by_step.insert(0,'action_offset',np.arange(len(by_step)))
        by_step.to_csv(output/f'{source}_imagination.csv',index=False)
        differences = np.stack(differences)
        branches = pd.DataFrame(dict(observation_index=selected,mean_delta=differences.mean(0),
            std_delta=differences.std(0),fraction_add_higher=(differences>0).mean(0)))
        branches.to_csv(output/f'{source}_branches.csv',index=False)
        answer['imagination'] = {k:float(np.stack([x[k] for x in imagination]).mean()) for k in imagination[0]}
        answer['branch_delta'] = dict(mean_by_state=distribution(differences.mean(0)),
            draw_std_by_state=distribution(differences.std(0)),
            states_add_higher=int((differences.mean(0)>0).sum()),
            estimate='15-step model rewards plus value bootstrap; NOT true Q or causal profit')
    if source == 'replay':
        returns, complete = completed_returns(data['reward'],data['is_terminal'],
                                              1-1/network.config.agent.horizon)
        select = complete & ~(data['is_first']|data['is_last']|data['is_terminal'])
        error = results.value.to_numpy()[select]-returns[select]
        answer['completed_behavior_returns'] = dict(rows=int(select.sum()),
            right_censored_rows=int((~complete).sum()),target=distribution(returns[select]),
            prediction=distribution(results.value[select]),MAE=float(np.abs(error).mean()) if len(error) else None,
            bias=float(error.mean()) if len(error) else None,
            basis='behavior return until real terminal, nonterminal chunks retained; not optimal V ground truth')
        long_rows = []
        for lag, pairs in cases.items():
            if not pairs:
                continue
            starts = {k:put(np.stack([collected[start][k] for start,_ in pairs])) for k in ('deter','stoch')}
            actions = put(np.stack([data['action'][start:end] for start,end in pairs]))
            predictions = np.stack([jax.device_get(network.open_loop(network.params,starts,actions,
                put(np.array([20261011,draw],np.uint32)))) for draw in range(draws)])
            for j,(start,end) in enumerate(pairs):
                long_rows.append(dict(lag=lag,start=start,event=end,target=float(data['reward'][end]),
                    prediction_mean=float(predictions[:,j].mean()),prediction_std=float(predictions[:,j].std()),
                    crosses_reset=bool(data['is_first'][start+1:end+1].any())))
        pd.DataFrame(long_rows).to_csv(output/'reward_open_loop.csv',index=False)
        answer['reward_open_loop'] = dict(rows=len(long_rows),selection='retrospective positive events',
            reset_protocol='free dynamics never reset inside rollout; cross-reset cases labelled separately',
            by_lag={str(lag):dict(count=len(rows),target=distribution([x['target'] for x in rows]),
                prediction=distribution([x['prediction_mean'] for x in rows]),
                crosses_reset=sum(x['crosses_reset'] for x in rows))
                for lag in cases if (rows:=[x for x in long_rows if x['lag']==lag])})
    return answer


def run_model(inputs, method, output, draws):
    import jax
    from embodied.run.fg_checkpoint import validate_bundle
    model_info = inputs['models'][method]; checkpoint = Path(model_info['checkpoint'])
    if file_hash(checkpoint/'manifest.json') != model_info['manifest_sha256']:
        raise ValueError('Checkpoint changed after diagnostic preparation')
    began = time.monotonic(); validate_bundle(checkpoint)
    network = ReadOnlyNetwork(checkpoint,inputs['validation_env'])
    directory = output/method; directory.mkdir()
    chunks = replay_chunks(checkpoint)
    data = {k:np.concatenate([x[k] for x in chunks]) for k in chunks[0]}
    del chunks; gc.collect()
    answer = dict(method=method,device='CUDA_GPU',checkpoint=model_info,
        actor_gate_enabled=bool(np.asarray(jax.device_get(network.params['avail_actor_gate/value'])).item()),
        config=dict(horizon=network.config.agent.horizon,imag_length=network.config.agent.imag_length,
                    contdisc=network.config.agent.contdisc,unimix=network.config.agent.policy.unimix),sources={})
    answer['sources']['replay'] = diagnose_source(network,data,directory,'replay',draws)
    del data; gc.collect()
    for reference, info in inputs['references'].items():
        path = Path(info['path'])
        if file_hash(path) != info['sha256']:
            raise ValueError('Reference observation data changed')
        with np.load(path,allow_pickle=False) as archive:
            data = {k:archive[k] for k in archive.files}
        answer['sources'][f'reference_{reference}'] = diagnose_source(
            network,data,directory,f'reference_{reference}',draws)
        del data; gc.collect()
    validate_bundle(checkpoint)
    if file_hash(checkpoint/'manifest.json') != model_info['manifest_sha256']:
        raise ValueError('Checkpoint changed during read-only inference')
    answer.update(checkpoint_unchanged=True,parameter_creation=False,parameter_updates=False,
                  elapsed_seconds=time.monotonic()-began)
    write_json(directory/'summary.json',answer)
    print(f'{method}: finished in {answer["elapsed_seconds"]:.1f}s',flush=True)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--root',default='artifacts/fg_abcd_v1/v3_shared_prefix')
    parser.add_argument('--output',required=True)
    parser.add_argument('--method',choices=list('ABC'))
    parser.add_argument('--draws',type=int,default=8)
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--from-prepared',help='Reuse verified reference inputs in a new attempt directory')
    args = parser.parse_args()
    if args.draws < 1:
        raise ValueError('draws must be positive')
    output = Path(args.output).resolve()
    if args.method:
        inputs = json.loads((output/'inputs.json').read_text())
        helper = Path(__file__).resolve().parents[1]/'src/gym_trading_env/research/fg_learning_diagnostics.py'
        if file_hash(__file__) != inputs['tool_sha256'] or file_hash(helper) != inputs['helper_sha256']:
            raise ValueError('Diagnostic implementation changed after preparation')
        for path, checksum in inputs['frozen_inputs'].items():
            if file_hash(path) != checksum:
                raise ValueError('Frozen input changed during diagnostics')
        run_model(inputs,args.method,output,args.draws)
        return
    output.mkdir(parents=True,exist_ok=False)
    if args.from_prepared:
        source = Path(args.from_prepared).resolve()/'inputs.json'
        inputs = json.loads(source.read_text())
        for info in inputs['references'].values():
            if file_hash(info['path']) != info['sha256']:
                raise ValueError('Prepared reference changed')
        inputs['preparation_receipt'] = dict(path=str(source),sha256=file_hash(source))
        inputs['tool_sha256'] = file_hash(__file__)
        write_json(output/'inputs.json',inputs)
    else:
        prepare(Path(args.root).resolve(),output)
    if not args.prepare_only:
        for method in 'ABC':
            subprocess.run([sys.executable,__file__,'--output',str(output),
                            '--method',method,'--draws',str(args.draws)],check=True)


if __name__ == '__main__':
    main()

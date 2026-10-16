"""Bounded TRAIN DAgger collector over the original Driver, Env and Replay."""
import copy
from collections import Counter
import hashlib
import json
from pathlib import Path
import pickle
import time
import uuid

import numpy as np

from gym_trading_env.envs.entry_contract import fingerprint
from .fg_replay import cpu_runtime, digest, execute_observed, read_manifest, signed, write_json


PROTOCOL = 'FG_S3_V2_DAGGER_D0_V1'


def observation_id(obs, *, observation_schema=None):
    from .fg_s3_e_observation import fields_for
    fields = fields_for(observation_schema)
    h = hashlib.sha256()
    for key in sorted(k for k in fields if k != 'action'):
        x = np.asarray(obs[key])
        h.update(key.encode() + str(x.dtype).encode() + str(x.shape).encode() + x.tobytes())
    return h.hexdigest()


def join_d0_teacher(batch, records, *, fold='TRAIN', catalog=None, observation_schema=None):
    """D0 sidecar join proof; V1 TeacherCatalog is deliberately unchanged."""
    if fold != 'TRAIN' or 'teacher_action' in batch or 'teacher_valid' in batch:
        raise ValueError('D0 teacher join requires an unjoined TRAIN batch')
    from .fg_s3_e_observation import fields_for
    fields = fields_for(observation_schema)
    ids = np.asarray(batch['stepid'])
    if ids.dtype != np.uint8 or ids.ndim != 3 or ids.shape[-1] != 20:
        raise ValueError('Native stepid contract differs')
    index = {r['stepid']: r for r in records}
    if len(index) != len(records):
        raise ValueError('Duplicate D0 native row')
    y, valid = np.zeros(ids.shape[:2], np.int32), np.zeros(ids.shape[:2], bool)
    for pos in np.ndindex(y.shape):
        row = index.get(ids[pos].tobytes().hex())
        if row is None and catalog is not None:
            if observation_schema is not None:
                raise ValueError('Legacy Replay cannot supply E historical snapshots')
            # Delegate historical P4 validation to its original shared adapter.
            # Undo only Replay.sample's documented first-position reset flag.
            from embodied.run.fg_s3_teacher_adapter import attach_teacher
            match = catalog.lookup(ids[pos])
            single = {k: np.asarray(v)[pos[0]:pos[0]+1, pos[1]:pos[1]+1].copy()
                      for k,v in batch.items()}
            if match.record is not None and pos[1] == 0 and bool(batch['is_first'][pos]):
                single['is_first'][0,0] = bool(match.record['is_first'])
            legacy = attach_teacher(single, catalog, mode='train', sampling_flags=False)
            y[pos], valid[pos] = legacy['teacher_action'][0,0], legacy['teacher_valid'][0,0]
            continue
        if row is None or row['source_type'] != 'dagger':
            raise ValueError('Unknown D0 Replay owner')
        obs = {k: np.asarray(batch[k][pos]) for k in fields if k != 'action'}
        if pos[1] == 0 and bool(obs['is_first']):
            obs['is_first'] = np.asarray(row['is_first'], np.bool_)
        if observation_id(obs, observation_schema=observation_schema) != row['actual_observation_id']:
            raise ValueError('D0 teacher observation alignment differs')
        if int(batch['action'][pos]) != row['executed_action']:
            raise ValueError('D0 Replay action is not the executed action')
        if int(batch['market_advance'][pos]) != row['market_advance']:
            raise ValueError('Replay market clock differs from the committed observation')
        proof = dict(action=row['oracle_action'], valid=row['oracle_valid'],
            certified=row['oracle_certified'], reason=row['oracle_reason'],
            q_add=row['q_add'], q_wait=row['q_wait'], q_margin=row['q_margin'],
            state_id=row['state_id'], oracle_identity=row['oracle_identity'])
        if fingerprint(proof) != row['oracle_proof_hash']:
            raise ValueError('D0 Oracle proof differs from the recorded query')
        if row['teacher_valid']:
            if (not row['oracle_certified'] or not row['oracle_valid']
                    or not row['execution_result']['advanced'] or not all(row['action_mask'])
                    or row['teacher_action'] != row['oracle_action']):
                raise ValueError('Uncertified or forced decision reached supervision')
            y[pos], valid[pos] = row['teacher_action'], True
    return {**batch, 'teacher_action': y, 'teacher_valid': valid}


class D0Collector:
    output_directory = 'fg_s3_v2_d0'
    max_env_steps = 1000
    protocol = PROTOCOL
    runtime = staticmethod(cpu_runtime)

    def __init__(self, env, oracle, directory, dreamer_root, *, mode='mixed', seed=202,
                 beta=.5, actor_probability=.5, actor=None, agent=None,
                 source_checkpoint_identity=None, existing=False,
                 observation_schema=None, e_provider=None, e_profile=None, teacher_enabled=True):
        if type(teacher_enabled) is not bool or (not teacher_enabled and beta != 0):
            raise ValueError('A disabled teacher requires beta=0')
        self.teacher_enabled = teacher_enabled
        if e_profile is not None:
            if e_provider is not None:
                raise ValueError('Choose native E profile or synthetic research provider')
            e_provider = e_profile
        embodied = self.runtime(dreamer_root)
        from embodied.envs.from_gymnasium import FromGymnasium
        from embodied.run.fg_s3_dagger_policy import MixedExecutionPolicy, ScriptedActor
        if mode not in ('wait', 'scripted', 'mixed'):
            raise ValueError('Unknown D0 collector mode')
        if oracle.identity['fold'] != 'TRAIN':
            raise ValueError('D0 collector is TRAIN-only')
        from .fg_s3_e_observation import fields_for, OBS_SCHEMA, observation_contract
        self.observation_schema = observation_schema
        self.fields = fields_for(observation_schema)
        if (observation_schema == OBS_SCHEMA) != (e_provider is not None):
            raise ValueError('E Collector requires matching schema and provider')
        self.root = Path(directory).resolve()
        if not self.output_allowed(self.root):
            raise ValueError('D0 output must be isolated from existing experiments and worktrees')
        self.root.mkdir(parents=True, exist_ok=existing)
        self.env, self.oracle = env, oracle
        self.records, self.pending, self.counts = [], None, Counter()
        self.query_seconds = []
        self.source_checkpoint_identity = source_checkpoint_identity
        self.identity = dict(protocol=self.protocol, oracle_identity=oracle.identity_hash,
            rules=oracle.rules, mode=mode, seed=seed,
            beta=beta if mode == 'mixed' else 0.,
            actor_probability=None if agent is not None else (0. if mode == 'wait' else actor_probability),
            actor_kind='checkpoint' if agent is not None else 'scripted',
            source_checkpoint_identity=source_checkpoint_identity,
            collector_code_sha256=digest(__file__),
            policy_code_sha256=digest(Path(dreamer_root)/'embodied/run/fg_s3_dagger_policy.py'))
        if observation_schema is not None:
            self.identity.update(e_observation=observation_contract(), e_data_identity=e_provider.identity)
        self.identity_hash = fingerprint(self.identity)
        owner = self

        self.observed = self.make_observed(env)
        if observation_schema is not None:
            native = all(k in self.observed.observation_space.spaces for k in
                ('market_seq_e_shared', 'market_context_15m'))
            if native:
                if not hasattr(self.observed, 'e_snapshot'):
                    raise ValueError('Native E Env lacks snapshot provenance')
            else:
                if e_profile is not None:
                    raise ValueError('Native E profile requires native Env observations')
                # Compatibility ONLY for the signed CPU SYNTHETIC E-model
                # fixture. Real TRAIN never uses an external EObserved wrapper.
                from .fg_replay import read_manifest
                fixture = getattr(e_provider, 'manifest', None)
                if fixture is None or read_manifest(fixture).get('schema') != 'FG_S3_E_MODEL_SYNTHETIC_V2':
                    raise ValueError('Real E observation must originate inside the Env')
                from .fg_s3_e_observation import wrap_e_observation
                self.observed = wrap_e_observation(self.observed, e_provider)
        wrapped = embodied.wrappers.CheckSpaces(embodied.wrappers.UnifyDtypes(
            FromGymnasium(self.observed, reset_seed=seed)))
        self.driver = embodied.Driver([lambda: wrapped], parallel=False)
        self.replay = self.make_replay(embodied, seed)

        def context(obs):
            state = owner.context_snapshot()
            one = {k: np.asarray(v[0]) for k, v in obs.items()}
            return dict(snapshot=state, state_id=fingerprint(state),
                observation_id=observation_id(one, observation_schema=observation_schema), is_first=bool(one['is_first']),
                is_last=bool(one['is_last']), is_terminal=bool(one['is_terminal']),
                observation_reward=float(one['reward']))

        def query(ctx, mask):
            if not owner.teacher_enabled:
                # Preserve the authenticated Replay owner/proof format without
                # consulting any future path or issuing a certified label.
                identity = oracle.identity.get('queries', {}).get(
                    ctx['snapshot']['contract'], oracle.identity_hash)
                answer = dict(action=0, valid=False, certified=False, reason='TEACHING_DISABLED',
                    q_add=None, q_wait=None, q_margin=None, state_id=ctx['state_id'], oracle_identity=identity)
                answer['proof_hash'] = fingerprint(answer)
                return answer
            started = time.perf_counter()
            answer = oracle.query(owner.decision_env, mask, snapshot=ctx['snapshot'])
            owner.query_seconds.append(time.perf_counter() - started)
            return answer

        actor = actor or ScriptedActor(seed, 0. if mode == 'wait' else actor_probability)
        self.policy = MixedExecutionPolicy(actor, query, context,
            beta=self.identity['beta'], seed=seed, dreamer_carry=agent is not None,
            observation_schema=observation_schema)
        if not teacher_enabled:
            self.identity['teacher_enabled'] = False
            self.identity_hash = fingerprint(self.identity)
        self.agent = agent
        if agent is not None:
            self.driver.reset(agent.init_policy)
        self.driver.on_step(self.capture)

    def output_allowed(self, root):
        allowed = Path(__file__).resolve().parents[3]/'artifacts'/self.output_directory
        return root.is_relative_to(allowed) or root.is_relative_to(Path('/tmp'))

    @property
    def decision_env(self):
        return self.env

    def make_replay(self, embodied, seed):
        return embodied.Replay(length=65, capacity=2048, online=True,
            directory=str(self.root/'replay'), chunksize=1024, save_wait=True, seed=seed)

    def context_snapshot(self):
        return self.decision_env.export_state()

    def make_observed(self, env):
        import gymnasium as gym
        class Observed(gym.Wrapper):
            @property
            def e_snapshot(self):
                return self.env.e_snapshot

            def reset(self, *, seed=None, options=None):
                if options:
                    return self.env.reset(seed=seed, options=options)
                obs, self.last = execute_observed(self.env, seed=seed)
                return obs, self.env._get_info()

            def step(self, action):
                obs, self.last = execute_observed(self.env, action)
                return obs, self.last['official_reward'], self.last['is_terminal'], self.env.truncated, self.env._get_info()
        return Observed(env)

    def _complete(self, result, next_context):
        r = self.pending
        if r is None:
            return
        advanced = bool(result.get('advanced', False))
        if result.get('incoming_action') is not None and result['incoming_action'] != r['executed_action']:
            raise RuntimeError('Driver executed an action different from the recorded choice')
        r.update(execution_result={k: result.get(k) for k in
            ('advanced', 'cursor', 'signal_time', 'time', 'reject_reason', 'status', 'fills', 'exits')},
            reward=float(np.float32(result.get('official_reward', 0.))),
            next_state_id=next_context['state_id'] if next_context else None,
            next_cursor=next_context['snapshot']['current_step'] if next_context else None,
            teacher_valid=bool(advanced and r['oracle_valid'] and r['oracle_certified']))
        r['teacher_action'] = r['oracle_action'] if r['teacher_valid'] else 0
        self.records.append(r)
        self.pending = None
        if advanced:
            self.counts['env_steps'] += 1
            self.counts['valid_market_minutes'] += int(result['valid_bar'])
            self.counts['actor_ADD'] += int(r['actor_proposal'] == 1)
            self.counts['oracle_ADD'] += int(r['oracle_valid'] and r['oracle_action'] == 1)
            self.counts['executed_ADD'] += int(r['executed_action'] == 1)
            self.counts['entries'] += sum(f['reason'] == 'ENTRY' for f in result['fills'])
            self.counts['TP'] += sum(e['reason'] == 'TAKE_PROFIT' for e in result['exits'])
            self.counts['SL'] += sum(e['reason'] == 'STOP_LOSS' for e in result['exits'])
            self.counts['rejects'] += int(bool(result['reject_reason']))
            self.counts['certified_labels'] += int(r['teacher_valid'])
            self.counts['learnable_decisions'] += int(all(r['action_mask']))
            self.counts['unproven_funds'] += int(r['oracle_reason'] == 'FUNDS_OUTSIDE_ORIGINAL_CERTIFICATE')

    def capture(self, step, worker):
        choice = self.policy.decision
        ctx, state = choice['context'], choice['context']['snapshot']
        self._complete(self.observed.last, ctx)
        native = {k: np.asarray(step[k]) for k in self.fields}
        native['market_advance'] = np.int32(self.observed.last['valid_bar'])
        if int(native['action']) != choice['executed_action']:
            raise RuntimeError('Driver changed the chosen action')
        stepid = self._add_native(native, worker)
        answer = choice['oracle']
        if answer['state_id'] != ctx['state_id']:
            raise RuntimeError('Oracle labeled a different account state')
        previous = choice['actor_prev_action']
        if self.pending is None and self.records and not ctx['is_first']:
            if previous != self.records[-1]['executed_action']:
                raise RuntimeError('Actor previous action differs from actual execution')
        episode = fingerprint(dict(identity=self.identity_hash, contract=state['contract'],
                                   account_sequence=state['account_sequence']))
        self.pending = dict(protocol=self.protocol, source_type='dagger', episode_id=episode,
            contract=state['contract'], cursor=state['current_step'], time=state['cursor_time'],
            actual_observation_id=ctx['observation_id'], state_id=ctx['state_id'],
            stepid=stepid,
            action_mask=native['action_mask'].tolist(),
            actor_proposal=choice['actor_proposal'], actor_prev_action=previous,
            oracle_action=answer['action'], oracle_valid=answer['valid'],
            oracle_certified=answer['certified'], oracle_reason=answer['reason'],
            oracle_proof_hash=answer['proof_hash'], oracle_identity=answer['oracle_identity'],
            executed_action=choice['executed_action'], oracle_selected=choice['oracle_selected'],
            beta=choice['beta'], beta_draw=choice['beta_draw'],
            q_add=None, q_wait=None, q_margin=None,
            is_first=ctx['is_first'], is_last=ctx['is_last'], is_terminal=ctx['is_terminal'],
            observation_reward=ctx['observation_reward'],
            market_advance=int(native['market_advance']), clock_protocol='S3_MARKET_MINUTE',
            equity_before=str(self.decision_env._calculate_equity()),
            source_checkpoint_identity=self.source_checkpoint_identity)
        if self.observation_schema is not None:
            from .fg_s3_e_observation import validate_e_snapshot
            snapshot = copy.deepcopy(self.observed.e_snapshot)
            validate_e_snapshot(snapshot,self.decision_env,self.identity['e_data_identity'],native)
            self.pending['e_snapshot'] = snapshot
        self.counts['native_rows'] += 1
        self.counts['resets'] += int(ctx['is_first'])

    def advance(self, target):
        if not 1 <= target <= self.max_env_steps or target < self.counts['env_steps']:
            raise ValueError('D0 cumulative environment budget must be 1..1000')
        while self.counts['env_steps'] < target:
            self.driver(self.policy, steps=1)
            if self.env.terminated or self.env.status in ('DATA_END', 'DATA_GAP'):
                break

    def _add_native(self, native, worker):
        self.replay.add(native, worker=worker)
        chunk, offset = self.replay.streams[worker][-1]
        return (bytes(chunk)+offset.to_bytes(4, 'big')).hex()

    def save_collection(self, directory):
        """A D0 collection checkpoint, not a transferable Dreamer checkpoint."""
        from embodied.run.fg_checkpoint import save_replay
        dest = Path(directory)
        if self.agent is not None:
            raise ValueError('D0 model state requires original FullCheckpoint registration; collection-only save refused')
        if not (dest.is_relative_to(self.root) or dest.resolve().is_relative_to(Path('/tmp'))):
            raise ValueError('Collection checkpoint must belong to this D0 output')
        if dest.exists():
            raise FileExistsError('D0 checkpoint retained; choose a fresh destination')
        temp = dest.with_name(dest.name+'.pending_'+uuid.uuid4().hex)
        temp.mkdir(parents=True)
        state = dict(driver=self.driver.save(), policy=self.policy.save(),
            replay=save_replay(self.replay, temp/'replay'),
            records=copy.deepcopy(self.records), pending=copy.deepcopy(self.pending),
            counts=copy.deepcopy(self.counts), query_seconds=list(self.query_seconds))
        (temp/'state.pkl').write_bytes(pickle.dumps(state, protocol=5))
        write_json(temp/'manifest.json', signed(dict(protocol=PROTOCOL+'_COLLECTION_STATE',
            identity=self.identity, identity_hash=self.identity_hash, model_present=False,
            env_steps=self.counts['env_steps'],
            files={str(p.relative_to(temp)): digest(p) for p in temp.rglob('*') if p.is_file()})))
        (temp/'done').touch()
        temp.rename(dest)

    def load_collection(self, directory):
        from embodied.run.fg_checkpoint import load_replay
        from .fg_replay import check_files
        directory = Path(directory)
        m = read_manifest(directory/'manifest.json')
        if (not (directory/'done').is_file() or m['protocol'] != PROTOCOL+'_COLLECTION_STATE'
                or m['identity'] != self.identity or m['identity_hash'] != self.identity_hash or m['model_present']):
            raise ValueError('D0 collection checkpoint identity mismatch')
        check_files({str(directory/f): h for f,h in m['files'].items()})
        state = pickle.loads((directory/'state.pkl').read_bytes())
        self.driver.load(state['driver'])
        self.policy.load(state['policy'])
        load_replay(self.replay, state['replay'], directory/'replay')
        self.records, self.pending = state['records'], state['pending']
        self.counts, self.query_seconds = state['counts'], state['query_seconds']

    def finish(self):
        self._complete(dict(status='BUDGET_END', advanced=False), None)
        path = self.root/'sidecar.jsonl'
        with path.open('x') as stream:
            for row in self.records:
                stream.write(json.dumps(row, sort_keys=True, allow_nan=False)+'\n')
        sample = self.replay.sample(1) if len(self.replay) else None
        join = join_d0_teacher(sample, self.records) if sample is not None else None
        if sample is not None:
            np.savez_compressed(self.root/'native_sample.npz', **sample)
            np.savez_compressed(self.root/'train_join_sample.npz', **join)
        self.replay.save()
        self.replay.workers.shutdown(wait=True)
        state = self.env.export_state()
        write_json(self.root/'env_state.json', state)
        semantic = [{k:v for k,v in r.items() if k != 'stepid'} for r in self.records]
        reasons = Counter(r['oracle_reason'] for r in self.records if r['execution_result']['advanced'])
        result = signed(dict(protocol=PROTOCOL, status='D0_ROLLOUT_COMPLETE', identity=self.identity,
            env_steps=self.counts['env_steps'], counts=dict(self.counts), oracle_reasons=dict(reasons),
            label_ratio_all_advances=self.counts['certified_labels']/max(1,self.counts['env_steps']),
            label_ratio_legal_decisions=self.counts['certified_labels']/max(1,self.counts['learnable_decisions']),
            semantic_trace_hash=fingerprint(semantic), final_state_id=fingerprint(state),
            query_seconds=dict(count=len(self.query_seconds), total=sum(self.query_seconds),
                mean=float(np.mean(self.query_seconds)), p95=float(np.quantile(self.query_seconds,.95))),
            native_sample='PASS' if sample is not None else 'NOT_RUN',
            sample_teacher_valid=int(join['teacher_valid'].sum()) if join is not None else 0,
            gradient_updates=0, gpu_training='NOT_RUN', dev='NOT_RUN', test='NOT_RUN',
            files={str(p.relative_to(self.root)):digest(p) for p in self.root.rglob('*') if p.is_file()}))
        write_json(self.root/'receipt.json', result)
        return result

    def close(self):
        self.replay.workers.shutdown(wait=True)
        self.driver.close()

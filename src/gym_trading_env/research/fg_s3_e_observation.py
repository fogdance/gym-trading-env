"""Versioned E observation contract; no aggregation or normalization here."""
from __future__ import annotations

import copy
import hashlib
from collections import OrderedDict

import numpy as np

from gym_trading_env.envs.entry_contract import fingerprint
from .fg_replay import FIELDS

OBS_SCHEMA = 'FG_S3_E_OBSERVATION_V2'
REPLAY_SCHEMA = 'FG_S3_E_REPLAY_MARKET_MINUTE_V2'
DECODER_SCHEMA = 'FG_S3_E_DECODER_V1'
E_FIELDS_V2 = {**FIELDS,
    'market_seq_e_shared': ((60, 18), np.dtype('float32')),
    'market_context_15m': ((24, 7), np.dtype('float32')),
    'market_macd_5m': ((4,), np.dtype('float32'))}
E_S3_FIELDS_V2 = {**E_FIELDS_V2, 'market_advance': ((), np.dtype('int32'))}
ENCODER_KEYS = ('market_seq_e_shared', 'position_state', 'action_mask', 'market_context_15m', 'market_macd_5m')
DECODER_KEYS = ('market_seq_e_shared', 'position_state')


class NativeEObservationCache:
    """Bounded market-only results for a private native counterfactual Env.

    Positions and action masks never enter this cache. Source objects remain
    strongly owned so a rebuilt/continued market cannot reuse an old entry.
    Every returned array and host snapshot has independent ownership.
    """
    def __init__(self, capacity=512):
        self.capacity = capacity
        self.sources = ()
        self.identity = None
        self.entries = OrderedDict()
        self.statistics = dict(hits=0, misses=0, evictions=0)

    def _bind(self, env):
        sources = (env._e_market, env._raw, env._features)
        identity = (env.contract, env.e_observation_identity, env._e_profile.identity)
        if (len(self.sources) != len(sources) or self.identity != identity or
                any(a is not b for a, b in zip(self.sources, sources))):
            self.sources, self.identity = sources, identity
            self.entries.clear()

    def get(self, env, market_seq):
        self._bind(env)
        item = self.entries.get(env.current_step)
        if item is None:
            self.statistics['misses'] += 1
            return None
        original, arrays, snapshot = item
        if market_seq.dtype != original.dtype or not np.array_equal(market_seq, original):
            raise ValueError('E observation changed the original 18 channels')
        self.entries.move_to_end(env.current_step)
        self.statistics['hits'] += 1
        return {k: v.copy() for k, v in arrays.items()}, copy.deepcopy(snapshot)

    def put(self, env, market_seq, arrays, snapshot):
        self._bind(env)
        self.entries[env.current_step] = (market_seq.copy(),
            {k: v.copy() for k, v in arrays.items()}, copy.deepcopy(snapshot))
        self.entries.move_to_end(env.current_step)
        if len(self.entries) > self.capacity:
            self.entries.popitem(last=False)
            self.statistics['evictions'] += 1


def fields_for(schema=None):
    if schema is None:
        return FIELDS
    if schema != OBS_SCHEMA:
        raise ValueError('Unknown E observation schema')
    return E_FIELDS_V2


def observation_contract():
    value = dict(observation_schema=OBS_SCHEMA, replay_schema=REPLAY_SCHEMA,
        decoder_schema=DECODER_SCHEMA,
        fields={k: dict(shape=list(shape), dtype=str(dtype))
                for k, (shape, dtype) in E_FIELDS_V2.items()},
        encoder_keys=list(ENCODER_KEYS), decoder_keys=list(DECODER_KEYS),
        audit_only=['market_seq'], context_fields=[
            'rel_open', 'rel_high', 'rel_low', 'rel_close', 'vol_relative', 'oi_change', 'valid'])
    from gym_trading_env.utils.macd_5m_causal import CONTRACT
    value['macd_5m'] = CONTRACT
    return {**value, 'contract_hash': fingerprint(value)}


def validate_e_observation(obs, *, batched=False):
    expected = set(E_FIELDS_V2) - {'action'}
    if set(obs) != expected:
        raise ValueError('E observation field set differs; future/teacher fields forbidden')
    leading = np.asarray(obs['is_first']).shape if batched else ()
    for k in expected:
        x = np.asarray(obs[k]); shape, dtype = E_FIELDS_V2[k]
        if x.shape != (*leading, *shape) or x.dtype != dtype or not np.isfinite(x).all():
            raise ValueError('Invalid E observation shape/dtype/finite: '+k)
    macd = np.asarray(obs['market_macd_5m'])
    if not np.isin(macd[..., 3], (0., 1.)).all() or np.any((macd[..., 3:4] == 0) & (macd[..., :3] != 0)):
        raise ValueError('E MACD invalid mask or nonzero invalid channels')
    context = np.asarray(obs['market_context_15m'])
    if not np.isin(context[..., 6], (0., 1.)).all():
        raise ValueError('E context valid mask must be binary')


def context_provenance(value):
    """Host-only identity; never returned as a policy input."""
    value = copy.deepcopy(value)
    required = {'source_contract', 'visible_cutoff', 'decision_time', 'data_identity'}
    if not required.issubset(value):
        raise ValueError('E provider provenance is incomplete')
    if any(k.startswith(('teacher', 'oracle', 'outcome', 'q_')) for k in value):
        raise ValueError('Teacher/outcome data in E provenance')
    return {**value, 'snapshot_identity': fingerprint(value)}


def e_array_hash(obs):
    h = hashlib.sha256()
    for k in ('market_seq_e_shared', 'market_context_15m', 'market_macd_5m'):
        x = np.asarray(obs[k])
        h.update(k.encode()+str(x.dtype).encode()+str(x.shape).encode()+x.tobytes())
    return h.hexdigest()


def validate_e_snapshot(value, decision_env, data_identity, arrays=None):
    """Authenticate host provenance at the Env/Collector boundary."""
    import pandas as pd
    meta = context_provenance({k:v for k,v in value.items()
                              if k not in ('snapshot_identity','arrays_hash')})
    if meta['data_identity'] != data_identity:
        raise ValueError('E provider data identity differs')
    if value.get('snapshot_identity',meta['snapshot_identity']) != meta['snapshot_identity']:
        raise ValueError('E snapshot provenance identity differs')
    actual = pd.Timestamp(decision_env._raw.index[decision_env.current_step])
    if meta['source_contract'] != decision_env.contract or pd.Timestamp(meta['decision_time']) != actual:
        raise ValueError('E provider contract/decision time differs from actual Env')
    cutoff = pd.Timestamp(meta['visible_cutoff'])
    if cutoff > actual or any(pd.Timestamp(t)>cutoff for t in meta.get('context_available_at',[])):
        raise ValueError('Future E context snapshot')
    if arrays is not None and value.get('arrays_hash') != e_array_hash(arrays):
        raise ValueError('E snapshot arrays differ from actual Env observation')
    return meta


def wrap_e_observation(observed, provider):
    """Adapt existing observed execution; provider owns all market formulas.

    provider.snapshot(decision_env, original_obs) returns (two arrays, metadata).
    It must be deterministic at a decision, including after account restoration.
    """
    import gymnasium as gym
    import pandas as pd

    if not hasattr(provider, 'identity') or not callable(getattr(provider, 'snapshot', None)):
        raise ValueError('E observation requires an identity-bound provider')

    class EObserved(gym.Wrapper):
        def __init__(self):
            super().__init__(observed)
            space = dict(observed.observation_space.spaces)
            for k in ('market_seq_e_shared', 'market_context_15m', 'market_macd_5m'):
                shape, dtype = E_FIELDS_V2[k]
                space[k] = gym.spaces.Box(-np.inf, np.inf, shape, dtype)
            self.observation_space = gym.spaces.Dict(space)
            self.e_snapshot = None

        @property
        def last(self):
            return self.env.last

        def decorate(self, obs):
            decision_env = getattr(self.env, 'decision_env', self.env.unwrapped)
            arrays, meta = provider.snapshot(decision_env, copy.deepcopy(obs))
            if set(arrays) != {'market_seq_e_shared', 'market_context_15m', 'market_macd_5m'}:
                raise ValueError('E provider returned foreign or missing fields')
            for k, x in arrays.items():
                shape, dtype = E_FIELDS_V2[k]
                if x.shape != shape or x.dtype != dtype or not np.isfinite(x).all():
                    raise ValueError('Invalid E provider array: '+k)
            if not np.isin(arrays['market_context_15m'][:, 6], (0., 1.)).all():
                raise ValueError('E provider mask must be binary')
            meta = validate_e_snapshot(meta,decision_env,provider.identity)
            self.e_snapshot = {**meta, 'arrays_hash': e_array_hash(arrays)}
            return {**obs, **{k: x.copy() for k, x in arrays.items()}}

        def reset(self, **kwargs):
            obs, info = self.env.reset(**kwargs)
            return self.decorate(obs), info

        def step(self, action):
            obs, reward, terminal, truncated, info = self.env.step(action)
            return self.decorate(obs), reward, terminal, truncated, info

    return EObserved()

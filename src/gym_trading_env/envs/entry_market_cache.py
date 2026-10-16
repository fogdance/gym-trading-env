"""Bounded process-local market reuse; accounts and checkpoints stay separate."""
from collections import OrderedDict
import copy
from dataclasses import dataclass
import hashlib
from pathlib import Path
from types import SimpleNamespace

from gym_trading_env.utils.s3_market_clock import MarketClock


def _stat_identity(path):
    stat = path.stat()
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def _file_identity(path):
    before = _stat_identity(path)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    if _stat_identity(path) != before:
        raise ValueError('Market input changed while hashing cache identity')
    return digest.hexdigest(), before


@dataclass(frozen=True)
class _MarketSnapshot:
    raw: object
    features: object
    bobs: object
    valid: object
    e_market: object = None

    @classmethod
    def capture(cls, env):
        # Never retain an Env/account or let an Env mutate the cache payload.
        # E V2 owns a causal MACD prefix cache. Never share that mutable
        # state between accounts, teacher forks or the pristine cache.
        return cls(env._raw.copy(deep=True), env._features.copy(),
                   env.bobs.copy(deep=True), env.valid_fill_clock.copy(),
                   copy.deepcopy(getattr(env, '_e_market', None)))

    def restore(self, env):
        env._raw = self.raw.copy(deep=True)
        env._features = self.features.copy()
        env.bobs = self.bobs.copy(deep=True)
        env.valid_fill_clock = self.valid.copy()
        env._e_market = copy.deepcopy(self.e_market)
        env.bar_source = SimpleNamespace(store=SimpleNamespace(
            index=env._raw.index, n_rows=len(env._raw)))
        env.market_clock_counter = (env._e_market.market_clock_counter if env._e_market is not None
                                    else MarketClock(env.valid_fill_clock))

    @property
    def nbytes(self):
        extra = 0
        if self.e_market is not None:
            # Account-free E data is part of the bounded market cache, not
            # financial checkpoint state. Include its dominant allocations.
            extra = (self.e_market.frame.memory_usage(deep=True).sum()
                + self.e_market.features.memory_usage(deep=True).sum()
                + self.e_market.obs_array.nbytes + self.e_market.aux_array.nbytes
                + self.e_market.price_array.nbytes)
        return int(self.raw.memory_usage(deep=True).sum() + self.features.nbytes
                   + self.bobs.nbytes + self.valid.nbytes + extra)


class EntryMarketCache:
    """Cache market snapshots by config identity and the current CSV bytes.

    Scope is one dataset pool, bounded by its member count. Every lookup hashes
    the source again; new bytes or configuration require normal validation and
    feature generation. Injected/continued data bypass this cache in the Env.
    No cache entries are serialized into the financial checkpoint.
    """

    def __init__(self, capacity):
        if type(capacity) is not int or capacity <= 0:
            raise ValueError('Market cache capacity must be a positive integer')
        self.capacity = capacity
        self._entries = OrderedDict()
        self.hits = self.misses = 0

    def load(self, env):
        source = Path(env.config.trading.data_path).expanduser()
        path = source.resolve(strict=True)
        digest, stat = _file_identity(path)
        if getattr(env, '_e_profile', None) is not None:
            # A current-byte cache key alone does not authenticate those bytes
            # against E admission. Apply the same check before hits or misses.
            env._validate_e_market_source(path, digest)
        key = (type(env), env.config_hash, getattr(env,'e_observation_identity',None), str(path), digest)
        snapshot = self._entries.get(key)
        if snapshot is None:
            env._load_market(None)
            snapshot = _MarketSnapshot.capture(env)
        else:
            snapshot.restore(env)
        if source.resolve(strict=True) != path or _stat_identity(path) != stat:
            raise ValueError('Market input changed while preparing cache snapshot')
        if key in self._entries:
            self.hits += 1
            self._entries.move_to_end(key)
        else:
            self.misses += 1
            self._entries[key] = snapshot
            while len(self._entries) > self.capacity:
                self._entries.popitem(last=False)

    def __len__(self):
        return len(self._entries)

    @property
    def nbytes(self):
        return sum(snapshot.nbytes for snapshot in self._entries.values())

    def clear(self):
        self._entries.clear()

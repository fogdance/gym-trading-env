"""Private native-account forks; public snapshots remain complete owned JSON.

Only immutable primitive tuples/bytes enter sealed history blocks. A fork owns
its hot JSON state and append tail; neither can mutate a source account or an
older version. These objects are process-local, never a checkpoint schema.
"""
from collections.abc import Sequence
from dataclasses import dataclass, field
import hashlib
import json
from types import MappingProxyType
import uuid

from gym_trading_env.envs.entry_contract import fingerprint

_FIELDS = ('time', 'equity', 'unrealized_pnl', 'net_lots', 'intrabar')
_BLOCK_ROWS = 256


def _encode(value):
    return json.dumps(value, sort_keys=True, default=str).encode()


def _freeze_row(row):
    if (type(row) is dict and row.keys() == frozenset(_FIELDS)
            and type(row['time']) is str and type(row['intrabar']) is bool
            and all(type(row[k]) in (int, float) for k in _FIELDS[1:4])):
        return (0, *(row[k] for k in _FIELDS))
    # Preserve the public JSON normalization for non-native rows too. Bytes
    # cannot retain a mutable nested object or lose -0.0 / bool / int spelling.
    return (1, _encode(row))


def _public_row(row):
    return dict(zip(_FIELDS, row[1:], strict=True)) if row[0] == 0 else json.loads(row[1])


def _readonly(value):
    if isinstance(value, dict): return MappingProxyType({k: _readonly(v) for k, v in value.items()})
    if isinstance(value, list): return tuple(_readonly(v) for v in value)
    return value


@dataclass(frozen=True, slots=True)
class _HistoryBlock:
    rows: tuple
    encoded: bytes = field(init=False, repr=False)
    commitment: str = field(init=False)

    def __post_init__(self):
        encoded = _encode([_public_row(row) for row in self.rows])[1:-1]
        object.__setattr__(self, 'encoded', encoded)
        object.__setattr__(self, 'commitment', hashlib.sha256(encoded).hexdigest())


@dataclass(frozen=True, slots=True)
class NativeHistoryVersion:
    owner: str
    source: str
    blocks: tuple
    tail: _HistoryBlock | None
    count: int = field(init=False)
    commitment: str = field(init=False)

    def __post_init__(self):
        blocks = (*self.blocks, *((self.tail,) if self.tail is not None else ()))
        count = sum(len(block.rows) for block in blocks)
        object.__setattr__(self, 'count', count)
        object.__setattr__(self, 'commitment', fingerprint(dict(
            protocol='NATIVE_PRIVATE_HISTORY_V1', owner=self.owner, source=self.source,
            count=count, blocks=[(len(b.rows), b.commitment) for b in blocks])))

    def json_parts(self):
        yield b'['
        first = True
        for block in (*self.blocks, *((self.tail,) if self.tail is not None else ())):
            if not block.rows: continue
            if not first: yield b', '
            yield block.encoded
            first = False
        yield b']'


class NativeHistory(Sequence):
    """Append-only owned tail over immutable blocks, for private Env use only.

    Index reads produce read-only rows. Past writes/deletes explicitly
    fail; callers requiring a mutable list use the unchanged public export.
    """
    def __init__(self, owner, source, blocks=(), tail=()):
        self.owner, self.source = owner, source
        self._blocks = tuple(blocks)
        self._tail = list(tail)
        self._tail_block = None
        self._count = sum(len(b.rows) for b in self._blocks) + len(self._tail)

    @classmethod
    def from_public(cls, rows, *, owner, source):
        result = cls(owner, source)
        result.extend(rows)
        return result

    @classmethod
    def from_version(cls, version, *, owner=None):
        return cls(version.owner if owner is None else owner, version.source, version.blocks,
                   () if version.tail is None else version.tail.rows)

    def append(self, row):
        self._append_frozen(_freeze_row(row))

    def _append_frozen(self, row):
        self._tail.append(row)
        self._count += 1
        self._tail_block = None
        if len(self._tail) == _BLOCK_ROWS:
            self._blocks += (_HistoryBlock(tuple(self._tail)),)
            self._tail = []

    def extend(self, rows):
        for row in rows: self.append(row)

    def version(self):
        if self._tail and self._tail_block is None:
            self._tail_block = _HistoryBlock(tuple(self._tail))
        return NativeHistoryVersion(self.owner, self.source, self._blocks, self._tail_block)

    def __len__(self): return self._count

    def __getitem__(self, key):
        if isinstance(key, slice): return [self[i] for i in range(*key.indices(len(self)))]
        return _readonly(_public_row(self._row_at(key)))

    def _row_at(self,key):
        if key < 0: key += len(self)
        if not 0 <= key < len(self): raise IndexError(key)
        block, offset = divmod(key, _BLOCK_ROWS)
        return self._blocks[block].rows[offset] if block < len(self._blocks) else self._tail[offset]

    def public_slice(self,start):
        if not 0<=start<=len(self):raise IndexError(start)
        return [_public_row(self._row_at(i)) for i in range(start,len(self))]

    def __iter__(self):
        return (_readonly(row) for row in self._public_rows())

    def to_public(self):
        return list(self._public_rows())

    def _public_rows(self):
        for block in self._blocks:
            for row in block.rows: yield _public_row(row)
        for row in self._tail: yield _public_row(row)

    def __setitem__(self, key, value): raise TypeError('Private native history is append-only')
    def __delitem__(self, key): raise TypeError('Private native history is append-only')


def native_owner(env,*,account_sequence=None):
    if not hasattr(env, '_native_history_owner'): env._native_history_owner = uuid.uuid4().hex
    sequence=env.account_sequence if account_sequence is None else account_sequence
    return env._native_history_owner+':'+str(sequence)


def native_source(env):
    return fingerprint(dict(contract=env.contract, config=env.config_hash,
        observation=getattr(env, 'e_observation_identity', None)))


@dataclass(frozen=True, slots=True)
class NativeAccountSnapshotView:
    owner: str
    source: str
    cursor: int
    history: NativeHistoryVersion
    hot_json: bytes = field(repr=False)
    _digest_cache: str | None = field(default=None, init=False, repr=False, compare=False)

    @classmethod
    def capture(cls, env, state, history):
        owner, source = native_owner(env), native_source(env)
        if (history.owner != owner or history.source != source
                or state['account_sequence']!=env.account_sequence):
            raise ValueError('Private history source/owner mismatch')
        return cls(owner, source, state['current_step'], history,
                   _encode({k: v for k, v in state.items() if k != 'equity_history'}))

    def restore(self, env, *, source_env):
        if (self.owner != native_owner(source_env) or self.source != native_source(source_env)
                or self.source != native_source(env) or self.history.owner != self.owner
                or self.history.source != self.source):
            raise ValueError('Private account source/owner mismatch')
        state = json.loads(self.hot_json)
        if state['account_sequence']!=source_env.account_sequence:
            raise ValueError('Private account source/owner mismatch')
        if state['current_step'] != self.cursor:
            raise ValueError('Private account cursor mismatch')
        state['equity_history'] = NativeHistory.from_version(self.history,
            owner=native_owner(env,account_sequence=state['account_sequence']))
        env._restore_account_state(state)

    def public_state(self):
        state=json.loads(self.hot_json)
        state['equity_history']=NativeHistory.from_version(self.history).to_public()
        return state

    def digest(self):
        """Hash the original sorted JSON bytes; never substitute a Merkle ID.

        The immutable block commitment proves ownership/version consistency.
        This SHA still consumes every canonical history byte, in original JSON
        order. No full row list, pickle comparison, or JSON history re-encoding.
        """
        if self._digest_cache is not None: return self._digest_cache
        hot=json.loads(self.hot_json)
        result=hashlib.sha256(b'{')
        for number,key in enumerate(sorted((*hot, 'equity_history'))):
            if number: result.update(b', ')
            result.update(_encode(key)); result.update(b': ')
            if key=='equity_history':
                for part in self.history.json_parts(): result.update(part)
            else: result.update(_encode(hot[key]))
        result.update(b'}')
        digest=result.hexdigest()
        # This memo contains only a deterministic value of frozen bytes.
        object.__setattr__(self,'_digest_cache',digest)
        return digest

    def same_content(self, other):
        # Hot bytes preserve JSON distinctions that Python float/int/bool
        # equality would miss. Sealed primitive blocks cannot be mutated.
        return (type(other) is NativeAccountSnapshotView and self.owner==other.owner
            and self.source==other.source and self.cursor==other.cursor
            and self.history.count==other.history.count
            and self.history.commitment==other.history.commitment and self.hot_json==other.hot_json)

"""Exact original JSON/SHA256 account hashes with immutable history prefixes."""
import hashlib
import json
import pickle
import weakref


def _encode(value):
    return json.dumps(value,sort_keys=True,default=str).encode()


class AccountFingerprint:
    def __init__(self,history):
        # Own the prefix: later caller mutations must never alter cached bytes.
        self.history=json.loads(json.dumps(history,default=str))
        self.encoded=_encode(self.history)[:-1]
        self.comparison=pickle.dumps(self.history,protocol=5)
        self._frozen_prefix=None

    def digest(self,state):
        return self._digest(state,self._reusable(state['equity_history']))

    def _reusable(self,history):
        count=len(self.history)
        return (len(history)>=count and
                pickle.dumps(history[:count],protocol=5)==self.comparison)

    def capture(self,state,*,owner,source):
        from .native_account import NativeHistory
        history=state['equity_history']
        reusable=self._reusable(history)
        digest=self._digest(state,reusable)
        if reusable:
            if (self._frozen_prefix is None or self._frozen_prefix.owner!=owner
                    or self._frozen_prefix.source!=source):
                self._frozen_prefix=NativeHistory.from_public(self.history,owner=owner,source=source).version()
            private=NativeHistory.from_version(self._frozen_prefix)
            private.extend(history[len(self.history):])
        else:
            private=NativeHistory.from_public(history,owner=owner,source=source)
        return digest,private.version()

    def _digest(self,state,reusable):
        history=state['equity_history'];count=len(self.history)
        # Compare every old row without decimal formatting. Python equality
        # alone misses JSON changes such as 0.0 -> -0.0 or 1 -> True. Pickle
        # is only a private comparison key, never an external input format.
        result=hashlib.sha256();result.update(b'{')
        for number,key in enumerate(sorted(state)):
            if number:result.update(b', ')
            result.update(_encode(key));result.update(b': ')
            if key!='equity_history' or not reusable:
                result.update(_encode(state[key]));continue
            result.update(self.encoded)
            if len(history)>count:
                if count:result.update(b', ')
                result.update(_encode(history[count:])[1:])
            else:result.update(b']')
        result.update(b'}')
        return result.hexdigest()


_HASHERS=weakref.WeakKeyDictionary()


def prime_account_fingerprint(env,state):
    if env not in _HASHERS:_HASHERS[env]=AccountFingerprint(state['equity_history'])


def account_fingerprint(env,state=None):
    from .native_account import NativeAccountSnapshotView,NativeHistory
    if state is None:
        state=env._export_private_state() if isinstance(env.equity_history,NativeHistory) else env.export_state()
    if isinstance(state,NativeAccountSnapshotView):state=state.public_state()
    prime_account_fingerprint(env,state)
    return _HASHERS[env].digest(state)


def capture_account_fingerprint(env,state):
    from .native_account import native_owner,native_source,NativeAccountSnapshotView
    prime_account_fingerprint(env,state)
    digest,history=_HASHERS[env].capture(state,owner=native_owner(env),source=native_source(env))
    return digest,NativeAccountSnapshotView.capture(env,state,history)

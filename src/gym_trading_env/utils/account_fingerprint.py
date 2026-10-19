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

    def digest(self,state):
        history=state['equity_history'];count=len(self.history)
        # Compare every old row without decimal formatting. Python equality
        # alone misses JSON changes such as 0.0 -> -0.0 or 1 -> True. Pickle
        # is only a private comparison key, never an external input format.
        reusable=(len(history)>=count and
                  pickle.dumps(history[:count],protocol=5)==self.comparison)
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
    state=env.export_state() if state is None else state
    prime_account_fingerprint(env,state)
    return _HASHERS[env].digest(state)

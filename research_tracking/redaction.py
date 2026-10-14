"""Explicit upload boundary, shared by configuration and all event payloads."""
import re

SECRET = re.compile(r'api.?key|token|password|secret|credential|private.?key|authorization|proxy', re.I)
TOKEN = re.compile(r'(?i)(?:bearer\s+\S+|(?:sk|wbd|wandb)_[a-z0-9_-]{16,}|[a-z0-9+/=_-]{40,})')
PRIVATE_KEY = re.compile(r'-----BEGIN [^-]*PRIVATE KEY-----[\s\S]*?-----END [^-]*PRIVATE KEY-----')
PATH = re.compile(r'(?<![\w])/(?:home|data|tmp|mnt|media|root|opt|usr)/[^\s\"\'<>;,]*')
HASH_KEY = re.compile(r'(?:sha256|hash|identity|commit|run_id|parent_training_run_id)$', re.I)
EXCLUDED = {'sampled_stepids', 'code', 'immutable_files', 'rng', 'balances',
            'ledger_entries', 'equity_history', 'market_seq', 'fills', 'entries'}


def redact(value, key='', *, excluded=True):
    if SECRET.search(key) and not (key in ('credential_present', 'credential_valid') and type(value) is bool):
        return '[REDACTED]'
    if isinstance(value, dict):
        return {str(k): redact(v, str(k), excluded=excluded) for k, v in value.items()
                if not excluded or k not in EXCLUDED}
    if isinstance(value, (list, tuple)):
        return [redact(v, key, excluded=excluded) for v in value]
    if isinstance(value, str):
        if (HASH_KEY.search(key) or key in ('gym-trading-env', 'dreamerv3')) and re.fullmatch(r'[0-9a-f]{16,64}', value):
            return value
        value = PRIVATE_KEY.sub('[REDACTED_PRIVATE_KEY]', value)
        value = PATH.sub('[LOCAL_PATH]', value)
        return TOKEN.sub('[REDACTED_TOKEN]', value)
    return value


def safe_error(exc):
    # Exception messages from SDK/network libraries can contain URLs or keys.
    return redact(str(exc), excluded=False)

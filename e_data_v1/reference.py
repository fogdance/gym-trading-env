"""Reuse the existing pure feature calculator without Gym's DB import hooks.

Only import targets are redirected into a private namespace; function bodies,
constants and arithmetic are compiled from the unchanged source bytes. This
keeps the independent CPU venv free of Torch, JAX, database and W&B packages.
"""
from __future__ import annotations

import ast
from functools import lru_cache
import hashlib
from pathlib import Path
import sys
from types import ModuleType
import warnings

import pandas as pd

from .schema import LEGACY_FIELDS

FILES = {
    "timebase": "utils/timebase.py", "time_contract": "utils/time_contract.py",
    "ohlcvi_contract": "utils/ohlcvi_contract.py", "s3_market_clock": "utils/s3_market_clock.py",
    "entry_market_cache": "envs/entry_market_cache.py", "session_fx": "utils/session_fx.py",
    "session_futures_strict": "utils/session_futures_strict.py", "market_features": "utils/market_features.py",
}


@lru_cache(maxsize=4)
def load_legacy(root=None):
    root = Path(root) if root else Path(__file__).resolve().parents[1]
    directory = root / "src/gym_trading_env"
    sources = {name: (directory / relative).read_bytes() for name,relative in FILES.items()}
    hashes = {"src/gym_trading_env/" + FILES[name]: hashlib.sha256(data).hexdigest()
              for name, data in sources.items()}
    namespace = "_s3_e_reference_" + hashlib.sha256(b"".join(sources.values())).hexdigest()[:20]
    if namespace not in sys.modules:
        package = ModuleType(namespace)
        package.__path__ = []
        sys.modules[namespace] = package
        for name, data in sources.items():
            filename = str(directory / FILES[name])
            tree = ast.parse(data, filename=filename)
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("gym_trading_env."):
                    target = node.module.removeprefix("gym_trading_env.utils.")
                    if target not in FILES:
                        raise ValueError("Unreviewed legacy import: " + node.module)
                    node.module = namespace + "." + target
            module = ModuleType(namespace + "." + name)
            module.__file__ = filename
            module.__package__ = namespace
            sys.modules[module.__name__] = module
            exec(compile(tree, filename, "exec"), module.__dict__)
    feature_module = sys.modules[namespace + ".market_features"]
    if tuple(feature_module.FEATURES_MARKET_OBS_OUTPUT) != LEGACY_FIELDS:
        raise ValueError("Existing 18-channel contract changed; review required")
    return feature_module, hashes


def legacy_helper(name, root=None):
    module, _ = load_legacy(root)
    return sys.modules[module.__package__ + "." + name]


def legacy_features(frame, schedule, root=None):
    module, _ = load_legacy(root)
    # Existing wide DataFrame construction emits pandas fragmentation warnings.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", pd.errors.PerformanceWarning)
        return module.build_market_features(frame, is_future=True,
            real_futures_clock=True, futures_schedule=schedule)

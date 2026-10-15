"""Exact, reviewed market-reader transitions; never a general identity bypass."""
import hashlib
from pathlib import Path
import subprocess


# c14fcdc market cache and 9eba965 shared DB connector. CSV normalization,
# causal observations, execution, market clock and reward math are unchanged.
PRODUCERS = {
    'gym_trading_env/envs/entry_dataset_env.py': '2647dd96dd4e7b3648262b52415d2e84a2f5ecea46959e01a1ece83b9794d4a8',
    'gym_trading_env/envs/entry_env.py': '2411131aaf3f84311ce3d54cca5ffdf530a1268ed4d583ba90b9708c28abbce9',
    'gym_trading_env/utils/bar_source.py': '9f847f8324811c48dcbbd8d980fb93fc400539d0bbf7f8ea0946df711ece0cdb',
}
TARGETS = {
    'gym_trading_env/envs/entry_dataset_env.py': '544fdaf162548d09324dfcd5b585737dfd71153cb40588dcc96f8e496e104d36',
    'gym_trading_env/envs/entry_env.py': '7d1fe65a34a3923575fb4d03c9b8586d44f2f10f81ba76922d6081c07f595a31',
    'gym_trading_env/envs/entry_market_cache.py': '0a57a84a2e0df5dfdeae3c7308ced3cb9c9bee74c05cfc20963447d1a9daf0e5',
    'gym_trading_env/utils/bar_source.py': '8520fe65904b4bd2e271283fc4cd03acce9b015f344b95b90213ff6444f87667',
}
# Native E adds optional observations and account-free cache payloads. The
# original financial/teacher identity is unchanged. Review evidence is in
# docs/e_native_env_20261015: all 488160 TRAIN rows preserve original fields
# and financial state, with one CSV read and one original feature build.
# Every byte in this bundle must match; E admission separately authenticates
# its additional observation/code/data identity. This does not relax the
# semantic-file checks for old B2 parameter warm-starts.
NATIVE_E_TARGETS = {
    'gym_trading_env/envs/entry_dataset_env.py': 'd55faff6d97412392a712117e9908110a5126ef70824d7cfd6d3589bbf19f998',
    'gym_trading_env/envs/entry_env.py': 'b72ecfd8051418954ac5c763ff6b403ba62b87c227289e54985ecd354de470cb',
    'gym_trading_env/envs/entry_market_cache.py': '1478f566cb27123efd13a974826e71454f46906bfe8c7e39c540e74714d2fe6f',
    'gym_trading_env/utils/bar_source.py': '8520fe65904b4bd2e271283fc4cd03acce9b015f344b95b90213ff6444f87667',
}
# E-only source authentication closes the admission/read gap. Non-E loading,
# feature math, financial rules and the previous reviewed bundles are retained.
# See docs/e_review_fix_20261015 for failing-before/passing-after regressions.
NATIVE_E_IDENTITY_TARGETS = {
    **NATIVE_E_TARGETS,
    'gym_trading_env/envs/entry_env.py': '2f21cae850be5790eae2c806415d8fa060a66b848732e7284602d0bd11e80418',
    'gym_trading_env/envs/entry_market_cache.py': '1f6aff8d038c8c83438106f1e314ca604a442e74639eae09d238e0594dff5bf2',
}


def reviewed_reader_transition(repo, name, expected, hash_file):
    """Authenticate both producer bytes and the complete current reader bundle.

    Callers opt into this seam only for parameter warm-start or loading the
    original Oracle receipts, never exact resume. Oracle loading also checks
    original market/candidates, funding certificate and suffix DP results.
    """
    repo = Path(repo)
    bundle = next((b for b in (TARGETS, NATIVE_E_TARGETS, NATIVE_E_IDENTITY_TARGETS)
                   if all(hash_file(repo/'src'/n) == h for n, h in b.items())), None)
    if expected != PRODUCERS.get(name) or bundle is None:
        raise ValueError('Unreviewed source reader transition: '+name)
    rel = 'src/'+name
    revisions = subprocess.check_output(['git', 'rev-list', '--all', '--', rel],
                                        cwd=repo, text=True).split()
    found = next((rev for rev in revisions if hashlib.sha256(
        subprocess.check_output(['git', 'show', rev+':'+rel], cwd=repo)).hexdigest() == expected), None)
    if found is None:
        raise ValueError('Reviewed reader producer bytes unavailable: '+name)
    return dict(source_sha256=expected, target_sha256=hash_file(repo/rel), source_git=found,
                approved_target_bundle=bundle.copy())

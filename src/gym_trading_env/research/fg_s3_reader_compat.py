"""Exact, reviewed market-reader transitions; never a general identity bypass."""
import hashlib
from pathlib import Path
import subprocess


# Exact reviewed reader bundle: market cache / shared DB connector plus an
# opt-in Quality TP profile. Legacy schema math, snapshots and observations
# are unchanged; new-target labels are rebuilt by the quality adapter.
PRODUCERS = {
    'gym_trading_env/envs/entry_dataset_env.py': '2647dd96dd4e7b3648262b52415d2e84a2f5ecea46959e01a1ece83b9794d4a8',
    'gym_trading_env/envs/entry_env.py': '2411131aaf3f84311ce3d54cca5ffdf530a1268ed4d583ba90b9708c28abbce9',
    'gym_trading_env/utils/bar_source.py': '9f847f8324811c48dcbbd8d980fb93fc400539d0bbf7f8ea0946df711ece0cdb',
    'gym_trading_env/rewards/single3_reward.py':'8985363ebb543ca19a86b03de94ebaec065101e621a19b22c52c76dfe9e61d68',
}
TARGETS = {
    'gym_trading_env/envs/entry_dataset_env.py': '544fdaf162548d09324dfcd5b585737dfd71153cb40588dcc96f8e496e104d36',
    'gym_trading_env/envs/entry_env.py': '9f12edd1d2bd625048cffc6aaa3b3edc2da97248e46acc9cdbae69dea88d726b',
    'gym_trading_env/envs/entry_market_cache.py': '0a57a84a2e0df5dfdeae3c7308ced3cb9c9bee74c05cfc20963447d1a9daf0e5',
    'gym_trading_env/utils/bar_source.py': '8520fe65904b4bd2e271283fc4cd03acce9b015f344b95b90213ff6444f87667',
    'gym_trading_env/rewards/single3_reward.py':'60c790ad065920a4f30efecaed60dfcb49abc8358491d96fc9b7d5548d715cb8',
    'gym_trading_env/research/fg_reward_candidates.py':'cbcfb6d8f040c8db17fa8fb93200213769d19028d30218c61ef9b8bc6b1b8c8b',
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
# This current bundle also binds the opt-in Quality reward/candidate modules;
# its Env includes both the E byte-identity check and Quality TP dispatch.
# See docs/e_review_fix_20261015 for failing-before/passing-after regressions.
NATIVE_E_IDENTITY_TARGETS = {
    **NATIVE_E_TARGETS,
    'gym_trading_env/envs/entry_env.py': 'dc36fe74096ef3b778c792040bfec8d364e2958b029b9fd25fd433af3fbae0a2',
    'gym_trading_env/envs/entry_market_cache.py': '1f6aff8d038c8c83438106f1e314ca604a442e74639eae09d238e0594dff5bf2',
    'gym_trading_env/rewards/single3_reward.py': TARGETS['gym_trading_env/rewards/single3_reward.py'],
    'gym_trading_env/research/fg_reward_candidates.py': TARGETS['gym_trading_env/research/fg_reward_candidates.py'],
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

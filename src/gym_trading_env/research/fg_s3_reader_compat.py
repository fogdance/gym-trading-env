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
    'gym_trading_env/research/fg_reward_candidates.py':'90df8ab4d6bad8ebef58a5e9caeb04499747887b5ef76d01dff9937ac034cb51',
}


def reviewed_reader_transition(repo, name, expected, hash_file):
    """Authenticate both producer bytes and the complete current reader bundle.

    Callers opt into this seam only for parameter warm-start or loading the
    original Oracle receipts, never exact resume. Oracle loading also checks
    original market/candidates, funding certificate and suffix DP results.
    """
    repo = Path(repo)
    if (expected != PRODUCERS.get(name) or any(
            hash_file(repo/'src'/n) != h for n, h in TARGETS.items())):
        raise ValueError('Unreviewed source reader transition: '+name)
    rel = 'src/'+name
    revisions = subprocess.check_output(['git', 'rev-list', '--all', '--', rel],
                                        cwd=repo, text=True).split()
    found = next((rev for rev in revisions if hashlib.sha256(
        subprocess.check_output(['git', 'show', rev+':'+rel], cwd=repo)).hexdigest() == expected), None)
    if found is None:
        raise ValueError('Reviewed reader producer bytes unavailable: '+name)
    return dict(source_sha256=expected, target_sha256=hash_file(repo/rel), source_git=found,
                approved_target_bundle=TARGETS.copy())

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

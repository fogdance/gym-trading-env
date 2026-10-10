import numpy as np
import pytest

from gym_trading_env.research.fg_history_fold_statistics import FIELDS, aggregate_job, summarize
from gym_trading_env.research.fg_history_parallel import run_job
from tests.entry_v1.test_fg_history_parallel import task_for


def test_censor_bounds_and_contract_clusters_are_explicit():
    values = dict(n=4, success=2, terminal=1, censored=1, reward_sum=1.5, reward_known=3,
                  censored_reward_upper=.5, wait_sum=62, **{'60m_hit': 2, '60m_unknown': 1})
    contract = [values.get(field, 0) for field in FIELDS]
    result = summarize([contract, contract], replicates=100, rng=np.random.default_rng(0))
    assert result['n'] == 8 and result['contracts'] == 2
    assert result['mean_known_reward'] == .5
    assert result['mean_reward_lower'] == .375
    assert result['mean_reward_upper'] == .5
    assert result['60m_lower'] == .5 and result['60m_upper'] == .75
    assert result['60m_cluster_ci_low'] == .5 and result['60m_cluster_ci_high'] == .75
    assert result['inference_status'] == 'insufficient_contract_clusters'


def test_event_aggregation_keeps_directions_and_detects_changed_outputs(tmp_path):
    task = task_for(tmp_path)
    run_job(task)
    result = aggregate_job(task)
    assert set(result['values']) == {'LONG', 'SHORT'}
    for direction in ['LONG', 'SHORT']:
        metrics = dict(zip(FIELDS, result['values'][direction]))
        assert metrics['n'] == 2
        assert metrics['n'] == metrics['success'] + metrics['terminal'] + metrics['censored']
        assert metrics['reward_known'] == metrics['success'] + metrics['terminal']
        assert result['rejected'][direction] == 1
        assert sum(item['values'][0] for item in result['daily'] if item['direction'] == direction) == metrics['n']
    from pathlib import Path
    (Path(task['directory']) / 'events.csv.gz').write_bytes(b'changed')
    with pytest.raises(ValueError, match='Event output changed'):
        aggregate_job(task)

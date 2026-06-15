import pytest

from gym_trading_env.envs.trading_env import CustomTradingEnv
from tests.factories.market_data_factory import MarketDataFactory


pytestmark = pytest.mark.integration


def _start_sequence(seed: int) -> list[int]:
    bundle = MarketDataFactory.make_futures_bundle(num_days=6, seed=11)
    env = CustomTradingEnv(df=bundle.df_1m, config_path="tests/test.yaml")
    try:
        env.config.training.randomize_start = True
        env.config.training.start_clock = "any"
        env.config.training.episode_length = 10
        env.config.training.episode_policy.truncate_on_session_end = False
        starts = []
        env.reset(seed=seed)
        starts.append(int(env.start_idx))
        for _ in range(5):
            env.reset()
            starts.append(int(env.start_idx))
        return starts
    finally:
        env.close()


def test_seeded_randomize_start_sequence_is_reproducible():
    first = _start_sequence(101)
    second = _start_sequence(101)

    assert first == second


def test_different_randomize_start_seed_changes_sequence_when_practical():
    first = _start_sequence(101)
    second = _start_sequence(202)

    assert len(set(first)) > 1
    assert first != second

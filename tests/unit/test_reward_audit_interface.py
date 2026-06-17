import pytest

from gym_trading_env.rewards.reward_audit import (
    REWARD_DEBUG_KEYS,
    RewardAuditError,
    RewardAuditMixin,
    validate_reward_audit,
)


pytestmark = pytest.mark.unit


class _GoodReward(RewardAuditMixin):
    def __init__(self):
        debug = self.default_reward_debug()
        debug["raw_total"] = 0.25
        debug["total"] = 0.25
        self._set_reward_debug(debug)


class _BadNoInterface:
    pass


class _MissingKeyReward(RewardAuditMixin):
    def reward_audit_debug(self):
        debug = self.default_reward_debug()
        debug.pop(REWARD_DEBUG_KEYS[0])
        return debug


class _MismatchedTotalReward(RewardAuditMixin):
    def __init__(self):
        debug = self.default_reward_debug()
        debug["total"] = 0.1
        self._set_reward_debug(debug)


def test_reward_audit_interface_accepts_valid_snapshot():
    debug = validate_reward_audit(
        _GoodReward(),
        reward_name="good_reward",
        returned_reward=0.25,
    )

    assert set(REWARD_DEBUG_KEYS).issubset(debug)
    assert debug["total"] == pytest.approx(0.25)


def test_reward_audit_interface_rejects_missing_interface():
    with pytest.raises(RewardAuditError, match="RewardAuditMixin"):
        validate_reward_audit(
            _BadNoInterface(),
            reward_name="bad_reward",
            returned_reward=0.0,
        )


def test_reward_audit_interface_rejects_missing_keys():
    with pytest.raises(RewardAuditError, match="missing keys"):
        validate_reward_audit(
            _MissingKeyReward(),
            reward_name="missing_key_reward",
            returned_reward=0.0,
        )


def test_reward_audit_interface_rejects_total_mismatch():
    with pytest.raises(RewardAuditError, match="does not match"):
        validate_reward_audit(
            _MismatchedTotalReward(),
            reward_name="mismatch_reward",
            returned_reward=0.2,
        )

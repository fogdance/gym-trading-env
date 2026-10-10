"""The FG business reward has no account, price forecast or penalty inputs."""
from .reward_audit import RewardAuditMixin
from gym_trading_env.envs.entry_contract import timestamp
from decimal import Decimal


class FastTakeProfitReward(RewardAuditMixin):
    def __init__(self):
        self.rewarded = set()

    def __call__(self, events):
        total = 0.0
        for event in events:
            if event["reason"] != "TAKE_PROFIT" or Decimal(str(event["closed_quantity"])) != 3:
                continue
            entry_id = event["entry_id"]
            if entry_id in self.rewarded:
                raise ValueError("Duplicate TP reward")
            sign = 1 if event["side"] == "LONG" else -1
            difference = sign * (Decimal(str(event["exit_price"])) - Decimal(str(event["entry_price"])))
            minutes = (timestamp(event["exit_time"]) - timestamp(event["entry_time"])).total_seconds() / 60
            if minutes < 0 or difference < 5 or not event.get("committed", False):
                raise ValueError("Reward requires a committed causal five-point full TP fill")
            self.rewarded.add(entry_id)
            event["elapsed_minutes"] = minutes
            event["reward"] = 60 / (60 + minutes)
            total += event["reward"]
        debug = self.default_reward_debug()
        debug.update(total=total, raw_total=total, tp_speed=total)
        self._set_reward_debug(debug)
        return total

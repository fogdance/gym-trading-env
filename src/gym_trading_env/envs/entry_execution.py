"""Single3 exit decision shared by the real Env and TRAIN Oracle indexes.

Quotes must come from the Env's executable-price convention. This module does
not post a fill, change liquidity, release margin, or settle an account.
"""
from dataclasses import dataclass
from decimal import Decimal


@dataclass(frozen=True)
class ExitDecision:
    reason: str | None
    price: Decimal | None
    sl_pending: bool
    sl_trigger_time: str | None
    metadata: dict


def single3_exit_decision(*, side, entry_time, stop_price, tp_price, opening_quote,
                         adverse_quote, favorable_quote, close_quote, bob, time,
                         tick, valid_bar, was_pending=False, trigger_time=None,
                         global_reason=None):
    sign = 1 if side == 'LONG' else -1
    tp_touch = sign * (favorable_quote-tp_price) >= tick
    sl_touch = valid_bar and sign * (adverse_quote-stop_price) <= 0
    pending = was_pending or sl_touch
    trigger_time = trigger_time or (str(time) if sl_touch else None)
    metadata = dict(ambiguous_tp_sl=bool(tp_touch and sl_touch),
                    same_entry_bar=bool(entry_time == bob),
                    price_reference_time=str(time))
    if global_reason:
        price = close_quote if global_reason == 'MAIN_ROLL' else adverse_quote
        return ExitDecision(global_reason, price, pending, trigger_time, metadata)
    if pending:
        gap = sign*(opening_quote-stop_price) < 0
        price = opening_quote if was_pending or gap else stop_price
        metadata.update(sl_trigger_time=trigger_time, sl_pending_retry=bool(was_pending),
            gap_slippage_points=float(max(Decimal(0), sign*(stop_price-price))),
            price_reference_time=str(bob if was_pending or gap else time))
        return ExitDecision('STOP_LOSS', price, pending, trigger_time, metadata)
    if tp_touch:
        return ExitDecision('TAKE_PROFIT', tp_price, False, trigger_time, metadata)
    return ExitDecision(None, None, False, trigger_time, metadata)

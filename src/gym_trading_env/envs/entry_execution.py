"""Single3 exit decision shared by the real Env and TRAIN Oracle indexes.

Quotes must come from the Env's executable-price convention. This module does
not post a fill, change liquidity, release margin, or settle an account.
"""
from collections import OrderedDict
from dataclasses import dataclass
from decimal import Decimal, getcontext


class NativeMarketExecutionCache:
    """Teacher-private reuse of market rows and exact executable quotes.

    This contains no account, margin, position, exit, reward or action result.
    The original Env still executes every financial transition. Source guards
    own the frozen frame; changing the frame object invalidates its row cache.
    Quotes depend on actual effective terms and Decimal context, never account
    funds or a timestamp stripped of its effective execution costs.
    """
    def __init__(self, row_capacity=1024, quote_capacity=8192, term_capacity=4096):
        self.row_capacity = row_capacity
        self.quote_capacity = quote_capacity
        self.term_capacity = term_capacity
        self.rows = OrderedDict()
        self.quotes = OrderedDict()
        self.effective_terms = OrderedDict()
        self.frame = None
        self._terms = None
        self._periods = ()
        self._period_layout = ()
        self.statistics = dict(row_hits=0, row_misses=0, quote_hits=0,
                               quote_misses=0, row_evictions=0, quote_evictions=0,
                               term_hits=0, term_misses=0, term_evictions=0)

    def terms_at(self, terms, time):
        # Keep actual period dictionaries, so a changed effective fee/rate is
        # observed immediately. Replaced periods or boundaries invalidate the
        # lookup; an old timestamp must never select a stale cost schedule.
        layout = tuple((start, end, id(values)) for start, end, values in terms.periods)
        if self._terms is not terms or self._period_layout != layout:
            self.effective_terms.clear()
            self._terms = terms
            self._periods = tuple(terms.periods)
            self._period_layout = layout
        if time in self.effective_terms:
            self.statistics['term_hits'] += 1
            self.effective_terms.move_to_end(time)
            return self.effective_terms[time]
        self.statistics['term_misses'] += 1
        value = terms.at(time)
        self.effective_terms[time] = value
        if len(self.effective_terms) > self.term_capacity:
            self.effective_terms.popitem(last=False)
            self.statistics['term_evictions'] += 1
        return value

    def row(self, frame, cursor):
        if self.frame is not frame:
            self.rows.clear()
            self.frame = frame  # Keep the identity alive, including across rewinds.
        if cursor in self.rows:
            self.statistics['row_hits'] += 1
            self.rows.move_to_end(cursor)
            return self.rows[cursor]
        self.statistics['row_misses'] += 1
        row = frame.iloc[cursor]
        self.rows[cursor] = row
        if len(self.rows) > self.row_capacity:
            self.rows.popitem(last=False)
            self.statistics['row_evictions'] += 1
        return row

    @staticmethod
    def quote_key(raw_price, side, opening, terms, tick):
        context = getcontext()
        # Decimal equality ignores its exponent; text retains the exact input
        # representation and the original rounded quote's tick exponent.
        return (str(raw_price), side, bool(opening), str(terms['spread']),
                str(terms['slippage']), str(tick), context.prec,
                context.rounding, context.Emin, context.Emax, context.clamp,
                tuple(context.traps.values()))

    def quote(self, key):
        value = self.quotes.get(key)
        if value is None:
            self.statistics['quote_misses'] += 1
        else:
            self.statistics['quote_hits'] += 1
            self.quotes.move_to_end(key)
        return value

    def put_quote(self, key, value):
        self.quotes[key] = value
        if len(self.quotes) > self.quote_capacity:
            self.quotes.popitem(last=False)
            self.statistics['quote_evictions'] += 1


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

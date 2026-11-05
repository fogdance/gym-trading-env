"""Live mode of the Single3 Entry env (user 2026-10-30), as the original env's
live_mode: completed bars of the real contract come from the market DB
(fut_bar_1m_v2), `step` blocks until the next bar, and committed fills are
sent to the existing order RPC. The Native engine stays the only ledger;
RPC acknowledgement means a queued signal, broker fills are not reconciled.
Every start, a restart too, is a first run (user 2026-10-30): an empty
account from the latest DB bar; the last history_days only fill the
observation window.
"""
import json
import time
from pathlib import Path

import pandas as pd

TZ = 'Asia/Shanghai'


class LiveFeed:
    """Completed bars of one contract in the CSV schema of the market exports."""

    def __init__(self, contract, settings, log):
        self.contract, self.log = contract, log
        self.history_days = float(settings.get('history_days', 10))
        self.poll_seconds = float(settings.get('poll_seconds', 2))
        self.max_signal_age = float(settings.get('max_signal_age_seconds', 90))
        self.rpc = None
        if settings.get('rpc_enabled'):
            from gym_trading_env.config.settings import RPC_ORDER_CONFIG
            from gym_trading_env.utils.rpc_client import LanOrderClient
            self.rpc = LanOrderClient(RPC_ORDER_CONFIG.order_endpoint, RPC_ORDER_CONFIG.order_token,
                                      RPC_ORDER_CONFIG.order_timeout_sec)
        self.action_log = Path(settings['action_log']) if settings.get('action_log') else None

    def _query(self, sql, args):
        import pymysql
        from gym_trading_env.utils.bar_source import connect_juejin_db
        wait = 2
        while True:  # a DB outage waits, it never skips or ends the account
            try:
                conn = connect_juejin_db(connect_timeout=5, read_timeout=30, write_timeout=5)
                try:
                    with conn.cursor() as c:
                        c.execute(sql, args)
                        return c.fetchall()
                finally:
                    conn.close()
            except pymysql.err.MySQLError as error:
                self.log(f'[LIVE] DB error, retry in {wait}s: {error}')
                time.sleep(wait)
                wait = min(wait * 2, 60)

    def read(self, after=None, floor=None):
        """Bars after `after` (tz-aware), or of the last history_days (not
        before `floor`, the first minute of the contract calendar)."""
        from gym_trading_env.utils.bar_source import JUEJIN_FUT_BAR_TABLE
        bound = after if after is not None else max(
            pd.Timestamp.now(tz=TZ) - pd.Timedelta(days=self.history_days), floor or pd.Timestamp(0, tz=TZ))
        rows = self._query(
            f'SELECT trading_date, bob, eob, `open`, `high`, `low`, `close`, volume, `position` '
            f'FROM {JUEJIN_FUT_BAR_TABLE} WHERE symbol=%s AND eob {">" if after is not None else ">="} %s '
            f'AND bob < eob ORDER BY eob',
            (self.contract, pd.Timestamp(bound).tz_convert(TZ).tz_localize(None).to_pydatetime()))
        if not rows:
            return pd.DataFrame()
        f = pd.DataFrame(rows)
        out = pd.DataFrame({
            'Date': pd.to_datetime(f.eob).dt.strftime('%Y-%m-%d %H:%M:%S'),
            **{k: pd.to_numeric(f[c]).astype(float) for k, c in (
                ('Open', 'open'), ('High', 'high'), ('Low', 'low'), ('Close', 'close'),
                ('Volume', 'volume'), ('OpenInterest', 'position'))},
            'symbol': self.contract,
            'bob': pd.to_datetime(f.bob).dt.strftime('%Y-%m-%d %H:%M:%S'),
            'trading_date': pd.to_datetime(f.trading_date).dt.strftime('%Y-%m-%d')})
        if after is None and bound > (floor or pd.Timestamp(0, tz=TZ)):
            # The history begins at a whole trading day (V2 live, 2026-11-05): the
            # 5m / 15m buckets of the E market and its causal MACD need a session
            # from its first bar, and now - history_days cuts one anywhere.
            out = out[out.trading_date != out.trading_date.iloc[0]].reset_index(drop=True)
        return out

    def wait(self, last):
        """Block until bars after `last` exist."""
        while True:
            rows = self.read(after=last)
            if not rows.empty:
                return rows
            time.sleep(self.poll_seconds)

    def send(self, env, decision_time, fills):
        """Mirror committed fills to the order RPC; a decision older than
        max_signal_age (a restart catching up) is logged, not sent."""
        from gym_trading_env.envs.action import Action, ForexCode
        from gym_trading_env.utils.rpc_protocol import TradeSignal
        age = (pd.Timestamp.now(tz=TZ) - decision_time).total_seconds()
        for fill in fills:
            action = Action[f"{fill['side']}_{'OPEN0' if fill['reason'] == 'ENTRY' else 'CLOSE0'}"]
            record = dict(time=str(decision_time), age=round(age, 1), fill=fill, sent=False)
            if self.rpc is not None and age <= self.max_signal_age:
                eob = env._raw.index[env.current_step].tz_convert(TZ).strftime('%Y-%m-%d %H:%M:%S')
                sig = TradeSignal(
                    signal_id=f"{env.contract}|{fill['entry_id']}|{fill['time']}|{action.name}",
                    symbol=env.contract, eob=eob, action_index=action.value, action_name=action.name,
                    volume=3, price=float(fill['price']), action_result=ForexCode.SUCCESS.value,
                    meta=dict(fill_reason=fill['reason'], simulated_fill_time=fill['time']))
                record.update(sent=True, response=self.rpc.send(sig))
            self.log(f'[LIVE] {json.dumps(record, default=str)}')
            if self.action_log is not None:
                with self.action_log.open('a') as f:
                    f.write(json.dumps(record, default=str) + '\n')

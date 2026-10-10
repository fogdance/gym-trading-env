"""Stdlib-only offline reward regression suite (run with pytest or unittest)."""
import copy
import importlib.util
import json
import math
from datetime import datetime,timedelta
from pathlib import Path
import sys
import unittest

P=Path(__file__).resolve().parents[2]/'src/gym_trading_env/research/fg_reward_candidates.py'
spec=importlib.util.spec_from_file_location('fg_candidates_for_tests',P)
m=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=m
spec.loader.exec_module(m)


def clock(n):
    return (datetime(2024,4,9,9,0)+timedelta(minutes=n)).isoformat()


def bar(index, *, price=100, high=None, low=None, side=None, entry_id='FG:1',
        exit_reason=None, exit_price=None, eligible=True, valid=True, chunk=False):
    t0=clock(index)
    t1=clock(index+1) if valid else t0
    row=dict(bar_open=t0,bar_end=t1,open=price,high=price if high is None else high,
             low=price if low is None else low,close=price,valid_bar=valid,
             eligible_market_minute=eligible,entries=[],exits=[],chunk_end=chunk)
    if side:
        row['entries']=[dict(entry_id=entry_id,reason='ENTRY',side=side,time=t0,
                             price=100,quantity=3,committed=True)]
    if exit_reason=='TAKE_PROFIT':
        # Actual committed fills must be supported by the observed bar range.
        target=exit_price if exit_price is not None else 105
        row['high']=max(row['high'],target)
    if exit_reason:
        row['exits']=[dict(entry_id=entry_id,side=side or 'LONG',entry_price=100,
                           exit_price=exit_price if exit_price is not None else 105,
                           entry_time=clock(1) if entry_id=='FG:2' else clock(0),exit_time=t1,closed_quantity=3,
                           reason=exit_reason,committed=True)]
    return row


class RewardCandidatesTests(unittest.TestCase):
    def test_tp_boundary_curve_and_monotone(self):
        for t,expected in [(5,60/65),(15,.8),(60,.5),
                           (240,(60/300)*(240/420)),(480,0),(600,0)]:
            self.assertAlmostEqual(m.timeboxed_tp(t),expected)
        values=[m.timeboxed_tp(t/2) for t in range(0,1201)]
        self.assertTrue(all(0 <= v <= 1 for v in values))
        self.assertTrue(all(a>=b for a,b in zip(values,values[1:])))
        self.assertAlmostEqual(m.timeboxed_tp(60),m.timeboxed_tp(60+1e-7),places=8)
        self.assertAlmostEqual(m.timeboxed_tp(480),m.timeboxed_tp(480-1e-7),places=8)
        with self.assertRaises(m.RewardInputError): m.timeboxed_tp(-1)

    def test_early_first_touch_boundaries_and_no_repeat(self):
        for first,expected in [(5,-.2),(6,-.06),(15,-.06),(16,0)]:
            scorer=m.FGRewardScorer(m.CandidateConfig(beta=.2,rho=.3))
            total=0
            for i in range(18):
                hit=i+1>=first
                row=bar(i,side='SHORT' if i==0 else None, high=110 if hit else 100,
                        low=100, price=100)
                total+=scorer.step(row)['early']
            self.assertAlmostEqual(total,expected)
            self.assertEqual(scorer.active['FG:1'].adverse_first_bar,first)
            self.assertTrue(scorer.active['FG:1'].adverse10_fired)

    def test_long_short_mirror(self):
        c=m.CandidateConfig(beta=.2,lam=.01,alpha=.05)
        long=m.FGRewardScorer(c)
        short=m.FGRewardScorer(c)
        a=bar(0,side='LONG',price=100,high=101,low=90)
        b=bar(0,side='SHORT',price=100,high=110,low=99)
        al=long.step(a);bs=short.step(b)
        for k in ('early','potential','v2','v3','capacity'):
            self.assertAlmostEqual(al[k],bs[k])

    def test_two_slots_and_same_bar_settlement(self):
        s=m.FGRewardScorer(m.CandidateConfig(lam=.2,alpha=.05))
        one=s.step(bar(0,side='LONG'))
        two=s.step(bar(1,side='LONG',entry_id='FG:2'))
        three=s.step(bar(2,exit_reason='TAKE_PROFIT',exit_price=105))
        four=s.step(bar(3,exit_reason='TAKE_PROFIT',exit_price=105,entry_id='FG:2'))
        self.assertAlmostEqual(one['capacity_basis'],.25)
        self.assertAlmostEqual(two['capacity_basis'],1)
        self.assertAlmostEqual(three['capacity_basis'],1)
        self.assertAlmostEqual(four['capacity_basis'],.25)
        self.assertEqual(s.summary()['tp_entries'],2)
        self.assertEqual(s.summary()['active_entries'],0)
        self.assertAlmostEqual(s.totals['capacity_basis'],2.5)
        self.assertAlmostEqual(sum(r['capacity_basis'] for r in s.trade_rows()),2.5)
        with self.assertRaises(m.RewardInputError): s.step(bar(4,exit_reason='TAKE_PROFIT'))

    def test_tp_one_shot_and_original_reconciliation(self):
        s=m.FGRewardScorer(m.CandidateConfig())
        s.step(bar(0,side='LONG'))
        result=s.step(bar(1,exit_reason='TAKE_PROFIT',exit_price=105))
        self.assertAlmostEqual(result['v0'],m.legacy_tp(2))
        self.assertAlmostEqual(result['v2'],m.timeboxed_tp(2))
        self.assertEqual(s.trade_rows()[0]['status'],'success')
        self.assertAlmostEqual(s.totals['v0'],result['v0'])
        for bad in ('TAKE_PROFIT','MARGIN_CALL'):
            before=s.snapshot()
            with self.assertRaises(m.RewardInputError): s.step(bar(2,exit_reason=bad))
            self.assertEqual(s.snapshot(),before)

    def test_tp_and_adverse_same_bar_is_ambiguity(self):
        s=m.FGRewardScorer(m.CandidateConfig(beta=.2))
        row=bar(0,side='LONG',high=105,low=90,price=100,exit_reason='TAKE_PROFIT')
        row['close']=105
        result=s.step(row)
        self.assertEqual(result['intrabar_ambiguous'],['FG:1'])
        self.assertAlmostEqual(result['early'],-.2)
        self.assertAlmostEqual(result['tp_timeboxed'],60/61)
        self.assertEqual(s.summary()['intrabar_ambiguous'],1)

    def test_meta_rows_do_not_increase_age_or_capacity(self):
        s=m.FGRewardScorer(m.CandidateConfig(beta=.2,lam=.01))
        s.step(bar(0,side='LONG'))
        row=bar(2,valid=False,eligible=True,high=101,low=99)
        r=s.step(row)
        self.assertEqual(r['capacity_basis'],0)
        self.assertEqual(s.active['FG:1'].bars_seen,1)

    def test_empty_wait_and_right_censoring(self):
        s=m.FGRewardScorer(m.CandidateConfig(beta=.2,lam=.1,alpha=.05))
        for i in range(10): s.step(bar(i))
        self.assertEqual(s.totals['v3'],0)
        self.assertEqual(s.trade_rows(),[])
        s.step(bar(10,side='LONG',low=90))
        self.assertEqual(s.trade_rows()[0]['status'],'right_censored')
        self.assertEqual(s.trade_rows()[0]['timeboxed_tp'],0)

    def test_potential_discounted_telescoping_chunk_restore(self):
        c=m.CandidateConfig(alpha=.05,lam=.0001)
        full=m.FGRewardScorer(c)
        one=m.FGRewardScorer(c)
        rows=[bar(0,side='LONG',price=100),bar(1,price=102,high=102,low=102,chunk=True),
              bar(2,price=98,high=98,low=98),bar(3,price=100,exit_reason='MARGIN_CALL',exit_price=97)]
        reference=[full.step(r) for r in rows]
        first=[one.step(r) for r in rows[:2]]
        restored=m.FGRewardScorer.restore(json.loads(json.dumps(one.snapshot())),c)
        other=[restored.step(r) for r in rows[2:]]
        self.assertEqual(reference,first+other)
        self.assertEqual(full.snapshot(),restored.snapshot())
        discounted_sum=sum(c.gamma**i*r['potential'] for i,r in enumerate(reference))
        self.assertAlmostEqual(discounted_sum,0,places=12)
        self.assertNotEqual(reference[1]['phi_after'],0)
        self.assertEqual(full.summary()['tp_entries'],0)
        self.assertEqual(full.trade_rows()[0]['status'],'terminal_failure')
        tampered=one.snapshot();tampered['step_number']+=1
        with self.assertRaises(m.RewardInputError):m.FGRewardScorer.restore(tampered,c)

    def test_invalid_atomic_inputs_and_capacity(self):
        s=m.FGRewardScorer(m.CandidateConfig())
        for bad in (dict(bar(0,side='LONG'),entries=[dict(bar(0,side='LONG')['entries'][0],quantity=2)]),
                    dict(bar(0,side='LONG'),high=99),
                    dict(bar(0,side='LONG'),eligible_market_minute=1)):
            with self.assertRaises(m.RewardInputError):s.step(bad)
            self.assertEqual(s.step_number,0)
        s.step(bar(0,side='LONG'))
        s.step(bar(1,side='LONG',entry_id='FG:2'))
        with self.assertRaises(m.RewardInputError):s.step(bar(2,side='LONG',entry_id='FG:3'))
        self.assertEqual(len(s.active),2)

    def test_rejects_impossible_claimed_tp_and_untrusted_clocks(self):
        s=m.FGRewardScorer(m.CandidateConfig())
        s.step(bar(0,side='LONG'))
        impossible=bar(1,exit_reason='TAKE_PROFIT')
        impossible['high']=104
        original=s.snapshot()
        with self.assertRaises(m.RewardInputError): s.step(impossible)
        self.assertEqual(s.snapshot(),original)
        invalid=bar(1,exit_reason='TAKE_PROFIT')
        invalid['exits'][0]['exit_time']=clock(4)
        with self.assertRaises(m.RewardInputError): s.step(invalid)
        self.assertEqual(s.snapshot(),original)

    def test_no_tp_reward_after_480(self):
        self.assertEqual(m.timeboxed_tp(480),0)
        self.assertEqual(m.timeboxed_tp(129583),0)
        self.assertGreater(m.legacy_tp(129583),0)

    def test_rejects_overlapping_bars_without_mutating_state(self):
        scorer=m.FGRewardScorer(m.CandidateConfig())
        scorer.step(bar(0))
        before=scorer.snapshot()
        overlapping=bar(1)
        overlapping['bar_open']='2024-04-09T09:00:30'
        overlapping['bar_end']='2024-04-09T09:01:30'
        with self.assertRaises(m.RewardInputError):
            scorer.step(overlapping)
        self.assertEqual(scorer.snapshot(),before)


if __name__=='__main__':
    unittest.main()

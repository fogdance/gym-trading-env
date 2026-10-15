"""TP5-only, versioned quality reward: offline no-network regression tests."""
from datetime import datetime, timedelta
from pathlib import Path
import copy
import importlib.util
import sys
import unittest
import random

SOURCE = Path(__file__).resolve().parents[2] / 'src/gym_trading_env/research/fg_reward_candidates.py'
SPEC = importlib.util.spec_from_file_location('fg_quality_tp_test', SOURCE)
m = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = m
SPEC.loader.exec_module(m)


class QualityTPRewardTests(unittest.TestCase):
    def test_score_and_reward_share_the_frozen_curve(self):
        from gym_trading_env.research.fg_s3_quality_score import tp_base_points
        rng=random.Random(202)
        values=[(t,d) for t in (1,10,11,74,75,120,121,240,241,10000) for d in (0,1,8)]
        values.extend((rng.randint(1,10000),rng.randint(0,10)) for _ in range(64))
        for minutes,days in values:
            self.assertEqual(m.quality_tp_reward(minutes,days),tp_base_points(minutes,days)/100)
        with self.assertRaises(m.RewardInputError):m.quality_tp_reward(1.5,0)

    def test_anchors_and_legacy_unchanged(self):
        for t, expected in ((4, 1), (10, 1), (30, .96875), (60, .921875),
                            (74, .9), (120, .85), (240, .75),
                            (480, .6203002924854919)):
            self.assertAlmostEqual(m.quality_tp_reward(t, 0), expected)
        self.assertAlmostEqual(m.quality_tp_reward(74, 1), .45)
        self.assertAlmostEqual(m.quality_tp_reward(74, 2), .40)
        self.assertAlmostEqual(m.quality_tp_reward(74, 8), .20)
        self.assertAlmostEqual(m.timeboxed_tp(74), (60 / 134) * (406 / 420))
        for value in (-1, 0, float('nan')):
            with self.assertRaises(m.RewardInputError):
                m.quality_tp_reward(value, 0)

    def test_native_bar_scorer_with_exact_restore(self):
        config = m.CandidateConfig(beta=.5, rho=.4, clock_protocol='S3_MARKET_MINUTE')
        for held, crossed in ((4, False), (74, False), (120, False),
                              (240, False), (74, True)):
            scorer = m.FGRewardScorer(config, tp_profile='QUALITY_TP_V1')
            begin = datetime(2026, 10, 15, 9, 0)
            for index in range(held):
                t0 = begin + timedelta(minutes=index)
                t1 = t0 + timedelta(minutes=1)
                last = index == held - 1
                day = '20261016' if crossed and index >= 20 else '20261015'
                row = dict(bar_open=str(t0), bar_end=str(t1), open=100, high=100,
                           low=95 if last else 100, close=95 if last else 100,
                           valid_bar=True, eligible_market_minute=False,
                           trading_day=day, true_terminal=last,
                           entries=[dict(entry_id='X', side='SHORT', time=str(t0),
                                         reason='ENTRY', price=100, quantity=3,
                                         committed=True)] if index == 0 else [],
                           exits=[dict(entry_id='X', side='SHORT', entry_time=str(begin),
                                       exit_time=str(t1), entry_price=100, exit_price=95,
                                       closed_quantity=3, reason='TAKE_PROFIT',
                                       committed=True)] if last else [])
                result = scorer.step(row)
                if index == 25 and held > 26:
                    scorer = m.FGRewardScorer.restore(
                        scorer.snapshot(), config, tp_profile='QUALITY_TP_V1')
            self.assertAlmostEqual(result['tp_timeboxed'],
                                   m.quality_tp_reward(held, int(crossed)))
            self.assertEqual(len(scorer.active), 0)

    def test_day_order_atomicity_and_snapshot_profile_isolation(self):
        config = m.CandidateConfig(clock_protocol='S3_MARKET_MINUTE')
        new = m.FGRewardScorer(config, tp_profile='QUALITY_TP_V1')
        t0 = datetime(2026, 10, 15, 9, 0)
        first = dict(bar_open=str(t0), bar_end=str(t0 + timedelta(minutes=1)),
                     open=100, high=100, low=100, close=100, valid_bar=True,
                     eligible_market_minute=False, trading_day='20261015',
                     entries=[dict(entry_id='X', side='SHORT', time=str(t0),
                                   price=100, quantity=3, committed=True, reason='ENTRY')], exits=[])
        new.step(first)
        before = copy.deepcopy(new.snapshot())
        backwards = dict(first, bar_open=str(t0 + timedelta(minutes=1)),
                         bar_end=str(t0 + timedelta(minutes=2)),
                         entries=[], trading_day='20261014')
        with self.assertRaises(m.RewardInputError):
            new.step(backwards)
        self.assertEqual(new.snapshot(), before)
        old = m.FGRewardScorer(config)
        with self.assertRaises(m.RewardInputError):
            m.FGRewardScorer.restore(new.snapshot(), config)
        self.assertNotEqual(old.SCHEMA, new.SCHEMA)


if __name__ == '__main__':
    unittest.main()

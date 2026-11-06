import copy
import json
from pathlib import Path
import subprocess
import sys
import unittest

# Copied from ~/Documents/work/review_future/python/tests (2026-11-06); only the package path
# (gym_trading_env.market_structure) and the fixture path differ.
from gym_trading_env.market_structure import analyze_market_structure

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1] / "src"
REFERENCE = json.loads((HERE / "fixtures/typescript-reference.json").read_text())


class MarketStructureTests(unittest.TestCase):
    def test_all_fields_match_current_typescript(self):
        for case in REFERENCE["cases"]:
            with self.subTest(case=case["name"]):
                before = copy.deepcopy(case["bars"])
                result = analyze_market_structure(case["bars"], **case["config"])
                self.assertEqual(result, case["expected"])
                self.assertEqual(case["bars"], before)
                if "strict_expected" in case:
                    self.assertEqual(analyze_market_structure(case["bars"], plateau_policy="strict"), case["strict_expected"])

    def test_real_plateaus_wait_for_confirmation(self):
        case = next(case for case in REFERENCE["cases"] if case["name"] == "fg501_equal_extrema")
        bars = case["bars"]
        full = analyze_market_structure(bars)
        for expected in case["required_pivots"]:
            with self.subTest(price=expected["price"]):
                pivot = next(i for i, bar in enumerate(bars) if bar["eob"] == expected["eob"])
                confirmation = next(i for i, bar in enumerate(bars) if bar["eob"] == expected["confirm_eob"])
                before = analyze_market_structure(bars[:confirmation])
                self.assertFalse(any(point["pivot_index"] == pivot for point in before["swings"]))
                after = analyze_market_structure(bars[:confirmation + 1])
                self.assertIn({"kind": expected["kind"], "price": expected["price"],
                               "pivot_index": pivot, "confirm_index": confirmation}, after["swings"])
                self.assertEqual(after["history"], full["history"][:confirmation + 1])
                self.assertEqual(after["events"], [event for event in full["events"] if event["index"] <= confirmation])

    def test_empty_short_and_flat_inputs(self):
        flat = {"open": 10, "high": 11, "low": 9, "close": 10}
        empty = analyze_market_structure([])
        self.assertIsNone(empty["current"])
        self.assertEqual(empty["swings"], [])
        self.assertEqual(empty["history"], [])
        for count in (1, 7):
            result = analyze_market_structure([flat] * count)
            self.assertEqual(result["swings"], [])
            self.assertEqual(result["current"]["trend"], "neutral")

    def test_invalid_inputs_and_ambiguous_raise(self):
        flat = {"open": 10, "high": 11, "low": 9, "close": 10}
        for bars, config in [
            ([{"open": 10}], {}), ([{**flat, "close": 12}], {}),
            ([{**flat, "high": float("nan")}], {}), ([{**flat, "close": "10"}], {}),
            ([], {"right": 0}), ([], {"left": True}), ([], {"plateau_policy": "unknown"}),
            ([], {"structure_tolerance_pct": -0.01}), ([], {"break_buffer_pct": float("inf")}),
            ([flat, {**flat, "high": 12, "low": 8}, flat], {"left": 1, "right": 1, "ambiguous_policy": "raise"}),
        ]:
            with self.subTest(bars=bars, config=config), self.assertRaises((TypeError, ValueError)):
                analyze_market_structure(bars, **config)

    def test_json_cli(self):
        case = REFERENCE["cases"][0]
        completed = subprocess.run(
            [sys.executable, "-m", "gym_trading_env.market_structure", "-", "--left", "1", "--right", "1"],
            input=json.dumps({"bars": case["bars"]}), text=True, capture_output=True, cwd=ROOT,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(json.loads(completed.stdout), case["expected"])
        invalid = subprocess.run([sys.executable, "-m", "gym_trading_env.market_structure", "-"],
                                 input='{"bars": [{"open": 10}]}', text=True, capture_output=True, cwd=ROOT)
        self.assertEqual(invalid.returncode, 2)
        self.assertIn("missing high", invalid.stderr)
        self.assertNotIn("Traceback", invalid.stderr)


if __name__ == "__main__":
    unittest.main()

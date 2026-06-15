import json

from tools import report_dreamer_multiseed_repeatability as report


def _summary(path, *, seed, pnl, fixed_exec, random_mean, split_hash):
    data = {
        "checkpoint": f"/tmp/ckpt/{seed}",
        "limitations": {"checkpoint_step": 900000},
        "seed_protocol": {
            "experiment_seed": seed,
            "dreamer_seed": seed,
            "train_env_seed": seed * 1000 + 101,
            "replay_seed": seed * 1000 + 202,
            "eval_env_seed": 0,
            "matched_random_seed": 20260615,
        },
        "entry_eval": {
            "version": "entry_eval_test",
            "execution_timing": "signal_on_close_plus_spread",
            "split_manifest_hash": split_hash,
        },
        "actual_dreamer": {
            "metrics": {"net_pnl": pnl, "profit_factor": 1.2},
        },
        "dreamer_entry_fixed_exit_candidate_join": {
            "metrics": {"net_pnl": fixed_exec - 10},
            "actual_matched": {"net_pnl": pnl},
        },
        "dreamer_entry_fixed_exit_executable_nonoverlap": {
            "metrics": {"net_pnl": fixed_exec},
        },
        "matched_random_fixed_exit_executable_nonoverlap": {
            "net_pnl_mean": random_mean,
        },
    }
    path.write_text(json.dumps(data))


def test_multiseed_report_records_seed_protocol_and_split_hash(tmp_path):
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    _summary(a, seed=101, pnl=100.0, fixed_exec=-20.0, random_mean=-50.0, split_hash="hash-a")
    _summary(b, seed=202, pnl=-10.0, fixed_exec=-30.0, random_mean=-40.0, split_hash="hash-b")

    summary = report.aggregate({"seed_a": a, "seed_b": b}, "exp")

    assert summary["aggregate"]["positive_test_seeds"] == 1
    assert summary["aggregate"]["fixed_exit_entry_negative_all_seeds"] is True
    assert summary["seeds"][0]["seed_protocol"]["dreamer_seed"] == 101
    assert summary["seeds"][1]["seed_protocol"]["replay_seed"] == 202202
    assert summary["seeds"][0]["entry_eval"]["split_manifest_hash"] == "hash-a"
    assert "matched_random_seed" in summary["manifest_fields_recorded"]


def test_multiseed_report_writes_markdown(tmp_path):
    a = tmp_path / "a.json"
    _summary(a, seed=101, pnl=100.0, fixed_exec=-20.0, random_mean=-50.0, split_hash="hash-a")
    summary = report.aggregate({"seed_a": a}, "exp")
    out = tmp_path / "report.md"

    report.write_report(out, summary)

    text = out.read_text()
    assert "Dreamer Walk-Forward Multi-Seed Repeatability" in text
    assert "101101" in text
    assert "hash-a" in text

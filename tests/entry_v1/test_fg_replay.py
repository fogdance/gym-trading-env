"""Replay public contract: original Env/Driver, causal policies, frozen data."""
import copy
import gzip
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from tests.entry_v1.test_entry_contract import make_env
from gym_trading_env.research.fg_replay import (
    BehaviorPolicy, SCHEMA, collect_trace, cpu_runtime, digest, iter_replay_rows,
    iter_sidecar, reconcile_trace, signed, trace_manifest, validate_training, write_json)
from gym_trading_env.research.fg_replay_coverage import (
    COLUMNS, INDEX, corpus_statistics, projection, row_labels, sample_statistics)


DREAMER = Path(os.environ.get("FG_REPLAY_DREAMER_ROOT", "/home/v/Documents/work/dreamerv3"))


@pytest.fixture
def original_driver():
    if not (DREAMER / "embodied/core/driver.py").is_file():
        pytest.skip("Original Dreamer checkout is required for the Replay contract")
    return cpu_runtime(DREAMER)


@pytest.fixture
def full_trace(tmp_path, original_driver):
    market = tmp_path / "market"; market.mkdir()
    calendar = [dict(start=f"2026-01-{day:02} {start}", end=f"2026-01-{day:02} {end}")
        for day in range(5, 9) for start, end in
        (("09:00", "10:15"), ("10:30", "11:30"), ("13:30", "15:00"), ("21:00", "23:00"))]
    times = pd.DatetimeIndex([minute for session in calendar for minute in
        pd.date_range(session["start"], session["end"], freq="min", inclusive="right")])[:1100]
    prices = [(1000, 1000, 1000, 1000)] * len(times)
    prices[2] = (1000, 1000, 990, 990)
    prices[1030] = (1000, 1006, 1000, 1006)
    def factory():
        return make_env(market, prices=prices, times=times, chunk=1024,
            calendar=calendar, liquidation="2026-01-09 15:00:00",
            intervals=[dict(contract="CZCE.FG2701", start="2026-01-05", end="2026-01-10", direction="LONG")],
            main_exit=str(times[-1]))
    spec = dict(id="P1", kind="always_add", seed_components=[0, 0, 0])
    directory = tmp_path / "trace"
    trace = collect_trace(market / "env.yaml", spec, directory, {"synthetic_fixture": True},
                          DREAMER, env_factory=factory)
    return directory, trace, factory


def test_original_driver_cross_1024_restore_tp_save_load_and_stream_isolation(full_trace, tmp_path, original_driver):
    directory, trace, factory = full_trace
    result = reconcile_trace(directory, DREAMER, env_factory=factory)
    assert result["status"] == "PASS" and result["account_restores"] == 1
    rows = list(iter_sidecar(directory))
    assert rows[1024]["status"] == "CHUNK_END" and rows[1024]["lots_after"] == 6
    assert rows[1025]["reset"] and rows[1025]["cursor"] == rows[1024]["cursor"]
    assert rows[1025]["lots_after"] == 6 and not rows[1024]["decision_executed"]
    rewarded = [r for r in rows if r["reward_events"]]
    assert len(rewarded) == 1 and len(rewarded[0]["reward_events"]) == 2
    assert rewarded[0]["cursor"] == 1030 and rewarded[0]["ordinal"] == 1031
    assert rewarded[0]["official_reward"] == pytest.approx(sum(
        60 / (60 + e["elapsed_minutes"]) for e in rewarded[0]["reward_events"]))
    assert sum(r["decision_executed"] for r in rows) == trace["advanced"] == 1099
    assert sum(r["diagnostics"]["early_first5"] for r in rows) == 2
    second = tmp_path / "second"
    rebuilt = collect_trace(directory / "env.yaml", trace["policy"], second, trace["identity"], DREAMER, env_factory=factory)
    assert rebuilt["replay_trace_hash"] == trace["replay_trace_hash"]
    assert rebuilt["array_hash"] == trace["array_hash"] and rebuilt["chunks"][0]["uuid"] != trace["chunks"][0]["uuid"]
    replay = original_driver.Replay(length=65, directory=str(tmp_path / "reader"), seed=202)
    identical = original_driver.Replay(length=65, directory=str(tmp_path / "reader2"), seed=202)
    try:
        for r in (replay, identical):
            r.load(directory=str(directory / "replay")); r.load(directory=str(second / "replay"))
            assert len(r) == 2 * trace["sequence_starts"]
        membership = {c["uuid"]: i for i, t in enumerate((trace, rebuilt)) for c in t["chunks"]}
        import elements
        crossed = 0
        for _ in range(30):
            batch = replay.sample(16); same = identical.sample(16)
            assert all(np.array_equal(batch[k], same[k]) for k in batch)
            assert batch["market_seq"].shape == (16, 65, 60, 18)
            for ids in batch["stepid"]:
                owners = {membership[str(elements.UUID(s[:16].tobytes()))] for s in ids}
                assert len(owners) == 1
                crossed += len({s[:16].tobytes() for s in ids}) > 1
        assert crossed > 0
    finally:
        replay.workers.shutdown(wait=True); identical.workers.shutdown(wait=True)


def test_shifted_feedback_is_rejected_even_with_updated_file_hash(full_trace):
    directory, trace, factory = full_trace
    chunk = trace["chunks"][1]
    path = directory / chunk["path"]
    with np.load(path) as data:
        arrays = {k: data[k] for k in data.files}
    arrays["reward"][7] += 1
    np.savez_compressed(path, **arrays)
    trace["files"][chunk["path"]] = digest(path)
    write_json(directory / "manifest.json", signed(trace))
    with pytest.raises(ValueError, match="feedback reward"):
        reconcile_trace(directory, DREAMER, env_factory=factory)


def obs(legal=True, last=False):
    return dict(action_mask=np.array([[1, legal]], np.float32), is_last=np.array([last]))


def test_p3_waits_after_capacity_release_skips_resets_and_does_not_wait_after_first_fill():
    policy = BehaviorPolicy(dict(kind="capacity_delay", wait_minutes=15, seed_components=[0, 0, 3]))
    for _ in range(14):
        policy.observe(advanced=True, valid_bar=True, interval_slots=0, lots_after=0)
    assert policy(None, obs())[1]["action"].item() == 0
    for advanced, valid in ((False, False), (True, False)):
        policy.observe(advanced=advanced, valid_bar=valid, interval_slots=0, lots_after=0)
    assert policy.waited == 14
    policy.observe(advanced=True, valid_bar=True, interval_slots=0, lots_after=0)
    assert policy(None, obs())[1]["action"].item() == 1
    policy.observe(advanced=True, valid_bar=True, interval_slots=1, lots_after=3)
    assert policy(None, obs())[1]["action"].item() == 1  # Continuous legal ADD until full.
    policy.observe(advanced=True, valid_bar=True, interval_slots=2, lots_after=3)
    assert policy.waited == 0 and policy.waiting  # Same-bar full->free release also waits.
    assert policy(None, obs())[1]["action"].item() == 0
    assert policy(None, obs(legal=False))[1]["action"].item() == 0


def test_p2_uses_only_legal_current_obs_and_preserves_rng_at_wrapper_boundaries():
    spec = dict(kind="probabilistic", probability=.5, seed_components=[0, 3, 2])
    a, b = BehaviorPolicy(spec), BehaviorPolicy(spec)
    a(None, obs(False)); a(None, obs(True, last=True))
    assert a.rng.bit_generator.state == b.rng.bit_generator.state
    assert [a(None, obs())[1]["action"].item() for _ in range(30)] == [
        b(None, obs())[1]["action"].item() for _ in range(30)]


def training_fixture(tmp_path):
    csv = tmp_path / "train.csv"
    frame = pd.DataFrame(dict(Date=["2024-04-08 09:01:00+08:00", "2024-04-08 09:02:00+08:00"],
        bob=["2024-04-08 09:00:00+08:00", "2024-04-08 09:01:00+08:00"],
        trading_date=["2024-04-08"] * 2, symbol=["CZCE.FG405"] * 2))
    frame.to_csv(csv, index=False)
    direction = tmp_path / "direction.yaml"; direction.write_text("product: FG\n")
    contract = tmp_path / "contract.yaml"; contract.write_text("contract: CZCE.FG405\n")
    env = tmp_path / "env.yaml"
    env.write_text(yaml.safe_dump(dict(strategy_mode="entry_only", entry=dict(contract="CZCE.FG405",
        direction_config=str(direction), contract_config=str(contract)), trading=dict(data_path=str(csv)))))
    row = dict(contract="CZCE.FG405", role="train", eligible_for_training=True,
        complete_lifecycle=True, calendar_complete=True, path=str(csv), sha256=digest(csv),
        env_config=str(env), env_sha256=digest(env), contract_config=str(contract), contract_sha256=digest(contract),
        rows=2, first_timestamp=frame.Date.iloc[0], last_timestamp=frame.Date.iloc[-1])
    body = dict(product="FG", roles=dict(train=[20240408], validation=[20240409], test=[20240820]),
        files=[row, dict(contract="CZCE.FG409", role="validation", eligible_for_training=False,
                         path=str(tmp_path / "sealed_prices.csv"))],
        direction_path=str(direction), direction_sha256=digest(direction))
    manifest = tmp_path / "manifest.json"; write_json(manifest, signed(body))
    return manifest, body, frame, csv


def test_train_hash_and_actual_days_checked_before_generation_without_reading_heldout(tmp_path, monkeypatch):
    manifest, body, frame, csv = training_fixture(tmp_path)
    original = pd.read_csv
    accessed = []
    def tracked(path, *args, **kwargs):
        accessed.append(Path(path))
        assert "sealed" not in str(path)
        return original(path, *args, **kwargs)
    monkeypatch.setattr(pd, "read_csv", tracked)
    _, rows, _, _ = validate_training(manifest)
    assert len(rows) == 1 and accessed == [csv]
    frame.loc[1, "trading_date"] = "2024-04-09"
    frame.to_csv(csv, index=False)
    with pytest.raises(ValueError, match="Frozen input/output changed"):
        validate_training(manifest)
    body["files"][0]["sha256"] = digest(csv); write_json(manifest, signed(body))
    with pytest.raises(ValueError, match="training time/row boundaries"):
        validate_training(manifest)


def test_naive_training_timestamps_use_the_frozen_environment_timezone(tmp_path):
    manifest, body, frame, csv = training_fixture(tmp_path)
    for column in ("Date", "bob"):
        frame[column] = frame[column].str.replace("+08:00", "", regex=False)
    frame.to_csv(csv, index=False)
    body["files"][0]["sha256"] = digest(csv)
    write_json(manifest, signed(body))
    _, rows, _, source = validate_training(manifest)
    assert len(rows) == 1 and source[0]["valid_minutes"] == 2


def test_native_sampling_counts_use_stepids_and_separate_context(full_trace, tmp_path, original_driver):
    directory, trace, _ = full_trace
    write_json(directory / "reconciliation.json", signed(dict(status="PASS", account_restores=1)))
    dataset = dict(traces=[dict(directory="trace", policy_id="P1", contract=trace["contract"],
        records=trace["records"], sequence_starts=trace["sequence_starts"])], views=dict(P1="trace/replay"))
    corpus = corpus_statistics(tmp_path, dataset)
    stats = sample_statistics(tmp_path, dataset, corpus, "P1", batches=50, batch_size=16, seed=202)
    assert stats["first_batch_reproduced"] and stats["selector"] == "Uniform"
    assert stats["cross_chunk_sequences"] > 0 and stats["discontinuous_sequences"] == 0
    assert stats["counts"].shape == (50, 2, len(COLUMNS))
    assert stats["counts"][:, 1, INDEX["tp"]].sum() > 0
    assert stats["counts"][:, 1, INDEX["legal_add_decision"]].sum() > 0
    reset = next(r for r in iter_sidecar(directory) if r["status"] == "CHUNK_END")
    values = row_labels(reset)
    assert values[INDEX["decision"]] == 0
    estimated = projection(trace["sequence_starts"], trace["records"], trace["advanced"], 100000)
    assert estimated["projected_gradient_updates"] == 6250
    assert estimated["sampled_rows_with_context"] == 6500000


def test_unfilled_main_roll_at_data_end_is_failure_not_a_free_reset_or_censor(tmp_path, original_driver):
    market = tmp_path / "market"; market.mkdir()
    def factory():
        env = make_env(market, main_exit="2026-01-05 09:38:00")
        env._raw.loc[env._raw.index[-1], "Volume"] = 0
        return env
    spec = dict(id="P1", kind="always_add", seed_components=[0, 0, 0])
    directory = tmp_path / "trace"
    trace = collect_trace(market / "env.yaml", spec, directory, {"synthetic_fixture": True}, DREAMER, env_factory=factory)
    receipt = reconcile_trace(directory, DREAMER, env_factory=factory)
    write_json(directory / "reconciliation.json", signed(receipt))
    assert trace["terminal_status"] == "DATA_END" and trace["pending_reason"] == "MAIN_ROLL"
    assert trace["open_lots"] == 6 and not trace["terminal"]
    dataset = dict(traces=[dict(directory="trace", policy_id="P1", contract=trace["contract"])])
    corpus = corpus_statistics(tmp_path, dataset)
    assert {t["status"] for t in corpus["trades"]} == {"terminal_failure_unfilled"}


def test_legal_add_rejection_is_counted_at_actual_feedback(tmp_path, original_driver):
    market = tmp_path / "market"; market.mkdir()
    def factory():
        env = make_env(market)
        env._raw.loc[env._raw.index[1], "Volume"] = 0
        return env
    spec = dict(id="P1", kind="always_add", seed_components=[0, 0, 0])
    directory = tmp_path / "trace"
    trace = collect_trace(market / "env.yaml", spec, directory, {"synthetic_fixture": True}, DREAMER, env_factory=factory)
    receipt = reconcile_trace(directory, DREAMER, env_factory=factory)
    write_json(directory / "reconciliation.json", signed(receipt))
    rows = list(iter_sidecar(directory))
    assert rows[0]["legal_add"] and rows[0]["action"] == 1
    assert row_labels(rows[0])[INDEX["add_rejected"]] == 0
    assert rows[1]["reject_reason"] == "INSUFFICIENT_BAR_VOLUME"
    assert row_labels(rows[1])[INDEX["add_rejected"]] == 1
    corpus = corpus_statistics(tmp_path, dict(traces=[dict(directory="trace", policy_id="P1", contract=trace["contract"])]))
    assert corpus["rejection_reasons"] == {("P1", "INSUFFICIENT_BAR_VOLUME"): 1}

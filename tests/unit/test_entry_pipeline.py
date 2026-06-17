from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from gym_trading_env.research import entry_pipeline


class FakeConfig(SimpleNamespace):
    def to_dict(self) -> dict:
        def convert(value):
            if isinstance(value, SimpleNamespace):
                return {k: convert(v) for k, v in value.__dict__.items()}
            return value

        return convert(self)


def test_dataset_only_writes_dataset_ready_manifest_and_status(tmp_path, monkeypatch):
    config_path = tmp_path / "entry_eval.yaml"
    data_path = tmp_path / "market.csv"
    output = tmp_path / "out"
    config_path.write_text("version: fake\n")
    data_path.write_text("timestamp,close\n")

    config = FakeConfig(
        version="fake_entry_eval_v1",
        data=SimpleNamespace(path=str(data_path), window_size=60),
        product=SimpleNamespace(max_entries_per_day=3, initial_balance=100000.0),
    )
    days = [20250101 + i for i in range(7)]
    market = pd.DataFrame({"trading_day": days})
    candidates = pd.DataFrame({
        "candidate_id": list(range(7)),
        "trading_day": days,
        "decision_row": [10 + i for i in range(7)],
    })
    outcomes = pd.DataFrame([
        {"candidate_id": candidate_id, "direction": direction}
        for candidate_id in range(7)
        for direction in ("LONG", "SHORT")
    ])
    features = np.arange(28, dtype=np.float32).reshape(7, 4)
    flat_features = np.arange(14, dtype=np.float32).reshape(7, 2)

    monkeypatch.setattr(entry_pipeline, "load_entry_eval_config", lambda path: config)
    monkeypatch.setattr(entry_pipeline, "load_market_frames", lambda cfg: (None, market))
    monkeypatch.setattr(
        entry_pipeline,
        "build_entry_dataset",
        lambda market_arg, cfg: (candidates.copy(), features.copy(), ["a", "b", "c", "d"]),
    )
    monkeypatch.setattr(entry_pipeline, "add_diagnostic_context", lambda c, x, n: c)
    monkeypatch.setattr(
        entry_pipeline,
        "build_flattened_entry_features",
        lambda market_arg, c, cfg: (flat_features.copy(), ["f0", "f1"]),
    )
    monkeypatch.setattr(entry_pipeline, "outcomes_from_candidates", lambda c: outcomes.copy())
    monkeypatch.setattr(
        entry_pipeline,
        "write_parquet",
        lambda df, path: df.to_csv(Path(path).with_suffix(".csv"), index=False),
    )

    result = entry_pipeline.run_entry_capability_pipeline(
        config_path=config_path,
        output=output,
        dataset_only=True,
        emit=lambda message: None,
    )

    manifest = json.loads((output / "manifest.json").read_text())
    status = json.loads((output / "pipeline_status.json").read_text())

    assert result["manifest_stage"] == "dataset_ready"
    assert manifest["manifest_stage"] == "dataset_ready"
    assert manifest["candidates"] == 7
    assert manifest["outcomes"] == 14
    assert manifest["feature_count"] == 4
    assert manifest["flat_feature_count"] == 2
    assert status["last_completed_stage"] == "dataset_ready"
    assert (output / "config.yaml").exists()
    assert (output / "candidates.csv").exists()
    assert (output / "features.npz").exists()
    assert (output / "flat_features.npz").exists()
    assert (output / "split_manifest.json").exists()


def test_resume_reuses_dataset_artifacts_when_hashes_match(tmp_path, monkeypatch):
    config_path = tmp_path / "entry_eval.yaml"
    data_path = tmp_path / "market.csv"
    output = tmp_path / "out"
    config_path.write_text("version: fake\n")
    data_path.write_text("timestamp,close\n")

    config = FakeConfig(
        version="fake_entry_eval_v1",
        data=SimpleNamespace(path=str(data_path), window_size=60),
        product=SimpleNamespace(max_entries_per_day=3, initial_balance=100000.0),
    )
    days = [20250101 + i for i in range(7)]
    market = pd.DataFrame({"trading_day": days})
    candidates = pd.DataFrame({
        "candidate_id": list(range(7)),
        "trading_day": days,
        "decision_row": [10 + i for i in range(7)],
    })
    outcomes = pd.DataFrame([
        {"candidate_id": candidate_id, "direction": direction}
        for candidate_id in range(7)
        for direction in ("LONG", "SHORT")
    ])
    features = np.arange(28, dtype=np.float32).reshape(7, 4)
    flat_features = np.arange(14, dtype=np.float32).reshape(7, 2)

    monkeypatch.setattr(entry_pipeline, "load_entry_eval_config", lambda path: config)
    monkeypatch.setattr(entry_pipeline, "load_market_frames", lambda cfg: (None, market))
    monkeypatch.setattr(
        entry_pipeline,
        "build_entry_dataset",
        lambda market_arg, cfg: (candidates.copy(), features.copy(), ["a", "b", "c", "d"]),
    )
    monkeypatch.setattr(entry_pipeline, "add_diagnostic_context", lambda c, x, n: c)
    monkeypatch.setattr(
        entry_pipeline,
        "build_flattened_entry_features",
        lambda market_arg, c, cfg: (flat_features.copy(), ["f0", "f1"]),
    )
    monkeypatch.setattr(entry_pipeline, "outcomes_from_candidates", lambda c: outcomes.copy())
    monkeypatch.setattr(
        entry_pipeline,
        "write_parquet",
        lambda df, path: df.to_csv(Path(path).with_suffix(".csv"), index=False),
    )

    entry_pipeline.run_entry_capability_pipeline(
        config_path=config_path,
        output=output,
        dataset_only=True,
        emit=lambda message: None,
    )

    def fail_build(*args, **kwargs):
        raise AssertionError("dataset should have been loaded, not rebuilt")

    monkeypatch.setattr(entry_pipeline, "build_entry_dataset", fail_build)
    result = entry_pipeline.run_entry_capability_pipeline(
        config_path=config_path,
        output=output,
        dataset_only=True,
        resume=True,
        emit=lambda message: None,
    )

    status = json.loads((output / "pipeline_status.json").read_text())
    assert result["manifest_stage"] == "dataset_ready"
    assert result["candidates"] == 7
    assert status["stage"] == "load_dataset_artifacts"
    assert status["state"] == "completed"


def test_resume_fails_fast_on_config_hash_mismatch(tmp_path, monkeypatch):
    config_path = tmp_path / "entry_eval.yaml"
    data_path = tmp_path / "market.csv"
    output = tmp_path / "out"
    config_path.write_text("version: fake\n")
    data_path.write_text("timestamp,close\n")

    config = FakeConfig(
        version="fake_entry_eval_v1",
        data=SimpleNamespace(path=str(data_path), window_size=60),
        product=SimpleNamespace(max_entries_per_day=3, initial_balance=100000.0),
    )
    days = [20250101 + i for i in range(7)]
    candidates = pd.DataFrame({
        "candidate_id": list(range(7)),
        "trading_day": days,
        "decision_row": [10 + i for i in range(7)],
    })
    outcomes = pd.DataFrame([
        {"candidate_id": candidate_id, "direction": direction}
        for candidate_id in range(7)
        for direction in ("LONG", "SHORT")
    ])
    features = np.arange(28, dtype=np.float32).reshape(7, 4)
    flat_features = np.arange(14, dtype=np.float32).reshape(7, 2)

    monkeypatch.setattr(entry_pipeline, "load_entry_eval_config", lambda path: config)
    monkeypatch.setattr(entry_pipeline, "load_market_frames", lambda cfg: (None, pd.DataFrame({"trading_day": days})))
    monkeypatch.setattr(
        entry_pipeline,
        "build_entry_dataset",
        lambda market_arg, cfg: (candidates.copy(), features.copy(), ["a", "b", "c", "d"]),
    )
    monkeypatch.setattr(entry_pipeline, "add_diagnostic_context", lambda c, x, n: c)
    monkeypatch.setattr(
        entry_pipeline,
        "build_flattened_entry_features",
        lambda market_arg, c, cfg: (flat_features.copy(), ["f0", "f1"]),
    )
    monkeypatch.setattr(entry_pipeline, "outcomes_from_candidates", lambda c: outcomes.copy())
    monkeypatch.setattr(
        entry_pipeline,
        "write_parquet",
        lambda df, path: df.to_csv(Path(path).with_suffix(".csv"), index=False),
    )

    entry_pipeline.run_entry_capability_pipeline(
        config_path=config_path,
        output=output,
        dataset_only=True,
        emit=lambda message: None,
    )

    config_path.write_text("version: changed\n")

    with pytest.raises(entry_pipeline.PipelineArtifactError, match="config hash mismatch"):
        entry_pipeline.run_entry_capability_pipeline(
            config_path=config_path,
            output=output,
            dataset_only=True,
            resume=True,
            emit=lambda message: None,
        )

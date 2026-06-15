from pathlib import Path

import pytest

from tools import dreamer_checkpoint_audit as audit


def _paths(tmp_path: Path) -> audit.AuditPaths:
    return audit.AuditPaths(
        dreamer_root=tmp_path / "dreamerv3",
        run_logdir=tmp_path / "run",
        checkpoint=tmp_path / "ckpt",
        entry_eval_dir=tmp_path / "entry_eval",
        output_dir=tmp_path / "out",
        env_config_path=None,
    )


def test_seed_metadata_reads_run_seed_protocol(tmp_path):
    paths = _paths(tmp_path)
    paths.run_logdir.mkdir(parents=True)
    (paths.run_logdir / "config.yaml").write_text(
        """
seed_protocol:
  experiment_seed: 101
  dreamer_seed: 101
  train_env_seed: 101101
  replay_seed: 101202
  eval_env_seed: 0
  matched_random_seed: 20260615
"""
    )

    metadata = audit._seed_metadata(paths, matched_random_seed=20260615)

    assert metadata["experiment_seed"] == 101
    assert metadata["dreamer_seed"] == 101
    assert metadata["train_env_seed"] == 101101
    assert metadata["replay_seed"] == 101202
    assert metadata["eval_env_seed"] == 0
    assert metadata["matched_random_seed"] == 20260615


def test_seed_metadata_rejects_missing_seed_protocol(tmp_path):
    paths = _paths(tmp_path)
    paths.run_logdir.mkdir(parents=True)
    (paths.run_logdir / "config.yaml").write_text("seed: 101\n")

    with pytest.raises(ValueError, match="missing seed_protocol"):
        audit._seed_metadata(paths, matched_random_seed=20260615)


def test_seed_metadata_can_read_retention_manifest(tmp_path):
    paths = _paths(tmp_path)
    manifest_dir = paths.run_logdir / "ckpt_retained"
    manifest_dir.mkdir(parents=True)
    (manifest_dir / "retention_manifest.json").write_text(
        json_text({
            "metadata": {
                "experiment_seed": 101,
                "dreamer_seed": 101,
                "train_env_seed": 101101,
                "replay_seed": 101202,
                "eval_env_seed": 0,
                "matched_random_seed": 20260615,
            }
        })
    )

    metadata = audit._seed_metadata(paths, matched_random_seed=20260615)

    assert metadata["dreamer_seed"] == 101
    assert metadata["source"]["retention_manifest"].endswith("retention_manifest.json")


def test_seed_metadata_rejects_config_retention_mismatch(tmp_path):
    paths = _paths(tmp_path)
    paths.run_logdir.mkdir(parents=True)
    (paths.run_logdir / "config.yaml").write_text(
        """
seed_protocol:
  experiment_seed: 101
  dreamer_seed: 101
  train_env_seed: 101101
  replay_seed: 101202
  eval_env_seed: 0
  matched_random_seed: 20260615
"""
    )
    manifest_dir = paths.run_logdir / "ckpt_retained"
    manifest_dir.mkdir()
    (manifest_dir / "retention_manifest.json").write_text(
        json_text({"metadata": {"dreamer_seed": 202}})
    )

    with pytest.raises(ValueError, match="Seed protocol mismatch"):
        audit._seed_metadata(paths, matched_random_seed=20260615)


def test_seed_metadata_rejects_matched_random_seed_mismatch(tmp_path):
    paths = _paths(tmp_path)
    paths.run_logdir.mkdir(parents=True)
    (paths.run_logdir / "config.yaml").write_text(
        """
seed_protocol:
  experiment_seed: 101
  dreamer_seed: 101
  train_env_seed: 101101
  replay_seed: 101202
  eval_env_seed: 0
  matched_random_seed: 20260615
"""
    )

    with pytest.raises(ValueError, match="matched_random_seed"):
        audit._seed_metadata(paths, matched_random_seed=1)


def json_text(value):
    import json

    return json.dumps(value)

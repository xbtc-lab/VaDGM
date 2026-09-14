from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

from vadgm.cli import main
from vadgm.data import DataConfig, build_zinc_dataset
from vadgm.workflow import (
    artifact_paths,
    cache_config_hash,
    legacy_config_hashes,
    resume_config_hash,
    training_is_ready,
)


def _base_config(epochs: int) -> dict[str, object]:
    return {
        "seed": 17,
        "model": {
            "hidden_dim": 12,
            "num_layers": 1,
            "dropout": 0.0,
            "max_atoms": 10,
        },
        "training": {
            "batch_size": 2,
            "epochs": epochs,
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "edge_loss_weight": 1.0,
            "validation_interval": 1,
            "patience": 2,
            "bucket_width": 2,
            "amp": False,
            "num_workers": 0,
        },
    }


def test_resume_config_hash_ignores_only_maximum_epoch_budget() -> None:
    first = _base_config(2)
    second = _base_config(10)
    assert resume_config_hash(first) == resume_config_hash(second)
    second["training"]["learning_rate"] = 0.01  # type: ignore[index]
    assert resume_config_hash(first) != resume_config_hash(second)


def test_artifact_paths_are_namespaced_for_targeted_guidance() -> None:
    base = artifact_paths(Path("artifacts/base/best.pt"))
    assert base["last"] == Path("artifacts/base/last.pt")
    guidance = artifact_paths(Path("artifacts/guidance/logp_q80_best.pt"))
    assert guidance["last"] == Path("artifacts/guidance/logp_q80_last.pt")
    assert guidance["history"] == Path("artifacts/guidance/logp_q80_history.csv")
    assert guidance["state"] == Path(
        "artifacts/guidance/logp_q80_training_state.json"
    )


def test_cache_hash_ignores_guidance_target_and_training() -> None:
    first = {
        "seed": 17,
        "target_quantile": 0.8,
        "cache": {"samples": 10, "batch_size": 2, "sampling_steps": 8},
        "training": {"epochs": 2},
    }
    second = {
        **first,
        "target_quantile": 0.9,
        "training": {"epochs": 20},
    }
    assert cache_config_hash(first) == cache_config_hash(second)
    second["cache"] = {"samples": 11, "batch_size": 2, "sampling_steps": 8}
    assert cache_config_hash(first) != cache_config_hash(second)


def test_legacy_hash_allows_only_epoch_and_new_completion_gate(tmp_path: Path) -> None:
    config_path = tmp_path / "base.yaml"
    original = "training:\n  epochs: 80\n  learning_rate: 0.001\n"
    config_path.write_text(original, encoding="utf-8")
    original_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()
    config_path.write_text(
        "training:\n  epochs: 300\n  require_full_epochs: true\n"
        "  learning_rate: 0.001\n",
        encoding="utf-8",
    )
    assert original_hash in legacy_config_hashes(config_path, completed_epochs=80)
    config_path.write_text(
        "training:\n  epochs: 300\n  require_full_epochs: true\n"
        "  learning_rate: 0.01\n",
        encoding="utf-8",
    )
    assert original_hash not in legacy_config_hashes(config_path, completed_epochs=80)


def test_early_stopped_run_is_ready_for_its_configured_epoch_target(
    tmp_path: Path,
) -> None:
    config = _base_config(300)
    best = tmp_path / "best.pt"
    best.write_bytes(b"checkpoint-placeholder")
    paths = artifact_paths(best)
    paths["state"].write_text(
        json.dumps(
            {
                "status": "complete",
                "completed_epochs": 150,
                "target_epochs": 300,
                "stopped_early": True,
                "completion_reason": "early_stopping",
                "ready_for_downstream": True,
                "resume_config_hash": resume_config_hash(config),
            }
        ),
        encoding="utf-8",
    )
    ready, reason = training_is_ready(best, config)
    assert ready, reason


def test_base_training_auto_resumes_when_only_epochs_increase(tmp_path: Path) -> None:
    source = tmp_path / "zinc.smi"
    source.write_text(
        "C1CC1\nC1CCC1\nC1CCCC1\nC1CCCCC1\nC1CCCCCC1\nC1CCCCCCC1\n",
        encoding="utf-8",
    )
    data_dir = tmp_path / "data"
    build_zinc_dataset(
        source,
        data_dir,
        DataConfig(
            seed=17,
            sample_size=5,
            train_fraction=0.6,
            validation_fraction=0.2,
            test_fraction=0.2,
            allowed_atomic_numbers=(6,),
            max_heavy_atoms=10,
        ),
    )
    config_path = tmp_path / "base.yaml"
    config = _base_config(1)
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    best = tmp_path / "artifacts" / "best.pt"
    arguments = [
        "train-base",
        "--manifest",
        str(data_dir / "manifest.csv"),
        "--vocabulary",
        str(data_dir / "atom_vocabulary.json"),
        "--output",
        str(best),
        "--config",
        str(config_path),
        "--device",
        "cpu",
    ]
    assert main(arguments) == 0
    ready, _ = training_is_ready(best, config)
    assert ready

    config = _base_config(2)
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    assert main(arguments) == 0
    paths = artifact_paths(best)
    state = json.loads(paths["state"].read_text(encoding="utf-8"))
    history = paths["history"].read_text(encoding="utf-8").splitlines()
    assert state["completed_epochs"] == 2
    assert state["ready_for_downstream"] is True
    assert paths["last"].exists()
    assert len(history) == 3  # header plus two epochs


def test_base_early_stopping_produces_downstream_ready_best(tmp_path: Path) -> None:
    source = tmp_path / "zinc.smi"
    source.write_text(
        "C1CC1\nC1CCC1\nC1CCCC1\nC1CCCCC1\nC1CCCCCC1\nC1CCCCCCC1\n",
        encoding="utf-8",
    )
    data_dir = tmp_path / "data"
    build_zinc_dataset(
        source,
        data_dir,
        DataConfig(
            seed=17,
            sample_size=5,
            train_fraction=0.6,
            validation_fraction=0.2,
            test_fraction=0.2,
            allowed_atomic_numbers=(6,),
            max_heavy_atoms=10,
        ),
    )
    config = _base_config(5)
    config["training"]["learning_rate"] = 0.0  # type: ignore[index]
    config["training"]["patience"] = 1  # type: ignore[index]
    config_path = tmp_path / "base.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    best = tmp_path / "artifacts" / "best.pt"
    assert (
        main(
            [
                "train-base",
                "--manifest",
                str(data_dir / "manifest.csv"),
                "--vocabulary",
                str(data_dir / "atom_vocabulary.json"),
                "--output",
                str(best),
                "--config",
                str(config_path),
                "--device",
                "cpu",
            ]
        )
        == 0
    )
    state = json.loads(
        artifact_paths(best)["state"].read_text(encoding="utf-8")
    )
    assert state["target_epochs"] == 5
    assert state["completed_epochs"] == 2
    assert state["completion_reason"] == "early_stopping"
    ready, reason = training_is_ready(best, config)
    assert ready, reason

"""Artifact readiness and resumable-training helpers for the VaDGM workflow."""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Mapping, Sequence

from vadgm.utils import stable_json_hash


def resume_config_hash(payload: Mapping[str, object]) -> str:
    """Hash all training semantics except the maximum epoch budget."""

    normalized = copy.deepcopy(dict(payload))
    training = normalized.get("training")
    if isinstance(training, dict):
        training.pop("epochs", None)
    return stable_json_hash(normalized)


def cache_config_hash(payload: Mapping[str, object]) -> str:
    """Hash only settings that change the frozen source distribution."""

    cache = payload.get("cache", {})
    if not isinstance(cache, Mapping):
        raise ValueError("guidance config cache must be a mapping")
    return stable_json_hash(
        {
            "seed": int(payload.get("seed", 2027)),
            "cache": dict(cache),
        }
    )


def legacy_config_hashes(
    config_path: str | Path,
    *,
    completed_epochs: int,
) -> set[str]:
    """Candidate hashes for a pre-workflow YAML changed only in duration policy.

    This is deliberately byte-level and strict.  It permits a one-time migration
    when ``epochs`` changed and/or ``require_full_epochs`` was newly inserted,
    while retaining the legacy checkpoint's original full-file SHA-256 guard.
    """

    raw = Path(config_path).read_bytes()
    text = raw.decode("utf-8")
    newline = "\r\n" if "\r\n" in text else "\n"
    lines = text.splitlines()

    variants = [lines]
    without_completion_policy = [
        line for line in lines if not re.match(r"^\s*require_full_epochs\s*:", line)
    ]
    variants.append(without_completion_policy)

    candidates: set[str] = set()
    for variant in variants:
        for replace_epochs in (False, True):
            updated = list(variant)
            if replace_epochs:
                for index, line in enumerate(updated):
                    match = re.match(r"^(\s*epochs\s*:\s*)\d+(\s*(?:#.*)?)$", line)
                    if match:
                        updated[index] = (
                            f"{match.group(1)}{int(completed_epochs)}{match.group(2)}"
                        )
                        break
            candidate = newline.join(updated)
            if text.endswith(("\n", "\r")):
                candidate += newline
            candidates.add(hashlib.sha256(candidate.encode("utf-8")).hexdigest())
    return candidates


def artifact_paths(best_path: str | Path) -> dict[str, Path]:
    best = Path(best_path)
    if best.stem == "best":
        prefix = ""
    elif best.stem.endswith("_best"):
        prefix = best.stem[: -len("best")]
    else:
        prefix = f"{best.stem}_"
    return {
        "best": best,
        "last": best.with_name(f"{prefix}last{best.suffix}"),
        "history": best.with_name(f"{prefix}history.csv"),
        "state": best.with_name(f"{prefix}training_state.json"),
    }


def write_json_atomic(path: str | Path, payload: Mapping[str, object]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".part")
    temporary.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, target)


def write_history(
    path: str | Path,
    rows: Sequence[Mapping[str, object]],
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".part")
    fieldnames = ["epoch", "train_loss", "validation_loss"]
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})
    os.replace(temporary, target)


def load_history(path: str | Path, *, before_epoch: int) -> list[dict[str, object]]:
    source = Path(path)
    if not source.exists():
        return []
    rows: list[dict[str, object]] = []
    with source.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            epoch = int(row["epoch"])
            if epoch >= before_epoch:
                continue
            rows.append(
                {
                    "epoch": epoch,
                    "train_loss": float(row["train_loss"]),
                    "validation_loss": (
                        float(row["validation_loss"])
                        if row.get("validation_loss") not in {None, "", "None"}
                        else None
                    ),
                }
            )
    return rows


def load_training_state(best_path: str | Path) -> dict[str, object] | None:
    state_path = artifact_paths(best_path)["state"]
    if not state_path.exists():
        return None
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid training state: {state_path}")
    return payload


def training_is_ready(
    best_path: str | Path,
    config_payload: Mapping[str, object],
) -> tuple[bool, str]:
    paths = artifact_paths(best_path)
    if not paths["best"].exists():
        return False, f"missing best checkpoint: {paths['best']}"
    state = load_training_state(best_path)
    if state is None:
        return False, f"missing completion state: {paths['state']}"
    if state.get("resume_config_hash") != resume_config_hash(config_payload):
        return False, "training configuration is incompatible with the completion state"
    training = config_payload.get("training", {})
    if not isinstance(training, Mapping):
        return False, "training config is not a mapping"
    requested_epochs = int(training.get("epochs", 0))
    completed_epochs = int(state.get("completed_epochs", 0))
    if int(state.get("target_epochs", 0)) != requested_epochs:
        return False, f"checkpoint target changed to {requested_epochs} epochs"
    if not bool(state.get("ready_for_downstream", False)):
        return False, "training run has not reached its configured completion gate"
    return True, "ready"

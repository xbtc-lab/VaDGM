"""Aggregate VaDGM paper experiments into traceable tables and Source Data."""

from __future__ import annotations

import csv
import json
import math
import statistics
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from rdkit import Chem
from tqdm.auto import tqdm

from vadgm.chemistry import AtomVocabulary
from vadgm.data import load_manifest
from vadgm.evaluation import SampleEvaluation, evaluate_generation_file


PAPER_METRICS = (
    "validity_percent",
    "connected_percent",
    "radical_free_percent",
    "terminal_valence_complete_percent",
    "terminal_gate_relaxation_percent",
    "strict_vthr_percent",
    "property_mae",
    "property_hit_percent",
    "vthr_percent",
    "final_capacity_violation_percent",
    "path_capacity_violation_percent",
    "uniqueness_percent",
    "novelty_percent",
    "mean_actual_nfe",
    "conflict_bin_percent",
    "conflict_molecule_percent",
    "mean_excess_capacity",
    "mean_logp",
)

METHOD_ORDER = {
    "Vanilla DGM": 0,
    "VaDGM (capacity-only)": 1,
    "VaDGM (no terminal atom phase)": 2,
    "VaDGM (strict terminal gate)": 3,
    "VaDGM": 4,
}

TABLE1_METRICS = (
    "validity_percent",
    "connected_percent",
    "radical_free_percent",
    "terminal_valence_complete_percent",
    "strict_vthr_percent",
    "property_mae",
    "uniqueness_percent",
    "novelty_percent",
    "final_capacity_violation_percent",
    "path_capacity_violation_percent",
    "mean_actual_nfe",
)

TABLE2_METRICS = (
    "connected_percent",
    "radical_free_percent",
    "terminal_valence_complete_percent",
    "terminal_gate_relaxation_percent",
    "strict_vthr_percent",
    "property_mae",
    "path_capacity_violation_percent",
    "mean_actual_nfe",
)

FIGURE4_METRICS = (
    "conflict_bin_percent",
    "conflict_molecule_percent",
    "path_capacity_violation_percent",
    "strict_vthr_percent",
    "mean_actual_nfe",
)

FIGURE3_METRICS = (
    "mean_logp",
    "property_mae",
    "property_hit_percent",
    "strict_vthr_percent",
)


def _canonical_smiles(smiles: str) -> str | None:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        return None
    return Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=False)


def _training_smiles(path: str | Path) -> set[str]:
    records = load_manifest(path, split="train")
    values = {
        canonical
        for record in tqdm(
            records,
            desc="Indexing training-set novelty",
            unit="mol",
        )
        if (canonical := _canonical_smiles(record.smiles)) is not None
    }
    if not values:
        raise ValueError("Training manifest contains no decodable train SMILES")
    return values


def _method(payload: Mapping[str, object]) -> str:
    explicit = payload.get("method")
    if explicit:
        return str(explicit)
    if isinstance(payload.get("graphs"), list):
        return "Vanilla DGM"
    terminal = payload.get("terminal_reachability", {})
    if not isinstance(terminal, Mapping) or not bool(terminal.get("enabled", False)):
        return "VaDGM (capacity-only)"
    if not bool(terminal.get("defer_atom_commit", False)):
        return "VaDGM (no terminal atom phase)"
    if not bool(terminal.get("relax_on_empty", True)):
        return "VaDGM (strict terminal gate)"
    return "VaDGM"


def _target(payload: Mapping[str, object]) -> float | None:
    value = payload.get("target")
    if isinstance(value, Mapping) and value.get("target") is not None:
        return float(value["target"])
    return None


def _optional_number(payload: Mapping[str, object], key: str) -> float | None:
    value = payload.get(key)
    return float(value) if value is not None else None


def _novelty(records: Sequence[SampleEvaluation], training: set[str]) -> float:
    unique = {record.smiles for record in records if record.valid and record.smiles}
    if not unique:
        return 0.0
    return 100.0 * sum(smiles not in training for smiles in unique) / len(unique)


def _ci95(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    critical = {
        1: 12.706,
        2: 4.303,
        3: 3.182,
        4: 2.776,
        5: 2.571,
        6: 2.447,
        7: 2.365,
        8: 2.306,
        9: 2.262,
        10: 2.228,
    }.get(len(values) - 1, 1.96)
    return critical * statistics.stdev(values) / math.sqrt(len(values))


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {key: "" if value is None else value for key, value in row.items()}
            )


def _paper_columns(
    rows: Sequence[Mapping[str, object]],
    metrics: Sequence[str],
    *,
    include_delta_s: bool = False,
) -> list[dict[str, object]]:
    """Keep paper-facing CSVs compact while retaining the full summary separately."""

    metadata = ["experiment", "condition", "method", "target"]
    if include_delta_s:
        metadata.append("delta_s")
    metadata.extend(("runs", "seeds", "samples_total"))
    fields = metadata + [
        field
        for metric in metrics
        for field in (f"{metric}_mean", f"{metric}_std", f"{metric}_ci95")
    ]
    return [{field: row.get(field) for field in fields} for row in rows]


def _summarize(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    group_fields = ("experiment", "condition", "method", "target", "delta_s", "steps")
    groups: dict[tuple[object, ...], list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(field) for field in group_fields)].append(row)
    summaries = []
    for key, members in groups.items():
        summary: dict[str, object] = dict(zip(group_fields, key))
        summary["runs"] = len(members)
        summary["seeds"] = len({member.get("seed") for member in members})
        summary["samples_total"] = sum(int(member["samples"]) for member in members)
        for metric in PAPER_METRICS:
            values = [
                float(member[metric])
                for member in members
                if member.get(metric) not in {None, ""}
                and math.isfinite(float(member[metric]))
            ]
            summary[f"{metric}_mean"] = statistics.fmean(values) if values else None
            summary[f"{metric}_std"] = (
                statistics.stdev(values) if len(values) >= 2 else 0.0 if values else None
            )
            summary[f"{metric}_ci95"] = _ci95(values) if values else None
        summaries.append(summary)
    return sorted(
        summaries,
        key=lambda row: (
            str(row["experiment"]),
            METHOD_ORDER.get(str(row["method"]), 99),
            float(row["target"]) if row["target"] is not None else math.inf,
            float(row["delta_s"]) if row["delta_s"] is not None else math.inf,
            str(row["condition"]),
        ),
    )


def _representatives(
    candidates: Iterable[tuple[Mapping[str, object], SampleEvaluation]],
    *,
    per_target: int = 2,
) -> list[dict[str, object]]:
    groups: dict[float, list[tuple[Mapping[str, object], SampleEvaluation]]] = defaultdict(list)
    for metadata, record in candidates:
        target = metadata.get("target")
        if target is None:
            continue
        groups[float(target)].append((metadata, record))
    output = []
    for target, members in sorted(groups.items()):
        seen = set()
        ordered = sorted(
            members,
            key=lambda item: (
                float(item[1].absolute_error)
                if item[1].absolute_error is not None
                else math.inf,
                item[1].sample_id,
                str(item[0]["source_run"]),
            ),
        )
        for metadata, record in ordered:
            if not record.smiles or record.smiles in seen:
                continue
            seen.add(record.smiles)
            output.append(
                {
                    "condition": metadata["condition"],
                    "target": target,
                    "seed": metadata["seed"],
                    "sample_id": record.sample_id,
                    "representative_id": (
                        f"{metadata['condition']}-s{metadata['seed']}-{record.sample_id}"
                    ),
                    "smiles": record.smiles,
                    "logp": record.logp,
                    "absolute_error": record.absolute_error,
                    "source_run": metadata["source_run"],
                }
            )
            if len(seen) >= per_target:
                break
    return output


def summarize_paper_experiments(
    input_dir: str | Path,
    vocabulary_path: str | Path,
    training_manifest: str | Path,
    output_dir: str | Path,
    *,
    hit_tolerance: float = 0.5,
) -> dict[str, str]:
    """Re-evaluate all completed runs and export the four paper outputs."""

    source = Path(input_dir)
    run_paths = sorted(
        path
        for path in source.rglob("*.json")
        if not path.name.endswith(".part") and not path.name.endswith(".meta.json")
    )
    if not run_paths:
        raise ValueError(f"No completed experiment JSON files found in {source}")
    vocabulary = AtomVocabulary.load(vocabulary_path)
    training = _training_smiles(training_manifest)
    run_rows: list[dict[str, object]] = []
    representative_candidates: list[tuple[Mapping[str, object], SampleEvaluation]] = []

    for run_path in run_paths:
        payload = json.loads(run_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or not any(
            isinstance(payload.get(key), list) for key in ("samples", "graphs")
        ):
            continue
        method = _method(payload)
        records, metrics = evaluate_generation_file(
            run_path,
            vocabulary,
            method=method,
            hit_tolerance=hit_tolerance,
            show_progress=True,
        )
        target = _target(payload)
        valid_logp = [float(record.logp) for record in records if record.logp is not None]
        row: dict[str, object] = {
            "source_run": str(run_path),
            "experiment": str(payload.get("experiment", "unspecified")),
            "condition": str(payload.get("condition", "unspecified")),
            "method": method,
            "seed": int(payload.get("seed", 0)),
            "target": target,
            "delta_s": _optional_number(payload, "delta_s"),
            "steps": int(payload["steps"]) if payload.get("steps") is not None else None,
            **asdict(metrics),
            "novelty_percent": _novelty(records, training),
            "mean_logp": statistics.fmean(valid_logp) if valid_logp else None,
        }
        row["method"] = method
        run_rows.append(row)

        if row["experiment"] in {"figure3", "table1"} and method == "VaDGM":
            eligible = [
                record
                for record in records
                if record.valid
                and record.connected
                and record.radical_free
                and record.capacity_safe
                and record.terminal_valence_complete
                and bool(record.property_hit)
            ]
            representative_candidates.extend(
                (row, record)
                for record in sorted(
                    eligible,
                    key=lambda record: (
                        float(record.absolute_error)
                        if record.absolute_error is not None
                        else math.inf,
                        record.sample_id,
                    ),
                )[:20]
            )

    if not run_rows:
        raise ValueError("No VaDGM-format experiment runs were found")

    run_keys: set[tuple[object, ...]] = set()
    for row in run_rows:
        key = tuple(
            row.get(field)
            for field in (
                "experiment",
                "condition",
                "method",
                "target",
                "delta_s",
                "steps",
                "seed",
            )
        )
        if key in run_keys:
            raise ValueError(
                "Duplicate experiment cell and seed; move superseded JSON files out "
                f"of the input directory: {key}"
            )
        run_keys.add(key)

    summaries = _summarize(run_rows)
    table1 = [row for row in summaries if row["experiment"] == "table1"]
    table2 = [
        row
        for row in summaries
        if row["experiment"] == "table2"
        or (
            row["experiment"] == "table1"
            and row["method"]
            in {"Vanilla DGM", "VaDGM (capacity-only)", "VaDGM"}
        )
    ]
    table2.sort(key=lambda row: METHOD_ORDER.get(str(row["method"]), 99))
    figure3 = [
        row
        for row in summaries
        if row["experiment"] == "figure3"
        or (row["experiment"] == "table1" and row["method"] == "VaDGM")
    ]
    figure3.sort(
        key=lambda row: (
            float(row["target"]) if row["target"] is not None else math.inf,
            str(row["condition"]),
        )
    )
    figure4 = [
        row
        for row in summaries
        if row["experiment"] == "figure4"
        or (row["experiment"] == "table1" and row["method"] == "VaDGM")
    ]
    figure4.sort(
        key=lambda row: (
            float(row["target"]) if row["target"] is not None else math.inf,
            float(row["delta_s"]) if row["delta_s"] is not None else math.inf,
        )
    )
    destination = Path(output_dir)
    files = {
        "runs": destination / "run_metrics.csv",
        "summary": destination / "summary_metrics.csv",
        "table1": destination / "table1_main_results.csv",
        "table2": destination / "table2_ablation.csv",
        "figure3": destination / "figure3_generation_source.csv",
        "figure4": destination / "figure4_conflict_source.csv",
        "representatives": destination / "figure3_representative_molecules.csv",
    }
    _write_csv(files["runs"], run_rows)
    _write_csv(files["summary"], summaries)
    _write_csv(files["table1"], _paper_columns(table1, TABLE1_METRICS))
    _write_csv(files["table2"], _paper_columns(table2, TABLE2_METRICS))
    _write_csv(
        files["figure3"],
        _paper_columns(
            figure3,
            FIGURE3_METRICS,
        ),
    )
    _write_csv(
        files["figure4"],
        _paper_columns(
            figure4,
            FIGURE4_METRICS,
            include_delta_s=True,
        ),
    )
    _write_csv(
        files["representatives"],
        _representatives(representative_candidates),
    )
    return {name: str(path) for name, path in files.items()}

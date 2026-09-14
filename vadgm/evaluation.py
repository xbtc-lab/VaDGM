"""Unified molecular evaluation and paper-table export for VaDGM stages."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

from rdkit import Chem
from rdkit.Chem import Crippen, Draw
from tqdm.auto import tqdm

from vadgm.chemistry import AtomVocabulary, CapacityEngine, GraphCodec, TokenGraph


@dataclass(frozen=True)
class SampleEvaluation:
    sample_id: str
    atom_ids: tuple[int, ...]
    bond_ids: tuple[int, ...]
    valid: bool
    capacity_safe: bool
    path_capacity_safe: bool | None
    connected: bool
    radical_free: bool
    radical_electrons: int
    terminal_valence_complete: bool
    strict_vthr: bool | None
    terminal_gate_deleted_categories: int
    terminal_gate_relaxations: int
    smiles: str | None
    logp: float | None
    target: float | None
    absolute_error: float | None
    property_hit: bool | None
    actual_nfe: int | None
    fallback_count: int
    conflict_bins: int
    total_bins: int
    mean_excess_capacity: float
    error: str | None = None


@dataclass(frozen=True)
class AggregateMetrics:
    method: str
    samples: int
    valid_samples: int
    validity_percent: float
    connected_samples: int
    connected_percent: float
    radical_free_samples: int
    radical_free_percent: float
    connected_radical_free_samples: int
    connected_radical_free_percent: float
    terminal_valence_complete_samples: int
    terminal_valence_complete_percent: float
    terminal_gate_relaxation_samples: int
    terminal_gate_relaxation_percent: float
    mean_terminal_gate_deleted_categories: float
    strict_vthr_samples: int | None
    strict_vthr_percent: float | None
    unique_valid_samples: int
    uniqueness_percent: float
    property_mae: float | None
    property_hit_percent: float | None
    vthr_percent: float | None
    final_capacity_violation_percent: float
    path_capacity_violation_percent: float | None
    mean_actual_nfe: float | None
    fallback_samples: int
    conflict_bin_percent: float | None
    conflict_molecule_percent: float | None
    mean_excess_capacity: float | None


def evaluate_graph(
    graph: TokenGraph,
    vocabulary: AtomVocabulary,
    capacity: CapacityEngine,
    *,
    sample_id: str,
    target: float | None = None,
    hit_tolerance: float = 0.5,
    diagnostics: Mapping[str, object] | None = None,
) -> SampleEvaluation:
    if hit_tolerance <= 0:
        raise ValueError("hit_tolerance must be positive")
    path_capacity_safe: bool | None = None
    actual_nfe: int | None = None
    fallback_count = 0
    conflict_bins = 0
    total_bins = 0
    excess_values: list[float] = []
    terminal_gate_deleted_categories = 0
    terminal_gate_relaxations = 0
    if diagnostics is not None:
        if "path_capacity_violations" in diagnostics:
            path_capacity_safe = int(diagnostics["path_capacity_violations"]) == 0
        if "base_forwards" in diagnostics:
            actual_nfe = int(diagnostics["base_forwards"])
        elif "model_forwards" in diagnostics:
            actual_nfe = int(diagnostics["model_forwards"])
        fallback_count = int(diagnostics.get("all_nonfinite_fallbacks", 0))
        terminal_gate_deleted_categories = int(
            diagnostics.get("terminal_gate_deleted_categories", 0)
        )
        terminal_gate_relaxations = int(diagnostics.get("terminal_gate_relaxations", 0))
        conflicts = diagnostics.get("joint_conflicts", [])
        if isinstance(conflicts, list):
            total_bins = len(conflicts)
            for conflict in conflicts:
                if not isinstance(conflict, dict) or not bool(conflict.get("occurred", False)):
                    continue
                conflict_bins += 1
                excess_pairs = conflict.get("excess_by_atom", [])
                if isinstance(excess_pairs, list) and excess_pairs:
                    excess_values.append(
                        sum(float(pair[1]) for pair in excess_pairs) / len(excess_pairs)
                    )
    try:
        capacity_safe = capacity.clean_graph_is_capacity_safe(graph)
    except (ValueError, RuntimeError):
        capacity_safe = False
    codec = GraphCodec(vocabulary)
    try:
        molecule = codec.decode_molecule(graph, sanitize=True)
        smiles = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=False)
        connected = len(Chem.GetMolFrags(molecule)) == 1
        radical_electrons = sum(
            atom.GetNumRadicalElectrons() for atom in molecule.GetAtoms()
        )
        radical_free = radical_electrons == 0
        terminal_valence_complete = capacity.clean_graph_is_terminal_complete(graph)
        logp = float(Crippen.MolLogP(molecule))
        absolute_error = abs(logp - target) if target is not None else None
        property_hit = absolute_error <= hit_tolerance if absolute_error is not None else None
        strict_vthr = (
            connected and radical_free and bool(property_hit)
            if property_hit is not None
            else None
        )
        return SampleEvaluation(
            sample_id=sample_id,
            atom_ids=graph.atom_ids,
            bond_ids=graph.bond_ids,
            valid=True,
            capacity_safe=capacity_safe,
            path_capacity_safe=path_capacity_safe,
            connected=connected,
            radical_free=radical_free,
            radical_electrons=radical_electrons,
            terminal_valence_complete=terminal_valence_complete,
            strict_vthr=strict_vthr,
            terminal_gate_deleted_categories=terminal_gate_deleted_categories,
            terminal_gate_relaxations=terminal_gate_relaxations,
            smiles=smiles,
            logp=logp,
            target=target,
            absolute_error=absolute_error,
            property_hit=property_hit,
            actual_nfe=actual_nfe,
            fallback_count=fallback_count,
            conflict_bins=conflict_bins,
            total_bins=total_bins,
            mean_excess_capacity=(
                sum(excess_values) / len(excess_values) if excess_values else 0.0
            ),
        )
    except Exception as error:
        return SampleEvaluation(
            sample_id=sample_id,
            atom_ids=graph.atom_ids,
            bond_ids=graph.bond_ids,
            valid=False,
            capacity_safe=capacity_safe,
            path_capacity_safe=path_capacity_safe,
            connected=False,
            radical_free=False,
            radical_electrons=0,
            terminal_valence_complete=False,
            strict_vthr=False if target is not None else None,
            terminal_gate_deleted_categories=terminal_gate_deleted_categories,
            terminal_gate_relaxations=terminal_gate_relaxations,
            smiles=None,
            logp=None,
            target=target,
            absolute_error=None,
            property_hit=False if target is not None else None,
            actual_nfe=actual_nfe,
            fallback_count=fallback_count,
            conflict_bins=conflict_bins,
            total_bins=total_bins,
            mean_excess_capacity=(
                sum(excess_values) / len(excess_values) if excess_values else 0.0
            ),
            error=f"{type(error).__name__}: {error}",
        )


def aggregate_evaluations(
    evaluations: Sequence[SampleEvaluation],
    *,
    method: str,
) -> AggregateMetrics:
    if not evaluations:
        raise ValueError("Cannot aggregate an empty evaluation set")
    total = len(evaluations)
    valid = [record for record in evaluations if record.valid]
    unique_smiles = {record.smiles for record in valid if record.smiles is not None}
    with_property = [record for record in valid if record.absolute_error is not None]
    target_present = any(record.target is not None for record in evaluations)
    hits = [record for record in with_property if record.property_hit]
    connected = [record for record in evaluations if record.connected]
    radical_free = [record for record in evaluations if record.radical_free]
    connected_radical_free = [
        record for record in evaluations if record.connected and record.radical_free
    ]
    terminal_complete = [
        record for record in evaluations if record.terminal_valence_complete
    ]
    terminal_relaxed = [
        record for record in evaluations if record.terminal_gate_relaxations > 0
    ]
    strict_hits = [record for record in evaluations if record.strict_vthr]
    nfe_values = [record.actual_nfe for record in evaluations if record.actual_nfe is not None]
    total_bins = sum(record.total_bins for record in evaluations)
    conflict_bins = sum(record.conflict_bins for record in evaluations)
    conflict_molecules = sum(record.conflict_bins > 0 for record in evaluations)
    conflict_excess_sum = sum(
        record.mean_excess_capacity * record.conflict_bins
        for record in evaluations
        if record.conflict_bins > 0
    )
    known_path = [
        record for record in evaluations if record.path_capacity_safe is not None
    ]
    return AggregateMetrics(
        method=method,
        samples=total,
        valid_samples=len(valid),
        validity_percent=100.0 * len(valid) / total,
        connected_samples=len(connected),
        connected_percent=100.0 * len(connected) / total,
        radical_free_samples=len(radical_free),
        radical_free_percent=100.0 * len(radical_free) / total,
        connected_radical_free_samples=len(connected_radical_free),
        connected_radical_free_percent=100.0 * len(connected_radical_free) / total,
        terminal_valence_complete_samples=len(terminal_complete),
        terminal_valence_complete_percent=100.0 * len(terminal_complete) / total,
        terminal_gate_relaxation_samples=len(terminal_relaxed),
        terminal_gate_relaxation_percent=100.0 * len(terminal_relaxed) / total,
        mean_terminal_gate_deleted_categories=(
            sum(record.terminal_gate_deleted_categories for record in evaluations) / total
        ),
        strict_vthr_samples=len(strict_hits) if target_present else None,
        strict_vthr_percent=(
            100.0 * len(strict_hits) / total if target_present else None
        ),
        unique_valid_samples=len(unique_smiles),
        uniqueness_percent=100.0 * len(unique_smiles) / len(valid) if valid else 0.0,
        property_mae=(
            sum(float(record.absolute_error) for record in with_property) / len(with_property)
            if with_property
            else None
        ),
        property_hit_percent=(100.0 * len(hits) / len(with_property) if with_property else None),
        vthr_percent=(100.0 * len(hits) / total if target_present else None),
        final_capacity_violation_percent=(
            100.0 * sum(not record.capacity_safe for record in evaluations) / total
        ),
        path_capacity_violation_percent=(
            100.0 * sum(record.path_capacity_safe is False for record in known_path) / len(known_path)
            if known_path
            else None
        ),
        mean_actual_nfe=(sum(int(value) for value in nfe_values) / len(nfe_values) if nfe_values else None),
        fallback_samples=sum(record.fallback_count > 0 for record in evaluations),
        conflict_bin_percent=(100.0 * conflict_bins / total_bins if total_bins else None),
        conflict_molecule_percent=(
            100.0 * conflict_molecules / total if total_bins else None
        ),
        mean_excess_capacity=(
            conflict_excess_sum / conflict_bins if conflict_bins else None
        ),
    )


def load_generation_file(
    path: str | Path,
) -> tuple[list[tuple[str, TokenGraph, Mapping[str, object] | None]], float | None]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    target_payload = payload.get("target")
    target: float | None = None
    if isinstance(target_payload, dict) and "target" in target_payload:
        target = float(target_payload["target"])
    graphs: list[tuple[str, TokenGraph, Mapping[str, object] | None]] = []
    if isinstance(payload.get("samples"), list):
        for index, sample in enumerate(payload["samples"]):
            graph_payload = sample["graph"]
            graph = TokenGraph(
                tuple(graph_payload["atom_ids"]),
                tuple(graph_payload["bond_ids"]),
            )
            diagnostics = sample.get("diagnostics")
            sample_id = str(sample.get("sample_id", f"sample-{index:06d}"))
            graphs.append(
                (sample_id, graph, diagnostics if isinstance(diagnostics, dict) else None)
            )
    elif isinstance(payload.get("graphs"), list):
        shared_diagnostics = payload.get("diagnostics")
        for index, graph_payload in enumerate(payload["graphs"]):
            graph = TokenGraph(
                tuple(graph_payload["atom_ids"]),
                tuple(graph_payload["bond_ids"]),
            )
            sample_id = str(graph_payload.get("sample_id", f"sample-{index:06d}"))
            diagnostics = (
                shared_diagnostics if isinstance(shared_diagnostics, dict) else None
            )
            graphs.append((sample_id, graph, diagnostics))
    else:
        raise ValueError("Generation file has neither samples nor graphs")
    return graphs, target


def evaluate_generation_file(
    path: str | Path,
    vocabulary: AtomVocabulary,
    *,
    method: str,
    target_override: float | None = None,
    hit_tolerance: float = 0.5,
    show_progress: bool = False,
) -> tuple[list[SampleEvaluation], AggregateMetrics]:
    graphs, stored_target = load_generation_file(path)
    target = target_override if target_override is not None else stored_target
    capacity = CapacityEngine(vocabulary)
    evaluations = []
    for index, (stored_sample_id, graph, diagnostics) in enumerate(
        tqdm(
            graphs,
            desc="Evaluating molecules",
            unit="mol",
            disable=not show_progress,
        )
    ):
        evaluations.append(
            evaluate_graph(
                graph,
                vocabulary,
                capacity,
                sample_id=stored_sample_id or f"{method}-{index:06d}",
                target=target,
                hit_tolerance=hit_tolerance,
                diagnostics=diagnostics,
            )
        )
    return evaluations, aggregate_evaluations(evaluations, method=method)


def _csv_value(value: object) -> object:
    if value is None:
        return ""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    return value


def write_sample_records(records: Sequence[SampleEvaluation], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(asdict(records[0]).keys()) if records else list(SampleEvaluation.__dataclass_fields__)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row = asdict(record)
            row["atom_ids"] = json.dumps(row["atom_ids"])
            row["bond_ids"] = json.dumps(row["bond_ids"])
            writer.writerow({key: _csv_value(value) for key, value in row.items()})


def write_metrics_csv(metrics: Sequence[AggregateMetrics], path: str | Path) -> None:
    if not metrics:
        raise ValueError("No metrics to write")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(asdict(metrics[0]).keys())
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for metric in metrics:
            writer.writerow(
                {key: _csv_value(value) for key, value in asdict(metric).items()}
            )


def write_metrics_markdown(metrics: Sequence[AggregateMetrics], path: str | Path) -> None:
    headers = [
        "Method",
        "Samples",
        "Decode Validity %",
        "Connected %",
        "Radical-free %",
        "Terminal Complete %",
        "Terminal Relax %",
        "Strict VTHR %",
        "Property MAE",
        "Hit Rate %",
        "Loose VTHR %",
        "Capacity Violation %",
        "Uniqueness %",
        "Actual NFE",
    ]

    def formatted(value: float | None) -> str:
        return "—" if value is None else f"{value:.4f}"

    rows = []
    for metric in metrics:
        rows.append(
            [
                metric.method,
                str(metric.samples),
                formatted(metric.validity_percent),
                formatted(metric.connected_percent),
                formatted(metric.radical_free_percent),
                formatted(metric.terminal_valence_complete_percent),
                formatted(metric.terminal_gate_relaxation_percent),
                formatted(metric.strict_vthr_percent),
                formatted(metric.property_mae),
                formatted(metric.property_hit_percent),
                formatted(metric.vthr_percent),
                formatted(metric.final_capacity_violation_percent),
                formatted(metric.uniqueness_percent),
                formatted(metric.mean_actual_nfe),
            ]
        )
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_joint_conflict_csv(
    metrics: AggregateMetrics,
    path: str | Path,
    *,
    target_value: float | None,
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "target",
                "mean_actual_nfe",
                "molecules",
                "conflict_bins_percent",
                "conflict_molecules_percent",
                "mean_excess_capacity",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "target": _csv_value(target_value),
                "mean_actual_nfe": _csv_value(metrics.mean_actual_nfe),
                "molecules": metrics.samples,
                "conflict_bins_percent": _csv_value(metrics.conflict_bin_percent),
                "conflict_molecules_percent": _csv_value(metrics.conflict_molecule_percent),
                "mean_excess_capacity": _csv_value(metrics.mean_excess_capacity),
            }
        )


def select_representative_samples(
    records: Sequence[SampleEvaluation],
    *,
    count: int = 8,
) -> list[SampleEvaluation]:
    if count <= 0:
        raise ValueError("count must be positive")
    hits = [record for record in records if record.valid and record.property_hit]
    candidates = hits or [record for record in records if record.valid]
    return sorted(
        candidates,
        key=lambda record: (
            record.absolute_error if record.absolute_error is not None else math.inf,
            record.sample_id,
        ),
    )[:count]


def render_representative_molecules(
    records: Sequence[SampleEvaluation],
    path: str | Path,
    *,
    count: int = 8,
    molecules_per_row: int = 4,
) -> list[str]:
    selected = select_representative_samples(records, count=count)
    molecules = []
    legends = []
    selected_ids = []
    for record in selected:
        if record.smiles is None:
            continue
        molecule = Chem.MolFromSmiles(record.smiles)
        if molecule is None:
            continue
        molecules.append(molecule)
        selected_ids.append(record.sample_id)
        if record.target is None:
            legends.append(f"{record.sample_id}\nlogP={record.logp:.2f}")
        else:
            legends.append(
                f"{record.sample_id}\ntarget={record.target:.2f}, logP={record.logp:.2f}"
            )
    if not molecules:
        raise ValueError("No valid molecules are available for rendering")
    image = Draw.MolsToGridImage(
        molecules,
        molsPerRow=molecules_per_row,
        subImgSize=(320, 260),
        legends=legends,
    )
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    image.save(str(target))
    return selected_ids


def export_evaluation_bundle(
    records: Sequence[SampleEvaluation],
    metrics: AggregateMetrics,
    output_dir: str | Path,
    *,
    render_examples: bool = True,
) -> dict[str, str]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    samples_path = destination / "samples.csv"
    metrics_path = destination / "metrics.csv"
    markdown_path = destination / "metrics.md"
    conflict_path = destination / "joint_conflict.csv"
    write_sample_records(records, samples_path)
    write_metrics_csv([metrics], metrics_path)
    write_metrics_markdown([metrics], markdown_path)
    target_value = next((record.target for record in records if record.target is not None), None)
    write_joint_conflict_csv(metrics, conflict_path, target_value=target_value)
    outputs = {
        "samples": str(samples_path),
        "metrics": str(metrics_path),
        "markdown": str(markdown_path),
        "joint_conflict": str(conflict_path),
    }
    if render_examples and any(record.valid for record in records):
        figure_path = destination / "representative_molecules.png"
        render_representative_molecules(records, figure_path)
        outputs["representative_molecules"] = str(figure_path)
    return outputs

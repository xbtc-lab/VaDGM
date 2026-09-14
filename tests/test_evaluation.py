from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import torch
from rdkit import Chem
from rdkit.Chem import Crippen

from vadgm.base_model import BaseGenerator, BaseModelConfig
from vadgm.chemistry import (
    AtomToken,
    AtomVocabulary,
    BondToken,
    CapacityEngine,
    GraphCodec,
    TokenGraph,
)
from vadgm.data import vocabulary_from_smiles
from vadgm.diagnostics import EventPlan
from vadgm.evaluation import (
    aggregate_evaluations,
    evaluate_generation_file,
    evaluate_graph,
    export_evaluation_bundle,
    render_representative_molecules,
    select_representative_samples,
)
from vadgm.guidance_model import GuidanceConfig, GuidanceNetwork
from vadgm.sampler import result_to_dict, sample_vadgm


def test_metric_denominators_keep_invalid_samples() -> None:
    vocabulary = AtomVocabulary([AtomToken(6, 0, 4)])
    capacity = CapacityEngine(vocabulary)
    methane = TokenGraph((0,), ())
    invalid = TokenGraph((0, 0), (BondToken.SINGLE,))
    target = float(Crippen.MolLogP(Chem.MolFromSmiles("C")))
    records = [
        evaluate_graph(
            methane,
            vocabulary,
            capacity,
            sample_id="a",
            target=target,
            hit_tolerance=1e-6,
        ),
        evaluate_graph(
            methane,
            vocabulary,
            capacity,
            sample_id="b",
            target=target,
            hit_tolerance=1e-6,
        ),
        evaluate_graph(
            invalid,
            vocabulary,
            capacity,
            sample_id="bad",
            target=target,
            hit_tolerance=1e-6,
            diagnostics={"path_capacity_violations": 1, "base_forwards": 3},
        ),
    ]
    metrics = aggregate_evaluations(records, method="VaDGM")
    assert metrics.samples == 3
    assert metrics.valid_samples == 2
    assert metrics.validity_percent == pytest.approx(200 / 3)
    assert metrics.connected_percent == pytest.approx(200 / 3)
    assert metrics.radical_free_percent == pytest.approx(200 / 3)
    assert metrics.strict_vthr_percent == pytest.approx(200 / 3)
    assert metrics.terminal_gate_relaxation_percent == 0.0
    assert metrics.uniqueness_percent == 50.0
    assert metrics.property_mae == pytest.approx(0.0)
    assert metrics.property_hit_percent == 100.0
    assert metrics.vthr_percent == pytest.approx(200 / 3)
    assert metrics.final_capacity_violation_percent == pytest.approx(100 / 3)
    # Only the third sample has a recorded path trace; unknown is not reported as safe.
    assert records[0].path_capacity_safe is None
    assert metrics.path_capacity_violation_percent == 100.0


def test_joint_conflict_aggregation_uses_bin_and_molecule_denominators() -> None:
    vocabulary = vocabulary_from_smiles(["C"])
    graph = GraphCodec(vocabulary).encode_smiles("C")
    capacity = CapacityEngine(vocabulary)
    conflict = {
        "occurred": True,
        "excess_by_atom": [[0, 2]],
    }
    no_conflict = {"occurred": False, "excess_by_atom": []}
    records = [
        evaluate_graph(
            graph,
            vocabulary,
            capacity,
            sample_id="a",
            diagnostics={
                "base_forwards": 2,
                "path_capacity_violations": 0,
                "joint_conflicts": [conflict, no_conflict],
            },
        ),
        evaluate_graph(
            graph,
            vocabulary,
            capacity,
            sample_id="b",
            diagnostics={
                "base_forwards": 2,
                "path_capacity_violations": 0,
                "joint_conflicts": [no_conflict, no_conflict],
            },
        ),
    ]
    metrics = aggregate_evaluations(records, method="VaDGM")
    assert metrics.conflict_bin_percent == 25.0
    assert metrics.conflict_molecule_percent == 50.0
    assert metrics.mean_excess_capacity == 2.0
    assert metrics.mean_actual_nfe == 2.0


def test_representative_selection_is_fixed_by_error_then_sample_id(tmp_path: Path) -> None:
    vocabulary = vocabulary_from_smiles(["C", "CC"])
    codec = GraphCodec(vocabulary)
    capacity = CapacityEngine(vocabulary)
    target = float(Crippen.MolLogP(Chem.MolFromSmiles("C")))
    records = [
        evaluate_graph(
            codec.encode_smiles("CC"),
            vocabulary,
            capacity,
            sample_id="z",
            target=target,
            hit_tolerance=10.0,
        ),
        evaluate_graph(
            codec.encode_smiles("C"),
            vocabulary,
            capacity,
            sample_id="b",
            target=target,
            hit_tolerance=10.0,
        ),
        evaluate_graph(
            codec.encode_smiles("C"),
            vocabulary,
            capacity,
            sample_id="a",
            target=target,
            hit_tolerance=10.0,
        ),
    ]
    selected = select_representative_samples(records, count=2)
    assert [record.sample_id for record in selected] == ["a", "b"]
    output = tmp_path / "molecules.png"
    selected_ids = render_representative_molecules(records, output, count=2)
    assert selected_ids == ["a", "b"]
    assert output.exists() and output.stat().st_size > 0


def test_generation_file_exports_traceable_tables(tmp_path: Path) -> None:
    vocabulary = vocabulary_from_smiles(["C"])
    graph = GraphCodec(vocabulary).encode_smiles("C")
    target = float(Crippen.MolLogP(Chem.MolFromSmiles("C")))
    generation = tmp_path / "generation.json"
    generation.write_text(
        json.dumps(
            {
                "target": {"target": target},
                "graphs": [
                    {"atom_ids": list(graph.atom_ids), "bond_ids": list(graph.bond_ids)},
                    {"atom_ids": list(graph.atom_ids), "bond_ids": list(graph.bond_ids)},
                ],
            }
        ),
        encoding="utf-8",
    )
    records, metrics = evaluate_generation_file(
        generation,
        vocabulary,
        method="Base+Guidance",
    )
    outputs = export_evaluation_bundle(records, metrics, tmp_path / "bundle")
    assert set(outputs) >= {"samples", "metrics", "markdown", "joint_conflict"}
    with Path(outputs["metrics"]).open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["method"] == "Base+Guidance"
    assert float(rows[0]["vthr_percent"]) == 100.0
    assert rows[0]["path_capacity_violation_percent"] == ""
    markdown = Path(outputs["markdown"]).read_text(encoding="utf-8")
    assert "VTHR %" in markdown and "Base+Guidance" in markdown
    assert "Connected %" in markdown
    assert "Radical-free %" in markdown
    assert "Strict VTHR %" in markdown


def test_radical_and_terminal_metrics_are_not_decode_validity() -> None:
    vocabulary = AtomVocabulary([AtomToken(6, 0, 0)])
    graph = TokenGraph((0,), ())
    record = evaluate_graph(
        graph,
        vocabulary,
        CapacityEngine(vocabulary),
        sample_id="bare-carbon",
        target=0.0,
        hit_tolerance=10.0,
    )

    assert record.valid
    assert record.connected
    assert not record.radical_free
    assert record.radical_electrons > 0
    assert not record.terminal_valence_complete
    assert not record.strict_vthr


def test_small_vadgm_sampling_to_evaluation_pipeline(tmp_path: Path) -> None:
    vocabulary = vocabulary_from_smiles(["C1CC1"])
    base = BaseGenerator(
        BaseModelConfig(vocabulary.clean_size, hidden_dim=12, num_layers=1, max_atoms=4)
    )
    guidance = GuidanceNetwork(
        GuidanceConfig(vocabulary.clean_size, hidden_dim=12, num_layers=1, max_atoms=4)
    )
    plan = EventPlan.sample(3, 0.3, generator=torch.Generator().manual_seed(12))
    result = sample_vadgm(
        base,
        guidance,
        vocabulary,
        CapacityEngine(vocabulary),
        plan,
        target=1.0,
        category_generator=torch.Generator().manual_seed(13),
        diagnostic_generator=torch.Generator().manual_seed(14),
    )
    raw = tmp_path / "vadgm.json"
    raw.write_text(
        json.dumps(
            {
                "target": {"target": 1.0},
                "samples": [result_to_dict(result)],
            }
        ),
        encoding="utf-8",
    )
    records, metrics = evaluate_generation_file(raw, vocabulary, method="VaDGM")
    outputs = export_evaluation_bundle(
        records,
        metrics,
        tmp_path / "results",
        render_examples=False,
    )
    assert len(records) == 1
    assert metrics.final_capacity_violation_percent == 0.0
    assert metrics.path_capacity_violation_percent == 0.0
    assert Path(outputs["samples"]).exists()

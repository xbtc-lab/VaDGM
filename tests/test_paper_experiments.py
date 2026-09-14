from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import pytest
import torch  # Load PyTorch DLLs before RDKit on Windows.
from rdkit import Chem
from rdkit.Chem import Crippen

from figures.experiments import plot_figure3, plot_figure4
from vadgm.chemistry import AtomVocabulary, GraphCodec
from vadgm.cli import build_parser
import vadgm.cli as cli
from vadgm.data import vocabulary_from_smiles
from vadgm.paper_experiments import _method, summarize_paper_experiments


SMILES = ("C", "CC", "CCC", "CCCC")


def _logp(smiles: str) -> float:
    molecule = Chem.MolFromSmiles(smiles)
    assert molecule is not None
    return float(Crippen.MolLogP(molecule))


def _graph(codec: GraphCodec, smiles: str) -> dict[str, object]:
    graph = codec.encode_smiles(smiles)
    return {"atom_ids": list(graph.atom_ids), "bond_ids": list(graph.bond_ids)}


def _diagnostics(
    *,
    nfe: int = 16,
    conflict: bool = False,
    terminal_deleted: int = 0,
) -> dict[str, object]:
    return {
        "base_forwards": nfe,
        "path_capacity_violations": 0,
        "all_nonfinite_fallbacks": 0,
        "terminal_gate_deleted_categories": terminal_deleted,
        "terminal_gate_relaxations": 0,
        "joint_conflicts": [
            {
                "occurred": conflict,
                "excess_by_atom": [[0, 1]] if conflict else [],
            },
            {"occurred": False, "excess_by_atom": []},
        ],
    }


def _write_run(
    path: Path,
    codec: GraphCodec,
    *,
    experiment: str,
    condition: str,
    method: str,
    target: float,
    seed: int,
    smiles: tuple[str, ...],
    delta_s: float | None = None,
    nfe: int = 16,
    conflict: bool = False,
) -> None:
    payload: dict[str, object] = {
        "experiment": experiment,
        "condition": condition,
        "method": method,
        "target": {"target": target},
        "seed": seed,
    }
    if method == "Vanilla DGM":
        payload.update(
            {
                "steps": nfe,
                "diagnostics": {"model_forwards": nfe},
                "graphs": [
                    {"sample_id": f"guided-{index:03d}", **_graph(codec, value)}
                    for index, value in enumerate(smiles)
                ],
            }
        )
    else:
        payload["delta_s"] = 0.25 if delta_s is None else delta_s
        payload["terminal_reachability"] = {
            "enabled": method != "VaDGM (capacity-only)",
            "relax_on_empty": True,
            "defer_atom_commit": method != "VaDGM (no terminal atom phase)",
        }
        payload["samples"] = [
            {
                "sample_id": f"vadgm-{index:03d}",
                "graph": _graph(codec, value),
                "diagnostics": _diagnostics(
                    nfe=nfe,
                    conflict=conflict and index == 0,
                    terminal_deleted=3 if method == "VaDGM" else 0,
                ),
            }
            for index, value in enumerate(smiles)
        ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _experiment_fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    vocabulary = vocabulary_from_smiles(SMILES)
    vocabulary_path = tmp_path / "atom_vocabulary.json"
    vocabulary.save(vocabulary_path)
    codec = GraphCodec(vocabulary)
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "sample_id,smiles,split,n_heavy_atoms\ntrain-c,C,train,1\n",
        encoding="utf-8",
    )
    runs = tmp_path / "runs"
    q80 = _logp("CC")

    _write_run(
        runs / "table1" / "vanilla_2027.json",
        codec,
        experiment="table1",
        condition="q80",
        method="Vanilla DGM",
        target=q80,
        seed=2027,
        smiles=("CC", "CCC"),
        nfe=128,
    )
    _write_run(
        runs / "table1" / "capacity_2027.json",
        codec,
        experiment="table1",
        condition="q80",
        method="VaDGM (capacity-only)",
        target=q80,
        seed=2027,
        smiles=("CC", "CCC"),
    )
    for seed, generated in ((2027, ("CC", "CCC")), (2028, ("CCCC", "CCCC"))):
        _write_run(
            runs / "table1" / f"full_{seed}.json",
            codec,
            experiment="table1",
            condition="q80",
            method="VaDGM",
            target=q80,
            seed=seed,
            smiles=generated,
        )
    _write_run(
        runs / "table2" / "no_terminal_phase_2027.json",
        codec,
        experiment="table2",
        condition="q80",
        method="VaDGM (no terminal atom phase)",
        target=q80,
        seed=2027,
        smiles=("CC", "CCC"),
    )

    for index, (delta_s, nfe) in enumerate(
        ((0.5, 8), (0.125, 32), (0.0625, 64))
    ):
        _write_run(
            runs / "figure4" / f"delta_{index}.json",
            codec,
            experiment="figure4",
            condition=f"delta_s={delta_s}",
            method="VaDGM",
            target=q80,
            seed=2027,
            smiles=("CC", "CCC"),
            delta_s=delta_s,
            nfe=nfe,
            conflict=True,
        )

    figure3_runs = (
        ("q50", "C", ("C", "CC")),
        ("q90", "CCCC", ("CCC", "CCCC")),
    )
    for condition, target_smiles, generated in figure3_runs:
        _write_run(
            runs / "figure3" / f"{condition}.json",
            codec,
            experiment="figure3",
            condition=condition,
            method="VaDGM",
            target=_logp(target_smiles),
            seed=2027,
            smiles=generated,
        )
    return runs, vocabulary_path, manifest, tmp_path / "paper"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_paper_summary_builds_four_outputs_and_seed_statistics(tmp_path: Path) -> None:
    runs, vocabulary, manifest, output = _experiment_fixture(tmp_path)
    files = summarize_paper_experiments(runs, vocabulary, manifest, output)

    table1 = _read_csv(Path(files["table1"]))
    assert {row["method"] for row in table1} == {
        "Vanilla DGM",
        "VaDGM (capacity-only)",
        "VaDGM",
    }
    full = next(row for row in table1 if row["method"] == "VaDGM")
    assert full["runs"] == "2"
    assert full["seeds"] == "2"
    assert float(full["strict_vthr_percent_mean"]) == pytest.approx(50.0)
    assert float(full["strict_vthr_percent_ci95"]) > 0.0

    table2 = _read_csv(Path(files["table2"]))
    assert [row["method"] for row in table2] == [
        "Vanilla DGM",
        "VaDGM (capacity-only)",
        "VaDGM (no terminal atom phase)",
        "VaDGM",
    ]
    vanilla = next(row for row in table1 if row["method"] == "Vanilla DGM")
    assert float(vanilla["novelty_percent_mean"]) == 100.0

    figure4 = _read_csv(Path(files["figure4"]))
    assert sorted(float(row["mean_actual_nfe_mean"]) for row in figure4) == [8, 16, 32, 64]
    representatives = _read_csv(Path(files["representatives"]))
    assert len(representatives) == 6
    assert {row["condition"] for row in representatives} == {"q50", "q80", "q90"}


def test_method_inference_and_cli_experiment_arguments() -> None:
    assert _method({"graphs": []}) == "Vanilla DGM"
    assert _method({"samples": [], "terminal_reachability": {"enabled": False}}) == (
        "VaDGM (capacity-only)"
    )
    assert _method(
        {
            "samples": [],
            "terminal_reachability": {
                "enabled": True,
                "defer_atom_commit": False,
            },
        }
    ) == "VaDGM (no terminal atom phase)"

    args = build_parser().parse_args(
        [
            "sample-vadgm",
            "--base-checkpoint",
            "base.pt",
            "--guidance-checkpoint",
            "guidance.pt",
            "--vocabulary",
            "vocabulary.json",
            "--output",
            "run.json",
            "--seed",
            "2029",
            "--delta-s",
            "0.125",
            "--experiment",
            "figure4",
            "--condition",
            "delta_s=0.125",
        ]
    )
    assert args.seed == 2029
    assert args.delta_s == pytest.approx(0.125)
    assert args.experiment == "figure4"


def test_run_table1_reads_yaml_and_reuses_existing_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = tmp_path / "table1"
    output_dir.mkdir()
    (output_dir / "vanilla_q80_seed2027.json").write_text(
        json.dumps(
            {
                "experiment": "table1",
                "condition": "q80",
                "seed": 2027,
                "graphs": [{}, {}],
            }
        ),
        encoding="utf-8",
    )
    config = tmp_path / "table1.yaml"
    config.write_text(
        """
experiment: table1
condition: q80
seeds: [2027, 2028]
settings:
  samples: 2
  steps: 128
  batch_size: 256
  device: cpu
  overwrite: false
paths:
  base_checkpoint: base.pt
  guidance_checkpoint: guidance.pt
  vocabulary: vocabulary.json
  output_dir: {output_dir}
sampler_configs:
  capacity_only: capacity.yaml
  vadgm: sampling.yaml
""".format(output_dir=output_dir.as_posix()),
        encoding="utf-8",
    )
    calls: list[tuple[str, int, Path]] = []

    def fake_guided(args: object) -> int:
        calls.append(("guided", int(args.seed), Path(args.output)))
        return 0

    def fake_vadgm(args: object) -> int:
        calls.append(("vadgm", int(args.seed), Path(args.output)))
        return 0

    monkeypatch.setattr(cli, "_sample_guided", fake_guided)
    monkeypatch.setattr(cli, "_sample_vadgm", fake_vadgm)
    assert cli._run_table1(type("Args", (), {"config": config})()) == 0
    assert len(calls) == 5
    assert sum(call[1] == 2027 for call in calls) == 2
    assert sum(call[1] == 2028 for call in calls) == 3


def test_run_table1_parser_defaults_to_experiment_yaml() -> None:
    args = build_parser().parse_args(["run-table1"])
    assert str(args.config).replace("\\", "/") == "configs/experiments/table1.yaml"


def _write_reusable_result(
    path: Path,
    *,
    experiment: str,
    condition: str,
    seed: int,
    samples: int = 2,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "experiment": experiment,
                "condition": condition,
                "seed": seed,
                "samples": [{} for _ in range(samples)],
            }
        ),
        encoding="utf-8",
    )


def test_run_table2_uses_compact_yaml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = tmp_path / "table2.yaml"
    config.write_text(
        f"""
experiment: table2
condition: q80
seeds: [2027, 2028]
settings:
  samples: 2
  batch_size: 4
  device: cpu
  overwrite: false
paths:
  base_checkpoint: base.pt
  guidance_checkpoint: guidance.pt
  vocabulary: vocabulary.json
  sampler_config: no_phase.yaml
  output_dir: {(tmp_path / 'table2').as_posix()}
""",
        encoding="utf-8",
    )
    calls: list[argparse.Namespace] = []
    monkeypatch.setattr(cli, "_sample_vadgm", lambda args: calls.append(args) or 0)
    assert cli._run_table2(type("Args", (), {"config": config})()) == 0
    assert [call.seed for call in calls] == [2027, 2028]
    assert all(call.config == Path("no_phase.yaml") for call in calls)


def test_run_figure3_reuses_q80_and_samples_other_targets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reuse_pattern = tmp_path / "table1" / "vadgm_q80_seed{seed}.json"
    for seed in (2027, 2028):
        _write_reusable_result(
            Path(str(reuse_pattern).format(seed=seed)),
            experiment="table1",
            condition="q80",
            seed=seed,
        )
    config = tmp_path / "figure3.yaml"
    config.write_text(
        f"""
experiment: figure3
seeds: [2027, 2028]
settings:
  samples: 2
  batch_size: 4
  device: cpu
  overwrite: false
paths:
  base_checkpoint: base.pt
  vocabulary: vocabulary.json
  sampler_config: sampling.yaml
  output_dir: {(tmp_path / 'figure3').as_posix()}
targets:
  q50:
    guidance_checkpoint: q50.pt
    guidance_config: q50.yaml
  q80:
    reuse_output_pattern: {reuse_pattern.as_posix()}
    reuse_experiment: table1
    reuse_condition: q80
  q90:
    guidance_checkpoint: q90.pt
    guidance_config: q90.yaml
""",
        encoding="utf-8",
    )
    calls: list[argparse.Namespace] = []
    monkeypatch.setattr(cli, "_sample_vadgm", lambda args: calls.append(args) or 0)
    monkeypatch.setattr(cli, "_require_training_ready", lambda *args, **kwargs: None)
    assert cli._run_figure3(type("Args", (), {"config": config})()) == 0
    assert [(call.condition, call.seed) for call in calls] == [
        ("q50", 2027),
        ("q50", 2028),
        ("q90", 2027),
        ("q90", 2028),
    ]


def test_train_figure3_only_trains_non_reused_targets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = tmp_path / "figure3.yaml"
    config.write_text(
        """
experiment: figure3
seeds: [2027, 2028]
settings:
  samples: 2
  batch_size: 4
  device: cpu
paths:
  source: zinc.csv
  manifest: manifest.csv
  vocabulary: vocabulary.json
  source_cache: source.jsonl
  base_checkpoint: base.pt
targets:
  q50:
    guidance_checkpoint: q50.pt
    guidance_config: q50.yaml
  q80:
    reuse_output_pattern: table1/vadgm_q80_seed{seed}.json
  q90:
    guidance_checkpoint: q90.pt
    guidance_config: q90.yaml
""",
        encoding="utf-8",
    )
    calls: list[argparse.Namespace] = []
    monkeypatch.setattr(cli, "_train_guidance", lambda args: calls.append(args) or 0)
    assert cli._train_figure3(type("Args", (), {"config": config})()) == 0
    assert [(call.output, call.config) for call in calls] == [
        (Path("q50.pt"), Path("q50.yaml")),
        (Path("q90.pt"), Path("q90.yaml")),
    ]


def test_run_figure4_reuses_table1_delta(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    reuse_pattern = tmp_path / "table1" / "vadgm_q80_seed{seed}.json"
    for seed in (2027, 2028):
        _write_reusable_result(
            Path(str(reuse_pattern).format(seed=seed)),
            experiment="table1",
            condition="q80",
            seed=seed,
        )
    config = tmp_path / "figure4.yaml"
    config.write_text(
        f"""
experiment: figure4
condition_prefix: delta_s=
seeds: [2027, 2028]
delta_s: [0.5, 0.25, 0.125]
settings:
  samples: 2
  batch_size: 4
  device: cpu
  overwrite: false
paths:
  base_checkpoint: base.pt
  guidance_checkpoint: guidance.pt
  vocabulary: vocabulary.json
  sampler_config: sampling.yaml
  output_dir: {(tmp_path / 'figure4').as_posix()}
reuse:
  delta_s: 0.25
  output_pattern: {reuse_pattern.as_posix()}
  experiment: table1
  condition: q80
""",
        encoding="utf-8",
    )
    calls: list[argparse.Namespace] = []
    monkeypatch.setattr(cli, "_sample_vadgm", lambda args: calls.append(args) or 0)
    assert cli._run_figure4(type("Args", (), {"config": config})()) == 0
    assert [(call.delta_s, call.seed) for call in calls] == [
        (0.5, 2027),
        (0.5, 2028),
        (0.125, 2027),
        (0.125, 2028),
    ]


@pytest.mark.parametrize(
    ("command", "config"),
    (
        ("run-table2", "configs/experiments/table2.yaml"),
        ("train-figure3", "configs/experiments/figure3.yaml"),
        ("run-figure3", "configs/experiments/figure3.yaml"),
        ("run-figure4", "configs/experiments/figure4.yaml"),
    ),
)
def test_remaining_paper_experiment_parser_defaults(command: str, config: str) -> None:
    args = build_parser().parse_args([command])
    assert str(args.config).replace("\\", "/") == config


def test_paper_figures_export_all_required_formats(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runs, vocabulary, manifest, output = _experiment_fixture(tmp_path)
    files = summarize_paper_experiments(runs, vocabulary, manifest, output)
    monkeypatch.setattr(plot_figure3, "RASTER_DPI", 72)
    monkeypatch.setattr(plot_figure4, "RASTER_DPI", 72)

    figure3_outputs = plot_figure3.plot_figure3(
        files["figure3"], files["representatives"], tmp_path / "figure3"
    )
    figure4_outputs = plot_figure4.plot_figure4(
        files["figure4"], tmp_path / "figure4"
    )
    for generated in figure3_outputs + figure4_outputs:
        path = Path(generated)
        assert path.suffix in {".svg", ".pdf", ".png", ".tiff"}
        assert path.exists() and path.stat().st_size > 0


def test_paper_summary_rejects_empty_input(tmp_path: Path) -> None:
    vocabulary = AtomVocabulary.load(_experiment_fixture(tmp_path)[1])
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="No completed experiment JSON"):
        summarize_paper_experiments(
            empty,
            tmp_path / "atom_vocabulary.json",
            tmp_path / "manifest.csv",
            tmp_path / "output",
        )
    assert vocabulary.clean_size > 0

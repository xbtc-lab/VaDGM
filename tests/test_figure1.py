from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
from matplotlib.colors import to_rgba
from matplotlib.text import Text

matplotlib.use("Agg")

from figures.figure1.plot_v11_terminal_reachability import build_figure as build_v11
from figures.figure1.plot_v3_conflict_anatomy import build_figure as build_v3
from figures.figure1.plot_v3b_conflict_composition import (
    build_figure as build_v3b,
)
from figures.figure1.plot_v3b_type_severity_map import build_figure as build_type_severity
from figures.figure1.plot_v3c_per_molecule_burden import build_figure as build_v3c
from figures.figure1.prepare_figure1_data import prepare_source_data
from figures.figure1.style import load_style, v3_section


def test_prepare_and_plot_ready_figure1_candidates(tmp_path: Path) -> None:
    raw = tmp_path / "run.json"
    raw.write_text(
        json.dumps(
            {
                "target": {"target": 3.0, "label": "Q80"},
                "seed": 12,
                "delta_s": 0.25,
                "samples": [
                    {
                        "sample_id": "a",
                        "diagnostics": {
                            "base_forwards": 2,
                            "path_capacity_violations": 0,
                            "final_capacity_safe": True,
                            "joint_conflicts": [
                                {
                                    "occurred": True,
                                    "conflict_type": "bond-bond",
                                    "proposal_count": 4,
                                    "affected_atoms": [0],
                                    "excess_by_atom": [[0, 1]],
                                },
                                {"occurred": False},
                            ],
                        },
                    },
                    {
                        "sample_id": "b",
                        "diagnostics": {
                            "base_forwards": 2,
                            "path_capacity_violations": 0,
                            "final_capacity_safe": True,
                            "joint_conflicts": [
                                {
                                    "occurred": True,
                                    "conflict_type": "multi-way",
                                    "proposal_count": 5,
                                    "affected_atoms": [0, 1],
                                    "excess_by_atom": [[0, 2], [1, 1]],
                                },
                                {"occurred": False},
                            ],
                        },
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    outputs = prepare_source_data(raw, tmp_path / "source")
    with outputs["run"].open("r", encoding="utf-8", newline="") as handle:
        run = next(csv.DictReader(handle))
    assert float(run["shadow_conflict_molecules_percent"]) == 100.0
    assert float(run["shadow_conflict_bins_percent"]) == 50.0

    metric_fields = {
        "samples": "2",
        "connected_percent": "75",
        "radical_free_percent": "50",
        "strict_vthr_percent": "25",
        "property_mae": "1.0",
        "property_hit_percent": "30",
        "mean_actual_nfe": "20",
    }
    before = tmp_path / "before.csv"
    after = tmp_path / "after.csv"
    for path, connected, radical_free, strict_vthr in (
        (before, "75", "50", "25"),
        (after, "90", "95", "40"),
    ):
        row = dict(metric_fields)
        row.update(
            connected_percent=connected,
            radical_free_percent=radical_free,
            strict_vthr_percent=strict_vthr,
        )
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)

    figures = [
        build_v3(outputs["sample"], outputs["conflict"]),
        build_v3b(outputs["conflict"]),
        build_type_severity(outputs["conflict"]),
        build_v3c(outputs["sample"]),
        build_v11(before, after),
    ]
    v3 = figures[0]
    assert [axis.get_title(loc="left") for axis in v3.axes] == ["", "", ""]
    figure_text = {text.get_text() for text in v3.texts}
    assert "b" in figure_text
    assert "Joint-conflict composition and severity" in figure_text
    assert "c" in figure_text
    assert "Per-molecule burden" in figure_text
    assert v3.axes[1].get_xlabel() == "Mean excess-capacity interval"
    assert v3.axes[2].get_ylabel() == "Molecules (%)"
    assert figures[1].axes[0].get_title(loc="left") == "Conflict composition"
    assert figures[2].axes[0].get_title(loc="left") == "Type–severity map"
    assert figures[3].axes[0].get_ylabel() == "Molecules (%)"
    assert "No conflict" not in {text.get_text() for text in figures[3].axes[0].texts}
    style = load_style()
    assert style.font_family == "Times New Roman"
    assert float(v3_section(style, "burden")["boundary_x"]) == 0.5
    assert set(style.v3_options) == {
        "combined",
        "composition",
        "type_severity",
        "burden",
    }
    for figure in figures[:4]:
        for text_artist in figure.findobj(match=Text):
            assert to_rgba(text_artist.get_color()) == to_rgba("black")
    for figure in figures:
        assert figure.axes
        matplotlib.pyplot.close(figure)

#!/usr/bin/env python3
"""Reproduce Table 2: Q80 component ablation study."""

from pathlib import Path
import pandas as pd

def main():
    root = Path(__file__).resolve().parent.parent
    data_path = root / "results" / "paper" / "source_data" / "table2_ablation.csv"
    
    if not data_path.exists():
        print(f"Error: Source data not found at {data_path}")
        return

    df = pd.read_csv(data_path)
    
    name_map = {
        "Vanilla DGM": "Vanilla",
        "VaDGM (capacity-only)": "C",
        "VaDGM (no terminal atom phase)": "C+R",
        "VaDGM": "C+R+A"
    }
    df["Setting"] = df["method"].map(name_map)
    df = df.sort_values(by="method", key=lambda x: x.map({"Vanilla DGM": 0, "VaDGM (capacity-only)": 1, "VaDGM (no terminal atom phase)": 2, "VaDGM": 3}))
    
    print("\n" + "=" * 65)
    print("Table 2. Q80 ablation study (Mean ± SD in percent)")
    print("=" * 65)
    header = f"{'Setting':<12} | {'Complete (%)':<16} | {'Relax. (%)':<16} | {'Strict VTHR (%)':<15}"
    print(header)
    print("-" * 65)
    
    for _, row in df.iterrows():
        comp = f"{row['terminal_valence_complete_percent_mean']:.2f} ± {row['terminal_valence_complete_percent_std']:.2f}"
        if row["Setting"] in ["Vanilla", "C"]:
            rel = "-"
        else:
            rel = f"{row['terminal_gate_relaxation_percent_mean']:.2f} ± {row['terminal_gate_relaxation_percent_std']:.2f}"
        vthr = f"{row['strict_vthr_percent_mean']:.2f} ± {row['strict_vthr_percent_std']:.2f}"
        print(f"{row['Setting']:<12} | {comp:<16} | {rel:<16} | {vthr:<15}")
        
    print("=" * 65 + "\n")

if __name__ == "__main__":
    main()

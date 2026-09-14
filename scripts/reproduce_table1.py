#!/usr/bin/env python3
"""Reproduce Table 1: Q80 logP comparison across methods."""

from pathlib import Path
import pandas as pd

def main():
    root = Path(__file__).resolve().parent.parent
    data_path = root / "results" / "paper" / "source_data" / "table1_main_results.csv"
    
    if not data_path.exists():
        print(f"Error: Source data not found at {data_path}")
        return

    df = pd.read_csv(data_path)
    
    # Method display mapping
    name_map = {
        "Vanilla DGM": "Vanilla DGM",
        "VaDGM (capacity-only)": "VaDGM (capacity-only)",
        "VaDGM": "VaDGM (full)"
    }
    df["Method"] = df["method"].map(name_map)
    df = df.sort_values(by="method", key=lambda x: x.map({"Vanilla DGM": 0, "VaDGM (capacity-only)": 1, "VaDGM": 2}))
    
    print("\n" + "=" * 105)
    print("Table 1. Q80 logP comparison across methods (Mean ± SD over 3 seeds, 10,000 outputs each)")
    print("=" * 105)
    header = f"{'Method':<25} | {'Connected (%)':<14} | {'Radical-free (%)':<18} | {'Strict VTHR (%)':<17} | {'MAE':<13} | {'Final viol. (%)':<16} | {'NFE':<10}"
    print(header)
    print("-" * 105)
    
    for _, row in df.iterrows():
        conn = f"{row['connected_percent_mean']:.2f} ± {row['connected_percent_std']:.2f}"
        rad = f"{row['radical_free_percent_mean']:.2f} ± {row['radical_free_percent_std']:.2f}"
        vthr = f"{row['strict_vthr_percent_mean']:.2f} ± {row['strict_vthr_percent_std']:.2f}"
        mae = f"{row['property_mae_mean']:.3f} ± {row['property_mae_std']:.3f}"
        viol = f"{row['final_capacity_violation_percent_mean']:.2f} ± {row['final_capacity_violation_percent_std']:.2f}"
        nfe = f"{row['mean_actual_nfe_mean']:.2f} ± {row['mean_actual_nfe_std']:.2f}"
        print(f"{row['Method']:<25} | {conn:<14} | {rad:<18} | {vthr:<17} | {mae:<13} | {viol:<16} | {nfe:<10}")
        
    print("=" * 105 + "\n")

if __name__ == "__main__":
    main()

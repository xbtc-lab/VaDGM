#!/usr/bin/env python3
"""Reproduce Table 3: Q80 bin-width sensitivity study."""

from pathlib import Path
import pandas as pd

def main():
    root = Path(__file__).resolve().parent.parent
    data_path = root / "results" / "paper" / "source_data" / "summary_metrics.csv"
    
    if not data_path.exists():
        print(f"Error: Source data not found at {data_path}")
        return

    df = pd.read_csv(data_path)
    fig4_df = df[df['experiment'] == 'figure4'].copy()
    
    # Delta s = 0.2500 is the default Q80 VaDGM setting from Table 1
    t1_vadgm = df[(df['experiment'] == 'table1') & (df['method'] == 'VaDGM')].copy()
    t1_vadgm['delta_s'] = 0.2500
    
    combined = pd.concat([fig4_df, t1_vadgm], ignore_index=True)
    combined = combined.sort_values(by='delta_s', ascending=False)
    
    print("\n" + "=" * 80)
    print("Table 3. Q80 bin-width sensitivity across four settings (Mean ± SD)")
    print("=" * 80)
    header = f"{'Δs':<8} | {'NFE':<15} | {'Shadow (%)':<16} | {'Connected (%)':<16} | {'Strict VTHR (%)':<15}"
    print(header)
    print("-" * 80)
    
    for _, row in combined.iterrows():
        ds = f"{row['delta_s']:.4f}"
        nfe = f"{row['mean_actual_nfe_mean']:.2f} ± {row['mean_actual_nfe_std']:.2f}"
        shad = f"{row['conflict_molecule_percent_mean']:.2f} ± {row['conflict_molecule_percent_std']:.2f}"
        conn = f"{row['connected_percent_mean']:.2f} ± {row['connected_percent_std']:.2f}"
        vthr = f"{row['strict_vthr_percent_mean']:.2f} ± {row['strict_vthr_percent_std']:.2f}"
        if ds == "0.2500":
            ds += " *" # Highlight default
        print(f"{ds:<8} | {nfe:<15} | {shad:<16} | {conn:<16} | {vthr:<15}")
        
    print("=" * 80)
    print("(* denotes the preset default setting in VaDGM; Path violations remain 0.00% everywhere)\n")

if __name__ == "__main__":
    main()

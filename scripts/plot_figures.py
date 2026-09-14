#!/usr/bin/env python3
"""Plot publication figures from experimental source data."""

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

def plot_figure4_sensitivity(source_path: Path, output_dir: Path):
    df = pd.read_csv(source_path)
    # Figure 4 sensitivity curves: delta_s vs NFE and Strict VTHR
    fig, ax1 = plt.subplots(figsize=(6, 4), dpi=300)
    
    delta_s = df['delta_s'].values
    nfe = df['mean_actual_nfe_mean'].values
    vthr = df['strict_vthr_percent_mean'].values
    shadow = df['conflict_molecule_percent_mean'].values
    
    color = 'tab:blue'
    ax1.set_xlabel(r'Hazard Bin Width ($\Delta s$)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Strict VTHR (%)', color=color, fontsize=11, fontweight='bold')
    line1 = ax1.plot(delta_s, vthr, marker='o', color=color, linewidth=2, label='Strict VTHR (%)')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Neural Evaluations (NFE)', color=color, fontsize=11, fontweight='bold')
    line2 = ax2.plot(delta_s, nfe, marker='s', color=color, linewidth=2, linestyle='--', label='NFE')
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title('VaDGM: Efficiency vs Target-Hit Trade-off across $\Delta s$', fontsize=12, fontweight='bold')
    fig.tight_layout()
    
    out_file = output_dir / "figure4_sensitivity.png"
    plt.savefig(out_file)
    plt.close()
    print(f"[OK] Saved Figure 4 sensitivity plot to: {out_file}")

def main():
    root = Path(__file__).resolve().parent.parent
    data_dir = root / "results" / "paper" / "source_data"
    output_dir = root / "results" / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig4_source = data_dir / "figure4_conflict_source.csv"
    if fig4_source.exists():
        plot_figure4_sensitivity(fig4_source, output_dir)
    else:
        print(f"[!] Source data not found at: {fig4_source}")

if __name__ == "__main__":
    main()

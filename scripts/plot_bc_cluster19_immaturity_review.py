#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TABLE_DIR = Path("results/tables/bc_cluster19_immaturity_review")
COUNTS_CSV = Path(
    "results/tables/bc_guca1b_gt2_mikiko_no_contam_26_28_gene_umaps/"
    "bc_guca1b_gt2_cluster_counts_by_renamed_samples.csv"
)
TOTAL_COUNTS_CSV = Path(
    "results/tables/bc_guca1b_gt2_mikiko_no_contam_26_28_gene_umaps/"
    "bc_guca1b_gt2_counts_by_renamed_samples.csv"
)
FIGURE_DIR = Path("results/figures/bc_cluster19_immaturity_review")
TARGET_CLUSTER = "19"


IMMATURE_GENES = [
    "stmn1b",
    "marcksl1b",
    "marcksl1a",
    "nfixa",
    "nfixb",
    "nfia",
    "id2a",
    "otx1",
    "hmgb2a",
    "hmgb2b",
    "neurod4",
]
CYCLING_GENES = ["mki67", "pcna", "top2a", "cdk1", "ccnd1", "ascl1a"]
MATURE_BC_GENES = ["cabp5a", "cabp5b", "grm6a", "grm6b", "trpm1a", "trpm1b", "nyx", "prkcaa"]


def scale_per_gene(df: pd.DataFrame) -> pd.DataFrame:
    scaled = df.copy()
    for col in scaled.columns:
        values = scaled[col].astype(float)
        lo = values.min()
        hi = values.max()
        if np.isfinite(hi - lo) and hi > lo:
            scaled[col] = (values - lo) / (hi - lo)
        else:
            scaled[col] = 0.0
    return scaled


def plot_dot_panel(
    ax: plt.Axes,
    mean_df: pd.DataFrame,
    pct_df: pd.DataFrame,
    genes: list[str],
    title: str,
    show_ylabel: bool = False,
) -> None:
    genes = [gene for gene in genes if gene in mean_df.columns]
    scaled = scale_per_gene(mean_df[genes])
    clusters = list(mean_df.index)
    x = np.arange(len(genes))
    y = np.arange(len(clusters))
    xx, yy = np.meshgrid(x, y)
    sizes = 6 + pct_df[genes].to_numpy().ravel() * 95
    sca = ax.scatter(
        xx.ravel(),
        yy.ravel(),
        c=scaled.to_numpy().ravel(),
        s=sizes,
        cmap="Reds",
        vmin=0,
        vmax=1,
        edgecolors="#d0d5dd",
        linewidths=0.25,
    )
    target_idx = clusters.index(TARGET_CLUSTER)
    ax.axhspan(target_idx - 0.5, target_idx + 0.5, color="#fff2a8", alpha=0.65, zorder=0)
    ax.set_title(title, fontsize=12, pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(genes, rotation=55, ha="right", fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels(clusters if show_ylabel else [], fontsize=8)
    ax.set_ylim(len(clusters) - 0.5, -0.5)
    ax.set_xlim(-0.6, len(genes) - 0.4)
    ax.grid(True, color="#eef2f6", linewidth=0.45)
    ax.set_axisbelow(True)
    if show_ylabel:
        ax.set_ylabel("BC cluster")
    return sca


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    stats = pd.read_csv(TABLE_DIR / "cluster_marker_expression_all_clusters.csv")
    stats["cluster"] = stats["cluster"].astype(str)
    clusters = sorted(stats["cluster"].unique(), key=int)
    mean_df = (
        stats.pivot(index="cluster", columns="gene", values="mean_expression")
        .reindex(clusters)
        .fillna(0.0)
    )
    pct_df = (
        stats.pivot(index="cluster", columns="gene", values="fraction_expressing")
        .reindex(clusters)
        .fillna(0.0)
    )

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(
        2,
        3,
        width_ratios=[1.25, 0.9, 0.95],
        height_ratios=[0.76, 0.24],
        wspace=0.25,
        hspace=0.36,
    )
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1], sharey=ax1)
    ax3 = fig.add_subplot(gs[0, 2], sharey=ax1)
    sca = plot_dot_panel(ax1, mean_df, pct_df, IMMATURE_GENES, "Immature / transitional markers", True)
    plot_dot_panel(ax2, mean_df, pct_df, CYCLING_GENES, "Cycling / progenitor markers")
    plot_dot_panel(ax3, mean_df, pct_df, MATURE_BC_GENES, "Mature BC markers")

    cbar = fig.colorbar(sca, ax=[ax1, ax2, ax3], fraction=0.018, pad=0.012)
    cbar.set_label("Mean expression scaled per gene", fontsize=9)

    legend_values = [0.25, 0.50, 0.75, 1.00]
    handles = [
        ax3.scatter([], [], s=6 + value * 95, color="#737373", edgecolors="#d0d5dd")
        for value in legend_values
    ]
    ax3.legend(
        handles,
        [f"{int(value * 100)}%" for value in legend_values],
        title="Fraction\nexpressing",
        loc="center left",
        bbox_to_anchor=(1.25, 0.4),
        frameon=False,
        fontsize=8,
        title_fontsize=8,
    )

    counts = pd.read_csv(COUNTS_CSV)
    total_counts = pd.read_csv(TOTAL_COUNTS_CSV)
    condition_order = ["Control", "LD", "NMDA"]
    cluster_row = counts[counts["cluster"].astype(str) == TARGET_CLUSTER].iloc[0]
    cluster_props = np.array([cluster_row[c] for c in condition_order], dtype=float)
    cluster_props = cluster_props / cluster_props.sum() * 100
    total_map = dict(zip(total_counts["sample"], total_counts["n_cells"]))
    total_props = np.array([total_map[c] for c in condition_order], dtype=float)
    total_props = total_props / total_props.sum() * 100

    ax4 = fig.add_subplot(gs[1, 0])
    xpos = np.arange(len(condition_order))
    width = 0.36
    ax4.bar(xpos - width / 2, total_props, width, label="All BC", color="#9aa4b2")
    ax4.bar(xpos + width / 2, cluster_props, width, label="Cluster 19", color="#d97706")
    ax4.set_xticks(xpos)
    ax4.set_xticklabels(condition_order)
    ax4.set_ylabel("Cells (%)")
    ax4.set_title("Condition composition")
    ax4.legend(frameon=False)
    ax4.spines[["top", "right"]].set_visible(False)

    ax5 = fig.add_subplot(gs[1, 1:])
    cluster19 = pd.read_csv(TABLE_DIR / "cluster19_marker_expression_summary.csv")
    selected = cluster19[
        cluster19["gene"].isin(["stmn1b", "nfixa", "id2a", "otx1", "mki67", "pcna", "cdk1"])
    ].copy()
    selected["label"] = selected.apply(
        lambda row: f"{row['gene']}  rank {int(row['mean_rank_desc'])}", axis=1
    )
    ax5.barh(selected["label"], selected["fraction_expressing"] * 100, color="#2563eb")
    ax5.set_xlabel("Cluster 19 cells expressing gene (%)")
    ax5.set_title("Cluster 19 marker highlights")
    ax5.invert_yaxis()
    ax5.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        "BC cluster 19 shows immature/transitional marker enrichment, not strong cycling",
        fontsize=16,
        y=0.98,
    )
    output = FIGURE_DIR / "bc_cluster19_immaturity_visual_summary.png"
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()

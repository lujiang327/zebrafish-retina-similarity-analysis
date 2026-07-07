#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc

from find_bc_guca1b_gt2_cluster_markers_ac_style import (
    compute_cluster_expression_stats,
    plot_top1_dotplot,
)


DEFAULT_H5AD = "results/h5ad/bc_9_sample_guca1b_gt2_mikiko_merged_clusters.h5ad"
DEFAULT_TABLES = "results/tables/bc_guca1b_gt2_mikiko_merged_gene_umaps"
DEFAULT_FIGURES = "results/figures/bc_guca1b_gt2_clusters_26_28_marker_review"
DEFAULT_OUT_TABLES = "results/tables/bc_guca1b_gt2_clusters_26_28_marker_review"
TARGET_CLUSTERS = ["26", "27", "28"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Review alternative DEG markers for BC merged clusters 26, 27, and 28."
    )
    parser.add_argument("--h5ad", default=DEFAULT_H5AD)
    parser.add_argument("--cluster-key", default="leiden")
    parser.add_argument("--source-tables-dir", default=DEFAULT_TABLES)
    parser.add_argument("--figures-dir", default=DEFAULT_FIGURES)
    parser.add_argument("--tables-dir", default=DEFAULT_OUT_TABLES)
    parser.add_argument("--top-n-source", type=int, default=50)
    parser.add_argument("--top-n-candidates", type=int, default=8)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def rank_candidate_markers(candidates: pd.DataFrame) -> pd.DataFrame:
    ranked = candidates.copy()
    for column in [
        "logfoldchanges",
        "pct_group_from_matrix",
        "mean_expr_group",
        "pct_gap_vs_max_other",
        "mean_expr_gap_vs_max_other",
        "max_other_pct_nz",
        "max_other_mean_expr",
        "specificity_score",
        "max_other_specificity_score",
        "scores",
    ]:
        if column not in ranked.columns:
            ranked[column] = 0.0
        ranked[column] = pd.to_numeric(ranked[column], errors="coerce").fillna(0.0)

    pct_gap = ranked["pct_gap_vs_max_other"].clip(lower=0)
    mean_gap = ranked["mean_expr_gap_vs_max_other"].clip(lower=0)
    target_strength = ranked["pct_group_from_matrix"].clip(lower=0) * ranked["mean_expr_group"].clip(lower=0)
    other_penalty = (ranked["max_other_pct_nz"].clip(lower=0) + 0.05) * (
        ranked["max_other_mean_expr"].clip(lower=0) + 0.05
    )
    ranked["clean_marker_score"] = (
        ranked["logfoldchanges"].clip(lower=0)
        * target_strength
        * pct_gap
        * mean_gap
        / other_penalty
    )
    ranked["clean_marker_pass"] = (
        (ranked["pct_group_from_matrix"] >= 0.10)
        & (ranked["pct_gap_vs_max_other"] > 0)
        & (ranked["mean_expr_gap_vs_max_other"] > 0)
    )
    ranked["other_expression_flag"] = np.select(
        [
            ranked["max_other_pct_nz"] >= 0.50,
            ranked["max_other_pct_nz"] >= 0.30,
            ranked["max_other_pct_nz"] >= 0.15,
        ],
        ["high_other_pct", "moderate_other_pct", "some_other_pct"],
        default="low_other_pct",
    )
    ranked["group_sort"] = ranked["group"].astype(int)
    return ranked.sort_values(
        [
            "group_sort",
            "clean_marker_pass",
            "clean_marker_score",
            "pct_gap_vs_max_other",
            "mean_expr_gap_vs_max_other",
            "max_other_specificity_score",
        ],
        ascending=[True, False, False, False, False, False],
    ).drop(columns="group_sort")


def candidate_rows(source: pd.DataFrame, target_clusters: list[str], top_n_source: int) -> pd.DataFrame:
    rows = []
    for cluster in target_clusters:
        subset = source[source["group"].astype(str) == cluster].head(top_n_source).copy()
        subset["source_top_n_limit"] = top_n_source
        rows.append(subset)
    return pd.concat(rows, ignore_index=True)


def label_top_candidates(df: pd.DataFrame, per_cluster: int) -> pd.DataFrame:
    selected = (
        df.groupby("group", observed=False)
        .head(per_cluster)
        .reset_index(drop=True)
    )
    selected["candidate_rank"] = selected.groupby("group", observed=False).cumcount() + 1
    return selected


def plot_candidate_dotplot(
    adata,
    cluster_key: str,
    candidates: pd.DataFrame,
    output_path: Path,
    dpi: int,
) -> None:
    plot_rows = candidates.copy()
    plot_rows["group"] = plot_rows["group"].astype(str)
    plot_rows["names"] = plot_rows["names"].astype(str)
    plot_rows["display_group"] = plot_rows["group"] + "." + plot_rows["candidate_rank"].astype(str)
    top_like = plot_rows.rename(columns={"display_group": "group"}).copy()
    top_like["group"] = plot_rows["display_group"]

    genes = top_like["names"].tolist()
    unique_genes = list(dict.fromkeys(genes))
    mean_df, pct_df = compute_cluster_expression_stats(adata, cluster_key, unique_genes)
    clusters = list(mean_df.columns)

    mean_columns = []
    pct_columns = []
    for gene in genes:
        values = mean_df.loc[gene, clusters].astype(float).to_numpy()
        value_range = float(np.nanmax(values) - np.nanmin(values))
        if np.isfinite(value_range) and value_range > 0:
            values = (values - float(np.nanmin(values))) / value_range
        else:
            values = np.zeros_like(values)
        mean_columns.append(values)
        pct_columns.append(pct_df.loc[gene, clusters].astype(float).to_numpy())

    import matplotlib.pyplot as plt

    color_values = np.vstack(mean_columns).T
    pct_values = np.vstack(pct_columns).T
    x = np.arange(len(genes))
    y = np.arange(len(clusters))
    xx, yy = np.meshgrid(x, y)
    sizes = 8 + pct_values.ravel() * 85

    fig, ax = plt.subplots(figsize=(max(14, 0.55 * len(genes) + 2), 9.2))
    scatter = ax.scatter(
        xx.ravel(),
        yy.ravel(),
        c=color_values.ravel(),
        s=sizes,
        cmap="Reds",
        vmin=0,
        vmax=1,
        edgecolors="#d0d5dd",
        linewidths=0.25,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(
        [
            f"{row.group} rank{row.candidate_rank}: {row.names}"
            for row in candidates.itertuples(index=False)
        ],
        rotation=90,
        ha="center",
        fontsize=8,
    )
    ax.set_yticks(y)
    ax.set_yticklabels(clusters, fontsize=9)
    ax.set_xlim(-0.75, len(genes) - 0.25)
    ax.set_ylim(len(clusters) - 0.5, -0.5)
    ax.grid(True, color="#eef2f6", linewidth=0.5)
    ax.set_axisbelow(True)
    ax.set_xlabel("Candidate markers for clusters 26-28")
    ax.set_ylabel(cluster_key)
    cbar = fig.colorbar(scatter, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Mean expression in group\nscaled per marker")
    for pct in [0.2, 0.4, 0.6, 0.8, 1.0]:
        ax.scatter([], [], s=8 + pct * 85, c="#7f7f7f", edgecolors="#4a4a4a", label=f"{int(pct * 100)}")
    ax.legend(
        title="Fraction of cells\nin group (%)",
        loc="center left",
        bbox_to_anchor=(1.08, 0.5),
        frameon=False,
    )
    fig.subplots_adjust(left=0.06, right=0.84, bottom=0.34, top=0.97)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", pad_inches=0.35)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    figures_dir = Path(args.figures_dir)
    tables_dir = Path(args.tables_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    source_tables = Path(args.source_tables_dir)
    max_other_path = source_tables / "bc_guca1b_gt2_deg_max_other_top100_markers.csv"
    top1_path = source_tables / "bc_guca1b_gt2_deg_max_other_top1_marker_per_cluster.csv"
    source = pd.read_csv(max_other_path)
    top1 = pd.read_csv(top1_path)

    candidates = candidate_rows(source, TARGET_CLUSTERS, args.top_n_source)
    ranked = rank_candidate_markers(candidates)
    ranked.to_csv(tables_dir / "clusters_26_28_top50_candidates_ranked_clean_marker_score.csv", index=False)

    selected = label_top_candidates(ranked, args.top_n_candidates)
    selected.to_csv(tables_dir / "clusters_26_28_selected_candidate_markers.csv", index=False)

    replacements = selected.groupby("group", observed=False).head(1).copy()
    replacement_map = {str(row.group): str(row.names) for row in replacements.itertuples(index=False)}

    replacement_top1 = top1.copy()
    replacement_top1["original_names"] = replacement_top1["names"]
    replacement_top1["marker_replaced_for_review"] = False
    for cluster, gene in replacement_map.items():
        source_row = selected[(selected["group"].astype(str) == cluster) & (selected["names"].astype(str) == gene)].iloc[0]
        mask = replacement_top1["group"].astype(str) == cluster
        for column in selected.columns:
            if column in replacement_top1.columns:
                replacement_top1.loc[mask, column] = source_row[column]
        replacement_top1.loc[mask, "names"] = gene
        replacement_top1.loc[mask, "marker_replaced_for_review"] = True
    replacement_top1.to_csv(tables_dir / "bc_guca1b_gt2_deg_max_other_top1_with_clusters_26_28_replaced.csv", index=False)

    adata = sc.read_h5ad(args.h5ad)
    plot_candidate_dotplot(
        adata,
        args.cluster_key,
        selected,
        figures_dir / "clusters_26_28_top_candidate_marker_dotplot.png",
        args.dpi,
    )
    plot_top1_dotplot(
        adata,
        replacement_top1,
        args.cluster_key,
        figures_dir / "max_other_top1_dotplot_clusters_26_28_replaced.png",
        args.dpi,
    )

    print(tables_dir / "clusters_26_28_top50_candidates_ranked_clean_marker_score.csv")
    print(tables_dir / "clusters_26_28_selected_candidate_markers.csv")
    print(figures_dir / "clusters_26_28_top_candidate_marker_dotplot.png")
    print(figures_dir / "max_other_top1_dotplot_clusters_26_28_replaced.png")


if __name__ == "__main__":
    main()

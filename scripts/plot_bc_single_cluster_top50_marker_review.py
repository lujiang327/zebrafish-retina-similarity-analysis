#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc

from find_bc_guca1b_gt2_cluster_markers_ac_style import compute_cluster_expression_stats


DEFAULT_H5AD = "results/h5ad/bc_9_sample_guca1b_gt2_mikiko_merged_clusters.h5ad"
DEFAULT_SOURCE = (
    "results/tables/bc_guca1b_gt2_mikiko_merged_gene_umaps/"
    "bc_guca1b_gt2_deg_max_other_top100_markers.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot top DEG candidates for one BC cluster.")
    parser.add_argument("--h5ad", default=DEFAULT_H5AD)
    parser.add_argument("--source-csv", default=DEFAULT_SOURCE)
    parser.add_argument("--cluster-key", default="leiden")
    parser.add_argument("--target-cluster", default="26")
    parser.add_argument("--top-n", type=int, default=50)
    parser.add_argument("--figures-dir", default="results/figures/bc_guca1b_gt2_single_cluster_marker_review")
    parser.add_argument("--tables-dir", default="results/tables/bc_guca1b_gt2_single_cluster_marker_review")
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def add_clean_metrics(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for column in [
        "logfoldchanges",
        "scores",
        "pct_group_from_matrix",
        "max_other_pct_nz",
        "pct_gap_vs_max_other",
        "mean_expr_group",
        "max_other_mean_expr",
        "mean_expr_gap_vs_max_other",
        "max_other_specificity_score",
    ]:
        if column not in out.columns:
            out[column] = 0.0
        out[column] = pd.to_numeric(out[column], errors="coerce").fillna(0.0)

    target_strength = out["pct_group_from_matrix"].clip(lower=0) * out["mean_expr_group"].clip(lower=0)
    clean_gap = out["pct_gap_vs_max_other"].clip(lower=0) * out["mean_expr_gap_vs_max_other"].clip(lower=0)
    other_penalty = (out["max_other_pct_nz"].clip(lower=0) + 0.05) * (
        out["max_other_mean_expr"].clip(lower=0) + 0.05
    )
    out["clean_background_score"] = (
        out["logfoldchanges"].clip(lower=0) * target_strength * clean_gap / other_penalty
    )
    out["clean_background_rank"] = out.sort_values(
        [
            "max_other_pct_nz",
            "max_other_mean_expr",
            "pct_gap_vs_max_other",
            "mean_expr_gap_vs_max_other",
            "scores",
        ],
        ascending=[True, True, False, False, False],
    ).reset_index().index + 1
    return out


def add_expression_specificity_metrics(adata, cluster_key: str, df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    genes = out["names"].astype(str).drop_duplicates().tolist()
    mean_df, pct_df = compute_cluster_expression_stats(adata, cluster_key, genes)
    mean_df.columns = mean_df.columns.astype(str)
    pct_df.columns = pct_df.columns.astype(str)
    clusters = list(mean_df.columns)
    rows = []
    for row in out.to_dict(orient="records"):
        gene = str(row["names"])
        group = str(int(float(row["group"])))
        try:
            other_clusters = [cluster for cluster in clusters if cluster != group]
            other_means = mean_df.loc[gene, other_clusters]
            other_pcts = pct_df.loc[gene, other_clusters]
            max_mean_cluster = str(other_means.idxmax())
            max_pct_cluster = str(other_pcts.idxmax())
            target_mean = float(mean_df.at[gene, group])
            target_pct = float(pct_df.at[gene, group])
            max_mean = float(other_means.loc[max_mean_cluster])
            max_pct = float(other_pcts.loc[max_pct_cluster])
        except KeyError:
            rows.append({})
            continue
        rows.append(
            {
                "pct_group_from_matrix": target_pct,
                "max_other_pct_nz": max_pct,
                "max_other_pct_cluster": max_pct_cluster,
                "pct_gap_vs_max_other": target_pct - max_pct,
                "mean_expr_group": target_mean,
                "max_other_mean_expr": max_mean,
                "max_other_mean_cluster": max_mean_cluster,
                "mean_expr_gap_vs_max_other": target_mean - max_mean,
            }
        )
    metrics = pd.DataFrame(rows)
    for column in metrics.columns:
        out[column] = metrics[column].to_numpy()
    out["max_other_specificity_score"] = (
        pd.to_numeric(out.get("logfoldchanges", 0), errors="coerce").fillna(0).clip(lower=0)
        * out["pct_gap_vs_max_other"].fillna(0).clip(lower=0)
        * out["mean_expr_gap_vs_max_other"].fillna(0).clip(lower=0)
    )
    return out


def sort_clean(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(
        [
            "max_other_pct_nz",
            "max_other_mean_expr",
            "pct_gap_vs_max_other",
            "mean_expr_gap_vs_max_other",
            "clean_background_score",
            "scores",
        ],
        ascending=[True, True, False, False, False, False],
    ).reset_index(drop=True)


def plot_dotplot(
    adata,
    cluster_key: str,
    target_cluster: str,
    candidates: pd.DataFrame,
    output_path: Path,
    title: str,
    dpi: int,
) -> None:
    genes = candidates["names"].astype(str).tolist()
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

    color_values = np.vstack(mean_columns).T
    pct_values = np.vstack(pct_columns).T
    x = np.arange(len(genes))
    y = np.arange(len(clusters))
    xx, yy = np.meshgrid(x, y)
    sizes = 6 + pct_values.ravel() * 80

    fig, ax = plt.subplots(figsize=(max(22, 0.42 * len(genes) + 4), 9.5))
    scatter = ax.scatter(
        xx.ravel(),
        yy.ravel(),
        c=color_values.ravel(),
        s=sizes,
        cmap="Reds",
        vmin=0,
        vmax=1,
        edgecolors="#d0d5dd",
        linewidths=0.2,
    )
    labels = [
        f"{i + 1}: {row.names}"
        for i, row in enumerate(candidates.itertuples(index=False))
    ]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90, ha="center", fontsize=7)
    ax.set_yticks(y)
    ax.set_yticklabels(clusters, fontsize=9)
    for tick in ax.get_yticklabels():
        if tick.get_text() == str(target_cluster):
            tick.set_color("#b42318")
            tick.set_fontweight("bold")
    target_index = clusters.index(str(target_cluster))
    ax.axhspan(target_index - 0.5, target_index + 0.5, color="#fff3cd", alpha=0.45, zorder=0)
    ax.set_xlim(-0.75, len(genes) - 0.25)
    ax.set_ylim(len(clusters) - 0.5, -0.5)
    ax.grid(True, color="#eef2f6", linewidth=0.5)
    ax.set_axisbelow(True)
    ax.set_xlabel(f"Top {len(genes)} candidate DEG genes for cluster {target_cluster}")
    ax.set_ylabel(cluster_key)
    ax.set_title(title, fontsize=14, pad=12)

    cbar = fig.colorbar(scatter, ax=ax, fraction=0.018, pad=0.018)
    cbar.set_label("Mean expression in group\nscaled per marker")
    for pct in [0.2, 0.4, 0.6, 0.8, 1.0]:
        ax.scatter([], [], s=6 + pct * 80, c="#7f7f7f", edgecolors="#4a4a4a", label=f"{int(pct * 100)}")
    ax.legend(
        title="Fraction of cells\nin group (%)",
        loc="center left",
        bbox_to_anchor=(1.06, 0.5),
        frameon=False,
    )
    fig.subplots_adjust(left=0.055, right=0.86, bottom=0.36, top=0.94)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", pad_inches=0.35)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    figures_dir = Path(args.figures_dir)
    tables_dir = Path(args.tables_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    adata = sc.read_h5ad(args.h5ad)
    source = pd.read_csv(args.source_csv)
    target = source[source["group"].astype(str) == str(args.target_cluster)].head(args.top_n).copy()
    target["original_top50_rank"] = np.arange(1, len(target) + 1)
    target = add_expression_specificity_metrics(adata, args.cluster_key, target)
    target = add_clean_metrics(target)
    clean_sorted = sort_clean(target)
    clean_sorted["clean_order_rank"] = np.arange(1, len(clean_sorted) + 1)

    table_path = tables_dir / f"cluster_{args.target_cluster}_top{args.top_n}_candidate_markers.csv"
    clean_sorted.to_csv(table_path, index=False)

    plot_dotplot(
        adata,
        args.cluster_key,
        args.target_cluster,
        target,
        figures_dir / f"cluster_{args.target_cluster}_top{args.top_n}_deg_order_dotplot.png",
        f"Cluster {args.target_cluster} top {args.top_n} DEG candidates, original max-other order",
        args.dpi,
    )
    plot_dotplot(
        adata,
        args.cluster_key,
        args.target_cluster,
        clean_sorted,
        figures_dir / f"cluster_{args.target_cluster}_top{args.top_n}_clean_background_order_dotplot.png",
        f"Cluster {args.target_cluster} top {args.top_n} DEG candidates, cleaner-background order",
        args.dpi,
    )

    print(table_path)
    print(figures_dir / f"cluster_{args.target_cluster}_top{args.top_n}_deg_order_dotplot.png")
    print(figures_dir / f"cluster_{args.target_cluster}_top{args.top_n}_clean_background_order_dotplot.png")


if __name__ == "__main__":
    main()

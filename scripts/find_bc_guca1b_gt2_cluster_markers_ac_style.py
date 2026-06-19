#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse


DEFAULT_H5AD = (
    "results/h5ad/"
    "bc_9_sample_guca1b_gt2_filtered.h5ad"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Find top marker genes for final BC guca1b > 2.0 clusters.")
    parser.add_argument("--h5ad", default=DEFAULT_H5AD, help="Input BC AnnData h5ad file.")
    parser.add_argument(
        "--cluster-key",
        default="leiden",
        help="BC cluster column in adata.obs.",
    )
    parser.add_argument(
        "--figures-dir",
        default="results/figures/bc_guca1b_gt2_ac_style_gene_umaps",
        help="Output figure directory.",
    )
    parser.add_argument(
        "--tables-dir",
        default="results/tables/bc_guca1b_gt2_ac_style_gene_umaps",
        help="Output table directory.",
    )
    parser.add_argument("--method", default="wilcoxon", help="Scanpy rank_genes_groups method.")
    parser.add_argument("--top-n", type=int, default=20, help="Top markers to keep per cluster.")
    parser.add_argument("--pval-adj", type=float, default=0.05, help="Adjusted p-value cutoff.")
    parser.add_argument("--min-logfc", type=float, default=0.25, help="Minimum log fold-change.")
    parser.add_argument(
        "--min-pct-group",
        type=float,
        default=0.10,
        help="Minimum fraction of expressing cells in the target cluster.",
    )
    parser.add_argument("--dpi", type=int, default=300, help="Output figure DPI.")
    return parser.parse_args()


def numeric_sort(values) -> list[str]:
    return sorted([str(value) for value in values], key=lambda value: int(value))


def pct_group_column(df: pd.DataFrame) -> str | None:
    for column in ["pct_nz_group", "pts"]:
        if column in df.columns:
            return column
    return None


def filter_markers(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    filtered = df.copy()
    if "pvals_adj" in filtered.columns:
        filtered = filtered[filtered["pvals_adj"] <= args.pval_adj]
    if "logfoldchanges" in filtered.columns:
        filtered = filtered[filtered["logfoldchanges"] >= args.min_logfc]
    pct_col = pct_group_column(filtered)
    if pct_col is not None:
        filtered = filtered[filtered[pct_col] >= args.min_pct_group]
    return filtered


def add_specificity_columns(df: pd.DataFrame) -> pd.DataFrame:
    annotated = df.copy()
    if "pct_nz_group" in annotated.columns and "pct_nz_reference" in annotated.columns:
        annotated["pct_diff"] = annotated["pct_nz_group"] - annotated["pct_nz_reference"]
        annotated["pct_ratio"] = (annotated["pct_nz_group"] + 0.01) / (
            annotated["pct_nz_reference"] + 0.01
        )
    else:
        annotated["pct_diff"] = 0.0
        annotated["pct_ratio"] = 1.0
    if "logfoldchanges" in annotated.columns:
        annotated["specificity_score"] = annotated["logfoldchanges"] * annotated["pct_diff"].clip(lower=0)
    else:
        annotated["specificity_score"] = annotated["pct_diff"].clip(lower=0)
    return annotated


def top_per_cluster(df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    ranked = df.copy()
    ranked["group_sort"] = ranked["group"].astype(int)
    return (
        ranked.sort_values(["group_sort", "scores"], ascending=[True, False])
        .groupby("group", observed=False)
        .head(top_n)
        .drop(columns=["group_sort"])
        .reset_index(drop=True)
    )


def top_specific_per_cluster(df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    ranked = add_specificity_columns(df)
    ranked["group_sort"] = ranked["group"].astype(int)
    return (
        ranked.sort_values(
            ["group_sort", "specificity_score", "logfoldchanges", "pct_diff", "scores"],
            ascending=[True, False, False, False, False],
        )
        .groupby("group", observed=False)
        .head(top_n)
        .drop(columns=["group_sort"])
        .reset_index(drop=True)
    )


def expression_source(adata):
    if adata.raw is not None:
        return adata.raw.X, pd.Index(adata.raw.var_names)
    return adata.X, pd.Index(adata.var_names)


def compute_cluster_expression_stats(
    adata,
    cluster_key: str,
    genes: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    matrix, var_names = expression_source(adata)
    present_genes = [gene for gene in genes if gene in var_names]
    gene_indices = var_names.get_indexer(present_genes)
    subset = matrix[:, gene_indices]
    cluster_labels = adata.obs[cluster_key].astype(str)
    clusters = numeric_sort(cluster_labels.unique())

    mean_rows = []
    pct_rows = []
    for cluster in clusters:
        mask = (cluster_labels == cluster).to_numpy()
        cluster_matrix = subset[mask, :]
        if sparse.issparse(cluster_matrix):
            means = np.asarray(cluster_matrix.mean(axis=0)).ravel()
            pcts = cluster_matrix.getnnz(axis=0) / cluster_matrix.shape[0]
        else:
            array = np.asarray(cluster_matrix)
            means = array.mean(axis=0)
            pcts = (array > 0).mean(axis=0)
        mean_rows.append(means)
        pct_rows.append(pcts)

    mean_df = pd.DataFrame(np.vstack(mean_rows).T, index=present_genes, columns=clusters)
    pct_df = pd.DataFrame(np.vstack(pct_rows).T, index=present_genes, columns=clusters)
    return mean_df, pct_df


def annotate_max_other_specificity(
    markers: pd.DataFrame,
    mean_df: pd.DataFrame,
    pct_df: pd.DataFrame,
) -> pd.DataFrame:
    annotated = add_specificity_columns(markers)
    clusters = list(mean_df.columns)
    rows = []
    for row in annotated.itertuples(index=False):
        gene = str(row.names)
        group = str(row.group)
        if gene not in mean_df.index or group not in clusters:
            rows.append({})
            continue
        other_clusters = [cluster for cluster in clusters if cluster != group]
        target_mean = float(mean_df.at[gene, group])
        target_pct = float(pct_df.at[gene, group])
        other_means = mean_df.loc[gene, other_clusters]
        other_pcts = pct_df.loc[gene, other_clusters]
        max_mean_cluster = str(other_means.idxmax())
        max_pct_cluster = str(other_pcts.idxmax())
        max_other_mean = float(other_means.loc[max_mean_cluster])
        max_other_pct = float(other_pcts.loc[max_pct_cluster])
        rows.append(
            {
                "mean_expr_group": target_mean,
                "max_other_mean_expr": max_other_mean,
                "max_other_mean_cluster": max_mean_cluster,
                "mean_expr_gap_vs_max_other": target_mean - max_other_mean,
                "pct_group_from_matrix": target_pct,
                "max_other_pct_nz": max_other_pct,
                "max_other_pct_cluster": max_pct_cluster,
                "pct_gap_vs_max_other": target_pct - max_other_pct,
            }
        )

    max_other = pd.DataFrame(rows)
    annotated = pd.concat([annotated.reset_index(drop=True), max_other], axis=1)
    pct_gap = annotated["pct_gap_vs_max_other"].fillna(0).clip(lower=0)
    mean_gap = annotated["mean_expr_gap_vs_max_other"].fillna(0).clip(lower=0)
    if "logfoldchanges" in annotated.columns:
        annotated["max_other_specificity_score"] = annotated["logfoldchanges"] * pct_gap * mean_gap
    else:
        annotated["max_other_specificity_score"] = pct_gap * mean_gap
    return annotated


def top_max_other_per_cluster(df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    ranked = df.copy()
    ranked["group_sort"] = ranked["group"].astype(int)
    return (
        ranked.sort_values(
            [
                "group_sort",
                "max_other_specificity_score",
                "pct_gap_vs_max_other",
                "mean_expr_gap_vs_max_other",
                "specificity_score",
                "scores",
            ],
            ascending=[True, False, False, False, False, False],
        )
        .groupby("group", observed=False)
        .head(top_n)
        .drop(columns=["group_sort"])
        .reset_index(drop=True)
    )


def plot_top1_dotplot(adata, top1: pd.DataFrame, cluster_key: str, output_path: Path, dpi: int) -> None:
    ordered = top1.copy()
    ordered["group_sort"] = ordered["group"].astype(int)
    ordered = ordered.sort_values("group_sort").drop(columns=["group_sort"]).reset_index(drop=True)
    genes = ordered["names"].astype(str).tolist()
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
    sizes = 8 + (pct_values.ravel() * 85)

    width = max(18.0, 0.45 * len(genes) + 2.0)
    height = max(8.8, 0.26 * len(clusters) + 1.4)
    fig, ax = plt.subplots(figsize=(width, height))
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
        [f"{row.group}: {row.names}" for row in ordered.itertuples(index=False)],
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
    ax.set_xlabel("Top1 marker per cluster")
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
        labelspacing=0.9,
    )
    fig.subplots_adjust(left=0.06, right=0.84, bottom=0.28, top=0.97)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", pad_inches=0.35)
    plt.close(fig)


def plot_cluster_umap(adata, cluster_key: str, output_path: Path, dpi: int) -> None:
    fig = sc.pl.umap(
        adata,
        color=cluster_key,
        legend_loc="on data",
        title="BC guca1b > 2.0 clusters",
        size=10,
        frameon=True,
        show=False,
        return_fig=True,
    )
    fig.set_size_inches(8.95, 6.0)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    figures_dir = Path(args.figures_dir)
    tables_dir = Path(args.tables_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    adata = sc.read_h5ad(args.h5ad)
    if args.cluster_key not in adata.obs:
        raise KeyError(f"Cluster key not found in adata.obs: {args.cluster_key}")

    sc.tl.rank_genes_groups(
        adata,
        groupby=args.cluster_key,
        method=args.method,
        use_raw=adata.raw is not None,
        pts=True,
    )
    all_markers = sc.get.rank_genes_groups_df(adata, group=None)
    all_markers["group"] = all_markers["group"].astype(str)
    filtered = filter_markers(all_markers, args)

    top_markers = top_per_cluster(filtered, args.top_n)
    top1 = top_per_cluster(filtered, 1)
    specific_top_markers = top_specific_per_cluster(filtered, args.top_n)
    specific_top1 = top_specific_per_cluster(filtered, 1)

    filtered_genes = filtered["names"].astype(str).drop_duplicates().tolist()
    mean_df, pct_df = compute_cluster_expression_stats(adata, args.cluster_key, filtered_genes)
    max_other_markers = annotate_max_other_specificity(filtered, mean_df, pct_df)
    max_other_top_markers = top_max_other_per_cluster(max_other_markers, args.top_n)
    max_other_top1 = top_max_other_per_cluster(max_other_markers, 1)

    prefix = "bc_guca1b_gt2_deg"
    all_markers.to_csv(tables_dir / f"{prefix}_markers_all.csv", index=False)
    filtered.to_csv(tables_dir / f"{prefix}_markers_filtered.csv", index=False)
    top_markers.to_csv(tables_dir / f"{prefix}_score_top{args.top_n}_markers.csv", index=False)
    top1.to_csv(tables_dir / f"{prefix}_score_top1_marker_per_cluster.csv", index=False)
    specific_top_markers.to_csv(
        tables_dir / f"{prefix}_specific_top{args.top_n}_markers.csv",
        index=False,
    )
    specific_top1.to_csv(tables_dir / f"{prefix}_specific_top1_marker_per_cluster.csv", index=False)
    max_other_markers.to_csv(tables_dir / f"{prefix}_max_other_annotated_filtered.csv", index=False)
    max_other_top_markers.to_csv(
        tables_dir / f"{prefix}_max_other_top{args.top_n}_markers.csv",
        index=False,
    )
    max_other_top1.to_csv(tables_dir / f"{prefix}_max_other_top1_marker_per_cluster.csv", index=False)
    (
        adata.obs[args.cluster_key]
        .astype(str)
        .value_counts()
        .rename_axis("cluster")
        .reset_index(name="n_cells")
        .assign(cluster_sort=lambda df: df["cluster"].astype(int))
        .sort_values("cluster_sort")
        .drop(columns="cluster_sort")
        .to_csv(tables_dir / f"{prefix}_cluster_cell_counts.csv", index=False)
    )
    pd.DataFrame(
        [
            {
                "method": args.method,
                "use_raw": adata.raw is not None,
                "pval_adj": args.pval_adj,
                "min_logfc": args.min_logfc,
                "min_pct_group": args.min_pct_group,
                "top_n": args.top_n,
                "cluster_key": args.cluster_key,
                "specificity_score": "logfoldchanges * max(pct_nz_group - pct_nz_reference, 0)",
                "max_other_specificity_score": (
                    "logfoldchanges * max(pct_group - max_other_pct, 0) "
                    "* max(mean_group - max_other_mean, 0)"
                ),
            }
        ]
    ).to_csv(tables_dir / f"{prefix}_parameters.csv", index=False)

    plot_cluster_umap(
        adata,
        args.cluster_key,
        figures_dir / "bc_umap_by_leiden.png",
        args.dpi,
    )
    plot_top1_dotplot(
        adata,
        top1,
        args.cluster_key,
        figures_dir / f"{prefix}_score_top1_marker_dotplot.png",
        args.dpi,
    )
    plot_top1_dotplot(
        adata,
        specific_top1,
        args.cluster_key,
        figures_dir / f"{prefix}_specific_top1_marker_dotplot.png",
        args.dpi,
    )
    plot_top1_dotplot(
        adata,
        max_other_top1,
        args.cluster_key,
        figures_dir / f"{prefix}_max_other_top1_marker_dotplot.png",
        args.dpi,
    )
    print(tables_dir / f"{prefix}_max_other_top1_marker_per_cluster.csv")


if __name__ == "__main__":
    main()

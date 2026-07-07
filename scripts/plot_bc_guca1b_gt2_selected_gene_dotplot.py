#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse


DEFAULT_H5AD = "results/h5ad/bc_9_sample_guca1b_gt2_filtered.h5ad"
DEFAULT_GENES = [
    "isl1",
    "gram6a",
    "gram6b",
    "trpm1a",
    "trpm1b",
    "prkcaa",
    "grik1a",
    "grik1b",
    "grik4",
]
ALIASES = {
    "gram6a": "grm6a",
    "gram6b": "grm6b",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot selected BC marker gene expression dotplot by Leiden cluster."
    )
    parser.add_argument("--h5ad", default=DEFAULT_H5AD)
    parser.add_argument("--cluster-key", default="leiden")
    parser.add_argument(
        "--figures-dir",
        default="results/figures/bc_guca1b_gt2_selected_marker_dotplot",
    )
    parser.add_argument(
        "--tables-dir",
        default="results/tables/bc_guca1b_gt2_selected_marker_dotplot",
    )
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--genes", nargs="+", default=DEFAULT_GENES)
    parser.add_argument("--cluster-order", nargs="+", default=None)
    parser.add_argument("--preserve-gene-order", action="store_true")
    parser.add_argument("--keep-duplicate-genes", action="store_true")
    parser.add_argument("--title", default="BC guca1b > 2.0 selected marker gene dotplot")
    parser.add_argument("--output-name", default="bc_guca1b_gt2_selected_marker_gene_dotplot.png")
    return parser.parse_args()


def numeric_sort(values) -> list[str]:
    return sorted([str(value) for value in values], key=lambda value: int(value))


def expression_matrix_and_names(adata):
    if adata.raw is not None:
        return adata.raw.X, pd.Index(adata.raw.var_names), "raw"
    return adata.X, pd.Index(adata.var_names), "X"


def gene_lookup(var_names: pd.Index) -> dict[str, str]:
    return {str(gene).lower(): str(gene) for gene in var_names}


def resolve_genes(
    requested_genes: list[str],
    var_names: pd.Index,
    keep_duplicates: bool = False,
) -> tuple[list[str], pd.DataFrame]:
    lookup = gene_lookup(var_names)
    records = []
    matched_genes = []
    for requested in requested_genes:
        requested_key = requested.lower()
        alias = ""
        matched = lookup.get(requested_key)
        if matched is None and requested_key in ALIASES:
            alias = ALIASES[requested_key]
            matched = lookup.get(alias.lower())
        records.append(
            {
                "requested_gene": requested,
                "alias_used": alias,
                "matched_gene": matched or "",
                "present": matched is not None,
            }
        )
        if matched is not None and (keep_duplicates or matched not in matched_genes):
            matched_genes.append(matched)
    return matched_genes, pd.DataFrame(records)


def compute_cluster_stats(
    adata,
    cluster_key: str,
    genes: list[str],
    cluster_order: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    matrix, var_names, _source = expression_matrix_and_names(adata)
    indices = var_names.get_indexer(genes)
    subset = matrix[:, indices]
    labels = adata.obs[cluster_key].astype(str)
    available_clusters = set(labels.unique())
    if cluster_order is None:
        clusters = numeric_sort(available_clusters)
    else:
        clusters = [str(cluster) for cluster in cluster_order]
        missing_clusters = [cluster for cluster in clusters if cluster not in available_clusters]
        if missing_clusters:
            raise ValueError(f"Requested cluster labels were not found: {', '.join(missing_clusters)}")

    mean_rows = []
    pct_rows = []
    for cluster in clusters:
        mask = (labels == cluster).to_numpy()
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

    mean_df = pd.DataFrame(np.vstack(mean_rows), index=clusters, columns=genes)
    pct_df = pd.DataFrame(np.vstack(pct_rows), index=clusters, columns=genes)
    return mean_df, pct_df


def order_genes_by_peak_cluster(mean_df: pd.DataFrame, genes: list[str]) -> list[str]:
    cluster_order = {cluster: idx for idx, cluster in enumerate(mean_df.index)}
    return sorted(
        genes,
        key=lambda gene: (
            cluster_order[str(mean_df[gene].idxmax())],
            -float(mean_df[gene].max()),
            gene,
        ),
    )


def plot_dotplot(
    mean_df: pd.DataFrame,
    pct_df: pd.DataFrame,
    genes: list[str],
    output_path: Path,
    title: str,
    dpi: int,
) -> None:
    mean_values = np.column_stack([mean_df[gene].to_numpy(dtype=float) for gene in genes])
    pct_values = np.column_stack([pct_df[gene].to_numpy(dtype=float) for gene in genes])
    scaled_values = mean_values.copy()
    for idx, gene in enumerate(genes):
        values = scaled_values[:, idx]
        value_range = float(np.nanmax(values) - np.nanmin(values))
        if np.isfinite(value_range) and value_range > 0:
            scaled_values[:, idx] = (values - float(np.nanmin(values))) / value_range
        else:
            scaled_values[:, idx] = 0.0

    clusters = list(mean_df.index)
    x = np.arange(len(genes))
    y = np.arange(len(clusters))
    xx, yy = np.meshgrid(x, y)
    sizes = 5 + pct_values.ravel() * 95

    width = max(8.5, 0.55 * len(genes) + 3.0)
    height = max(8.0, 0.24 * len(clusters) + 1.8)
    fig, ax = plt.subplots(figsize=(width, height))
    scatter = ax.scatter(
        xx.ravel(),
        yy.ravel(),
        c=scaled_values.ravel(),
        s=sizes,
        cmap="Reds",
        vmin=0,
        vmax=1,
        edgecolors="#d0d5dd",
        linewidths=0.25,
    )

    ax.set_title(title, fontsize=14, pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(genes, rotation=45, ha="right", fontsize=10)
    ax.set_yticks(y)
    ax.set_yticklabels(clusters, fontsize=8)
    ax.set_xlabel("Gene")
    ax.set_ylabel("Leiden cluster")
    ax.set_xlim(-0.6, len(genes) - 0.4)
    ax.set_ylim(len(clusters) - 0.5, -0.5)
    ax.grid(True, color="#eef2f6", linewidth=0.5)
    ax.set_axisbelow(True)

    colorbar = fig.colorbar(scatter, ax=ax, fraction=0.025, pad=0.02)
    colorbar.set_label("Mean expression scaled per gene", fontsize=9)

    legend_values = [0.25, 0.50, 0.75, 1.00]
    handles = [
        plt.scatter([], [], s=5 + value * 95, color="#737373", edgecolors="#d0d5dd")
        for value in legend_values
    ]
    ax.legend(
        handles,
        [f"{int(value * 100)}%" for value in legend_values],
        title="Fraction\nexpressing",
        loc="center left",
        bbox_to_anchor=(1.08, 0.35),
        frameon=False,
        fontsize=8,
        title_fontsize=8,
    )

    fig.tight_layout()
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
        raise ValueError(f"Cluster key {args.cluster_key!r} is not present in adata.obs")

    _matrix, var_names, source = expression_matrix_and_names(adata)
    genes, report = resolve_genes(args.genes, var_names, args.keep_duplicate_genes)
    report["expression_source"] = source
    report.to_csv(tables_dir / "bc_selected_marker_dotplot_gene_match_report.csv", index=False)
    if not genes:
        raise ValueError("None of the requested genes were found.")

    stats_genes = list(dict.fromkeys(genes))
    mean_df, pct_df = compute_cluster_stats(adata, args.cluster_key, stats_genes, args.cluster_order)
    ordered_genes = genes if args.preserve_gene_order else order_genes_by_peak_cluster(mean_df, genes)
    ordered_mean_df = pd.DataFrame(
        np.column_stack([mean_df[gene].to_numpy(dtype=float) for gene in ordered_genes]),
        index=mean_df.index,
        columns=ordered_genes,
    )
    ordered_pct_df = pd.DataFrame(
        np.column_stack([pct_df[gene].to_numpy(dtype=float) for gene in ordered_genes]),
        index=pct_df.index,
        columns=ordered_genes,
    )
    ordered_mean_df.to_csv(tables_dir / "bc_selected_marker_dotplot_mean_expression.csv")
    ordered_pct_df.to_csv(tables_dir / "bc_selected_marker_dotplot_fraction_expressing.csv")
    pd.DataFrame({"gene": ordered_genes}).to_csv(
        tables_dir / "bc_selected_marker_dotplot_gene_order.csv",
        index=False,
    )

    output_path = figures_dir / args.output_name
    plot_dotplot(
        mean_df,
        pct_df,
        ordered_genes,
        output_path,
        args.title,
        args.dpi,
    )
    print(output_path)
    print(tables_dir / "bc_selected_marker_dotplot_gene_match_report.csv")


if __name__ == "__main__":
    main()

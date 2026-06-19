#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import scanpy as sc


DEFAULT_H5AD = (
    "/Users/hoanglab/Desktop/vscode_projects/zebrafish-singlecell-portal/"
    "BC_annotated_clustered_corrected_doubletRemoved_Zebrafishes.h5ad"
)

DEFAULT_MARKER_GENES = [
    "Aqp4",
    "Arr3a",
    "Ascl1a",
    "Atoh7",
    "Cabp5a",
    "Cabp5b",
    "Calb1",
    "Calb2a",
    "Calb2b",
    "Ccnd1",
    "Cdk1",
    "Crx",
    "Elavl3",
    "Gfap",
    "Gnat2",
    "Guca1b",
    "Isl2b",
    "Mbpa",
    "Mitfa",
    "Mpeg1.1",
    "Neurod1",
    "Nr2e3",
    "Nrl",
    "Ompa",
    "Pde6a",
    "Rbpms2b",
    "Tie1",
    "acta2",
    "Vsx1",
    "Vsx2",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create quick review plots/counts for BC h5ad.")
    parser.add_argument("--h5ad", default=DEFAULT_H5AD, help="Input BC h5ad file.")
    parser.add_argument(
        "--figures-dir",
        default="results/figures/bc_h5ad_review",
        help="Output figure directory.",
    )
    parser.add_argument(
        "--tables-dir",
        default="results/tables/bc_h5ad_review",
        help="Output table directory.",
    )
    parser.add_argument("--genes", nargs="*", default=DEFAULT_MARKER_GENES, help="Marker genes to plot.")
    parser.add_argument("--dpi", type=int, default=300, help="Output figure DPI.")
    return parser.parse_args()


def count_table(adata, column: str, tables_dir: Path) -> pd.DataFrame:
    counts = (
        adata.obs[column]
        .astype(str)
        .value_counts()
        .rename_axis(column)
        .reset_index(name="n_cells")
    )
    counts.to_csv(tables_dir / f"counts_by_{column}.csv", index=False)
    return counts


def plot_counts(counts: pd.DataFrame, column: str, figures_dir: Path, dpi: int) -> None:
    plot_data = counts.copy()
    if len(plot_data) > 12:
        plot_data = plot_data.sort_values(column, key=lambda s: pd.to_numeric(s, errors="coerce"))
    fig_width = max(6, min(16, 0.32 * len(plot_data) + 3))
    fig, ax = plt.subplots(figsize=(fig_width, 4.2))
    ax.bar(plot_data[column].astype(str), plot_data["n_cells"], color="#4C78A8")
    ax.set_title(f"Cells by {column}")
    ax.set_xlabel(column)
    ax.set_ylabel("Number of cells")
    ax.tick_params(axis="x", rotation=90 if len(plot_data) > 8 else 0)
    fig.tight_layout()
    fig.savefig(figures_dir / f"bc_counts_by_{column}.png", dpi=dpi)
    plt.close(fig)


def plot_umap(adata, column: str, figures_dir: Path, dpi: int, legend_loc: str | None = None) -> None:
    kwargs = {
        "color": column,
        "show": False,
        "return_fig": True,
        "size": 3 if adata.n_obs > 20000 else 8,
        "frameon": True,
    }
    if legend_loc:
        kwargs["legend_loc"] = legend_loc
    fig = sc.pl.umap(adata, **kwargs)
    fig.set_size_inches(8.95, 6.0)
    fig.savefig(figures_dir / f"bc_umap_by_{column}.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def condition_order(values) -> list[str]:
    present = {str(value) for value in values}
    ordered = [condition for condition in ["Control", "LD", "NMDA"] if condition in present]
    ordered.extend(sorted(condition for condition in present if condition not in ordered))
    return ordered


def plot_condition_umaps(adata, condition_column: str, figures_dir: Path, dpi: int) -> None:
    for condition in condition_order(adata.obs[condition_column].astype(str)):
        subset = adata[adata.obs[condition_column].astype(str) == condition]
        fig = sc.pl.umap(
            subset,
            color=condition_column,
            title=f"{condition_column}: {condition}",
            show=False,
            return_fig=True,
            size=3 if subset.n_obs > 20000 else 8,
            frameon=True,
        )
        fig.set_size_inches(8.95, 6.0)
        fig.savefig(
            figures_dir / f"bc_umap_by_{condition_column}_{condition}.png",
            dpi=dpi,
            bbox_inches="tight",
        )
        plt.close(fig)


def plot_cluster_counts_by_condition(
    adata,
    condition_column: str,
    cluster_column: str,
    tables_dir: Path,
    figures_dir: Path,
    dpi: int,
) -> None:
    counts = (
        adata.obs[[condition_column, cluster_column]]
        .astype(str)
        .value_counts()
        .rename("n_cells")
        .reset_index()
    )
    counts.to_csv(tables_dir / f"counts_by_{cluster_column}_and_{condition_column}.csv", index=False)

    all_clusters = sorted(
        adata.obs[cluster_column].astype(str).unique(),
        key=lambda value: int(value) if value.isdigit() else value,
    )
    for condition in condition_order(adata.obs[condition_column].astype(str)):
        condition_counts = (
            counts[counts[condition_column] == condition]
            .set_index(cluster_column)
            .reindex(all_clusters, fill_value=0)
            .reset_index()
        )
        condition_counts.to_csv(
            tables_dir / f"counts_by_{cluster_column}_{condition}.csv",
            index=False,
        )
        fig_width = max(9, min(18, 0.32 * len(all_clusters) + 3))
        fig, ax = plt.subplots(figsize=(fig_width, 4.2))
        ax.bar(condition_counts[cluster_column].astype(str), condition_counts["n_cells"], color="#4C78A8")
        ax.set_title(f"{condition}: cells by {cluster_column}")
        ax.set_xlabel(cluster_column)
        ax.set_ylabel("Number of cells")
        ax.tick_params(axis="x", rotation=90)
        fig.tight_layout()
        fig.savefig(
            figures_dir / f"bc_counts_by_{cluster_column}_{condition}.png",
            dpi=dpi,
        )
        plt.close(fig)


def gene_lookup(var_names) -> dict[str, str]:
    return {str(gene).lower(): str(gene) for gene in var_names}


def plot_gene_umap(adata, gene: str, figures_dir: Path, dpi: int, use_raw: bool) -> None:
    fig = sc.pl.umap(
        adata,
        color=gene,
        show=False,
        return_fig=True,
        use_raw=use_raw,
        cmap="viridis",
        size=3 if adata.n_obs > 20000 else 16,
        frameon=True,
    )
    fig.set_size_inches(8.95, 6.0)
    ax = fig.axes[0]
    ax.set_title(gene, fontsize=24, pad=12)
    ax.set_xlabel("UMAP1", fontsize=22)
    ax.set_ylabel("UMAP2", fontsize=22)
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    fig.savefig(figures_dir / f"bc_umap_gene_{gene}.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    figures_dir = Path(args.figures_dir)
    tables_dir = Path(args.tables_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    adata = sc.read_h5ad(args.h5ad)
    if "X_umap" not in adata.obsm:
        raise ValueError("Input h5ad does not contain adata.obsm['X_umap']")

    summary = pd.DataFrame(
        [
            {
                "h5ad": args.h5ad,
                "n_cells": adata.n_obs,
                "n_genes": adata.n_vars,
                "raw_present": adata.raw is not None,
                "obs_columns": ";".join(map(str, adata.obs.columns)),
                "obsm_keys": ";".join(map(str, adata.obsm.keys())),
            }
        ]
    )
    summary.to_csv(tables_dir / "bc_h5ad_summary.csv", index=False)

    columns = [
        "renamed_samples",
        "sample",
        "celltype",
        "major_celltype",
        "leiden",
        "original_leiden",
        "combined_leiden",
        "bc_subtype_cluster",
    ]
    present_columns = [column for column in columns if column in adata.obs]
    for column in present_columns:
        counts = count_table(adata, column, tables_dir)
        plot_counts(counts, column, figures_dir, args.dpi)

    cluster_columns = {"leiden", "original_leiden", "combined_leiden", "bc_subtype_cluster"}
    for column in present_columns:
        legend_loc = "on data" if column in cluster_columns else None
        plot_umap(adata, column, figures_dir, args.dpi, legend_loc=legend_loc)
    if "renamed_samples" in adata.obs:
        plot_condition_umaps(adata, "renamed_samples", figures_dir, args.dpi)
    if "renamed_samples" in adata.obs and "leiden" in adata.obs:
        plot_cluster_counts_by_condition(
            adata,
            "renamed_samples",
            "leiden",
            tables_dir,
            figures_dir,
            args.dpi,
        )

    expression_names = adata.raw.var_names if adata.raw is not None else adata.var_names
    lookup = gene_lookup(expression_names)
    gene_records = []
    present_genes = []
    for requested in args.genes:
        matched = lookup.get(requested.lower())
        gene_records.append(
            {
                "requested_gene": requested,
                "matched_gene": matched or "",
                "present": matched is not None,
                "expression_source": "raw" if adata.raw is not None else "X",
            }
        )
        if matched is not None and matched not in present_genes:
            present_genes.append(matched)
    pd.DataFrame(gene_records).to_csv(tables_dir / "bc_marker_gene_match_report.csv", index=False)

    for gene in present_genes:
        plot_gene_umap(adata, gene, figures_dir, args.dpi, use_raw=adata.raw is not None)

    print(f"Wrote BC review figures: {figures_dir}")
    print(f"Wrote BC review tables: {tables_dir}")


if __name__ == "__main__":
    main()

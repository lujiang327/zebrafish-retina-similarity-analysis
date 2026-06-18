#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import scanpy as sc


DEFAULT_H5AD = (
    "/Users/hoanglab/Desktop/vscode_projects/zebrafish-singlecell-portal/"
    "AC_subtypes_reproduced.h5ad"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create quick review plots/counts for AC h5ad.")
    parser.add_argument("--h5ad", default=DEFAULT_H5AD, help="Input AC h5ad file.")
    parser.add_argument(
        "--figures-dir",
        default="results/figures/ac_subtype_h5ad_review",
        help="Output figure directory.",
    )
    parser.add_argument(
        "--tables-dir",
        default="results/tables/ac_subtype_h5ad_review",
        help="Output table directory.",
    )
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
    fig.savefig(figures_dir / f"ac_counts_by_{column}.png", dpi=dpi)
    plt.close(fig)


def plot_umap(adata, column: str, figures_dir: Path, dpi: int, legend_loc: str | None = None) -> None:
    kwargs = {"color": column, "show": False, "return_fig": True, "size": 8}
    if legend_loc:
        kwargs["legend_loc"] = legend_loc
    fig = sc.pl.umap(adata, **kwargs)
    fig.savefig(figures_dir / f"ac_umap_by_{column}.png", dpi=dpi, bbox_inches="tight")


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
                "obs_columns": ";".join(map(str, adata.obs.columns)),
                "obsm_keys": ";".join(map(str, adata.obsm.keys())),
            }
        ]
    )
    summary.to_csv(tables_dir / "ac_h5ad_summary.csv", index=False)

    columns = [
        "renamed_samples",
        "sample",
        "celltype",
        "major_celltype",
        "leiden",
        "original_leiden",
        "combined_leiden",
        "ac_subtype_cluster",
    ]
    present_columns = [column for column in columns if column in adata.obs]
    for column in present_columns:
        counts = count_table(adata, column, tables_dir)
        plot_counts(counts, column, figures_dir, args.dpi)

    cluster_columns = {"leiden", "original_leiden", "combined_leiden", "ac_subtype_cluster"}
    for column in present_columns:
        legend_loc = "on data" if column in cluster_columns else None
        plot_umap(adata, column, figures_dir, args.dpi, legend_loc=legend_loc)

    print(f"Wrote AC review figures: {figures_dir}")
    print(f"Wrote AC review tables: {tables_dir}")


if __name__ == "__main__":
    main()

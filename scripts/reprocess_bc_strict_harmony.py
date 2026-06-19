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
    "/Users/hoanglab/Desktop/vscode_projects/zebrafish-singlecell-portal/"
    "BC_annotated_clustered_corrected_doubletRemoved_Zebrafishes.h5ad"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reprocess the full BC object with Sherine-style strict Harmony parameters."
    )
    parser.add_argument("--h5ad", default=DEFAULT_H5AD, help="Input BC h5ad file.")
    parser.add_argument(
        "--out-h5ad",
        default="results/h5ad/bc_strict_harmony_reprocessed.h5ad",
        help="Output corrected h5ad path.",
    )
    parser.add_argument(
        "--figures-dir",
        default="results/figures/bc_strict_harmony_review",
        help="Output figure directory.",
    )
    parser.add_argument(
        "--tables-dir",
        default="results/tables/bc_strict_harmony_review",
        help="Output table directory.",
    )
    parser.add_argument("--condition-key", default="renamed_samples", help="Batch/condition column.")
    parser.add_argument(
        "--split-zebra-sample",
        action="store_true",
        help=(
            "Create a sample_split_zebra batch column where sample Zebra is split by "
            "renamed_samples, e.g. Zebra_LD and Zebra_NMDA. Overrides --condition-key."
        ),
    )
    parser.add_argument("--n-top-genes", type=int, default=3000, help="Number of HVGs.")
    parser.add_argument("--n-pcs", type=int, default=50, help="Number of PCs.")
    parser.add_argument("--n-neighbors", type=int, default=15, help="Neighbors for graph.")
    parser.add_argument("--resolution", type=float, default=1.0, help="Leiden resolution.")
    parser.add_argument("--random-state", type=int, default=0, help="Random seed.")
    parser.add_argument("--dpi", type=int, default=300, help="Figure DPI.")
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


def save_umap(adata, color: str, output_path: Path, dpi: int, legend_loc: str | None = None) -> None:
    kwargs = {
        "color": color,
        "show": False,
        "return_fig": True,
        "size": 3 if adata.n_obs > 20000 else 8,
        "frameon": True,
    }
    if legend_loc:
        kwargs["legend_loc"] = legend_loc
    fig = sc.pl.umap(adata, **kwargs)
    fig.set_size_inches(8.95, 6.0)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_counts_bar(counts: pd.DataFrame, column: str, output_path: Path, dpi: int) -> None:
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
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    import harmonypy

    out_h5ad = Path(args.out_h5ad)
    figures_dir = Path(args.figures_dir)
    tables_dir = Path(args.tables_dir)
    out_h5ad.parent.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    adata = sc.read_h5ad(args.h5ad)
    if args.split_zebra_sample:
        if "sample" not in adata.obs or "renamed_samples" not in adata.obs:
            raise KeyError("--split-zebra-sample requires adata.obs['sample'] and adata.obs['renamed_samples']")
        sample_split = adata.obs["sample"].astype(str).copy()
        condition = adata.obs["renamed_samples"].astype(str)
        zebra_mask = sample_split == "Zebra"
        sample_split.loc[zebra_mask] = "Zebra_" + condition.loc[zebra_mask]
        adata.obs["sample_split_zebra"] = sample_split.astype("category")
        args.condition_key = "sample_split_zebra"
    if args.condition_key not in adata.obs:
        raise KeyError(f"Condition key not found in adata.obs: {args.condition_key}")

    # Use raw expression when present. The BC object's current X can contain many zeroed rows,
    # which recreates the small 6,489-cell object; raw preserves the full cell set.
    if adata.raw is not None:
        adata.X = adata.raw.X.copy()

    # Keep the full object for expression/raw review, but preprocess on HVGs for memory safety.
    sc.pp.filter_cells(adata, min_counts=1)
    if sparse.issparse(adata.X):
        adata.X.data = np.clip(adata.X.data, 0, None)
    else:
        adata.X = np.clip(adata.X, 0, None)
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    if sparse.issparse(adata.X):
        adata.X.data = np.nan_to_num(
            adata.X.data,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).astype("float32", copy=False)
    else:
        adata.X = np.nan_to_num(adata.X, nan=0.0, posinf=0.0, neginf=0.0).astype("float32", copy=False)
    sc.pp.highly_variable_genes(adata, n_top_genes=args.n_top_genes, flavor="seurat")
    hvg_mask = adata.var["highly_variable"].to_numpy()
    adata_hvg = adata[:, hvg_mask].copy()
    sc.pp.scale(adata_hvg, max_value=10)
    sc.tl.pca(
        adata_hvg,
        n_comps=args.n_pcs,
        svd_solver="arpack",
        random_state=args.random_state,
    )

    harmony_out = harmonypy.run_harmony(
        adata_hvg.obsm["X_pca"],
        adata_hvg.obs,
        args.condition_key,
        theta=3,
        max_iter_harmony=20,
        epsilon_cluster=1e-6,
        epsilon_harmony=1e-5,
        sigma=0.1,
        tau=0,
    )
    corrected = harmony_out.Z_corr
    if corrected.shape[0] != adata.n_obs and corrected.shape[1] == adata.n_obs:
        corrected = corrected.T
    if corrected.shape[0] != adata.n_obs:
        raise ValueError(
            "Harmony output shape does not match cells: "
            f"{corrected.shape}, expected first dimension {adata.n_obs}"
        )

    adata.obsm["X_pca_uncorrected_strict"] = adata_hvg.obsm["X_pca"].astype("float32")
    adata.obsm["X_pca_harmony_strict"] = corrected.astype("float32")
    adata.obsm["X_pca_harmony"] = adata.obsm["X_pca_harmony_strict"]
    adata.obsm["X_pca"] = adata.obsm["X_pca_harmony_strict"]
    sc.pp.neighbors(
        adata,
        n_neighbors=args.n_neighbors,
        n_pcs=args.n_pcs,
        use_rep="X_pca_harmony_strict",
        random_state=args.random_state,
    )
    sc.tl.umap(adata, random_state=args.random_state)
    sc.tl.leiden(
        adata,
        resolution=args.resolution,
        key_added="leiden_strict_harmony",
        random_state=args.random_state,
    )
    adata.obs["leiden"] = adata.obs["leiden_strict_harmony"].astype("category")

    pd.DataFrame(
        [
            {
                "input_h5ad": args.h5ad,
                "output_h5ad": str(out_h5ad),
                "n_cells": adata.n_obs,
                "n_genes": adata.n_vars,
                "condition_key": args.condition_key,
                "n_top_genes": args.n_top_genes,
                "n_pcs": args.n_pcs,
                "n_neighbors": args.n_neighbors,
                "resolution": args.resolution,
                "random_state": args.random_state,
                "harmony_theta": 3,
                "harmony_max_iter_harmony": 20,
                "harmony_epsilon_cluster": 1e-6,
                "harmony_epsilon_harmony": 1e-5,
                "harmony_sigma": 0.1,
                "harmony_tau": 0,
            }
        ]
    ).to_csv(tables_dir / "bc_strict_harmony_parameters.csv", index=False)

    for column in [args.condition_key, "sample", "celltype", "leiden"]:
        if column in adata.obs:
            counts = count_table(adata, column, tables_dir)
            save_counts_bar(counts, column, figures_dir / f"bc_strict_counts_by_{column}.png", args.dpi)

    save_umap(
        adata,
        args.condition_key,
        figures_dir / f"bc_strict_umap_by_{args.condition_key}.png",
        args.dpi,
    )
    save_umap(
        adata,
        "leiden",
        figures_dir / "bc_strict_umap_by_leiden.png",
        args.dpi,
        legend_loc="on data",
    )
    adata.write_h5ad(out_h5ad, compression="gzip")
    print(out_h5ad)
    print(figures_dir)
    print(tables_dir)


if __name__ == "__main__":
    main()

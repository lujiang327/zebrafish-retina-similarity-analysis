#!/usr/bin/env python
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import harmonypy
import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
from scipy import sparse

from fidelity_utils import load_config, normalize_labels, resolve_metadata_columns, setup_logging


LOGGER = logging.getLogger("retina_fidelity")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce Sherine's two-step RGC subtype UMAP workflow."
    )
    parser.add_argument("--config", default="config/config.yaml", help="Path to config YAML.")
    parser.add_argument("--h5ad", default=None, help="Override input .h5ad path from config.")
    parser.add_argument(
        "--rgc-h5ad",
        default=None,
        help="Use an existing RGC-only intermediate .h5ad and skip the subset.py-style stage.",
    )
    parser.add_argument("--cell-type-label", default="RGC", help="Cell type to subset.")
    parser.add_argument(
        "--figures-dir",
        default="results/figures/rgc_sherine_reproduction",
        help="Directory for reproduced figures.",
    )
    parser.add_argument(
        "--intermediate-dir",
        default="results/intermediate/rgc_sherine_reproduction",
        help="Directory for intermediate .h5ad files.",
    )
    parser.add_argument(
        "--write-corrected-h5ad",
        action="store_true",
        help="Also write the corrected RGC object after strict Harmony.",
    )
    parser.add_argument(
        "--skip-batch-normalize-log",
        action="store_true",
        help=(
            "Skip normalize_total/log1p in the batch_subset.py-style stage. "
            "Use this to test whether the input object is already normalized/log-transformed."
        ),
    )
    parser.add_argument("--verbose", action="store_true", help="Print debug logging.")
    return parser.parse_args()


def remove_unused_categories(adata) -> None:
    for col in adata.obs.select_dtypes(include="category").columns:
        adata.obs[col] = adata.obs[col].cat.remove_unused_categories()


def reproduce_subset_py(
    adata,
    cell_type_label: str,
    cell_type_column: str,
    condition_column: str,
    input_stem: str,
    figures_dir: Path,
    intermediate_dir: Path,
) -> Path:
    new_object = intermediate_dir / f"{cell_type_label}_{input_stem}.h5ad"
    subset = adata[normalize_labels(adata.obs[cell_type_column]) == cell_type_label, :].copy()
    LOGGER.info("subset.py-style %s subset before filtering: %s cells", cell_type_label, subset.n_obs)

    sc.pp.pca(subset)
    sc.pp.neighbors(subset)
    sc.tl.umap(subset)
    sc.tl.leiden(subset, resolution=1.0)

    fig = sc.pl.umap(subset, color="leiden", legend_loc="on data", show=False, return_fig=True)
    fig.set_size_inches(12, 12)
    fig.savefig(figures_dir / f"umap{cell_type_label}.png", dpi=600, bbox_inches="tight")
    plt.close(fig)

    adata.obs[condition_column] = adata.obs[condition_column].astype("category")
    renamed_samples = adata.obs[condition_column].cat.categories
    for sample in renamed_samples:
        sample_adata = adata[adata.obs[condition_column] == sample].copy()
        sample_adata.obs[condition_column] = sample_adata.obs[condition_column].astype("category")
        sample_adata.obs[condition_column] = sample_adata.obs[
            condition_column
        ].cat.remove_unused_categories()
        fig = sc.pl.umap(
            sample_adata,
            color=condition_column,
            title=f"Sample: {sample}",
            size=20,
            show=False,
            return_fig=True,
        )
        fig.set_size_inches(12, 12)
        fig.savefig(figures_dir / f"umap_{cell_type_label}_{sample}.png", dpi=600, bbox_inches="tight")
        plt.close(fig)

    fig = sc.pl.umap(adata, color=condition_column, size=2, show=False, return_fig=True)
    fig.set_size_inches(12, 12)
    fig.savefig(figures_dir / f"umap_merged_{cell_type_label}.png", dpi=600, bbox_inches="tight")
    plt.close(fig)

    remove_unused_categories(subset)
    if subset.raw is not None:
        subset.raw = None
    subset.write(new_object, compression="gzip")
    LOGGER.info("Wrote subset.py-style intermediate: %s", new_object)
    return new_object


def harmony_integrate_compatible(adata, condition_column: str) -> None:
    harmony_out = harmonypy.run_harmony(
        adata.obsm["X_pca"],
        adata.obs,
        condition_column,
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
        raise ValueError(f"Unexpected Harmony shape {corrected.shape} for {adata.n_obs} cells")
    adata.obsm["X_pca_harmony"] = corrected


def reproduce_batch_subset_py(
    rgc_h5ad: Path,
    condition_column: str,
    figures_dir: Path,
    write_corrected_h5ad: bool,
    skip_batch_normalize_log: bool,
) -> None:
    combined_adata = sc.read(rgc_h5ad)
    base_name = rgc_h5ad.stem

    sc.pp.filter_cells(combined_adata, min_counts=1)
    LOGGER.info("batch_subset.py-style subset after min_counts=1: %s cells", combined_adata.n_obs)

    if sparse.issparse(combined_adata.X):
        combined_adata.X = combined_adata.X.toarray()

    combined_adata.X = np.clip(combined_adata.X, 0, None)
    if skip_batch_normalize_log:
        LOGGER.info("Skipping batch_subset.py normalize_total/log1p test step")
        LOGGER.info("Skipping highly_variable_genes because existing X is not clean log-expression")
    else:
        sc.pp.normalize_total(combined_adata)
        sc.pp.log1p(combined_adata)
        sc.pp.highly_variable_genes(combined_adata)
    combined_adata.X = np.nan_to_num(combined_adata.X, nan=0.0, posinf=0.0, neginf=0.0)
    combined_adata.raw = combined_adata.copy()
    sc.pp.scale(combined_adata)
    sc.tl.pca(combined_adata, svd_solver="arpack")

    harmony_integrate_compatible(combined_adata, condition_column)
    combined_adata.obsm["X_pca"] = combined_adata.obsm["X_pca_harmony"]
    sc.pp.neighbors(combined_adata, random_state=0)
    sc.tl.umap(combined_adata, random_state=0)
    sc.tl.leiden(combined_adata, resolution=1.0, random_state=0)

    fig = sc.pl.umap(combined_adata, color=condition_column, size=2, show=False, return_fig=True)
    fig.savefig(figures_dir / f"{base_name}_sHarmony.png", dpi=600, bbox_inches="tight")
    plt.close(fig)

    samples = combined_adata.obs[condition_column].unique()
    for sample in samples:
        fig = sc.pl.umap(
            combined_adata[combined_adata.obs[condition_column] == sample],
            color=condition_column,
            title=f"Sample: {sample}",
            size=20,
            show=False,
            return_fig=True,
        )
        fig.savefig(figures_dir / f"{base_name}_sHarmony_{sample}.png", dpi=600, bbox_inches="tight")
        plt.close(fig)

    fig = sc.pl.umap(
        combined_adata,
        color="leiden",
        title="Leiden clusters",
        size=20,
        legend_loc="on data",
        show=False,
        return_fig=True,
    )
    fig.savefig(figures_dir / f"{base_name}_sHarmony_leiden.png", dpi=600, bbox_inches="tight")
    plt.close(fig)

    sc.pl.violin(
        combined_adata,
        keys=["n_genes_by_counts", "total_counts", "pct_counts_mt"],
        groupby="leiden",
        jitter=0.4,
        rotation=45,
        multi_panel=False,
        show=False,
    )
    plt.savefig(figures_dir / f"violin_{base_name}_sHarmony_QC.png", dpi=600, bbox_inches="tight")
    plt.close()

    if write_corrected_h5ad:
        corrected_path = rgc_h5ad.with_name(f"corrected_{rgc_h5ad.name}")
        combined_adata.write(corrected_path)
        LOGGER.info("Wrote corrected batch_subset.py-style object: %s", corrected_path)


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)
    config = load_config(args.config)
    h5ad_path = Path(args.h5ad or config["data"]["h5ad_path"])
    figures_dir = Path(args.figures_dir)
    intermediate_dir = Path(args.intermediate_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    intermediate_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Loading full AnnData: %s", h5ad_path)
    if args.rgc_h5ad:
        rgc_h5ad = Path(args.rgc_h5ad)
        adata = sc.read_h5ad(rgc_h5ad, backed="r")
        condition_column, _ = resolve_metadata_columns(adata, config)
        del adata
        LOGGER.info("Using existing RGC intermediate: %s", rgc_h5ad)
    else:
        adata = sc.read_h5ad(h5ad_path)
        condition_column, cell_type_column = resolve_metadata_columns(adata, config)

        rgc_h5ad = reproduce_subset_py(
            adata=adata,
            cell_type_label=args.cell_type_label,
            cell_type_column=cell_type_column,
            condition_column=condition_column,
            input_stem=h5ad_path.stem,
            figures_dir=figures_dir,
            intermediate_dir=intermediate_dir,
        )
        del adata

    reproduce_batch_subset_py(
        rgc_h5ad=rgc_h5ad,
        condition_column=condition_column,
        figures_dir=figures_dir,
        write_corrected_h5ad=args.write_corrected_h5ad,
        skip_batch_normalize_log=args.skip_batch_normalize_log,
    )
    LOGGER.info("Saved Sherine reproduction figures to %s", figures_dir)


if __name__ == "__main__":
    main()

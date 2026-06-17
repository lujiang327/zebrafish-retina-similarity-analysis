#!/usr/bin/env python
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse

from fidelity_utils import (
    ensure_output_dirs,
    load_config,
    normalize_labels,
    resolve_metadata_columns,
    setup_logging,
)


LOGGER = logging.getLogger("retina_fidelity")


DEFAULT_GENES = [
    "ephb2b",
    "ephb3a",
    "ephnb4a",
    "epha4b",
    "efna2",
    "efnb2a",
    "efna5a",
    "efna5b",
    "hmx1",
    "hmx4",
    "cntn2",
    "efnb1",
    "vax1",
    "vax2",
    "tbx5a",
    "zic2",
    "isl2b",
    "rbpms2a",
    "rbpms2b",
    "mafaa",
    "eomesa",
    "tbr1b",
    "elavl3",
    "cdk1",
    "nr2e3",
    "guca1b",
    "gnat2",
    "cabp5a",
    "ompa",
    "mpeg1.1",
    "mbpa",
    "rpe65a",
    "aqp4",
    "tie1",
    "acta2",
    "mitfa",
    "efna2a",
    "efna2b",
    "zic2a",
    "zic2b",
    "gria3a",
    "crhb",
    "ccdc80",
    "tubb5",
    "six3a",
    "sgk1",
    "tacr2",
    "matn3b",
    "rasgef1ba",
    "tac1",
    "pcdh11",
    "sdk1a",
    "ptprt",
    "gpd1b",
    "rprma",
    "tbx3a",
    "si:dkey-1h6.8",
    "neurod2",
    "pgam2",
    "mfap4",
    "tox",
    "id3",
    "emid1",
    "ccka",
    "gulp1b",
    "plppr2",
    "pyyb",
    "dio1",
    "gapdh",
    "npb",
    "nmbb",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recreate RGC Harmony subclustering and plot requested gene UMAPs."
    )
    parser.add_argument("--config", default="config/config.yaml", help="Path to config YAML.")
    parser.add_argument("--h5ad", default=None, help="Override input .h5ad path from config.")
    parser.add_argument("--genes", nargs="*", default=DEFAULT_GENES, help="Genes to plot.")
    parser.add_argument(
        "--extra-genes-csv",
        nargs="*",
        default=[],
        help="CSV files with a names column; genes are appended to --genes.",
    )
    parser.add_argument("--cell-type-label", default="RGC", help="Cell type label for RGCs.")
    parser.add_argument(
        "--use-existing-embedding",
        action="store_true",
        help="Use the input object's existing X_umap/leiden instead of rerunning Sherine strict Harmony.",
    )
    parser.add_argument("--resolution", type=float, default=1.0, help="Leiden resolution.")
    parser.add_argument("--random-state", type=int, default=0, help="Random seed.")
    parser.add_argument(
        "--output-subdir",
        default="rgc_subtype_gene_umaps",
        help="Subdirectory under results/tables and results/figures.",
    )
    parser.add_argument("--verbose", action="store_true", help="Print debug logging.")
    return parser.parse_args()


def append_extra_genes(genes: list[str], csv_paths: list[str]) -> list[str]:
    ordered = list(genes)
    seen = {gene.lower() for gene in ordered}
    for csv_path in csv_paths:
        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(f"Extra genes CSV does not exist: {path}")
        df = pd.read_csv(path)
        if "names" not in df.columns:
            raise KeyError(f"Extra genes CSV must contain a names column: {path}")
        for gene in df["names"].dropna().astype(str):
            key = gene.lower()
            if key not in seen:
                ordered.append(gene)
                seen.add(key)
    return ordered


def gene_lookup(var_names: np.ndarray) -> dict[str, str]:
    return {str(gene).lower(): str(gene) for gene in var_names}


def dense_vector(matrix) -> np.ndarray:
    if sparse.issparse(matrix):
        return np.asarray(matrix.toarray()).ravel()
    return np.asarray(matrix).ravel()


def expression_vector(adata, gene: str, config: dict) -> np.ndarray:
    lookup = gene_lookup(adata.var_names.to_numpy())
    actual = lookup[gene.lower()]
    idx = int(np.where(adata.var_names.astype(str) == actual)[0][0])
    return dense_vector(adata.X[:, idx])


def run_sherine_strict_harmony_subset(
    adata,
    condition_column: str,
    resolution: float,
    random_state: int,
):
    import harmonypy

    sc.pp.filter_cells(adata, min_counts=1)
    if sparse.issparse(adata.X):
        adata.X = adata.X.toarray()
    adata.X = np.clip(adata.X, 0, None)
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    adata.X = np.nan_to_num(adata.X, nan=0.0, posinf=0.0, neginf=0.0)
    sc.pp.highly_variable_genes(adata)
    adata.raw = adata.copy()
    sc.pp.scale(adata)
    sc.tl.pca(adata, svd_solver="arpack")
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
        raise ValueError(
            "Harmony output shape does not match cells: "
            f"{corrected.shape}, expected first dimension {adata.n_obs}"
        )
    adata.obsm["X_pca_harmony"] = corrected
    adata.obsm["X_pca"] = adata.obsm["X_pca_harmony"]
    sc.pp.neighbors(adata, random_state=random_state)
    sc.tl.umap(adata, random_state=random_state)
    sc.tl.leiden(
        adata,
        resolution=resolution,
        key_added="leiden",
        random_state=random_state,
    )
    return adata


def save_overview_umap(adata, color: str, output_path: Path, dpi: int) -> None:
    sc.pl.umap(adata, color=color, frameon=False, show=False)
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()


def save_sample_highlight_umaps(
    adata,
    condition_column: str,
    figures_dir: Path,
    dpi: int,
) -> None:
    coords = adata.obsm["X_umap"]
    condition_values = adata.obs[condition_column].astype(str).to_numpy()
    sample_colors = {
        "Control": "#1f77b4",
        "LD": "#ff7f0e",
        "NMDA": "#2ca02c",
    }
    ordered_samples = [
        sample for sample in ["Control", "LD", "NMDA"] if sample in set(condition_values)
    ]
    ordered_samples.extend(
        sorted(sample for sample in set(condition_values) if sample not in ordered_samples)
    )

    for sample in ordered_samples:
        mask = condition_values == sample
        fig, ax = plt.subplots(figsize=(8.95, 6.0))
        ax.scatter(
            coords[:, 0],
            coords[:, 1],
            c="#d9d9d9",
            s=4,
            linewidths=0,
            alpha=0.35,
        )
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            c=sample_colors.get(sample, "#1f77b4"),
            s=8,
            linewidths=0,
            label=sample,
        )
        ax.set_title(f"{condition_column}: {sample}", fontsize=20)
        ax.set_xlabel("UMAP1", fontsize=16)
        ax.set_ylabel("UMAP2", fontsize=16)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)
        fig.savefig(
            figures_dir / f"rgc_umap_by_{condition_column}_{sample}_highlight.png",
            dpi=dpi,
            bbox_inches="tight",
        )
        plt.close(fig)


def save_sherine_batch_subset_figures(
    adata,
    base_name: str,
    condition_column: str,
    figures_dir: Path,
) -> None:
    fig = sc.pl.umap(
        adata,
        color=condition_column,
        size=2,
        show=False,
        return_fig=True,
    )
    fig.savefig(figures_dir / f"{base_name}_sHarmony.png", dpi=600, bbox_inches="tight")
    plt.close(fig)

    samples = adata.obs[condition_column].unique()
    for sample in samples:
        fig = sc.pl.umap(
            adata[adata.obs[condition_column] == sample],
            color=condition_column,
            title=f"Sample: {sample}",
            size=20,
            show=False,
            return_fig=True,
        )
        fig.savefig(
            figures_dir / f"{base_name}_sHarmony_{sample}.png",
            dpi=600,
            bbox_inches="tight",
        )
        plt.close(fig)

    fig = sc.pl.umap(
        adata,
        color="leiden",
        title="Leiden clusters",
        size=20,
        legend_loc="on data",
        show=False,
        return_fig=True,
    )
    fig.savefig(figures_dir / f"{base_name}_sHarmony_leiden.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def expression_limits(expr: np.ndarray) -> tuple[float, float]:
    finite = expr[np.isfinite(expr)]
    if finite.size == 0:
        return 0.0, 1.0
    vmin = float(np.nanmin(finite))
    vmax = float(np.nanmax(finite))
    if vmax <= vmin:
        vmax = float(np.nanmax(expr))
    return vmin, vmax


def plot_gene_all_conditions(
    adata,
    gene: str,
    config: dict,
    output_path: Path,
    dpi: int,
) -> None:
    fig = sc.pl.umap(
        adata,
        color=gene,
        frameon=True,
        cmap="viridis",
        use_raw=False,
        size=16,
        show=False,
        return_fig=True,
    )
    fig.set_size_inches(8.95, 6.0)
    ax = fig.axes[0]
    ax.set_title(gene, fontsize=24, pad=12)
    ax.set_xlabel("UMAP1", fontsize=22)
    ax.set_ylabel("UMAP2", fontsize=22)
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_gene_by_condition(
    adata,
    gene: str,
    condition_column: str,
    condition_panels: list[tuple[str, tuple[str, ...]]],
    config: dict,
    output_path: Path,
    dpi: int,
) -> None:
    coords = adata.obsm["X_umap"]
    expr = expression_vector(adata, gene, config)
    vmin = np.nanpercentile(expr, 1)
    vmax = np.nanpercentile(expr, 99)
    if not np.isfinite(vmin):
        vmin = float(np.nanmin(expr))
    if not np.isfinite(vmax) or vmax <= vmin:
        vmax = float(np.nanmax(expr))

    n_panels = len(condition_panels)
    n_cols = 2 if n_panels > 1 else 1
    n_rows = int(np.ceil(n_panels / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.4 * n_cols + 0.8, 3.8 * n_rows),
    )
    axes = np.atleast_1d(axes).ravel()

    sca = None
    condition_values = adata.obs[condition_column].astype(str).to_numpy()
    for ax, (label, values) in zip(axes[:n_panels], condition_panels):
        mask = np.isin(condition_values, values)
        ax.scatter(
            coords[:, 0],
            coords[:, 1],
            c="#d9d9d9",
            s=4,
            linewidths=0,
            alpha=0.35,
        )
        sca = ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            c=expr[mask],
            s=7,
            linewidths=0,
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(f"{gene} | {label}")
        ax.set_xlabel("UMAP1")
        ax.set_ylabel("UMAP2")
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes[n_panels:]:
        ax.set_visible(False)
    fig.subplots_adjust(left=0.08, right=0.88, bottom=0.08, top=0.92, wspace=0.28, hspace=0.35)
    if sca is not None:
        cbar_ax = fig.add_axes([0.91, 0.2, 0.02, 0.6])
        cbar = fig.colorbar(sca, cax=cbar_ax)
        cbar.set_label("Expression")
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)
    config = load_config(args.config)
    tables_dir, figures_dir = ensure_output_dirs(config)
    tables_dir = tables_dir / args.output_subdir
    figures_dir = figures_dir / args.output_subdir
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    h5ad_path = Path(args.h5ad or config["data"]["h5ad_path"])
    base_name = f"{args.cell_type_label}_{h5ad_path.stem}"
    LOGGER.info("Loading AnnData: %s", h5ad_path)
    adata = sc.read_h5ad(h5ad_path)
    condition_column, cell_type_column = resolve_metadata_columns(adata, config)

    if cell_type_column in adata.obs:
        rgc_mask = normalize_labels(adata.obs[cell_type_column]) == args.cell_type_label
        adata_rgc = adata[rgc_mask].copy()
    else:
        adata_rgc = adata.copy()
    LOGGER.info("RGC subset: %s cells", adata_rgc.n_obs)

    if args.use_existing_embedding:
        if "X_umap" not in adata_rgc.obsm:
            raise ValueError("--use-existing-embedding requires X_umap in adata.obsm")
        if "leiden" not in adata_rgc.obs:
            raise ValueError("--use-existing-embedding requires leiden in adata.obs")
        LOGGER.info("Using existing X_umap and leiden from input object")
    else:
        adata_rgc = run_sherine_strict_harmony_subset(
            adata_rgc,
            condition_column=condition_column,
            resolution=args.resolution,
            random_state=args.random_state,
        )
    adata_rgc.obs["rgc_leiden_strict"] = adata_rgc.obs["leiden"].astype(str)

    output_cfg = config.get("outputs", {})
    dpi = int(output_cfg.get("figure_dpi", 300))
    fmt = output_cfg.get("figure_format", "png")
    requested_genes = append_extra_genes(list(args.genes), args.extra_genes_csv)

    save_overview_umap(
        adata_rgc,
        condition_column,
        figures_dir / f"rgc_umap_by_{condition_column}_strict.{fmt}",
        dpi,
    )
    save_sample_highlight_umaps(
        adata_rgc,
        condition_column,
        figures_dir,
        dpi,
    )
    save_overview_umap(
        adata_rgc,
        "rgc_leiden_strict",
        figures_dir / f"rgc_umap_by_rgc_leiden_strict.{fmt}",
        dpi,
    )
    save_sherine_batch_subset_figures(
        adata_rgc,
        base_name,
        condition_column,
        figures_dir,
    )

    var_names = adata_rgc.var_names.to_numpy()
    source_name = "X_after_subset_log_norm_scale"
    lookup = gene_lookup(var_names)
    gene_records = []
    present_genes = []
    for requested in requested_genes:
        actual = lookup.get(requested.lower())
        gene_records.append(
            {
                "requested_gene": requested,
                "matched_gene": actual,
                "present": actual is not None,
                "expression_source": source_name,
            }
        )
        if actual is not None:
            present_genes.append(actual)

    pd.DataFrame(gene_records).to_csv(tables_dir / "rgc_gene_match_report.csv", index=False)
    adata_rgc.obs[[condition_column, cell_type_column, "rgc_leiden_strict"]].to_csv(
        tables_dir / "rgc_subcluster_assignments.csv"
    )

    cluster_counts = (
        adata_rgc.obs.groupby([condition_column, "rgc_leiden_strict"], observed=False)
        .size()
        .rename("n_cells")
        .reset_index()
    )
    cluster_counts.to_csv(tables_dir / "rgc_subcluster_counts_by_condition.csv", index=False)

    available_conditions = set(adata_rgc.obs[condition_column].astype(str))
    condition_panels = [
        (condition, (condition,))
        for condition in ["Control", "LD", "NMDA"]
        if condition in available_conditions
    ]
    if {"LD", "NMDA"}.issubset(available_conditions):
        condition_panels.append(("LD + NMDA", ("LD", "NMDA")))
    if not condition_panels:
        condition_panels = [
            (condition, (condition,))
            for condition in sorted(available_conditions)
        ]

    for gene in present_genes:
        plot_gene_all_conditions(
            adata_rgc,
            gene,
            config,
            figures_dir / f"rgc_umap_gene_{gene}_all_conditions.{fmt}",
            dpi,
        )
        plot_gene_by_condition(
            adata_rgc,
            gene,
            condition_column,
            condition_panels,
            config,
            figures_dir / f"rgc_umap_gene_{gene}_split_by_condition.{fmt}",
            dpi,
        )

    LOGGER.info("Saved outputs to %s and %s", tables_dir, figures_dir)


if __name__ == "__main__":
    main()

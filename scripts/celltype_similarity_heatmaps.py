#!/usr/bin/env python
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from scipy import sparse

from fidelity_utils import (
    ensure_output_dirs,
    finite_similarity,
    get_expression_source,
    load_config,
    mean_expression_by_mask,
    normalize_labels,
    resolve_metadata_columns,
    retinal_cell_type_mask,
    setup_logging,
)


LOGGER = logging.getLogger("retina_fidelity")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Make cross-cell-type similarity heatmaps like manuscript panels A/B/C."
    )
    parser.add_argument("--config", default="config/config.yaml", help="Path to config YAML.")
    parser.add_argument("--h5ad", default=None, help="Override input .h5ad path from config.")
    parser.add_argument(
        "--gene-sets",
        nargs="*",
        default=["shared_expressed", "hvg_1000", "hvg_2000"],
        help="Gene sets to plot.",
    )
    parser.add_argument(
        "--metrics",
        nargs="*",
        default=None,
        help="Similarity metrics to plot. Defaults to config metrics.",
    )
    parser.add_argument("--verbose", action="store_true", help="Print debug logging.")
    return parser.parse_args()


def expressed_gene_mask(matrix, min_mean_expression: float) -> np.ndarray:
    if sparse.issparse(matrix):
        means = np.asarray(matrix.mean(axis=0)).ravel()
    else:
        means = np.asarray(matrix).mean(axis=0).ravel()
    return means > min_mean_expression


def variance_gene_indices(matrix, n_top: int) -> np.ndarray:
    if sparse.issparse(matrix):
        means = np.asarray(matrix.mean(axis=0)).ravel()
        means_sq = np.asarray(matrix.power(2).mean(axis=0)).ravel()
        variances = means_sq - np.square(means)
    else:
        variances = np.asarray(matrix).var(axis=0)
    n = min(int(n_top), matrix.shape[1])
    return np.argsort(variances)[-n:]


def scanpy_hvg_gene_indices(matrix, n_top: int, flavor: str) -> np.ndarray:
    n = min(int(n_top), matrix.shape[1])
    if n <= 0:
        return np.array([], dtype=int)
    try:
        hvg_adata = ad.AnnData(X=matrix.copy())
        sc.pp.highly_variable_genes(
            hvg_adata,
            n_top_genes=n,
            flavor=flavor,
            inplace=True,
        )
        indices = np.where(hvg_adata.var["highly_variable"].to_numpy())[0]
        if indices.size > 0:
            return indices[:n]
    except Exception as exc:  # pragma: no cover - data dependent fallback
        LOGGER.warning(
            "Scanpy HVG selection failed for n_top=%s, flavor=%s; using variance fallback. Error: %s",
            n_top,
            flavor,
            exc,
        )
    return variance_gene_indices(matrix, n)


def comparison_pairs(condition_order: list[str]) -> list[tuple[str, str, str, str]]:
    labels = []
    if "Control" in condition_order and "LD" in condition_order:
        labels.append(("Control_vs_LD", "Control", "LD", "control-LD"))
    if "Control" in condition_order and "NMDA" in condition_order:
        labels.append(("Control_vs_NMDA", "Control", "NMDA", "control-NMDA"))
    if "LD" in condition_order and "NMDA" in condition_order:
        labels.append(("LD_vs_NMDA", "LD", "NMDA", "LD-NMDA"))
    return labels


def plot_three_panel_heatmap(
    matrices: dict[str, pd.DataFrame],
    metric: str,
    gene_set: str,
    output_path: Path,
    dpi: int,
) -> None:
    panels = [
        ("A", "Control_vs_LD", "control-LD", "Reference: control", "Query: LD"),
        ("B", "Control_vs_NMDA", "control-NMDA", "Reference: control", "Query: NMDA"),
        ("C", "LD_vs_NMDA", "LD-NMDA", "Reference: LD", "Query: NMDA"),
    ]
    available = [panel for panel in panels if panel[1] in matrices]
    if not available:
        return

    vmin, vmax = 0, 1
    cmap = "viridis"

    n_panels = len(available)
    fig, axes = plt.subplots(
        n_panels,
        1,
        figsize=(5.0, 4.6 * n_panels),
        squeeze=False,
    )
    axes = axes.ravel()

    for ax, (letter, key, title_suffix, y_label, x_label) in zip(axes, available):
        matrix = matrices[key]
        sns.heatmap(
            matrix,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            annot=True,
            fmt=".2f",
            annot_kws={"fontsize": 5.5},
            linewidths=0.2,
            square=True,
            cbar=True,
            cbar_kws={"label": f"{metric.title()} similarity", "shrink": 0.75},
            ax=ax,
        )
        ax.set_title(
            f"Cell Type Similarity: {title_suffix}\n({metric.title()} - {gene_set})",
            fontsize=10,
        )
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.tick_params(axis="x", labelrotation=90)
        ax.tick_params(axis="y", labelrotation=0)
        ax.text(
            -0.18,
            1.08,
            letter,
            transform=ax.transAxes,
            fontsize=18,
            fontweight="bold",
            va="top",
            ha="right",
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)
    config = load_config(args.config)
    tables_dir, figures_dir = ensure_output_dirs(config)

    h5ad_path = Path(args.h5ad or config["data"]["h5ad_path"])
    LOGGER.info("Loading AnnData: %s", h5ad_path)
    adata = sc.read_h5ad(h5ad_path)

    condition_column, cell_type_column = resolve_metadata_columns(adata, config)
    expression, _, source_name = get_expression_source(adata, config)
    LOGGER.info("Using condition column: %s", condition_column)
    LOGGER.info("Using cell-type column: %s", cell_type_column)
    LOGGER.info("Using expression source: %s", source_name)

    metadata = config.get("metadata", {})
    condition_order = metadata.get("condition_order", ["Control", "LD", "NMDA"])
    metrics = args.metrics or config.get("analysis", {}).get(
        "similarity_metrics", ["pearson", "spearman", "cosine"]
    )
    analysis_cfg = config.get("analysis", {})
    min_cells = int(analysis_cfg.get("min_cells_per_group", 30))
    gene_cfg = analysis_cfg.get("gene_sets", {})
    hvg_flavor = gene_cfg.get("hvg_flavor", "seurat")
    min_mean_expression = float(gene_cfg.get("min_mean_expression", 0.0))

    obs = adata.obs.copy()
    obs["_condition"] = normalize_labels(obs[condition_column])
    obs["_cell_type"] = normalize_labels(obs[cell_type_column])

    retinal_mask = retinal_cell_type_mask(obs, "_cell_type", config).to_numpy()
    retinal_obs = obs.loc[retinal_mask]
    cell_types = sorted(retinal_obs["_cell_type"].dropna().unique())
    LOGGER.info("Retinal cell types: %s", ", ".join(cell_types))

    retinal_expression = expression[retinal_mask]
    expressed_idx = np.where(expressed_gene_mask(retinal_expression, min_mean_expression))[0]
    gene_sets: dict[str, np.ndarray] = {"shared_expressed": expressed_idx}
    for gene_set in args.gene_sets:
        if gene_set.startswith("hvg_"):
            n_top = int(gene_set.split("_", 1)[1])
            local_idx = scanpy_hvg_gene_indices(
                retinal_expression[:, expressed_idx],
                n_top,
                hvg_flavor,
            )
            gene_sets[gene_set] = expressed_idx[local_idx]

    gene_sets = {name: idx for name, idx in gene_sets.items() if name in args.gene_sets}

    centroids: dict[tuple[str, str], np.ndarray] = {}
    group_records = []
    for condition in condition_order:
        for cell_type in cell_types:
            group_mask = (
                (obs["_condition"] == condition).to_numpy()
                & (obs["_cell_type"] == cell_type).to_numpy()
            )
            n_cells = int(group_mask.sum())
            group_records.append(
                {"condition": condition, "cell_type": cell_type, "n_cells": n_cells}
            )
            if n_cells == 0:
                continue
            centroids[(condition, cell_type)] = mean_expression_by_mask(expression, group_mask)

    pd.DataFrame(group_records).to_csv(
        tables_dir / "celltype_similarity_heatmap_group_counts.csv",
        index=False,
    )

    records = []
    matrix_outputs: dict[tuple[str, str], dict[str, pd.DataFrame]] = {}
    comparisons = comparison_pairs(condition_order)

    for metric in metrics:
        for gene_set, gene_idx in gene_sets.items():
            matrices: dict[str, pd.DataFrame] = {}
            for comparison, ref_condition, query_condition, _ in comparisons:
                matrix = pd.DataFrame(index=cell_types, columns=cell_types, dtype=float)
                for ref_cell_type in cell_types:
                    ref_key = (ref_condition, ref_cell_type)
                    if ref_key not in centroids:
                        continue
                    for query_cell_type in cell_types:
                        query_key = (query_condition, query_cell_type)
                        if query_key not in centroids:
                            continue
                        score = finite_similarity(
                            centroids[ref_key][gene_idx],
                            centroids[query_key][gene_idx],
                            metric,
                        )
                        matrix.loc[ref_cell_type, query_cell_type] = score
                        ref_n = next(
                            item["n_cells"]
                            for item in group_records
                            if item["condition"] == ref_condition and item["cell_type"] == ref_cell_type
                        )
                        query_n = next(
                            item["n_cells"]
                            for item in group_records
                            if item["condition"] == query_condition and item["cell_type"] == query_cell_type
                        )
                        records.append(
                            {
                                "comparison": comparison,
                                "reference_condition": ref_condition,
                                "query_condition": query_condition,
                                "reference_cell_type": ref_cell_type,
                                "query_cell_type": query_cell_type,
                                "metric": metric,
                                "gene_set": gene_set,
                                "n_genes": int(gene_idx.size),
                                "similarity": score,
                                "reference_n_cells": ref_n,
                                "query_n_cells": query_n,
                                "low_cell_count_warning": bool(
                                    ref_n < min_cells or query_n < min_cells
                                ),
                            }
                        )
                matrices[comparison] = matrix
                matrix.to_csv(
                    tables_dir
                    / f"celltype_similarity_matrix_{metric}_{gene_set}_{comparison}.csv"
                )
            matrix_outputs[(metric, gene_set)] = matrices

    scores = pd.DataFrame.from_records(records)
    scores.to_csv(tables_dir / "celltype_similarity_heatmap_scores.csv", index=False)
    LOGGER.info("Saved cross-cell-type similarity scores.")

    output_cfg = config.get("outputs", {})
    dpi = int(output_cfg.get("figure_dpi", 300))
    fmt = output_cfg.get("figure_format", "png")

    for (metric, gene_set), matrices in matrix_outputs.items():
        plot_three_panel_heatmap(
            matrices,
            metric=metric,
            gene_set=gene_set,
            output_path=figures_dir / f"celltype_similarity_panels_{metric}_{gene_set}.{fmt}",
            dpi=dpi,
        )

    LOGGER.info("Done.")


if __name__ == "__main__":
    main()

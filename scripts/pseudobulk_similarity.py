#!/usr/bin/env python
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import anndata as ad
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
        description="Compute cell-type-specific pseudobulk expression similarities."
    )
    parser.add_argument("--config", default="config/config.yaml", help="Path to config YAML.")
    parser.add_argument("--h5ad", default=None, help="Override input .h5ad path from config.")
    parser.add_argument(
        "--cell-types",
        nargs="*",
        default=None,
        help="Optional explicit cell-type labels to analyze. Defaults to config retinal include list.",
    )
    parser.add_argument("--verbose", action="store_true", help="Print debug logging.")
    return parser.parse_args()


def expressed_gene_mask(matrix, min_mean_expression: float) -> np.ndarray:
    if sparse.issparse(matrix):
        means = np.asarray(matrix.mean(axis=0)).ravel()
    else:
        means = np.asarray(matrix).mean(axis=0).ravel()
    return means > min_mean_expression


def top_variable_gene_indices(matrix, n_top: int) -> np.ndarray:
    if matrix.shape[1] == 0:
        return np.array([], dtype=int)
    if sparse.issparse(matrix):
        means = np.asarray(matrix.mean(axis=0)).ravel()
        means_sq = np.asarray(matrix.power(2).mean(axis=0)).ravel()
        variances = means_sq - np.square(means)
    else:
        variances = np.asarray(matrix).var(axis=0)
    n = min(int(n_top), matrix.shape[1])
    if n <= 0:
        return np.array([], dtype=int)
    return np.argsort(variances)[-n:]


def scanpy_hvg_gene_indices(matrix, n_top: int, flavor: str) -> np.ndarray:
    if matrix.shape[1] == 0:
        return np.array([], dtype=int)
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
        mask = hvg_adata.var["highly_variable"].to_numpy()
        indices = np.where(mask)[0]
        if indices.size > 0:
            return indices[:n]
    except Exception as exc:  # pragma: no cover - fallback is data dependent
        LOGGER.warning(
            "Scanpy HVG selection failed for n_top=%s, flavor=%s; using variance fallback. Error: %s",
            n_top,
            flavor,
            exc,
        )
    return top_variable_gene_indices(matrix, n)


def marker_gene_indices(var_names: np.ndarray, markers: list[str]) -> np.ndarray:
    if not markers:
        return np.array([], dtype=int)
    lookup = {gene.upper(): idx for idx, gene in enumerate(var_names.astype(str))}
    indices = [lookup[gene.upper()] for gene in markers if gene.upper() in lookup]
    return np.array(sorted(set(indices)), dtype=int)


def condition_pairs(condition_order: list[str]) -> list[tuple[str, str]]:
    pairs = []
    for i, left in enumerate(condition_order):
        for right in condition_order[i + 1 :]:
            pairs.append((left, right))
    return pairs


def plot_similarity_heatmap(
    table: pd.DataFrame,
    metric: str,
    gene_set: str,
    comparison: str,
    output_path: Path,
    dpi: int,
) -> None:
    subset = table[
        (table["metric"] == metric)
        & (table["gene_set"] == gene_set)
        & (table["comparison"] == comparison)
    ]
    if subset.empty:
        return

    pivot = subset.pivot_table(
        index="cell_type",
        columns="comparison",
        values="similarity",
        observed=False,
    )
    if pivot.empty:
        return

    fig_height = max(3.0, 0.38 * len(pivot.index) + 1.2)
    fig, ax = plt.subplots(figsize=(4.2, fig_height))
    sns.heatmap(
        pivot,
        vmin=-1,
        vmax=1,
        cmap="vlag",
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        cbar_kws={"label": "Similarity"},
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title(f"{metric}: {gene_set} ({comparison})")
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_combined_heatmap(
    table: pd.DataFrame,
    metric: str,
    gene_set: str,
    output_path: Path,
    dpi: int,
) -> None:
    subset = table[(table["metric"] == metric) & (table["gene_set"] == gene_set)]
    if subset.empty:
        return

    pivot = subset.pivot_table(
        index="cell_type",
        columns="comparison",
        values="similarity",
        observed=False,
    )
    if pivot.empty:
        return

    fig_width = max(5.0, 1.25 * len(pivot.columns) + 2.0)
    fig_height = max(3.0, 0.38 * len(pivot.index) + 1.2)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    sns.heatmap(
        pivot,
        vmin=-1,
        vmax=1,
        cmap="vlag",
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        cbar_kws={"label": "Similarity"},
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title(f"Pseudobulk similarity: {metric}, {gene_set}")
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
    LOGGER.info("Using condition column: %s", condition_column)
    LOGGER.info("Using cell-type column: %s", cell_type_column)

    metadata = config.get("metadata", {})
    condition_order = metadata.get("condition_order", ["Control", "LD", "NMDA"])
    metrics = config.get("analysis", {}).get("similarity_metrics", ["pearson", "spearman", "cosine"])
    min_cells = int(config.get("analysis", {}).get("min_cells_per_group", 30))
    gene_cfg = config.get("analysis", {}).get("gene_sets", {})
    hvg_ns = gene_cfg.get("hvg_n_top_genes", [500, 1000, 2000])
    hvg_flavor = gene_cfg.get("hvg_flavor", "seurat")
    min_mean_expression = float(gene_cfg.get("min_mean_expression", 0.0))
    marker_cfg = gene_cfg.get("marker_genes") or {}

    expression, var_names, source_name = get_expression_source(adata, config)
    LOGGER.info("Using expression source: %s (%s genes)", source_name, len(var_names))

    obs = adata.obs.copy()
    obs["_condition"] = normalize_labels(obs[condition_column])
    obs["_cell_type"] = normalize_labels(obs[cell_type_column])

    retinal_mask = retinal_cell_type_mask(obs, "_cell_type", config)
    observed_cell_types = sorted(obs.loc[retinal_mask, "_cell_type"].dropna().unique())
    requested_cell_types = args.cell_types or observed_cell_types

    records = []
    pseudobulk_records = []
    skipped_records = []

    for cell_type in requested_cell_types:
        cell_mask = (obs["_cell_type"] == cell_type).to_numpy()
        if not cell_mask.any():
            skipped_records.append(
                {"cell_type": cell_type, "reason": "cell type not found", "n_cells": 0}
            )
            continue

        condition_counts = (
            obs.loc[cell_mask, "_condition"].value_counts().reindex(condition_order, fill_value=0)
        )
        low_conditions = condition_counts[condition_counts < min_cells]
        if not low_conditions.empty:
            LOGGER.warning(
                "Cell type %s has low cell counts for %s",
                cell_type,
                ", ".join(f"{idx}={val}" for idx, val in low_conditions.items()),
            )

        present_conditions = [condition for condition in condition_order if condition_counts[condition] > 0]
        if len(present_conditions) < 2:
            skipped_records.append(
                {
                    "cell_type": cell_type,
                    "reason": "fewer than two conditions present",
                    "n_cells": int(cell_mask.sum()),
                }
            )
            continue

        cell_expression = expression[cell_mask]
        expressed_idx = np.where(expressed_gene_mask(cell_expression, min_mean_expression))[0]
        if expressed_idx.size == 0:
            skipped_records.append(
                {"cell_type": cell_type, "reason": "no expressed genes after filtering", "n_cells": int(cell_mask.sum())}
            )
            continue

        gene_sets: dict[str, np.ndarray] = {"shared_expressed": expressed_idx}
        for n_top in hvg_ns:
            variable_local_idx = scanpy_hvg_gene_indices(
                cell_expression[:, expressed_idx],
                int(n_top),
                hvg_flavor,
            )
            gene_sets[f"hvg_{int(n_top)}"] = expressed_idx[variable_local_idx]

        markers = marker_cfg.get(cell_type, [])
        marker_idx = marker_gene_indices(var_names, markers)
        marker_idx = np.intersect1d(marker_idx, expressed_idx)
        if marker_idx.size > 0:
            gene_sets["markers"] = marker_idx

        centroids: dict[str, np.ndarray] = {}
        for condition in present_conditions:
            group_mask = cell_mask & (obs["_condition"] == condition).to_numpy()
            centroid = mean_expression_by_mask(expression, group_mask)
            centroids[condition] = centroid
            pseudobulk_records.append(
                {
                    "cell_type": cell_type,
                    "condition": condition,
                    "n_cells": int(group_mask.sum()),
                    "expression_source": source_name,
                }
            )

        for gene_set_name, gene_idx in gene_sets.items():
            if gene_idx.size < 2:
                continue
            for left, right in condition_pairs(condition_order):
                if left not in centroids or right not in centroids:
                    continue
                for metric in metrics:
                    similarity = finite_similarity(
                        centroids[left][gene_idx],
                        centroids[right][gene_idx],
                        metric,
                    )
                    records.append(
                        {
                            "cell_type": cell_type,
                            "comparison": f"{left}_vs_{right}",
                            "condition_a": left,
                            "condition_b": right,
                            "metric": metric,
                            "gene_set": gene_set_name,
                            "n_genes": int(gene_idx.size),
                            "similarity": similarity,
                            "n_cells_a": int(condition_counts[left]),
                            "n_cells_b": int(condition_counts[right]),
                            "min_cells_per_group": min_cells,
                            "low_cell_count_warning": bool(
                                condition_counts[left] < min_cells or condition_counts[right] < min_cells
                            ),
                            "expression_source": source_name,
                        }
                    )

    similarity = pd.DataFrame.from_records(records)
    similarity_path = tables_dir / "pseudobulk_similarity_scores.csv"
    similarity.to_csv(similarity_path, index=False)
    LOGGER.info("Saved similarity scores: %s", similarity_path)

    pseudobulk_meta = pd.DataFrame.from_records(pseudobulk_records)
    pseudobulk_meta_path = tables_dir / "pseudobulk_group_metadata.csv"
    pseudobulk_meta.to_csv(pseudobulk_meta_path, index=False)
    LOGGER.info("Saved pseudobulk metadata: %s", pseudobulk_meta_path)

    skipped = pd.DataFrame.from_records(skipped_records)
    skipped_path = tables_dir / "pseudobulk_skipped_cell_types.csv"
    skipped.to_csv(skipped_path, index=False)
    LOGGER.info("Saved skipped cell types: %s", skipped_path)

    if similarity.empty:
        LOGGER.warning("No similarity scores were produced.")
        return

    output_cfg = config.get("outputs", {})
    dpi = int(output_cfg.get("figure_dpi", 300))
    fmt = output_cfg.get("figure_format", "png")

    preferred_gene_sets = ["shared_expressed"] + [f"hvg_{int(n)}" for n in hvg_ns]
    if "markers" in similarity["gene_set"].unique():
        preferred_gene_sets.append("markers")

    for metric in metrics:
        for gene_set in preferred_gene_sets:
            if gene_set not in similarity["gene_set"].unique():
                continue
            plot_combined_heatmap(
                similarity,
                metric=metric,
                gene_set=gene_set,
                output_path=figures_dir / f"pseudobulk_similarity_{metric}_{gene_set}.{fmt}",
                dpi=dpi,
            )
            for comparison in similarity["comparison"].dropna().unique():
                plot_similarity_heatmap(
                    similarity,
                    metric=metric,
                    gene_set=gene_set,
                    comparison=comparison,
                    output_path=figures_dir
                    / f"pseudobulk_similarity_{metric}_{gene_set}_{comparison}.{fmt}",
                    dpi=dpi,
                )

    LOGGER.info("Done.")


if __name__ == "__main__":
    main()

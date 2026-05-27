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
        description="Pool LD and NMDA as injury/regenerated and compare with Control."
    )
    parser.add_argument("--config", default="config/config.yaml", help="Path to config YAML.")
    parser.add_argument("--h5ad", default=None, help="Override input .h5ad path from config.")
    parser.add_argument(
        "--gene-sets",
        nargs="*",
        default=["shared_expressed", "hvg_2000"],
        help="Gene sets to analyze.",
    )
    parser.add_argument(
        "--metrics",
        nargs="*",
        default=None,
        help="Similarity metrics to plot. Defaults to config metrics.",
    )
    parser.add_argument("--combined-label", default="LD_NMDA", help="Name for pooled LD+NMDA.")
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


def build_gene_sets(
    expression,
    retinal_mask: np.ndarray,
    requested_gene_sets: list[str],
    hvg_flavor: str,
    min_mean_expression: float,
) -> dict[str, np.ndarray]:
    retinal_expression = expression[retinal_mask]
    expressed_idx = np.where(expressed_gene_mask(retinal_expression, min_mean_expression))[0]
    gene_sets: dict[str, np.ndarray] = {}
    if "shared_expressed" in requested_gene_sets:
        gene_sets["shared_expressed"] = expressed_idx
    for gene_set in requested_gene_sets:
        if gene_set.startswith("hvg_"):
            n_top = int(gene_set.split("_", 1)[1])
            local_idx = scanpy_hvg_gene_indices(
                retinal_expression[:, expressed_idx],
                n_top,
                hvg_flavor,
            )
            gene_sets[gene_set] = expressed_idx[local_idx]
    return gene_sets


def plot_control_vs_combined(
    scores: pd.DataFrame,
    metric: str,
    gene_set: str,
    combined_label: str,
    figures_dir: Path,
    dpi: int,
    fmt: str,
) -> None:
    subset = scores[(scores["metric"] == metric) & (scores["gene_set"] == gene_set)]
    if subset.empty:
        return
    heat = subset.pivot_table(
        index="cell_type",
        columns="query_group",
        values="similarity",
        observed=False,
    )
    heat = heat[[combined_label]]

    fig, ax = plt.subplots(figsize=(3.8, max(3.2, 0.36 * len(heat) + 1.2)))
    sns.heatmap(
        heat,
        vmin=0,
        vmax=1,
        cmap="viridis",
        annot=True,
        fmt=".2f",
        linewidths=0.4,
        cbar_kws={"label": f"{metric.title()} similarity to Control"},
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title(f"Control vs pooled injury\n{metric}, {gene_set}")
    fig.tight_layout()
    fig.savefig(
        figures_dir / f"combined_injury_control_vs_{combined_label}_{metric}_{gene_set}.{fmt}",
        dpi=dpi,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_cross_type_heatmap(
    matrix: pd.DataFrame,
    metric: str,
    gene_set: str,
    combined_label: str,
    figures_dir: Path,
    dpi: int,
    fmt: str,
) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 4.8))
    sns.heatmap(
        matrix,
        vmin=0,
        vmax=1,
        cmap="viridis",
        annot=True,
        fmt=".2f",
        annot_kws={"fontsize": 5.5},
        linewidths=0.25,
        square=True,
        cbar_kws={"label": f"{metric.title()} similarity"},
        ax=ax,
    )
    ax.set_title(f"Cell Type Similarity: Control vs {combined_label}\n{metric}, {gene_set}")
    ax.set_xlabel(f"Query: {combined_label}")
    ax.set_ylabel("Reference: Control")
    ax.tick_params(axis="x", labelrotation=90)
    ax.tick_params(axis="y", labelrotation=0)
    fig.tight_layout()
    fig.savefig(
        figures_dir
        / f"combined_injury_cross_type_control_vs_{combined_label}_{metric}_{gene_set}.{fmt}",
        dpi=dpi,
        bbox_inches="tight",
    )
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
    control_label = metadata.get("control_label", "Control")
    injury_labels = list(metadata.get("injury_labels", ["LD", "NMDA"]))
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
    obs["_condition_combined"] = obs["_condition"].where(
        ~obs["_condition"].isin(injury_labels),
        args.combined_label,
    )

    retinal_mask = retinal_cell_type_mask(obs, "_cell_type", config).to_numpy()
    cell_types = sorted(obs.loc[retinal_mask, "_cell_type"].dropna().unique())
    LOGGER.info("Retinal cell types: %s", ", ".join(cell_types))

    gene_sets = build_gene_sets(
        expression,
        retinal_mask,
        args.gene_sets,
        hvg_flavor=hvg_flavor,
        min_mean_expression=min_mean_expression,
    )

    records = []
    cross_type_records = []
    group_records = []
    cross_type_matrices: dict[tuple[str, str], pd.DataFrame] = {}
    for gene_set, gene_idx in gene_sets.items():
        control_centroids = {}
        injury_centroids = {}
        for cell_type in cell_types:
            control_mask = (
                (obs["_cell_type"] == cell_type).to_numpy()
                & (obs["_condition_combined"] == control_label).to_numpy()
            )
            injury_mask = (
                (obs["_cell_type"] == cell_type).to_numpy()
                & (obs["_condition_combined"] == args.combined_label).to_numpy()
            )
            n_control = int(control_mask.sum())
            n_injury = int(injury_mask.sum())
            group_records.extend(
                [
                    {
                        "gene_set": gene_set,
                        "cell_type": cell_type,
                        "group": control_label,
                        "n_cells": n_control,
                    },
                    {
                        "gene_set": gene_set,
                        "cell_type": cell_type,
                        "group": args.combined_label,
                        "n_cells": n_injury,
                    },
                ]
            )
            if n_control == 0 or n_injury == 0:
                continue
            control_centroid = mean_expression_by_mask(expression[:, gene_idx], control_mask)
            injury_centroid = mean_expression_by_mask(expression[:, gene_idx], injury_mask)
            control_centroids[cell_type] = control_centroid
            injury_centroids[cell_type] = injury_centroid
            for metric in metrics:
                records.append(
                    {
                        "cell_type": cell_type,
                        "comparison": f"{control_label}_vs_{args.combined_label}",
                        "reference_group": control_label,
                        "query_group": args.combined_label,
                        "metric": metric,
                        "gene_set": gene_set,
                        "n_genes": int(len(gene_idx)),
                        "similarity": finite_similarity(
                            control_centroid,
                            injury_centroid,
                            metric,
                        ),
                        "n_control": n_control,
                        "n_combined_injury": n_injury,
                        "injury_labels_pooled": ",".join(injury_labels),
                        "low_cell_count_warning": bool(
                            n_control < min_cells or n_injury < min_cells
                        ),
                        "expression_source": source_name,
                    }
                )

        for metric in metrics:
            matrix = pd.DataFrame(index=cell_types, columns=cell_types, dtype=float)
            for reference_cell_type in cell_types:
                if reference_cell_type not in control_centroids:
                    continue
                for query_cell_type in cell_types:
                    if query_cell_type not in injury_centroids:
                        continue
                    similarity = finite_similarity(
                        control_centroids[reference_cell_type],
                        injury_centroids[query_cell_type],
                        metric,
                    )
                    matrix.loc[reference_cell_type, query_cell_type] = similarity
                    cross_type_records.append(
                        {
                            "comparison": f"{control_label}_vs_{args.combined_label}",
                            "reference_group": control_label,
                            "query_group": args.combined_label,
                            "reference_cell_type": reference_cell_type,
                            "query_cell_type": query_cell_type,
                            "metric": metric,
                            "gene_set": gene_set,
                            "n_genes": int(len(gene_idx)),
                            "similarity": similarity,
                            "expression_source": source_name,
                        }
                    )
            cross_type_matrices[(metric, gene_set)] = matrix
            matrix.to_csv(
                tables_dir
                / f"combined_injury_cross_type_matrix_{metric}_{gene_set}.csv"
            )

    scores = pd.DataFrame.from_records(records)
    scores.to_csv(tables_dir / "combined_injury_control_similarity.csv", index=False)
    pd.DataFrame.from_records(cross_type_records).to_csv(
        tables_dir / "combined_injury_cross_type_similarity.csv",
        index=False,
    )
    pd.DataFrame.from_records(group_records).to_csv(
        tables_dir / "combined_injury_group_counts.csv",
        index=False,
    )
    LOGGER.info("Saved combined injury similarity tables.")

    output_cfg = config.get("outputs", {})
    dpi = int(output_cfg.get("figure_dpi", 300))
    fmt = output_cfg.get("figure_format", "png")
    for metric in metrics:
        for gene_set in gene_sets:
            plot_control_vs_combined(
                scores,
                metric=metric,
                gene_set=gene_set,
                combined_label=args.combined_label,
                figures_dir=figures_dir,
                dpi=dpi,
                fmt=fmt,
            )
            plot_cross_type_heatmap(
                cross_type_matrices[(metric, gene_set)],
                metric=metric,
                gene_set=gene_set,
                combined_label=args.combined_label,
                figures_dir=figures_dir,
                dpi=dpi,
                fmt=fmt,
            )

    LOGGER.info("Done.")


if __name__ == "__main__":
    main()

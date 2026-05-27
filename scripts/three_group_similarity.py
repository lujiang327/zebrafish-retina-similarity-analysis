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
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

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
        description="Compare Control, LD, and NMDA together using heatmaps and centroid PCA."
    )
    parser.add_argument("--config", default="config/config.yaml", help="Path to config YAML.")
    parser.add_argument("--h5ad", default=None, help="Override input .h5ad path from config.")
    parser.add_argument(
        "--gene-sets",
        nargs="*",
        default=["shared_expressed", "hvg_1000", "hvg_2000"],
        help="Gene sets to analyze.",
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
        if not gene_set.startswith("hvg_"):
            continue
        n_top = int(gene_set.split("_", 1)[1])
        local_idx = scanpy_hvg_gene_indices(
            retinal_expression[:, expressed_idx],
            n_top,
            hvg_flavor,
        )
        gene_sets[gene_set] = expressed_idx[local_idx]
    return gene_sets


def plot_control_referenced_heatmap(
    summary: pd.DataFrame,
    metric: str,
    gene_set: str,
    output_path: Path,
    dpi: int,
) -> None:
    subset = summary[
        (summary["metric"] == metric)
        & (summary["gene_set"] == gene_set)
        & summary["comparison"].isin(["Control_vs_LD", "Control_vs_NMDA"])
    ].copy()
    if subset.empty:
        return
    subset["query_condition"] = subset["condition_b"]
    heat = subset.pivot_table(
        index="cell_type",
        columns="query_condition",
        values="similarity",
        observed=False,
    )
    heat = heat[[col for col in ["LD", "NMDA"] if col in heat.columns]]

    fig, ax = plt.subplots(figsize=(4.2, max(3.2, 0.36 * len(heat) + 1.2)))
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
    ax.set_title(f"Control-referenced fidelity: {metric}, {gene_set}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_delta_heatmap(
    summary: pd.DataFrame,
    metric: str,
    gene_set: str,
    output_path: Path,
    dpi: int,
) -> None:
    subset = summary[
        (summary["metric"] == metric)
        & (summary["gene_set"] == gene_set)
        & summary["comparison"].isin(["Control_vs_LD", "Control_vs_NMDA"])
    ].copy()
    if subset.empty:
        return
    wide = subset.pivot_table(
        index="cell_type",
        columns="comparison",
        values="similarity",
        observed=False,
    )
    if not {"Control_vs_LD", "Control_vs_NMDA"}.issubset(wide.columns):
        return
    delta = pd.DataFrame(
        {
            "NMDA_minus_LD": wide["Control_vs_NMDA"] - wide["Control_vs_LD"],
        }
    )
    max_abs = float(np.nanmax(np.abs(delta.to_numpy())))
    max_abs = max(max_abs, 0.01)

    fig, ax = plt.subplots(figsize=(3.6, max(3.2, 0.36 * len(delta) + 1.2)))
    sns.heatmap(
        delta,
        vmin=-max_abs,
        vmax=max_abs,
        cmap="vlag",
        annot=True,
        fmt=".2f",
        linewidths=0.4,
        cbar_kws={"label": "Control similarity difference"},
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title(f"Delta fidelity: NMDA - LD\n{metric}, {gene_set}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_centroid_pca(
    centroid_table: pd.DataFrame,
    gene_set: str,
    output_path: Path,
    dpi: int,
) -> None:
    subset = centroid_table[centroid_table["gene_set"] == gene_set].copy()
    if subset.empty:
        return
    feature_columns = [
        col for col in subset.columns if col.startswith("gene_") and col[5:].isdigit()
    ]
    feature_columns = [col for col in feature_columns if subset[col].notna().all()]
    if len(feature_columns) < 2:
        LOGGER.warning("Skipping PCA for %s: fewer than two complete gene columns.", gene_set)
        return
    x = subset[feature_columns].to_numpy()
    x = StandardScaler().fit_transform(x)
    pca = PCA(n_components=2, random_state=0)
    coords = pca.fit_transform(x)
    subset["PC1"] = coords[:, 0]
    subset["PC2"] = coords[:, 1]

    fig, ax = plt.subplots(figsize=(7, 5.4))
    sns.scatterplot(
        data=subset,
        x="PC1",
        y="PC2",
        hue="condition",
        style="cell_type",
        s=85,
        edgecolor="black",
        linewidth=0.4,
        ax=ax,
    )
    for _, row in subset.iterrows():
        ax.text(
            row["PC1"],
            row["PC2"],
            f" {row['cell_type']}",
            fontsize=7,
            va="center",
        )
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)")
    ax.set_title(f"Three-condition pseudobulk centroid PCA: {gene_set}")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
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
    control_label = metadata.get("control_label", "Control")
    injury_labels = list(metadata.get("injury_labels", ["LD", "NMDA"]))
    condition_order = [control_label] + injury_labels
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
    centroid_records = []
    group_records = []

    for gene_set, gene_idx in gene_sets.items():
        centroids: dict[tuple[str, str], np.ndarray] = {}
        for cell_type in cell_types:
            for condition in condition_order:
                group_mask = (
                    (obs["_cell_type"] == cell_type).to_numpy()
                    & (obs["_condition"] == condition).to_numpy()
                )
                n_cells = int(group_mask.sum())
                group_records.append(
                    {
                        "gene_set": gene_set,
                        "cell_type": cell_type,
                        "condition": condition,
                        "n_cells": n_cells,
                    }
                )
                if n_cells == 0:
                    continue
                centroid = mean_expression_by_mask(expression[:, gene_idx], group_mask)
                centroids[(cell_type, condition)] = centroid
                centroid_records.append(
                    {
                        "gene_set": gene_set,
                        "cell_type": cell_type,
                        "condition": condition,
                        "n_cells": n_cells,
                        **{f"gene_{i}": value for i, value in enumerate(centroid)},
                    }
                )

            for condition_b in injury_labels:
                if (cell_type, control_label) not in centroids or (
                    cell_type,
                    condition_b,
                ) not in centroids:
                    continue
                for metric in metrics:
                    similarity = finite_similarity(
                        centroids[(cell_type, control_label)],
                        centroids[(cell_type, condition_b)],
                        metric,
                    )
                    records.append(
                        {
                            "cell_type": cell_type,
                            "comparison": f"{control_label}_vs_{condition_b}",
                            "condition_a": control_label,
                            "condition_b": condition_b,
                            "metric": metric,
                            "gene_set": gene_set,
                            "n_genes": int(len(gene_idx)),
                            "similarity": similarity,
                            "n_cells_a": int(
                                (
                                    (obs["_cell_type"] == cell_type)
                                    & (obs["_condition"] == control_label)
                                ).sum()
                            ),
                            "n_cells_b": int(
                                (
                                    (obs["_cell_type"] == cell_type)
                                    & (obs["_condition"] == condition_b)
                                ).sum()
                            ),
                            "low_cell_count_warning": bool(
                                (
                                    (
                                        (obs["_cell_type"] == cell_type)
                                        & (obs["_condition"] == control_label)
                                    ).sum()
                                    < min_cells
                                )
                                or (
                                    (
                                        (obs["_cell_type"] == cell_type)
                                        & (obs["_condition"] == condition_b)
                                    ).sum()
                                    < min_cells
                                )
                            ),
                            "expression_source": source_name,
                        }
                    )

    summary = pd.DataFrame.from_records(records)
    summary.to_csv(tables_dir / "three_group_control_referenced_similarity.csv", index=False)
    pd.DataFrame.from_records(group_records).to_csv(
        tables_dir / "three_group_centroid_group_counts.csv",
        index=False,
    )
    centroid_table = pd.DataFrame.from_records(centroid_records)
    centroid_table.to_csv(tables_dir / "three_group_centroids.csv", index=False)
    LOGGER.info("Saved three-group tables.")

    if summary.empty:
        LOGGER.warning("No three-group similarity scores were produced.")
        return

    delta_rows = []
    for (cell_type, metric, gene_set), group in summary.groupby(
        ["cell_type", "metric", "gene_set"], observed=False
    ):
        values = group.set_index("comparison")["similarity"]
        if {"Control_vs_LD", "Control_vs_NMDA"}.issubset(values.index):
            delta_rows.append(
                {
                    "cell_type": cell_type,
                    "metric": metric,
                    "gene_set": gene_set,
                    "control_vs_ld": values["Control_vs_LD"],
                    "control_vs_nmda": values["Control_vs_NMDA"],
                    "nmda_minus_ld": values["Control_vs_NMDA"]
                    - values["Control_vs_LD"],
                }
            )
    pd.DataFrame(delta_rows).to_csv(
        tables_dir / "three_group_nmda_minus_ld_delta.csv",
        index=False,
    )

    output_cfg = config.get("outputs", {})
    dpi = int(output_cfg.get("figure_dpi", 300))
    fmt = output_cfg.get("figure_format", "png")

    for metric in metrics:
        for gene_set in gene_sets:
            plot_control_referenced_heatmap(
                summary,
                metric=metric,
                gene_set=gene_set,
                output_path=figures_dir
                / f"three_group_control_referenced_{metric}_{gene_set}.{fmt}",
                dpi=dpi,
            )
            plot_delta_heatmap(
                summary,
                metric=metric,
                gene_set=gene_set,
                output_path=figures_dir
                / f"three_group_delta_nmda_minus_ld_{metric}_{gene_set}.{fmt}",
                dpi=dpi,
            )
    for gene_set in gene_sets:
        plot_centroid_pca(
            centroid_table,
            gene_set=gene_set,
            output_path=figures_dir / f"three_group_centroid_pca_{gene_set}.{fmt}",
            dpi=dpi,
        )

    LOGGER.info("Done.")


if __name__ == "__main__":
    main()

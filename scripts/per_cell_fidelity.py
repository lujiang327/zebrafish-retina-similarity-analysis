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
from scipy.stats import rankdata

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
        description="Compute per-cell fidelity to same-cell-type Control centroids."
    )
    parser.add_argument("--config", default="config/config.yaml", help="Path to config YAML.")
    parser.add_argument("--h5ad", default=None, help="Override input .h5ad path from config.")
    parser.add_argument("--metrics", nargs="*", default=None, help="Override metrics.")
    parser.add_argument("--gene-sets", nargs="*", default=None, help="Override gene sets.")
    parser.add_argument(
        "--output-subdir",
        default=None,
        help="Optional subdirectory name under results/tables and results/figures.",
    )
    parser.add_argument("--verbose", action="store_true", help="Print debug logging.")
    return parser.parse_args()


def expressed_gene_mask(matrix, min_mean_expression: float) -> np.ndarray:
    if sparse.issparse(matrix):
        means = np.asarray(matrix.mean(axis=0)).ravel()
    else:
        means = np.asarray(matrix).mean(axis=0).ravel()
    return means > min_mean_expression


def fraction_expressed_gene_indices(matrix, min_fraction: float) -> np.ndarray:
    if sparse.issparse(matrix):
        fractions = np.asarray((matrix > 0).mean(axis=0)).ravel()
    else:
        fractions = (np.asarray(matrix) > 0).mean(axis=0)
    return np.where(fractions >= min_fraction)[0]


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
    except Exception as exc:  # pragma: no cover - fallback is data dependent
        LOGGER.warning(
            "Scanpy HVG selection failed for n_top=%s, flavor=%s; using variance fallback. Error: %s",
            n_top,
            flavor,
            exc,
        )
    return variance_gene_indices(matrix, n)


def enriched_gene_indices(
    target_matrix,
    background_matrix,
    n_top: int,
    min_target_fraction: float = 0.1,
    min_log2_fc: float = 0.25,
) -> np.ndarray:
    if sparse.issparse(target_matrix):
        target_mean = np.asarray(target_matrix.mean(axis=0)).ravel()
        target_fraction = np.asarray((target_matrix > 0).mean(axis=0)).ravel()
    else:
        target = np.asarray(target_matrix)
        target_mean = target.mean(axis=0)
        target_fraction = (target > 0).mean(axis=0)

    if sparse.issparse(background_matrix):
        background_mean = np.asarray(background_matrix.mean(axis=0)).ravel()
    else:
        background_mean = np.asarray(background_matrix).mean(axis=0)

    log2_fc = np.log2((target_mean + 1e-9) / (background_mean + 1e-9))
    keep = (target_fraction >= min_target_fraction) & (log2_fc >= min_log2_fc)
    candidate_idx = np.where(keep)[0]
    if candidate_idx.size == 0:
        return np.array([], dtype=int)
    ordered = candidate_idx[np.argsort(log2_fc[candidate_idx])[::-1]]
    return ordered[: min(int(n_top), ordered.size)]


def dense_matrix(matrix) -> np.ndarray:
    if sparse.issparse(matrix):
        return matrix.toarray()
    return np.asarray(matrix)


def score_matrix_to_centroid(matrix, centroid: np.ndarray, metric: str) -> np.ndarray:
    x = dense_matrix(matrix).astype(float, copy=False)
    y = np.asarray(centroid, dtype=float)

    valid = np.isfinite(y)
    x = x[:, valid]
    y = y[valid]
    if y.size < 2 or np.allclose(y, y[0]):
        return np.full(x.shape[0], np.nan)

    if metric == "cosine":
        y_norm = np.linalg.norm(y)
        x_norm = np.linalg.norm(x, axis=1)
        denom = x_norm * y_norm
        scores = np.full(x.shape[0], np.nan)
        nonzero = denom > 0
        scores[nonzero] = (x[nonzero] @ y) / denom[nonzero]
        return scores

    if metric == "pearson":
        y_centered = y - y.mean()
        y_norm = np.linalg.norm(y_centered)
        x_centered = x - x.mean(axis=1, keepdims=True)
        x_norm = np.linalg.norm(x_centered, axis=1)
        denom = x_norm * y_norm
        scores = np.full(x.shape[0], np.nan)
        nonzero = denom > 0
        scores[nonzero] = (x_centered[nonzero] @ y_centered) / denom[nonzero]
        return scores

    if metric == "spearman":
        y_rank = rankdata(y)
        scores = np.empty(x.shape[0], dtype=float)
        for i in range(x.shape[0]):
            scores[i] = finite_similarity(rankdata(x[i, :]), y_rank, "pearson")
        return scores

    raise ValueError(f"Unsupported metric: {metric}")


def select_gene_sets(
    control_expression,
    requested_gene_sets: list[str],
    hvg_flavor: str,
    min_mean_expression: float,
    cell_type: str,
    var_names: np.ndarray,
    marker_cfg: dict,
    background_expression=None,
) -> dict[str, np.ndarray]:
    expressed_idx = np.where(expressed_gene_mask(control_expression, min_mean_expression))[0]
    selected: dict[str, np.ndarray] = {}
    if "shared_expressed" in requested_gene_sets:
        selected["shared_expressed"] = expressed_idx
    if "expressed_frac_5pct" in requested_gene_sets:
        selected["expressed_frac_5pct"] = fraction_expressed_gene_indices(
            control_expression,
            0.05,
        )
    if "expressed_frac_10pct" in requested_gene_sets:
        selected["expressed_frac_10pct"] = fraction_expressed_gene_indices(
            control_expression,
            0.10,
        )
    for gene_set in requested_gene_sets:
        if gene_set.startswith("enriched_top_"):
            if background_expression is None or background_expression.shape[0] == 0:
                continue
            n_top = int(gene_set.split("_")[-1])
            selected[gene_set] = enriched_gene_indices(
                control_expression,
                background_expression,
                n_top=n_top,
                min_target_fraction=0.10,
                min_log2_fc=0.25,
            )
    for gene_set in requested_gene_sets:
        if not gene_set.startswith("hvg_"):
            continue
        n_top = int(gene_set.split("_", 1)[1])
        local_idx = scanpy_hvg_gene_indices(
            control_expression[:, expressed_idx],
            n_top,
            hvg_flavor,
        )
        selected[gene_set] = expressed_idx[local_idx]
    if "identity_markers" in requested_gene_sets or "markers" in requested_gene_sets:
        marker_gene_set_name = (
            "identity_markers" if "identity_markers" in requested_gene_sets else "markers"
        )
        markers = marker_cfg.get(cell_type, [])
        lookup = {gene.upper(): idx for idx, gene in enumerate(var_names.astype(str))}
        marker_idx = [
            lookup[marker.upper()] for marker in markers if marker.upper() in lookup
        ]
        marker_idx = np.array(sorted(set(marker_idx)), dtype=int)
        marker_idx = np.intersect1d(marker_idx, expressed_idx)
        if marker_idx.size > 0:
            selected[marker_gene_set_name] = marker_idx
    return selected


def plot_combined_violin(
    scores: pd.DataFrame,
    metric: str,
    gene_set: str,
    figures_dir: Path,
    dpi: int,
    fmt: str,
) -> None:
    subset = scores[(scores["metric"] == metric) & (scores["gene_set"] == gene_set)].copy()
    if subset.empty:
        return
    subset["condition"] = pd.Categorical(
        subset["condition"], categories=["Control_heldout", "LD", "NMDA"], ordered=True
    )
    fig, ax = plt.subplots(figsize=(max(8, 0.8 * subset["cell_type"].nunique()), 5))
    sns.violinplot(
        data=subset,
        x="cell_type",
        y="fidelity_score",
        hue="condition",
        cut=0,
        inner="quartile",
        linewidth=0.7,
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel(f"{metric.title()} similarity to Control centroid")
    ax.set_title(f"Per-cell fidelity: {metric}, {gene_set}")
    ax.tick_params(axis="x", labelrotation=45)
    ax.legend(title="Condition", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(
        figures_dir / f"per_cell_fidelity_violin_{metric}_{gene_set}.{fmt}",
        dpi=dpi,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_cell_type_facets(
    scores: pd.DataFrame,
    metric: str,
    gene_set: str,
    figures_dir: Path,
    dpi: int,
    fmt: str,
) -> None:
    subset = scores[(scores["metric"] == metric) & (scores["gene_set"] == gene_set)].copy()
    if subset.empty:
        return
    subset["condition"] = pd.Categorical(
        subset["condition"], categories=["Control_heldout", "LD", "NMDA"], ordered=True
    )
    grid = sns.catplot(
        data=subset,
        x="condition",
        y="fidelity_score",
        col="cell_type",
        col_wrap=3,
        kind="box",
        sharey=True,
        height=2.6,
        aspect=1.1,
        fliersize=0.8,
        linewidth=0.8,
    )
    grid.set_axis_labels("", f"{metric.title()} similarity")
    grid.set_titles("{col_name}")
    for ax in grid.axes.flat:
        ax.tick_params(axis="x", labelrotation=45)
    grid.fig.suptitle(f"Per-cell fidelity by cell type: {metric}, {gene_set}", y=1.02)
    grid.fig.savefig(
        figures_dir / f"per_cell_fidelity_box_by_cell_type_{metric}_{gene_set}.{fmt}",
        dpi=dpi,
        bbox_inches="tight",
    )
    plt.close(grid.fig)


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)
    config = load_config(args.config)
    tables_dir, figures_dir = ensure_output_dirs(config)
    if args.output_subdir:
        tables_dir = tables_dir / args.output_subdir
        figures_dir = figures_dir / args.output_subdir
        tables_dir.mkdir(parents=True, exist_ok=True)
        figures_dir.mkdir(parents=True, exist_ok=True)

    h5ad_path = Path(args.h5ad or config["data"]["h5ad_path"])
    LOGGER.info("Loading AnnData: %s", h5ad_path)
    adata = sc.read_h5ad(h5ad_path)

    condition_column, cell_type_column = resolve_metadata_columns(adata, config)
    expression, var_names, source_name = get_expression_source(adata, config)
    LOGGER.info("Using condition column: %s", condition_column)
    LOGGER.info("Using cell-type column: %s", cell_type_column)
    LOGGER.info("Using expression source: %s", source_name)

    metadata = config.get("metadata", {})
    control_label = metadata.get("control_label", "Control")
    injury_labels = metadata.get("injury_labels", ["LD", "NMDA"])
    condition_order = [control_label] + list(injury_labels)

    analysis_cfg = config.get("analysis", {})
    min_cells = int(analysis_cfg.get("min_cells_per_group", 30))
    random_state = int(analysis_cfg.get("random_state", 0))
    gene_cfg = analysis_cfg.get("gene_sets", {})
    hvg_flavor = gene_cfg.get("hvg_flavor", "seurat")
    min_mean_expression = float(gene_cfg.get("min_mean_expression", 0.0))
    marker_cfg = gene_cfg.get("marker_genes") or {}
    per_cell_cfg = analysis_cfg.get("per_cell_fidelity", {})
    metrics = args.metrics or per_cell_cfg.get("metrics") or analysis_cfg.get(
        "similarity_metrics", ["pearson", "spearman", "cosine"]
    )
    requested_gene_sets = args.gene_sets or per_cell_cfg.get("gene_sets", ["hvg_1000"])
    reference_fraction = float(per_cell_cfg.get("control_reference_fraction", 0.7))
    reference_fraction = min(max(reference_fraction, 0.1), 0.9)

    obs = adata.obs.copy()
    obs["_condition"] = normalize_labels(obs[condition_column])
    obs["_cell_type"] = normalize_labels(obs[cell_type_column])
    retinal_mask = retinal_cell_type_mask(obs, "_cell_type", config)
    cell_types = sorted(obs.loc[retinal_mask, "_cell_type"].dropna().unique())

    rng = np.random.default_rng(random_state)
    records = []
    summary_records = []
    skipped_records = []
    marker_records = []
    gene_set_records = []

    control_retinal_mask = (
        retinal_mask.to_numpy() & (obs["_condition"] == control_label).to_numpy()
    )

    for cell_type in cell_types:
        cell_mask = (obs["_cell_type"] == cell_type).to_numpy()
        control_mask = cell_mask & (obs["_condition"] == control_label).to_numpy()
        control_indices = np.where(control_mask)[0]
        if control_indices.size < min_cells:
            skipped_records.append(
                {
                    "cell_type": cell_type,
                    "reason": "too few Control cells for reference",
                    "n_control": int(control_indices.size),
                }
            )
            continue

        shuffled = rng.permutation(control_indices)
        n_reference = max(2, int(np.floor(reference_fraction * shuffled.size)))
        n_reference = min(n_reference, shuffled.size - 1)
        reference_indices = shuffled[:n_reference]
        heldout_indices = shuffled[n_reference:]

        reference_mask = np.zeros(adata.n_obs, dtype=bool)
        reference_mask[reference_indices] = True

        control_expression = expression[reference_mask]
        background_mask = control_retinal_mask & (obs["_cell_type"] != cell_type).to_numpy()
        background_expression = expression[background_mask]
        gene_sets = select_gene_sets(
            control_expression,
            requested_gene_sets,
            hvg_flavor=hvg_flavor,
            min_mean_expression=min_mean_expression,
            cell_type=cell_type,
            var_names=var_names,
            marker_cfg=marker_cfg,
            background_expression=background_expression,
        )
        if not gene_sets:
            skipped_records.append(
                {
                    "cell_type": cell_type,
                    "reason": "no usable gene sets",
                    "n_control": int(control_indices.size),
                }
            )
            continue

        for gene_set, gene_idx in gene_sets.items():
            if gene_idx.size < 2:
                continue
            gene_set_records.append(
                {
                    "cell_type": cell_type,
                    "gene_set": gene_set,
                    "n_genes": int(gene_idx.size),
                    "genes": ";".join(map(str, var_names[gene_idx])),
                }
            )
            if gene_set in {"identity_markers", "markers"}:
                configured = marker_cfg.get(cell_type, [])
                matched = [str(var_names[idx]) for idx in gene_idx]
                marker_records.append(
                    {
                        "cell_type": cell_type,
                        "gene_set": gene_set,
                        "n_configured_markers": len(configured),
                        "n_matched_expressed_markers": len(matched),
                        "configured_markers": ";".join(configured),
                        "matched_expressed_markers": ";".join(matched),
                    }
                )
            centroid = mean_expression_by_mask(expression[:, gene_idx], reference_mask)

            score_indices = list(heldout_indices)
            for condition in injury_labels:
                condition_indices = np.where(
                    cell_mask & (obs["_condition"] == condition).to_numpy()
                )[0]
                score_indices.extend(condition_indices.tolist())

            score_indices = np.array(score_indices, dtype=int)
            if score_indices.size == 0:
                continue

            score_matrix = expression[score_indices][:, gene_idx]
            for metric in metrics:
                scores = score_matrix_to_centroid(score_matrix, centroid, metric)
                for index, score in zip(score_indices, scores):
                    condition = obs.iloc[index]["_condition"]
                    output_condition = (
                        "Control_heldout" if condition == control_label else condition
                    )
                    records.append(
                        {
                            "cell_id": obs.index[index],
                            "cell_type": cell_type,
                            "condition": output_condition,
                            "original_condition": condition,
                            "metric": metric,
                            "gene_set": gene_set,
                            "n_genes": int(gene_idx.size),
                            "fidelity_score": score,
                            "reference_condition": control_label,
                            "reference_n_cells": int(reference_indices.size),
                            "heldout_control_n_cells": int(heldout_indices.size),
                            "expression_source": source_name,
                        }
                    )

                metric_summary = pd.DataFrame(
                    {
                        "condition": [
                            "Control_heldout" if obs.iloc[index]["_condition"] == control_label else obs.iloc[index]["_condition"]
                            for index in score_indices
                        ],
                        "score": scores,
                    }
                )
                for condition, group in metric_summary.groupby("condition"):
                    summary_records.append(
                        {
                            "cell_type": cell_type,
                            "condition": condition,
                            "metric": metric,
                            "gene_set": gene_set,
                            "n_cells": int(group.shape[0]),
                            "mean_fidelity": float(group["score"].mean()),
                            "median_fidelity": float(group["score"].median()),
                            "std_fidelity": float(group["score"].std()),
                            "reference_n_cells": int(reference_indices.size),
                            "low_cell_count_warning": bool(group.shape[0] < min_cells),
                        }
                    )

    scores = pd.DataFrame.from_records(records)
    scores_path = tables_dir / "per_cell_fidelity_scores.csv"
    scores.to_csv(scores_path, index=False)
    LOGGER.info("Saved per-cell scores: %s", scores_path)

    summary = pd.DataFrame.from_records(summary_records)
    summary_path = tables_dir / "per_cell_fidelity_summary.csv"
    summary.to_csv(summary_path, index=False)
    LOGGER.info("Saved per-cell summary: %s", summary_path)

    skipped = pd.DataFrame.from_records(skipped_records)
    skipped_path = tables_dir / "per_cell_fidelity_skipped_cell_types.csv"
    skipped.to_csv(skipped_path, index=False)
    LOGGER.info("Saved skipped cell types: %s", skipped_path)

    if gene_set_records:
        gene_set_report = pd.DataFrame.from_records(gene_set_records).drop_duplicates(
            subset=["cell_type", "gene_set"]
        )
        gene_set_report_path = tables_dir / "per_cell_fidelity_gene_sets.csv"
        gene_set_report.to_csv(gene_set_report_path, index=False)
        LOGGER.info("Saved gene set report: %s", gene_set_report_path)

    if marker_records:
        marker_report = pd.DataFrame.from_records(marker_records).drop_duplicates()
        marker_report_path = tables_dir / "per_cell_fidelity_marker_gene_matches.csv"
        marker_report.to_csv(marker_report_path, index=False)
        LOGGER.info("Saved marker gene match report: %s", marker_report_path)

    if scores.empty:
        LOGGER.warning("No per-cell fidelity scores were produced.")
        return

    output_cfg = config.get("outputs", {})
    dpi = int(output_cfg.get("figure_dpi", 300))
    fmt = output_cfg.get("figure_format", "png")

    for metric in metrics:
        for gene_set in requested_gene_sets:
            plot_combined_violin(scores, metric, gene_set, figures_dir, dpi, fmt)
            plot_cell_type_facets(scores, metric, gene_set, figures_dir, dpi, fmt)

    default_metric = per_cell_cfg.get("default_plot_metric", metrics[0])
    default_gene_set = per_cell_cfg.get("default_plot_gene_set", requested_gene_sets[0])
    umap_subset = scores[
        (scores["metric"] == default_metric) & (scores["gene_set"] == default_gene_set)
    ]
    if "X_umap" in adata.obsm and not umap_subset.empty:
        score_map = umap_subset.set_index("cell_id")["fidelity_score"]
        adata.obs["fidelity_score"] = adata.obs_names.map(score_map)
        sc.pl.umap(
            adata,
            color="fidelity_score",
            frameon=False,
            cmap="viridis",
            show=False,
        )
        plt.savefig(
            figures_dir / f"umap_per_cell_fidelity_{default_metric}_{default_gene_set}.{fmt}",
            dpi=dpi,
            bbox_inches="tight",
        )
        plt.close()

    LOGGER.info("Done.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse


DEFAULT_H5AD = "results/h5ad/bc_9_sample_guca1b_gt2_filtered.h5ad"
DEFAULT_GROUPS = [
    ("merge_28_33", ["28", "33"]),
    ("merge_12_36", ["12", "36"]),
    ("merge_2_29_35", ["2", "29", "35"]),
    ("merge_9_32", ["9", "32"]),
    ("merge_3_31", ["3", "31"]),
    ("merge_14_37_38", ["14", "37", "38"]),
    ("merge_5_26", ["5", "26"]),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Review proposed BC Leiden cluster merges.")
    parser.add_argument("--h5ad", default=DEFAULT_H5AD)
    parser.add_argument("--cluster-key", default="leiden")
    parser.add_argument("--condition-key", default="renamed_samples")
    parser.add_argument("--pca-key", default="X_pca_harmony_strict")
    parser.add_argument("--figures-dir", default="results/figures/bc_guca1b_gt2_cluster_merge_review")
    parser.add_argument("--tables-dir", default="results/tables/bc_guca1b_gt2_cluster_merge_review")
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def numeric_sort(values) -> list[str]:
    return sorted([str(value) for value in values], key=lambda value: int(value))


def cluster_counts(adata, cluster_key: str, condition_key: str) -> pd.DataFrame:
    table = pd.crosstab(
        adata.obs[cluster_key].astype(str),
        adata.obs[condition_key].astype(str),
    )
    for condition in ["Control", "LD", "NMDA"]:
        if condition not in table.columns:
            table[condition] = 0
    table = table[["Control", "LD", "NMDA"]]
    table["total"] = table.sum(axis=1)
    proportions = table[["Control", "LD", "NMDA"]].div(table["total"], axis=0)
    proportions = proportions.rename(columns={col: f"{col}_fraction" for col in proportions.columns})
    out = pd.concat([table, proportions], axis=1).reset_index().rename(columns={cluster_key: "cluster"})
    out["cluster_sort"] = out["cluster"].astype(int)
    return out.sort_values("cluster_sort").drop(columns="cluster_sort")


def centroid_frame(labels: pd.Series, coords: np.ndarray, prefix: str) -> pd.DataFrame:
    rows = []
    for cluster in numeric_sort(labels.unique()):
        mask = (labels == cluster).to_numpy()
        centroid = coords[mask].mean(axis=0)
        row = {"cluster": cluster}
        row.update({f"{prefix}_{idx + 1}": value for idx, value in enumerate(centroid)})
        rows.append(row)
    return pd.DataFrame(rows).set_index("cluster")


def cluster_mean_expression(adata, cluster_key: str) -> pd.DataFrame:
    labels = adata.obs[cluster_key].astype(str)
    hvg = np.asarray(adata.var["highly_variable"].fillna(False), dtype=bool)
    matrix = adata.X[:, hvg]
    genes = pd.Index(adata.var_names[hvg])
    rows = []
    clusters = numeric_sort(labels.unique())
    for cluster in clusters:
        mask = (labels == cluster).to_numpy()
        sub = matrix[mask, :]
        if sparse.issparse(sub):
            values = np.asarray(sub.mean(axis=0)).ravel()
        else:
            values = np.asarray(sub).mean(axis=0)
        rows.append(values)
    return pd.DataFrame(np.vstack(rows), index=clusters, columns=genes)


def pair_ranks(distance_or_similarity: pd.DataFrame, higher_is_better: bool) -> dict[tuple[str, str], tuple[float, int, int]]:
    clusters = list(distance_or_similarity.index)
    output = {}
    for a, b in combinations(clusters, 2):
        a_values = distance_or_similarity.loc[a].drop(index=a)
        b_values = distance_or_similarity.loc[b].drop(index=b)
        if higher_is_better:
            a_sorted = a_values.sort_values(ascending=False)
            b_sorted = b_values.sort_values(ascending=False)
        else:
            a_sorted = a_values.sort_values(ascending=True)
            b_sorted = b_values.sort_values(ascending=True)
        value = float(distance_or_similarity.at[a, b])
        output[(a, b)] = (value, int(a_sorted.index.get_loc(b) + 1), int(b_sorted.index.get_loc(a) + 1))
    return output


def euclidean_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    values = frame.to_numpy(dtype=float)
    diff = values[:, None, :] - values[None, :, :]
    dist = np.sqrt((diff * diff).sum(axis=2))
    return pd.DataFrame(dist, index=frame.index, columns=frame.index)


def condition_distance(counts: pd.DataFrame) -> pd.DataFrame:
    props = counts.set_index("cluster")[["Control_fraction", "LD_fraction", "NMDA_fraction"]]
    values = props.to_numpy(dtype=float)
    dist = np.abs(values[:, None, :] - values[None, :, :]).sum(axis=2) / 2
    return pd.DataFrame(dist, index=props.index, columns=props.index)


def top_marker_map(tables_dir: Path) -> dict[str, str]:
    path = tables_dir / "bc_guca1b_gt2_deg_max_other_top1_marker_per_cluster.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    return {str(row.group): str(row.names) for row in df.itertuples(index=False)}


def summarize_groups(
    groups: list[tuple[str, list[str]]],
    counts: pd.DataFrame,
    umap_pairs: dict[tuple[str, str], tuple[float, int, int]],
    pca_pairs: dict[tuple[str, str], tuple[float, int, int]],
    expr_pairs: dict[tuple[str, str], tuple[float, int, int]],
    cond_pairs: dict[tuple[str, str], tuple[float, int, int]],
    marker_map: dict[str, str],
) -> pd.DataFrame:
    count_lookup = counts.set_index("cluster")
    rows = []
    for group_name, clusters in groups:
        for a, b in combinations(clusters, 2):
            key = tuple(sorted((a, b), key=int))
            umap_value, umap_rank_a, umap_rank_b = umap_pairs[key]
            pca_value, pca_rank_a, pca_rank_b = pca_pairs[key]
            expr_value, expr_rank_a, expr_rank_b = expr_pairs[key]
            cond_value, cond_rank_a, cond_rank_b = cond_pairs[key]
            rows.append(
                {
                    "proposed_merge": group_name,
                    "cluster_a": a,
                    "cluster_b": b,
                    "cells_a": int(count_lookup.at[a, "total"]),
                    "cells_b": int(count_lookup.at[b, "total"]),
                    "marker_a": marker_map.get(a, ""),
                    "marker_b": marker_map.get(b, ""),
                    "umap_centroid_distance": umap_value,
                    "umap_neighbor_rank_a_to_b": umap_rank_a,
                    "umap_neighbor_rank_b_to_a": umap_rank_b,
                    "harmony_pca_centroid_distance": pca_value,
                    "harmony_neighbor_rank_a_to_b": pca_rank_a,
                    "harmony_neighbor_rank_b_to_a": pca_rank_b,
                    "hvg_mean_expression_correlation": expr_value,
                    "expression_similarity_rank_a_to_b": expr_rank_a,
                    "expression_similarity_rank_b_to_a": expr_rank_b,
                    "condition_fraction_distance": cond_value,
                    "condition_similarity_rank_a_to_b": cond_rank_a,
                    "condition_similarity_rank_b_to_a": cond_rank_b,
                }
            )
    return pd.DataFrame(rows)


def plot_merge_umap(adata, cluster_key: str, groups: list[tuple[str, list[str]]], output: Path, dpi: int) -> None:
    labels = adata.obs[cluster_key].astype(str)
    merge_label = pd.Series("other", index=adata.obs_names)
    for group_name, clusters in groups:
        merge_label.loc[labels.isin(clusters)] = group_name.replace("merge_", "")

    coords = adata.obsm["X_umap"]
    fig, ax = plt.subplots(figsize=(12, 9))
    other = merge_label == "other"
    ax.scatter(coords[other, 0], coords[other, 1], s=1, c="#d7dce2", linewidths=0, alpha=0.35)
    palette = plt.get_cmap("tab10")
    for idx, (group_name, clusters) in enumerate(groups):
        label = group_name.replace("merge_", "")
        mask = merge_label == label
        ax.scatter(coords[mask, 0], coords[mask, 1], s=4, c=[palette(idx)], linewidths=0, label=label, alpha=0.95)
        for cluster in clusters:
            cluster_mask = labels == cluster
            centroid = coords[cluster_mask].mean(axis=0)
            ax.text(
                centroid[0],
                centroid[1],
                cluster,
                ha="center",
                va="center",
                fontsize=9,
                fontweight="bold",
                bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "#111827", "lw": 0.6, "alpha": 0.85},
            )
    ax.set_title("BC proposed Leiden cluster merge review", fontsize=16)
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.legend(title="Proposed merge", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    figures_dir = Path(args.figures_dir)
    tables_dir = Path(args.tables_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    adata = sc.read_h5ad(args.h5ad)
    labels = adata.obs[args.cluster_key].astype(str)
    counts = cluster_counts(adata, args.cluster_key, args.condition_key)
    counts.to_csv(tables_dir / "bc_guca1b_gt2_cluster_counts_by_condition.csv", index=False)

    umap_centroids = centroid_frame(labels, adata.obsm["X_umap"], "umap")
    pca_centroids = centroid_frame(labels, adata.obsm[args.pca_key][:, :30], "harmony_pc")
    expr_means = cluster_mean_expression(adata, args.cluster_key)

    umap_dist = euclidean_matrix(umap_centroids)
    pca_dist = euclidean_matrix(pca_centroids)
    expr_corr = expr_means.T.corr(method="pearson")
    cond_dist = condition_distance(counts)

    for name, frame in [
        ("umap_centroid_distances.csv", umap_dist),
        ("harmony_pca_centroid_distances.csv", pca_dist),
        ("hvg_mean_expression_correlations.csv", expr_corr),
        ("condition_fraction_distances.csv", cond_dist),
    ]:
        frame.to_csv(tables_dir / name)

    marker_map = top_marker_map(Path("results/tables/bc_guca1b_gt2_ac_style_gene_umaps"))
    summary = summarize_groups(
        DEFAULT_GROUPS,
        counts,
        pair_ranks(umap_dist, higher_is_better=False),
        pair_ranks(pca_dist, higher_is_better=False),
        pair_ranks(expr_corr, higher_is_better=True),
        pair_ranks(cond_dist, higher_is_better=False),
        marker_map,
    )
    summary.to_csv(tables_dir / "bc_guca1b_gt2_proposed_merge_pairwise_review.csv", index=False)

    group_summary = (
        summary.groupby("proposed_merge", observed=False)
        .agg(
            n_pairwise_comparisons=("cluster_a", "size"),
            mean_umap_neighbor_rank=("umap_neighbor_rank_a_to_b", "mean"),
            mean_harmony_neighbor_rank=("harmony_neighbor_rank_a_to_b", "mean"),
            mean_expression_correlation=("hvg_mean_expression_correlation", "mean"),
            max_condition_fraction_distance=("condition_fraction_distance", "max"),
        )
        .reset_index()
    )
    group_summary.to_csv(tables_dir / "bc_guca1b_gt2_proposed_merge_group_summary.csv", index=False)

    plot_merge_umap(
        adata,
        args.cluster_key,
        DEFAULT_GROUPS,
        figures_dir / "bc_guca1b_gt2_proposed_merge_groups_umap.png",
        args.dpi,
    )

    print(tables_dir / "bc_guca1b_gt2_proposed_merge_pairwise_review.csv")
    print(tables_dir / "bc_guca1b_gt2_proposed_merge_group_summary.csv")
    print(figures_dir / "bc_guca1b_gt2_proposed_merge_groups_umap.png")


if __name__ == "__main__":
    main()

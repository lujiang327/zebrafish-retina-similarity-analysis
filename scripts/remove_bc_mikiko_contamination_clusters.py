#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import scanpy as sc


DEFAULT_INPUT = "results/h5ad/bc_9_sample_guca1b_gt2_mikiko_merged_clusters.h5ad"
DEFAULT_OUTPUT = "results/h5ad/bc_9_sample_guca1b_gt2_mikiko_no_contam_26_28.h5ad"
DEFAULT_TABLES = "results/tables/bc_guca1b_gt2_mikiko_no_contam_26_28_gene_umaps"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Remove BC Mikiko-merged contamination clusters and relabel remaining clusters."
    )
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--cluster-key", default="leiden")
    parser.add_argument("--remove-clusters", nargs="+", default=["26", "27", "28"])
    parser.add_argument("--tables-dir", default=DEFAULT_TABLES)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    tables_dir = Path(args.tables_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    adata = sc.read_h5ad(args.input)
    labels = adata.obs[args.cluster_key].astype(str)
    remove_clusters = [str(cluster) for cluster in args.remove_clusters]
    keep_mask = ~labels.isin(remove_clusters)

    removed_counts = (
        labels[~keep_mask]
        .value_counts()
        .rename_axis("removed_cluster")
        .reset_index(name="n_removed_cells")
        .assign(cluster_sort=lambda df: df["removed_cluster"].astype(int))
        .sort_values("cluster_sort")
        .drop(columns="cluster_sort")
    )
    removed_counts.to_csv(tables_dir / "bc_removed_contamination_cluster_counts.csv", index=False)

    filtered = adata[keep_mask].copy()
    old_labels = filtered.obs[args.cluster_key].astype(str)
    old_counts = old_labels.value_counts()
    old_to_new = {
        old_cluster: str(new_cluster)
        for new_cluster, old_cluster in enumerate(
            old_counts.sort_values(ascending=False).index.astype(str).tolist()
        )
    }
    new_labels = old_labels.map(old_to_new)
    categories = [str(idx) for idx in range(len(old_to_new))]

    filtered.obs["leiden_before_contam_removal"] = pd.Categorical(
        old_labels,
        categories=sorted(old_to_new, key=int),
        ordered=True,
    )
    filtered.obs[args.cluster_key] = pd.Categorical(new_labels, categories=categories, ordered=True)
    filtered.obs["leiden_no_contam_26_28"] = filtered.obs[args.cluster_key]
    filtered.uns["removed_bc_contamination_clusters"] = remove_clusters
    filtered.uns["bc_no_contam_relabel_by_count"] = old_to_new

    pd.DataFrame(
        [
            {
                "old_cluster": old_cluster,
                "new_cluster": new_cluster,
                "n_cells": int(old_counts.loc[old_cluster]),
            }
            for old_cluster, new_cluster in old_to_new.items()
        ]
    ).to_csv(tables_dir / "bc_no_contam_cluster_relabel_by_count.csv", index=False)

    (
        filtered.obs[args.cluster_key]
        .astype(str)
        .value_counts()
        .rename_axis("cluster")
        .reset_index(name="n_cells")
        .assign(cluster_sort=lambda df: df["cluster"].astype(int))
        .sort_values("cluster_sort")
        .drop(columns="cluster_sort")
        .to_csv(tables_dir / "bc_no_contam_cluster_counts.csv", index=False)
    )

    filtered.write_h5ad(output)
    print(output)
    print(tables_dir / "bc_removed_contamination_cluster_counts.csv")
    print(tables_dir / "bc_no_contam_cluster_relabel_by_count.csv")


if __name__ == "__main__":
    main()

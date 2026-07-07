#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import scanpy as sc


DEFAULT_INPUT = "results/h5ad/bc_9_sample_guca1b_gt2_filtered.h5ad"
DEFAULT_OUTPUT = "results/h5ad/bc_9_sample_guca1b_gt2_mikiko_merged_clusters.h5ad"
MERGE_GROUPS = {
    "28": ["28", "33"],
    "12": ["12", "36"],
    "2": ["2", "29", "35"],
    "9": ["9", "32"],
    "3": ["3", "31"],
    "14": ["14", "37", "38"],
    "5": ["5", "26"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Add Mikiko-reviewed merged BC cluster labels.")
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--cluster-key", default="leiden")
    parser.add_argument("--merged-key", default="leiden_mikiko_merged")
    parser.add_argument("--tables-dir", default="results/tables/bc_guca1b_gt2_mikiko_merged_gene_umaps")
    return parser.parse_args()


def numeric_sort(values) -> list[str]:
    return sorted([str(value) for value in values], key=lambda value: int(value))


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    tables_dir = Path(args.tables_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    adata = sc.read_h5ad(args.input)
    original = adata.obs[args.cluster_key].astype(str)
    merged = original.copy()

    rows = []
    for merged_label, original_clusters in MERGE_GROUPS.items():
        for original_cluster in original_clusters:
            merged.loc[original == original_cluster] = merged_label
            rows.append(
                {
                    "original_cluster": original_cluster,
                    "mikiko_merged_cluster": merged_label,
                    "merge_group": "_".join(original_clusters),
                }
            )

    unchanged = [cluster for cluster in numeric_sort(original.unique()) if cluster not in {r["original_cluster"] for r in rows}]
    for cluster in unchanged:
        rows.append(
                {
                    "original_cluster": cluster,
                    "mikiko_merged_cluster": cluster,
                    "merge_group": cluster,
                }
        )

    old_counts = merged.value_counts()
    old_to_final = {
        old_cluster: str(new_cluster)
        for new_cluster, old_cluster in enumerate(
            old_counts.sort_values(ascending=False).index.astype(str).tolist()
        )
    }
    final = merged.map(old_to_final)
    final_categories = [str(idx) for idx in range(len(old_to_final))]
    adata.obs[f"{args.merged_key}_original_labels"] = pd.Categorical(
        merged,
        categories=numeric_sort(merged.unique()),
        ordered=True,
    )
    adata.obs[args.merged_key] = pd.Categorical(final, categories=final_categories, ordered=True)
    adata.obs["leiden"] = adata.obs[args.merged_key]
    adata.uns["mikiko_bc_cluster_merges"] = MERGE_GROUPS
    adata.uns["mikiko_bc_merged_cluster_relabel_by_count"] = old_to_final

    merge_map = pd.DataFrame(rows)
    merge_map["final_cluster"] = merge_map["mikiko_merged_cluster"].map(old_to_final)
    merge_map.assign(
        original_sort=lambda df: df["original_cluster"].astype(int),
        final_sort=lambda df: df["final_cluster"].astype(int),
    ).sort_values(["final_sort", "original_sort"]).drop(columns=["original_sort", "final_sort"]).to_csv(
        tables_dir / "bc_guca1b_gt2_mikiko_cluster_merge_map.csv",
        index=False,
    )

    pd.DataFrame(
        [
            {
                "mikiko_merged_cluster": old_cluster,
                "final_cluster": final_cluster,
                "n_cells": int(old_counts.loc[old_cluster]),
            }
            for old_cluster, final_cluster in old_to_final.items()
        ]
    ).to_csv(tables_dir / "bc_guca1b_gt2_mikiko_cluster_relabel_by_count.csv", index=False)

    (
        adata.obs[args.merged_key]
        .astype(str)
        .value_counts()
        .rename_axis("cluster")
        .reset_index(name="n_cells")
        .assign(cluster_sort=lambda df: df["cluster"].astype(int))
        .sort_values("cluster_sort")
        .drop(columns="cluster_sort")
        .to_csv(tables_dir / "bc_guca1b_gt2_mikiko_merged_cluster_counts.csv", index=False)
    )

    adata.write_h5ad(output)
    print(output)
    print(tables_dir / "bc_guca1b_gt2_mikiko_cluster_merge_map.csv")


if __name__ == "__main__":
    main()

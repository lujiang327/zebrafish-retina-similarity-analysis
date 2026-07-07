#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import scanpy as sc
from scanpy.plotting.palettes import default_102


DEFAULT_H5AD = "results/h5ad/bc_9_sample_guca1b_gt2_mikiko_no_contam_26_28.h5ad"
DEFAULT_REFERENCE_H5AD = "results/h5ad/bc_9_sample_guca1b_gt2_mikiko_merged_clusters.h5ad"
DEFAULT_OUTPUT = (
    "results/figures/bc_guca1b_gt2_mikiko_no_contam_26_28_gene_umaps/"
    "bc_umap_by_leiden.png"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot no-contamination BC Leiden UMAP while preserving the old pre-removal "
            "cluster colors."
        )
    )
    parser.add_argument("--h5ad", default=DEFAULT_H5AD)
    parser.add_argument(
        "--reference-h5ad",
        default=DEFAULT_REFERENCE_H5AD,
        help="Pre-removal h5ad used to recover the original Leiden category color order.",
    )
    parser.add_argument("--cluster-key", default="leiden")
    parser.add_argument("--old-cluster-key", default="leiden_before_contam_removal")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def numeric_categories(values) -> list[str]:
    return sorted([str(value) for value in values], key=lambda value: int(value))


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    adata = sc.read_h5ad(args.h5ad)
    if "X_umap" not in adata.obsm:
        raise ValueError("Input h5ad does not contain adata.obsm['X_umap']")
    if args.cluster_key not in adata.obs:
        raise KeyError(f"Missing cluster column: {args.cluster_key}")
    if args.old_cluster_key not in adata.obs:
        raise KeyError(f"Missing old cluster column: {args.old_cluster_key}")

    new_labels = adata.obs[args.cluster_key].astype(str)
    old_labels = adata.obs[args.old_cluster_key].astype(str)

    reference = sc.read_h5ad(args.reference_h5ad)
    if args.cluster_key in reference.obs:
        old_categories = numeric_categories(reference.obs[args.cluster_key].astype(str).unique())
    else:
        old_categories = [str(idx) for idx in range(max(old_labels.astype(int)) + 1)]

    if f"{args.cluster_key}_colors" in reference.uns:
        reference_colors = list(reference.uns[f"{args.cluster_key}_colors"])
    else:
        reference_colors = [default_102[idx % len(default_102)] for idx in range(len(old_categories))]

    old_color_by_cluster = {
        cluster: reference_colors[idx % len(reference_colors)]
        for idx, cluster in enumerate(old_categories)
    }
    new_categories = numeric_categories(new_labels.unique())
    new_colors = []
    for cluster in new_categories:
        old_for_new = old_labels[new_labels == cluster].iloc[0]
        new_colors.append(old_color_by_cluster[old_for_new])

    adata.obs[args.cluster_key] = adata.obs[args.cluster_key].cat.set_categories(
        new_categories,
        ordered=True,
    )
    adata.uns[f"{args.cluster_key}_colors"] = new_colors

    fig = sc.pl.umap(
        adata,
        color=args.cluster_key,
        legend_loc="on data",
        title="BC guca1b > 2.0 clusters",
        size=10,
        frameon=True,
        show=False,
        return_fig=True,
    )
    fig.set_size_inches(8.95, 6.0)
    fig.savefig(output, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()

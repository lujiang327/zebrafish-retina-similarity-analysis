#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc


H5AD = "results/h5ad/bc_strict_harmony_by_9_samples_reprocessed.h5ad"
FIGURES_DIR = Path("results/figures/bc_strict_harmony_by_9_samples_marker_comparison")
TABLES_DIR = Path("results/tables/bc_strict_harmony_by_9_samples_marker_comparison")


def cluster_sort_key(value: str):
    return int(value) if value.isdigit() else value


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    adata = sc.read_h5ad(H5AD)
    coords = adata.obsm["X_umap"]
    clusters = adata.obs["leiden"].astype(str)
    ordered_clusters = sorted(clusters.unique(), key=cluster_sort_key)

    cmap = plt.get_cmap("tab20")
    colors = {cluster: cmap(i % 20) for i, cluster in enumerate(ordered_clusters)}

    fig, ax = plt.subplots(figsize=(11.5, 8.0))
    for cluster in ordered_clusters:
        mask = clusters == cluster
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=3,
            c=[colors[cluster]],
            linewidths=0,
            alpha=0.72,
        )

    label_records = []
    for cluster in ordered_clusters:
        mask = clusters == cluster
        xy = coords[mask]
        x = float(np.median(xy[:, 0]))
        y = float(np.median(xy[:, 1]))
        label_records.append(
            {
                "leiden": cluster,
                "n_cells": int(mask.sum()),
                "label_x": x,
                "label_y": y,
            }
        )
        ax.text(
            x,
            y,
            cluster,
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
            color="black",
            bbox={
                "boxstyle": "round,pad=0.22",
                "facecolor": "white",
                "edgecolor": "black",
                "linewidth": 0.5,
                "alpha": 0.82,
            },
        )

    pd.DataFrame(label_records).to_csv(TABLES_DIR / "bc_9_sample_leiden_label_positions.csv", index=False)

    ax.set_title("BC 9-Sample Harmony Leiden Clusters", fontsize=24)
    ax.set_xlabel("UMAP1", fontsize=18)
    ax.set_ylabel("UMAP2", fontsize=18)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "bc_9_sample_harmony_leiden_cluster_numbers.png", dpi=300)
    fig.savefig(FIGURES_DIR / "bc_9_sample_harmony_leiden_cluster_numbers.pdf", dpi=300)
    plt.close(fig)

    # A monochrome label-only version is useful when checking clusters against marker panels.
    fig, ax = plt.subplots(figsize=(11.5, 8.0))
    ax.scatter(coords[:, 0], coords[:, 1], s=2, c="#cfcfcf", linewidths=0, alpha=0.45)
    for record in label_records:
        ax.text(
            record["label_x"],
            record["label_y"],
            record["leiden"],
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
            color="black",
            bbox={
                "boxstyle": "round,pad=0.22",
                "facecolor": "white",
                "edgecolor": "black",
                "linewidth": 0.5,
                "alpha": 0.86,
            },
        )
    ax.set_title("BC 9-Sample Harmony Cluster Number Reference", fontsize=24)
    ax.set_xlabel("UMAP1", fontsize=18)
    ax.set_ylabel("UMAP2", fontsize=18)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "bc_9_sample_harmony_cluster_number_reference.png", dpi=300)
    fig.savefig(FIGURES_DIR / "bc_9_sample_harmony_cluster_number_reference.pdf", dpi=300)
    plt.close(fig)

    print(FIGURES_DIR / "bc_9_sample_harmony_leiden_cluster_numbers.png")
    print(FIGURES_DIR / "bc_9_sample_harmony_cluster_number_reference.png")
    print(TABLES_DIR / "bc_9_sample_leiden_label_positions.csv")


if __name__ == "__main__":
    main()

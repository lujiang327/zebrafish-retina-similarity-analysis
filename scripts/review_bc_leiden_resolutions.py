#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc


DEFAULT_H5AD = "results/h5ad/bc_strict_harmony_by_9_samples_reprocessed.h5ad"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Review alternate Leiden resolutions for BC 9-sample Harmony.")
    parser.add_argument("--h5ad", default=DEFAULT_H5AD)
    parser.add_argument("--figures-dir", default="results/figures/bc_9_sample_leiden_resolution_review")
    parser.add_argument("--tables-dir", default="results/tables/bc_9_sample_leiden_resolution_review")
    parser.add_argument("--resolutions", nargs="+", type=float, default=[1.3, 1.5, 1.8, 2.0])
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def res_tag(resolution: float) -> str:
    return str(resolution).replace(".", "p")


def sort_key(value: str):
    return int(value) if value.isdigit() else value


def add_cluster_labels(ax, coords: np.ndarray, clusters: pd.Series, fontsize: int = 9) -> None:
    for cluster in sorted(clusters.astype(str).unique(), key=sort_key):
        mask = clusters.astype(str) == cluster
        xy = coords[mask.to_numpy()]
        ax.text(
            float(np.median(xy[:, 0])),
            float(np.median(xy[:, 1])),
            cluster,
            ha="center",
            va="center",
            fontsize=fontsize,
            fontweight="bold",
            color="black",
            bbox={
                "boxstyle": "round,pad=0.16",
                "facecolor": "white",
                "edgecolor": "black",
                "linewidth": 0.4,
                "alpha": 0.78,
            },
        )


def save_cluster_umap(
    adata,
    cluster_key: str,
    resolution: float,
    figures_dir: Path,
    dpi: int,
) -> None:
    fig = sc.pl.umap(
        adata,
        color=cluster_key,
        legend_loc=None,
        show=False,
        return_fig=True,
        size=3,
        frameon=True,
    )
    fig.set_size_inches(11.5, 8.0)
    ax = fig.axes[0]
    add_cluster_labels(ax, adata.obsm["X_umap"], adata.obs[cluster_key], fontsize=9)
    ax.set_title(f"BC 9-Sample Harmony Leiden resolution {resolution}", fontsize=22)
    ax.set_xlabel("UMAP1", fontsize=16)
    ax.set_ylabel("UMAP2", fontsize=16)
    tag = res_tag(resolution)
    fig.savefig(figures_dir / f"bc_9_sample_leiden_res_{tag}_cluster_numbers.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(figures_dir / f"bc_9_sample_leiden_res_{tag}_cluster_numbers.pdf", dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    coords = adata.obsm["X_umap"]
    clusters = adata.obs[cluster_key].astype(str)
    fig, ax = plt.subplots(figsize=(11.5, 8.0))
    ax.scatter(coords[:, 0], coords[:, 1], s=2, c="#d0d0d0", linewidths=0, alpha=0.42)
    add_cluster_labels(ax, coords, clusters, fontsize=10)
    ax.set_title(f"Cluster Number Reference | Leiden resolution {resolution}", fontsize=22)
    ax.set_xlabel("UMAP1", fontsize=16)
    ax.set_ylabel("UMAP2", fontsize=16)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(1.1)
    fig.tight_layout()
    fig.savefig(figures_dir / f"bc_9_sample_leiden_res_{tag}_label_reference.png", dpi=dpi)
    fig.savefig(figures_dir / f"bc_9_sample_leiden_res_{tag}_label_reference.pdf", dpi=dpi)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    figures_dir = Path(args.figures_dir)
    tables_dir = Path(args.tables_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    adata = sc.read_h5ad(args.h5ad)
    summary_records = []
    assignment = pd.DataFrame(index=adata.obs_names)

    for resolution in args.resolutions:
        tag = res_tag(resolution)
        key = f"leiden_res_{tag}"
        sc.tl.leiden(
            adata,
            resolution=resolution,
            key_added=key,
            random_state=args.random_state,
        )
        counts = (
            adata.obs[key]
            .astype(str)
            .value_counts()
            .rename_axis(key)
            .reset_index(name="n_cells")
            .sort_values(key, key=lambda s: s.map(sort_key))
        )
        counts.to_csv(tables_dir / f"counts_by_{key}.csv", index=False)
        assignment[key] = adata.obs[key].astype(str)
        summary_records.append(
            {
                "resolution": resolution,
                "cluster_key": key,
                "n_clusters": int(adata.obs[key].nunique()),
                "min_cluster_size": int(counts["n_cells"].min()),
                "median_cluster_size": float(counts["n_cells"].median()),
                "max_cluster_size": int(counts["n_cells"].max()),
            }
        )
        save_cluster_umap(adata, key, resolution, figures_dir, args.dpi)

    assignment.to_csv(tables_dir / "leiden_resolution_assignments.csv")
    pd.DataFrame(summary_records).to_csv(tables_dir / "leiden_resolution_summary.csv", index=False)
    print(figures_dir)
    print(tables_dir)


if __name__ == "__main__":
    main()

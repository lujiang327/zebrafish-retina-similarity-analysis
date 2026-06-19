#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse


DEFAULT_H5AD = "results/h5ad/bc_strict_harmony_by_9_samples_reprocessed.h5ad"
PHOTO_GENES = ["nr2e3", "pde6a", "guca1b"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter BC cells with high photoreceptor-like marker expression and replot gene UMAPs."
    )
    parser.add_argument("--h5ad", default=DEFAULT_H5AD)
    parser.add_argument("--figures-dir", default="results/figures/bc_9_sample_photo_gene_filtered_review")
    parser.add_argument("--tables-dir", default="results/tables/bc_9_sample_photo_gene_filtered_review")
    parser.add_argument("--out-h5ad", default="results/h5ad/bc_9_sample_photo_gene_filtered.h5ad")
    parser.add_argument("--genes", nargs="+", default=PHOTO_GENES)
    parser.add_argument(
        "--plot-genes",
        nargs="+",
        default=None,
        help="Genes to plot after filtering. Defaults to the filtering genes.",
    )
    parser.add_argument("--threshold", type=float, default=1.5)
    parser.add_argument("--mode", choices=["any", "all"], default="any")
    parser.add_argument("--cluster-key", default="leiden")
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def sort_key(value: str):
    return int(value) if str(value).isdigit() else str(value)


def get_gene_expression(adata, genes: list[str]) -> tuple[np.ndarray, list[str]]:
    source = adata.raw if adata.raw is not None else adata
    lookup = {str(g).lower(): str(g) for g in source.var_names}
    matched = []
    missing = []
    for gene in genes:
        match = lookup.get(gene.lower())
        if match is None:
            missing.append(gene)
        else:
            matched.append(match)
    if missing:
        raise KeyError(f"Missing genes: {', '.join(missing)}")
    X = source[:, matched].X
    if sparse.issparse(X):
        X = X.toarray()
    return np.asarray(X, dtype=float), matched


def add_cluster_labels(ax, coords: np.ndarray, clusters: pd.Series, fontsize: int = 9) -> None:
    clusters = clusters.astype(str)
    for cluster in sorted(clusters.unique(), key=sort_key):
        mask = clusters == cluster
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


def plot_gene_umap(adata, gene: str, cluster_key: str, figures_dir: Path, dpi: int) -> None:
    fig = sc.pl.umap(
        adata,
        color=gene,
        use_raw=adata.raw is not None,
        cmap="viridis",
        show=False,
        return_fig=True,
        size=3 if adata.n_obs > 20000 else 8,
        frameon=True,
    )
    fig.set_size_inches(10.5, 7.0)
    ax = fig.axes[0]
    add_cluster_labels(ax, adata.obsm["X_umap"], adata.obs[cluster_key], fontsize=9)
    ax.set_title(f"{gene} after filtering | {cluster_key} res 1.0", fontsize=22, pad=10)
    ax.set_xlabel("UMAP1", fontsize=18)
    ax.set_ylabel("UMAP2", fontsize=18)
    fig.savefig(figures_dir / f"bc_filtered_umap_gene_{gene}_leiden_labels.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    figures_dir = Path(args.figures_dir)
    tables_dir = Path(args.tables_dir)
    out_h5ad = Path(args.out_h5ad)
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    out_h5ad.parent.mkdir(parents=True, exist_ok=True)

    adata = sc.read_h5ad(args.h5ad)
    if args.cluster_key not in adata.obs:
        raise KeyError(f"Cluster key not found: {args.cluster_key}")
    X, matched_genes = get_gene_expression(adata, args.genes)

    high = X > args.threshold
    remove_mask = high.any(axis=1) if args.mode == "any" else high.all(axis=1)
    keep_mask = ~remove_mask

    per_gene_removed = high.sum(axis=0)
    summary = pd.DataFrame(
        [
            {
                "input_h5ad": args.h5ad,
                "output_h5ad": str(out_h5ad),
                "threshold": args.threshold,
                "mode": args.mode,
                "n_cells_before": int(adata.n_obs),
                "n_cells_removed": int(remove_mask.sum()),
                "n_cells_after": int(keep_mask.sum()),
                "removed_fraction": float(remove_mask.mean()),
                **{f"n_cells_{gene}_gt_threshold": int(count) for gene, count in zip(matched_genes, per_gene_removed)},
            }
        ]
    )
    summary.to_csv(tables_dir / "photo_gene_filter_summary.csv", index=False)

    cell_flags = pd.DataFrame(index=adata.obs_names)
    for i, gene in enumerate(matched_genes):
        cell_flags[f"{gene}_expression"] = X[:, i]
        cell_flags[f"{gene}_gt_{args.threshold}"] = high[:, i]
    cell_flags["remove_photo_like"] = remove_mask
    cell_flags.to_csv(tables_dir / "photo_gene_filter_cell_flags.csv")

    cluster_before = adata.obs[args.cluster_key].astype(str).value_counts().rename("n_before")
    cluster_removed = adata.obs.loc[remove_mask, args.cluster_key].astype(str).value_counts().rename("n_removed")
    cluster_counts = (
        pd.concat([cluster_before, cluster_removed], axis=1)
        .fillna(0)
        .astype(int)
        .reset_index(names=args.cluster_key)
        .sort_values(args.cluster_key, key=lambda s: s.map(sort_key))
    )
    cluster_counts["n_after"] = cluster_counts["n_before"] - cluster_counts["n_removed"]
    cluster_counts["removed_fraction"] = cluster_counts["n_removed"] / cluster_counts["n_before"]
    cluster_counts.to_csv(tables_dir / "photo_gene_filter_counts_by_leiden.csv", index=False)

    filtered = adata[keep_mask].copy()
    for col in filtered.obs.select_dtypes(include="category").columns:
        filtered.obs[col] = filtered.obs[col].cat.remove_unused_categories()
    filtered.write_h5ad(out_h5ad, compression="gzip")

    plot_genes = args.plot_genes or matched_genes
    _, matched_plot_genes = get_gene_expression(filtered, plot_genes)
    for gene in matched_plot_genes:
        plot_gene_umap(filtered, gene, args.cluster_key, figures_dir, args.dpi)

    print(figures_dir)
    print(tables_dir)
    print(out_h5ad)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from scipy import sparse


DEFAULT_H5AD = "results/h5ad/bc_strict_harmony_by_9_samples_reprocessed.h5ad"
PHOTO_GENES = ["nr2e3", "pde6a", "guca1b"]
BC_GENES = ["vsx1", "vsx2", "cabp5a", "cabp5b", "grm6a", "grm6b", "pcp4a"]
ALL_GENES = PHOTO_GENES + BC_GENES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Review BC clusters for possible photoreceptor-like removal.")
    parser.add_argument("--h5ad", default=DEFAULT_H5AD)
    parser.add_argument("--figures-dir", default="results/figures/bc_9_sample_cluster_removal_review")
    parser.add_argument("--tables-dir", default="results/tables/bc_9_sample_cluster_removal_review")
    parser.add_argument("--cluster-key", default="leiden")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--gene-fig-width", type=float, default=9.8)
    parser.add_argument("--gene-fig-height", type=float, default=6.6)
    parser.add_argument("--gene-label-fontsize", type=int, default=9)
    parser.add_argument("--gene-output-suffix", default="")
    return parser.parse_args()


def sort_key(value: str):
    return int(value) if str(value).isdigit() else str(value)


def get_expression_matrix(adata, genes: list[str]) -> tuple[np.ndarray, list[str]]:
    source = adata.raw if adata.raw is not None else adata
    lookup = {str(g).lower(): str(g) for g in source.var_names}
    matched = [lookup[g.lower()] for g in genes if g.lower() in lookup]
    if not matched:
        raise ValueError("None of the requested genes were found.")
    X = source[:, matched].X
    if sparse.issparse(X):
        X = X.toarray()
    return np.asarray(X, dtype=float), matched


def cluster_centers(adata, cluster_key: str) -> pd.DataFrame:
    coords = adata.obsm["X_umap"]
    clusters = adata.obs[cluster_key].astype(str)
    records = []
    for cluster in sorted(clusters.unique(), key=sort_key):
        mask = clusters == cluster
        records.append(
            {
                cluster_key: cluster,
                "n_cells": int(mask.sum()),
                "x": float(np.median(coords[mask, 0])),
                "y": float(np.median(coords[mask, 1])),
            }
        )
    return pd.DataFrame(records)


def add_cluster_labels(ax, centers: pd.DataFrame, cluster_key: str, fontsize: int = 9) -> None:
    for row in centers.itertuples(index=False):
        cluster = getattr(row, cluster_key)
        ax.text(
            row.x,
            row.y,
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
                "alpha": 0.76,
            },
        )


def plot_labeled_gene_umap(
    adata,
    gene: str,
    centers: pd.DataFrame,
    cluster_key: str,
    out_png: Path,
    out_pdf: Path,
    dpi: int,
    width: float,
    height: float,
    label_fontsize: int,
) -> None:
    fig = sc.pl.umap(
        adata,
        color=gene,
        use_raw=adata.raw is not None,
        cmap="viridis",
        size=3,
        frameon=True,
        show=False,
        return_fig=True,
    )
    fig.set_size_inches(width, height)
    ax = fig.axes[0]
    ax.set_title(f"{gene} with {cluster_key} labels", fontsize=22, pad=10)
    ax.set_xlabel("UMAP1", fontsize=18)
    ax.set_ylabel("UMAP2", fontsize=18)
    add_cluster_labels(ax, centers, cluster_key, fontsize=label_fontsize)
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    fig.savefig(out_pdf, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def cluster_expression_summary(adata, cluster_key: str, genes: list[str]) -> pd.DataFrame:
    X, matched = get_expression_matrix(adata, genes)
    clusters = adata.obs[cluster_key].astype(str).to_numpy()
    records = []
    for cluster in sorted(pd.unique(clusters), key=sort_key):
        mask = clusters == cluster
        Xm = X[mask, :]
        for i, gene in enumerate(matched):
            expr = Xm[:, i]
            records.append(
                {
                    cluster_key: cluster,
                    "gene": gene,
                    "n_cells": int(mask.sum()),
                    "mean_expression": float(np.mean(expr)),
                    "median_expression": float(np.median(expr)),
                    "fraction_expressing": float(np.mean(expr > 0)),
                }
            )
    return pd.DataFrame(records)


def plot_heatmap(df: pd.DataFrame, value: str, title: str, out: Path, cluster_key: str, dpi: int) -> None:
    pivot = df.pivot(index=cluster_key, columns="gene", values=value)
    pivot = pivot.loc[sorted(pivot.index, key=sort_key), ALL_GENES]
    fig_height = max(7, 0.22 * len(pivot) + 2.5)
    fig, ax = plt.subplots(figsize=(10.5, fig_height))
    sns.heatmap(
        pivot,
        cmap="viridis",
        ax=ax,
        linewidths=0.2,
        linecolor="white",
        cbar_kws={"label": value.replace("_", " ")},
    )
    ax.set_title(title, fontsize=18)
    ax.set_xlabel("")
    ax.set_ylabel(cluster_key)
    fig.tight_layout()
    fig.savefig(out, dpi=dpi)
    plt.close(fig)


def plot_score_scatter(summary: pd.DataFrame, cluster_key: str, out: Path, dpi: int) -> pd.DataFrame:
    mean = summary.pivot(index=cluster_key, columns="gene", values="mean_expression").fillna(0)
    frac = summary.pivot(index=cluster_key, columns="gene", values="fraction_expressing").fillna(0)
    score = pd.DataFrame(index=mean.index)
    score["n_cells"] = summary.groupby(cluster_key)["n_cells"].first()
    score["photo_mean_score"] = mean[[g for g in PHOTO_GENES if g in mean.columns]].mean(axis=1)
    score["bc_mean_score"] = mean[[g for g in BC_GENES if g in mean.columns]].mean(axis=1)
    score["photo_fraction_score"] = frac[[g for g in PHOTO_GENES if g in frac.columns]].mean(axis=1)
    score["bc_fraction_score"] = frac[[g for g in BC_GENES if g in frac.columns]].mean(axis=1)
    score["photo_minus_bc_mean"] = score["photo_mean_score"] - score["bc_mean_score"]
    score = score.reset_index().sort_values(cluster_key, key=lambda s: s.map(sort_key))

    fig, ax = plt.subplots(figsize=(8.2, 6.2))
    sizes = np.clip(score["n_cells"].to_numpy(), 40, 900)
    sizes = 30 + 260 * (sizes - sizes.min()) / max(1, sizes.max() - sizes.min())
    ax.scatter(
        score["bc_mean_score"],
        score["photo_mean_score"],
        s=sizes,
        c=score["photo_minus_bc_mean"],
        cmap="coolwarm",
        edgecolor="black",
        linewidth=0.35,
    )
    for row in score.itertuples(index=False):
        ax.text(row.bc_mean_score, row.photo_mean_score, getattr(row, cluster_key), fontsize=8, ha="center", va="center")
    ax.set_xlabel("Mean BC marker score")
    ax.set_ylabel("Mean photoreceptor-like marker score")
    ax.set_title("Cluster-Level Marker Score Comparison")
    ax.axline((0, 0), slope=1, color="#777777", linestyle="--", linewidth=1)
    fig.tight_layout()
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    return score


def plot_flagged_reference(adata, centers: pd.DataFrame, score: pd.DataFrame, cluster_key: str, out: Path, dpi: int) -> None:
    coords = adata.obsm["X_umap"]
    clusters = adata.obs[cluster_key].astype(str)
    top_photo = set(
        score.sort_values(["photo_minus_bc_mean", "photo_mean_score"], ascending=False)
        .head(8)[cluster_key]
        .astype(str)
    )
    colors = clusters.map(lambda c: "#d62728" if c in top_photo else "#d8d8d8")
    sizes = clusters.map(lambda c: 5 if c in top_photo else 2)
    fig, ax = plt.subplots(figsize=(10.5, 7.2))
    ax.scatter(coords[:, 0], coords[:, 1], c=colors, s=sizes, linewidths=0, alpha=0.75)
    add_cluster_labels(ax, centers, cluster_key, fontsize=9)
    ax.set_title("Top Photoreceptor-Like Candidate Clusters Highlighted", fontsize=20)
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out, dpi=dpi)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    figures_dir = Path(args.figures_dir)
    tables_dir = Path(args.tables_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    adata = sc.read_h5ad(args.h5ad)
    if args.cluster_key not in adata.obs:
        raise KeyError(f"Cluster key not found: {args.cluster_key}")

    centers = cluster_centers(adata, args.cluster_key)
    centers.to_csv(tables_dir / "cluster_label_positions.csv", index=False)

    for gene in PHOTO_GENES:
        suffix = f"_{args.gene_output_suffix}" if args.gene_output_suffix else ""
        plot_labeled_gene_umap(
            adata,
            gene,
            centers,
            args.cluster_key,
            figures_dir / f"bc_9_sample_umap_gene_{gene}_cluster_labels{suffix}.png",
            figures_dir / f"bc_9_sample_umap_gene_{gene}_cluster_labels{suffix}.pdf",
            args.dpi,
            args.gene_fig_width,
            args.gene_fig_height,
            args.gene_label_fontsize,
        )

    summary = cluster_expression_summary(adata, args.cluster_key, ALL_GENES)
    summary.to_csv(tables_dir / "cluster_marker_expression_summary.csv", index=False)

    plot_heatmap(
        summary,
        "mean_expression",
        "Mean Expression by Cluster",
        figures_dir / "cluster_marker_mean_expression_heatmap.png",
        args.cluster_key,
        args.dpi,
    )
    plot_heatmap(
        summary,
        "fraction_expressing",
        "Fraction Expressing by Cluster",
        figures_dir / "cluster_marker_fraction_expressing_heatmap.png",
        args.cluster_key,
        args.dpi,
    )
    score = plot_score_scatter(
        summary,
        args.cluster_key,
        figures_dir / "cluster_photo_vs_bc_marker_score_scatter.png",
        args.dpi,
    )
    score.to_csv(tables_dir / "cluster_photo_vs_bc_marker_scores.csv", index=False)
    plot_flagged_reference(
        adata,
        centers,
        score,
        args.cluster_key,
        figures_dir / "top_photo_like_candidate_clusters_reference.png",
        args.dpi,
    )

    print(figures_dir)
    print(tables_dir)


if __name__ == "__main__":
    main()

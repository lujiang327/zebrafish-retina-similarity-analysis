#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path

import argparse

import matplotlib.pyplot as plt
import pandas as pd
import scanpy as sc


DEFAULT_H5AD = "results/h5ad/bc_strict_harmony_by_sample_reprocessed.h5ad"
DEFAULT_FIGURES_DIR = Path("results/figures/bc_strict_harmony_by_sample_review")
DEFAULT_TABLES_DIR = Path("results/tables/bc_strict_harmony_by_sample_review")

SAMPLE9_COLORS = {
    "TH115": "#ff7f0e",
    "TH44": "#2ca02c",
    "TH54": "#d62728",
    "TH55": "#9467bd",
    "TH56": "#8c564b",
    "TH57": "#e377c2",
    "TH71": "#7f7f7f",
    "Zebra_LD": "#bcbd22",
    "Zebra_NMDA": "#17becf",
}


def split_zebra_samples(adata) -> pd.Series:
    samples = adata.obs["sample"].astype(str).copy()
    conditions = adata.obs["renamed_samples"].astype(str)
    zebra = samples == "Zebra"
    samples.loc[zebra] = "Zebra_" + conditions.loc[zebra]
    return samples


def style_axis(ax) -> None:
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot BC Harmony UMAPs with Zebra split into LD/NMDA.")
    parser.add_argument("--h5ad", default=DEFAULT_H5AD)
    parser.add_argument("--figures-dir", default=str(DEFAULT_FIGURES_DIR))
    parser.add_argument("--tables-dir", default=str(DEFAULT_TABLES_DIR))
    parser.add_argument("--prefix", default="bc_strict_harmony_by_sample")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    figures_dir = Path(args.figures_dir)
    tables_dir = Path(args.tables_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    adata = sc.read_h5ad(args.h5ad)
    coords = adata.obsm["X_umap"]
    sample9 = split_zebra_samples(adata)
    ordered = [sample for sample in SAMPLE9_COLORS if sample in set(sample9)]
    ordered.extend(sorted(sample for sample in set(sample9) if sample not in ordered))

    counts = sample9.value_counts().reindex(ordered).rename_axis("sample_split_zebra").reset_index(name="n_cells")
    counts.to_csv(tables_dir / "counts_by_sample_split_zebra.csv", index=False)

    fig, ax = plt.subplots(figsize=(9.6, 6.2))
    for sample in ordered:
        mask = sample9 == sample
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=3,
            c=SAMPLE9_COLORS.get(sample, "#333333"),
            label=f"{sample} ({int(mask.sum()):,})",
            linewidths=0,
            alpha=0.9,
        )
    ax.set_title("BC Harmony by 9 samples", fontsize=18)
    style_axis(ax)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, markerscale=3)
    fig.tight_layout()
    fig.savefig(figures_dir / f"{args.prefix}_umap_by_9_samples.png", dpi=300)
    fig.savefig(figures_dir / f"{args.prefix}_umap_by_9_samples.pdf", dpi=300)
    plt.close(fig)

    xlim = (coords[:, 0].min(), coords[:, 0].max())
    ylim = (coords[:, 1].min(), coords[:, 1].max())
    fig, axes = plt.subplots(3, 3, figsize=(16, 13.2), constrained_layout=True)
    axes = axes.ravel()
    for ax, sample in zip(axes, ordered):
        mask = sample9 == sample
        ax.scatter(coords[~mask, 0], coords[~mask, 1], s=1.1, c="#d9d9d9", linewidths=0, alpha=0.2)
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=2.8,
            c=SAMPLE9_COLORS.get(sample, "#333333"),
            linewidths=0,
            alpha=0.95,
        )
        ax.set_title(f"{sample} ({int(mask.sum()):,})", fontsize=14)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        style_axis(ax)

    fig.suptitle("BC Harmony by 9 Samples: Split Samples", fontsize=22)
    fig.savefig(figures_dir / f"{args.prefix}_split_9_samples_grid.png", dpi=300)
    fig.savefig(figures_dir / f"{args.prefix}_split_9_samples_grid.pdf", dpi=300)
    plt.close(fig)

    print(figures_dir / f"{args.prefix}_umap_by_9_samples.png")
    print(figures_dir / f"{args.prefix}_split_9_samples_grid.png")
    print(tables_dir / "counts_by_sample_split_zebra.csv")


if __name__ == "__main__":
    main()

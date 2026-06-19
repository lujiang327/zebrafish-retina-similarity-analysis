#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import scanpy as sc


H5AD = "results/h5ad/bc_strict_harmony_by_sample_reprocessed.h5ad"
OUT_DIR = Path("results/figures/bc_strict_harmony_by_sample_review")

SAMPLE_COLORS = {
    "Zebra": "#1f77b4",
    "TH115": "#ff7f0e",
    "TH44": "#2ca02c",
    "TH54": "#d62728",
    "TH55": "#9467bd",
    "TH56": "#8c564b",
    "TH57": "#e377c2",
    "TH71": "#7f7f7f",
}


def style_axis(ax) -> None:
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(1)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    adata = sc.read_h5ad(H5AD)
    coords = adata.obsm["X_umap"]
    samples = adata.obs["sample"].astype(str)
    ordered = [sample for sample in SAMPLE_COLORS if sample in set(samples)]
    ordered.extend(sorted(sample for sample in set(samples) if sample not in ordered))

    xlim = (coords[:, 0].min(), coords[:, 0].max())
    ylim = (coords[:, 1].min(), coords[:, 1].max())

    for sample in ordered:
        mask = samples == sample
        fig, ax = plt.subplots(figsize=(8.95, 6.0))
        ax.scatter(coords[~mask, 0], coords[~mask, 1], s=2, c="#d9d9d9", linewidths=0, alpha=0.28)
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=4,
            c=SAMPLE_COLORS.get(sample, "#333333"),
            linewidths=0,
            alpha=0.95,
        )
        ax.set_title(f"Harmony by sample | {sample} ({int(mask.sum()):,} cells)", fontsize=18)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        style_axis(ax)
        fig.tight_layout()
        fig.savefig(OUT_DIR / f"bc_strict_harmony_by_sample_umap_{sample}.png", dpi=300)
        fig.savefig(OUT_DIR / f"bc_strict_harmony_by_sample_umap_{sample}.pdf", dpi=300)
        plt.close(fig)

    fig, axes = plt.subplots(2, 4, figsize=(18, 8.8), constrained_layout=True)
    axes = axes.ravel()
    for ax, sample in zip(axes, ordered):
        mask = samples == sample
        ax.scatter(coords[~mask, 0], coords[~mask, 1], s=1.3, c="#d9d9d9", linewidths=0, alpha=0.22)
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=2.7,
            c=SAMPLE_COLORS.get(sample, "#333333"),
            linewidths=0,
            alpha=0.95,
        )
        ax.set_title(f"{sample} ({int(mask.sum()):,})", fontsize=15)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        style_axis(ax)

    for ax in axes[len(ordered) :]:
        ax.axis("off")

    fig.suptitle("BC Harmony by Sample: Split Original Samples", fontsize=22)
    fig.savefig(OUT_DIR / "bc_strict_harmony_by_sample_split_samples_grid.png", dpi=300)
    fig.savefig(OUT_DIR / "bc_strict_harmony_by_sample_split_samples_grid.pdf", dpi=300)
    plt.close(fig)

    print(OUT_DIR / "bc_strict_harmony_by_sample_split_samples_grid.png")
    print(OUT_DIR / "bc_strict_harmony_by_sample_split_samples_grid.pdf")


if __name__ == "__main__":
    main()

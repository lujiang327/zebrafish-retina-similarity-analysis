#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import scanpy as sc


H5AD = "results/h5ad/bc_strict_harmony_by_sample_reprocessed.h5ad"
OUT_DIR = Path("results/figures/bc_strict_harmony_by_sample_review")

CONDITION_COLORS = {
    "Control": "#1f77b4",
    "LD": "#ff7f0e",
    "NMDA": "#2ca02c",
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
    conditions = adata.obs["renamed_samples"].astype(str)
    ordered = [condition for condition in CONDITION_COLORS if condition in set(conditions)]
    ordered.extend(sorted(condition for condition in set(conditions) if condition not in ordered))

    fig, ax = plt.subplots(figsize=(8.95, 6.0))
    for condition in ordered:
        mask = conditions == condition
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=3,
            c=CONDITION_COLORS.get(condition, "#333333"),
            label=f"{condition} ({int(mask.sum()):,})",
            linewidths=0,
            alpha=0.9,
        )
    ax.set_title("BC Harmony by sample | renamed_samples", fontsize=18)
    style_axis(ax)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, markerscale=3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "bc_strict_harmony_by_sample_umap_by_renamed_samples.png", dpi=300)
    fig.savefig(OUT_DIR / "bc_strict_harmony_by_sample_umap_by_renamed_samples.pdf", dpi=300)
    plt.close(fig)

    xlim = (coords[:, 0].min(), coords[:, 0].max())
    ylim = (coords[:, 1].min(), coords[:, 1].max())
    fig, axes = plt.subplots(1, len(ordered), figsize=(5.5 * len(ordered), 5.0), constrained_layout=True)
    if len(ordered) == 1:
        axes = [axes]
    for ax, condition in zip(axes, ordered):
        mask = conditions == condition
        ax.scatter(coords[~mask, 0], coords[~mask, 1], s=1.5, c="#d9d9d9", linewidths=0, alpha=0.22)
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=4,
            c=CONDITION_COLORS.get(condition, "#333333"),
            linewidths=0,
            alpha=0.95,
        )
        ax.set_title(f"{condition} ({int(mask.sum()):,})", fontsize=16)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        style_axis(ax)
    fig.suptitle("BC Harmony by Sample: Split Conditions", fontsize=20)
    fig.savefig(OUT_DIR / "bc_strict_harmony_by_sample_split_renamed_samples.png", dpi=300)
    fig.savefig(OUT_DIR / "bc_strict_harmony_by_sample_split_renamed_samples.pdf", dpi=300)
    plt.close(fig)

    print(OUT_DIR / "bc_strict_harmony_by_sample_umap_by_renamed_samples.png")
    print(OUT_DIR / "bc_strict_harmony_by_sample_split_renamed_samples.png")


if __name__ == "__main__":
    main()

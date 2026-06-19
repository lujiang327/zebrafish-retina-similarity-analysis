#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import scanpy as sc


H5AD = "results/h5ad/bc_strict_harmony_by_9_samples_reprocessed.h5ad"
OUT_DIR = Path("results/figures/bc_strict_harmony_by_9_samples_marker_comparison")

GENE_ROWS = [
    ("Photoreceptor-like", ["nr2e3", "pde6a", "guca1b"]),
    ("BC markers", ["vsx1", "vsx2", "cabp5a", "cabp5b", "grm6a", "grm6b", "pcp4a"]),
]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    adata = sc.read_h5ad(H5AD)
    genes = [gene for _, row_genes in GENE_ROWS for gene in row_genes]

    fig, axes = plt.subplots(3, 4, figsize=(17, 11.0), constrained_layout=True)
    axes = axes.ravel()
    for ax in axes:
        ax.axis("off")

    ordered_genes = ["nr2e3", "pde6a", "guca1b", "vsx1", "vsx2", "cabp5a", "cabp5b", "grm6a", "grm6b", "pcp4a"]
    positions = [0, 1, 2, 4, 5, 6, 7, 8, 9, 10]
    for gene, pos in zip(ordered_genes, positions):
        ax = axes[pos]
        ax.axis("on")
        sc.pl.umap(
            adata,
            color=gene,
            use_raw=adata.raw is not None,
            cmap="viridis",
            size=3,
            frameon=True,
            show=False,
            ax=ax,
            colorbar_loc="right",
        )
        ax.set_title(gene, fontsize=16)
        ax.set_xlabel("UMAP1", fontsize=10)
        ax.set_ylabel("UMAP2", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    axes[0].text(
        -0.08,
        1.08,
        "Photoreceptor-like",
        transform=axes[0].transAxes,
        fontsize=17,
        fontweight="bold",
        va="bottom",
    )
    axes[4].text(
        -0.08,
        1.08,
        "BC markers",
        transform=axes[4].transAxes,
        fontsize=17,
        fontweight="bold",
        va="bottom",
    )
    fig.suptitle("BC 9-Sample Harmony: Photoreceptor-Like vs BC Marker Expression", fontsize=22)
    fig.savefig(OUT_DIR / "bc_9_sample_harmony_photoreceptor_vs_bc_marker_grid.png", dpi=300)
    fig.savefig(OUT_DIR / "bc_9_sample_harmony_photoreceptor_vs_bc_marker_grid.pdf", dpi=300)
    plt.close(fig)

    print(OUT_DIR / "bc_9_sample_harmony_photoreceptor_vs_bc_marker_grid.png")
    print(OUT_DIR / "bc_9_sample_harmony_photoreceptor_vs_bc_marker_grid.pdf")


if __name__ == "__main__":
    main()

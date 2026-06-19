#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import scanpy as sc


DATASETS = [
    (
        "BC original",
        "/Users/hoanglab/Desktop/vscode_projects/zebrafish-singlecell-portal/"
        "BC_annotated_clustered_corrected_doubletRemoved_Zebrafishes.h5ad",
    ),
    (
        "Harmony by condition",
        "results/h5ad/bc_strict_harmony_reprocessed.h5ad",
    ),
    (
        "Harmony by sample",
        "results/h5ad/bc_strict_harmony_by_sample_reprocessed.h5ad",
    ),
]

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


def main() -> None:
    figures_dir = Path("results/figures/bc_harmony_strategy_comparison")
    figures_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(21, 6.6), constrained_layout=True)
    for ax, (label, path) in zip(axes, DATASETS):
        adata = sc.read_h5ad(path)
        coords = adata.obsm["X_umap"]
        samples = adata.obs["sample"].astype(str)
        ordered = [sample for sample in SAMPLE_COLORS if sample in set(samples)]
        ordered.extend(sorted(sample for sample in set(samples) if sample not in ordered))

        for sample in ordered:
            mask = samples == sample
            ax.scatter(
                coords[mask, 0],
                coords[mask, 1],
                s=3,
                c=SAMPLE_COLORS.get(sample, "#333333"),
                label=sample,
                linewidths=0,
                alpha=0.9,
            )

        ax.set_title(f"{label}\n{adata.n_obs:,} cells", fontsize=17)
        ax.set_xlabel("UMAP1")
        ax.set_ylabel("UMAP2")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(1)

    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center right", frameon=False, markerscale=3, fontsize=12)
    fig.suptitle("BC UMAPs Colored by Original Sample", fontsize=22)
    fig.savefig(figures_dir / "bc_original_vs_harmony_condition_vs_harmony_sample_by_sample.png", dpi=300)
    fig.savefig(figures_dir / "bc_original_vs_harmony_condition_vs_harmony_sample_by_sample.pdf", dpi=300)
    plt.close(fig)

    print(figures_dir / "bc_original_vs_harmony_condition_vs_harmony_sample_by_sample.png")
    print(figures_dir / "bc_original_vs_harmony_condition_vs_harmony_sample_by_sample.pdf")


if __name__ == "__main__":
    main()

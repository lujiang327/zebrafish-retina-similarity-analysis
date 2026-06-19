#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import scanpy as sc


DATASETS = [
    (
        "RGC",
        "/Users/hoanglab/Desktop/vscode_projects/zebrafish-singlecell-portal/"
        "corrected_RGC_annotated_clustered_corrected_doubletRemoved_Zebrafishes.h5ad",
    ),
    (
        "AC",
        "/Users/hoanglab/Desktop/vscode_projects/zebrafish-singlecell-portal/"
        "AC_subtypes_reproduced.h5ad",
    ),
    (
        "BC original",
        "/Users/hoanglab/Desktop/vscode_projects/zebrafish-singlecell-portal/"
        "BC_annotated_clustered_corrected_doubletRemoved_Zebrafishes.h5ad",
    ),
    (
        "BC reprocessed",
        "results/h5ad/bc_strict_harmony_reprocessed.h5ad",
    ),
]

SAMPLE_COLORS = {
    "Control": "#1f77b4",
    "LD": "#ff7f0e",
    "NMDA": "#2ca02c",
}

ORIGINAL_SAMPLE_COLORS = {
    "TH44": "#1f77b4",
    "TH54": "#ff7f0e",
    "TH55": "#2ca02c",
    "TH56": "#d62728",
    "TH57": "#9467bd",
    "TH71": "#8c564b",
    "TH115": "#e377c2",
    "Zebra": "#7f7f7f",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare sample-colored UMAPs across cell-type h5ads.")
    parser.add_argument(
        "--figures-dir",
        default="results/figures/celltype_sample_umap_comparison",
        help="Output figure directory.",
    )
    parser.add_argument(
        "--color-key",
        default="renamed_samples",
        choices=["renamed_samples", "sample"],
        help="Observation column used to color UMAPs.",
    )
    parser.add_argument("--dpi", type=int, default=300, help="PNG/PDF DPI.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12), constrained_layout=True)
    axes = axes.ravel()

    for ax, (label, path) in zip(axes, DATASETS):
        adata = sc.read_h5ad(path)
        if "X_umap" not in adata.obsm:
            raise ValueError(f"{label} does not contain adata.obsm['X_umap']: {path}")
        if args.color_key not in adata.obs:
            raise KeyError(f"{label} does not contain adata.obs['{args.color_key}']: {path}")

        coords = adata.obsm["X_umap"]
        samples = adata.obs[args.color_key].astype(str)
        point_size = 3 if adata.n_obs > 20000 else 8
        color_map = SAMPLE_COLORS if args.color_key == "renamed_samples" else ORIGINAL_SAMPLE_COLORS
        ordered_samples = [sample for sample in color_map if sample in set(samples)]
        ordered_samples.extend(sorted(sample for sample in set(samples) if sample not in ordered_samples))
        for sample in ordered_samples:
            mask = samples == sample
            if not mask.any():
                continue
            ax.scatter(
                coords[mask, 0],
                coords[mask, 1],
                s=point_size,
                c=color_map.get(sample, "#333333"),
                label=f"{sample} ({int(mask.sum()):,})",
                linewidths=0,
                alpha=0.9,
            )

        ax.set_title(f"{label}: {adata.n_obs:,} cells", fontsize=18)
        ax.set_xlabel("UMAP1")
        ax.set_ylabel("UMAP2")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend(loc="best", frameon=False, fontsize=10, markerscale=2)
        for spine in ax.spines.values():
            spine.set_linewidth(1.0)

    fig.suptitle(f"UMAP by {args.color_key}: RGC, AC, BC original, BC reprocessed", fontsize=22)
    out_png = figures_dir / f"{args.color_key}_umap_comparison_RGC_AC_BC_original_BC_reprocessed.png"
    out_pdf = figures_dir / f"{args.color_key}_umap_comparison_RGC_AC_BC_original_BC_reprocessed.pdf"
    fig.savefig(out_png, dpi=args.dpi, bbox_inches="tight")
    fig.savefig(out_pdf, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(out_png)
    print(out_pdf)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse


DEFAULT_H5AD = (
    "results/h5ad/"
    "bc_9_sample_guca1b_gt2_filtered.h5ad"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot BC guca1b > 2.0 DEG top1 gene UMAPs.")
    parser.add_argument("--h5ad", default=DEFAULT_H5AD, help="Input BC AnnData h5ad file.")
    parser.add_argument(
        "--gene-csv",
        nargs="+",
        default=[
            "results/tables/bc_guca1b_gt2_ac_style_gene_umaps/bc_guca1b_gt2_deg_max_other_top1_marker_per_cluster.csv",
            "results/tables/bc_guca1b_gt2_ac_style_gene_umaps/bc_guca1b_gt2_deg_specific_top1_marker_per_cluster.csv",
        ],
        help="CSV files with a names column to use as the gene list.",
    )
    parser.add_argument(
        "--cluster-key",
        default="leiden",
        help="BC cluster column in adata.obs.",
    )
    parser.add_argument(
        "--condition-key",
        default="renamed_samples",
        help="Condition/sample column in adata.obs.",
    )
    parser.add_argument(
        "--figures-dir",
        default="results/figures/bc_guca1b_gt2_ac_style_gene_umaps",
        help="Output figure directory.",
    )
    parser.add_argument(
        "--tables-dir",
        default="results/tables/bc_guca1b_gt2_ac_style_gene_umaps",
        help="Output table directory.",
    )
    parser.add_argument("--dpi", type=int, default=300, help="Output figure DPI.")
    return parser.parse_args()


def gene_lookup(var_names) -> dict[str, str]:
    return {str(gene).lower(): str(gene) for gene in var_names}


def read_gene_list(csv_paths: list[str]) -> list[str]:
    genes = []
    seen = set()
    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)
        if "names" not in df.columns:
            raise KeyError(f"Gene CSV must contain a names column: {csv_path}")
        for gene in df["names"].dropna().astype(str):
            key = gene.lower()
            if key not in seen:
                genes.append(gene)
                seen.add(key)
    return genes


def expression_matrix_and_names(adata):
    if adata.raw is not None:
        return adata.raw.X, pd.Index(adata.raw.var_names)
    return adata.X, pd.Index(adata.var_names)


def dense_vector(matrix, column_index: int) -> np.ndarray:
    column = matrix[:, column_index]
    if sparse.issparse(column):
        return column.toarray().ravel()
    return np.asarray(column).ravel()


def expression_vector(adata, gene: str) -> np.ndarray:
    matrix, var_names = expression_matrix_and_names(adata)
    idx = int(var_names.get_loc(gene))
    return dense_vector(matrix, idx)


def save_overview_umap(adata, color: str, output_path: Path, dpi: int) -> None:
    fig = sc.pl.umap(
        adata,
        color=color,
        legend_loc="on data" if "cluster" in color or color.endswith("leiden") else None,
        size=10,
        frameon=True,
        show=False,
        return_fig=True,
    )
    fig.set_size_inches(8.95, 6.0)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_sample_highlight_umaps(adata, condition_key: str, figures_dir: Path, dpi: int) -> None:
    coords = adata.obsm["X_umap"]
    condition_values = adata.obs[condition_key].astype(str).to_numpy()
    sample_colors = {"Control": "#1f77b4", "LD": "#ff7f0e", "NMDA": "#2ca02c"}
    ordered_samples = [sample for sample in ["Control", "LD", "NMDA"] if sample in set(condition_values)]
    ordered_samples.extend(sorted(sample for sample in set(condition_values) if sample not in ordered_samples))

    for sample in ordered_samples:
        mask = condition_values == sample
        fig, ax = plt.subplots(figsize=(8.95, 6.0))
        ax.scatter(coords[:, 0], coords[:, 1], c="#d9d9d9", s=4, linewidths=0, alpha=0.35)
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            c=sample_colors.get(sample, "#1f77b4"),
            s=8,
            linewidths=0,
            label=sample,
        )
        ax.set_title(f"{condition_key}: {sample}", fontsize=20)
        ax.set_xlabel("UMAP1", fontsize=16)
        ax.set_ylabel("UMAP2", fontsize=16)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)
        fig.savefig(
            figures_dir / f"bc_umap_by_{condition_key}_{sample}_highlight.png",
            dpi=dpi,
            bbox_inches="tight",
        )
        plt.close(fig)


def plot_gene_all_conditions(adata, gene: str, output_path: Path, dpi: int) -> None:
    fig = sc.pl.umap(
        adata,
        color=gene,
        frameon=True,
        cmap="viridis",
        use_raw=adata.raw is not None,
        size=16,
        show=False,
        return_fig=True,
    )
    fig.set_size_inches(8.95, 6.0)
    ax = fig.axes[0]
    ax.set_title(gene, fontsize=24, pad=12)
    ax.set_xlabel("UMAP1", fontsize=22)
    ax.set_ylabel("UMAP2", fontsize=22)
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_gene_by_condition(
    adata,
    gene: str,
    condition_key: str,
    condition_panels: list[tuple[str, tuple[str, ...]]],
    output_path: Path,
    dpi: int,
) -> None:
    coords = adata.obsm["X_umap"]
    expr = expression_vector(adata, gene)
    finite_expr = expr[np.isfinite(expr)]
    vmin = float(np.nanpercentile(finite_expr, 1)) if finite_expr.size else 0.0
    vmax = float(np.nanpercentile(finite_expr, 99)) if finite_expr.size else 1.0
    if not np.isfinite(vmax) or vmax <= vmin:
        vmax = float(np.nanmax(finite_expr)) if finite_expr.size else 1.0

    n_panels = len(condition_panels)
    n_cols = 2 if n_panels > 1 else 1
    n_rows = int(np.ceil(n_panels / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.4 * n_cols + 0.8, 3.8 * n_rows))
    axes = np.atleast_1d(axes).ravel()

    condition_values = adata.obs[condition_key].astype(str).to_numpy()
    sca = None
    for ax, (label, values) in zip(axes[:n_panels], condition_panels):
        mask = np.isin(condition_values, values)
        ax.scatter(coords[:, 0], coords[:, 1], c="#d9d9d9", s=4, linewidths=0, alpha=0.35)
        sca = ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            c=expr[mask],
            s=7,
            linewidths=0,
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(f"{gene} | {label}")
        ax.set_xlabel("UMAP1")
        ax.set_ylabel("UMAP2")
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes[n_panels:]:
        ax.set_visible(False)
    fig.subplots_adjust(left=0.08, right=0.88, bottom=0.08, top=0.92, wspace=0.28, hspace=0.35)
    if sca is not None:
        cbar_ax = fig.add_axes([0.91, 0.2, 0.02, 0.6])
        cbar = fig.colorbar(sca, cax=cbar_ax)
        cbar.set_label("Expression")
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    figures_dir = Path(args.figures_dir)
    tables_dir = Path(args.tables_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    adata = sc.read_h5ad(args.h5ad)
    if "X_umap" not in adata.obsm:
        raise ValueError("Input h5ad does not contain adata.obsm['X_umap']")

    lookup = gene_lookup(expression_matrix_and_names(adata)[1])
    requested_genes = read_gene_list(args.gene_csv)
    records = []
    present_genes = []
    for requested in requested_genes:
        matched = lookup.get(requested.lower())
        records.append(
            {
                "requested_gene": requested,
                "matched_gene": matched or "",
                "present": matched is not None,
                "expression_source": "raw" if adata.raw is not None else "X",
            }
        )
        if matched is not None and matched not in present_genes:
            present_genes.append(matched)
    pd.DataFrame(records).to_csv(tables_dir / "bc_guca1b_gt2_deg_top1_gene_match_report.csv", index=False)

    save_overview_umap(
        adata,
        args.condition_key,
        figures_dir / f"bc_umap_by_{args.condition_key}.png",
        args.dpi,
    )
    save_sample_highlight_umaps(adata, args.condition_key, figures_dir, args.dpi)
    save_overview_umap(
        adata,
        args.cluster_key,
        figures_dir / f"bc_umap_by_{args.cluster_key}.png",
        args.dpi,
    )

    available_conditions = set(adata.obs[args.condition_key].astype(str))
    condition_panels = [
        (condition, (condition,))
        for condition in ["Control", "LD", "NMDA"]
        if condition in available_conditions
    ]
    if {"LD", "NMDA"}.issubset(available_conditions):
        condition_panels.append(("LD + NMDA", ("LD", "NMDA")))

    for gene in present_genes:
        plot_gene_all_conditions(
            adata,
            gene,
            figures_dir / f"bc_umap_gene_{gene}_all_conditions.png",
            args.dpi,
        )
        plot_gene_by_condition(
            adata,
            gene,
            args.condition_key,
            condition_panels,
            figures_dir / f"bc_umap_gene_{gene}_split_by_condition.png",
            args.dpi,
        )

    print(tables_dir / "bc_guca1b_gt2_deg_top1_gene_match_report.csv")


if __name__ == "__main__":
    main()

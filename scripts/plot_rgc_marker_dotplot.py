#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc


DOTPLOT_GENES = [
    "emid1",
    "eomesa",
    "gpd1b",
    "gria3a",
    "gulp1b",
    "maffa",
    "matn3b",
    "neurod2",
    "npb",
    "pcdh11",
    "pgam2",
    "ptprt",
    "rprma",
    "si:dkey-1h6.8",
    "tacr2",
    "tbr1b",
    "tbx5a",
    "tox",
    "tubb5",
    "zic2a",
    "rplp1",
    "gipr",
    "bcl11ba",
    "dacha",
    "foxp2",
    "nr2f2",
    "meis1b",
    "tubb5",
    "usp43a",
    "pdyn",
    "rprm3",
    "sox6",
    "lrtm41l",
    "mdga1",
    "si:ch211-263k4.2-204",
    "sdk1a",
]

GENE_ALIASES = {
    "maffa": "mafaa",
    "si:ch211-263k4.2-204": "si:ch211-263k4.2",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot RGC marker dotplot after merging clusters 1 and 35."
    )
    parser.add_argument(
        "--h5ad",
        default="/Users/hoanglab/Desktop/vscode_projects/zebrafish-singlecell-portal/corrected_RGC_annotated_clustered_corrected_doubletRemoved_Zebrafishes.h5ad",
        help="Input AnnData h5ad file.",
    )
    parser.add_argument("--cluster-key", default="leiden", help="Cluster column in adata.obs.")
    parser.add_argument(
        "--figures-dir",
        default="results/figures/rgc_subtype_gene_umaps",
        help="Output figure directory.",
    )
    parser.add_argument(
        "--tables-dir",
        default="results/tables/rgc_subtype_gene_umaps",
        help="Output table directory.",
    )
    parser.add_argument("--dpi", type=int, default=300, help="Output figure DPI.")
    return parser.parse_args()


def gene_lookup(var_names) -> dict[str, str]:
    return {str(gene).lower(): str(gene) for gene in var_names}


def resolve_dotplot_genes(adata) -> tuple[list[str], pd.DataFrame]:
    lookup = gene_lookup(adata.var_names)
    seen = set()
    resolved = []
    records = []
    for requested in DOTPLOT_GENES:
        alias = GENE_ALIASES.get(requested, requested)
        matched = lookup.get(alias.lower())
        present = matched is not None
        if present and matched not in seen:
            resolved.append(matched)
            seen.add(matched)
        records.append(
            {
                "requested_gene": requested,
                "alias_used": alias if alias != requested else "",
                "matched_gene": matched or "",
                "present": present,
                "included_once": present and matched in resolved,
            }
        )
    return resolved, pd.DataFrame(records)


def add_merged_cluster_labels(
    adata,
    cluster_key: str,
    output_key: str = "leiden_1_35_merged_renamed",
) -> pd.DataFrame:
    original = adata.obs[cluster_key].astype(str)
    numeric_clusters = sorted({int(value) for value in original.unique()})
    collapsed = [1 if value == 35 else value for value in numeric_clusters]
    ordered_collapsed = []
    for value in collapsed:
        if value not in ordered_collapsed:
            ordered_collapsed.append(value)
    rename_map = {str(value): str(index) for index, value in enumerate(ordered_collapsed)}

    merged_original = original.replace({"35": "1"})
    adata.obs[output_key] = pd.Categorical(
        merged_original.map(rename_map),
        categories=[str(index) for index in range(len(ordered_collapsed))],
        ordered=True,
    )

    records = []
    for original_cluster in numeric_clusters:
        merged_cluster = 1 if original_cluster == 35 else original_cluster
        records.append(
            {
                "original_cluster": str(original_cluster),
                "merged_before_rename": str(merged_cluster),
                "renamed_cluster": rename_map[str(merged_cluster)],
            }
        )
    return pd.DataFrame(records)


def order_genes_by_peak_cluster(adata, genes: list[str], cluster_key: str) -> tuple[list[str], pd.DataFrame]:
    expr = adata[:, genes].to_df()
    expr[cluster_key] = adata.obs[cluster_key].astype(str).to_numpy()
    means = expr.groupby(cluster_key, observed=False)[genes].mean()
    means = means.reindex(sorted(means.index, key=lambda value: int(value)))

    records = []
    for gene in genes:
        values = means[gene]
        peak_cluster = str(values.idxmax())
        records.append(
            {
                "gene": gene,
                "peak_cluster": peak_cluster,
                "peak_mean_expression": float(values.loc[peak_cluster]),
                "overall_mean_expression": float(np.nanmean(expr[gene].to_numpy())),
            }
        )
    order_df = pd.DataFrame(records).sort_values(
        ["peak_cluster", "peak_mean_expression", "gene"],
        key=lambda col: col.astype(int) if col.name == "peak_cluster" else col,
        ascending=[True, False, True],
    )
    return order_df["gene"].tolist(), order_df


def main() -> None:
    args = parse_args()
    figures_dir = Path(args.figures_dir)
    tables_dir = Path(args.tables_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    adata = sc.read_h5ad(args.h5ad)
    if args.cluster_key not in adata.obs:
        raise KeyError(f"Cluster key not found in adata.obs: {args.cluster_key}")

    cluster_map = add_merged_cluster_labels(adata, args.cluster_key)
    genes, gene_report = resolve_dotplot_genes(adata)
    if not genes:
        raise ValueError("None of the requested dotplot genes were found in adata.var_names.")

    dotplot_key = "leiden_1_35_merged_renamed"
    out_png = figures_dir / "rgc_marker_dotplot_leiden_1_35_merged.png"
    out_pdf = figures_dir / "rgc_marker_dotplot_leiden_1_35_merged.pdf"
    out_umap = figures_dir / "rgc_umap_leiden_1_35_merged_renamed.png"
    cluster_map.to_csv(tables_dir / "rgc_dotplot_cluster_1_35_merge_map.csv", index=False)
    gene_report.to_csv(tables_dir / "rgc_dotplot_gene_match_report.csv", index=False)
    genes, gene_order = order_genes_by_peak_cluster(adata, genes, dotplot_key)
    gene_order.to_csv(tables_dir / "rgc_dotplot_gene_order_by_peak_cluster.csv", index=False)

    umap_fig = sc.pl.umap(
        adata,
        color=dotplot_key,
        legend_loc="on data",
        title="Leiden clusters: 1 and 35 merged, then renamed",
        size=20,
        frameon=True,
        show=False,
        return_fig=True,
    )
    umap_fig.set_size_inches(8.95, 6.0)
    for ax in umap_fig.axes:
        ax.set_xlabel("UMAP1", fontsize=16)
        ax.set_ylabel("UMAP2", fontsize=16)
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)
    umap_fig.savefig(out_umap, dpi=args.dpi, bbox_inches="tight")
    plt.close(umap_fig)

    width = max(13.0, 0.42 * len(genes))
    height = max(7.5, 0.24 * adata.obs[dotplot_key].nunique())
    dotplot = sc.pl.dotplot(
        adata,
        var_names=genes,
        groupby=dotplot_key,
        use_raw=False,
        standard_scale="var",
        dot_max=1.0,
        dot_min=0.0,
        color_map="Reds",
        dendrogram=False,
        show=False,
        return_fig=True,
    )
    dotplot.style(
        cmap="Reds",
        dot_edge_color="#d0d5dd",
        dot_edge_lw=0.35,
        largest_dot=90,
        smallest_dot=0,
    )
    dotplot.legend(width=1.2)
    dotplot.make_figure()
    fig = dotplot.fig
    fig.set_size_inches(width, height)
    fig.savefig(out_png, dpi=args.dpi, bbox_inches="tight")
    fig.savefig(out_pdf, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(out_png)


if __name__ == "__main__":
    main()

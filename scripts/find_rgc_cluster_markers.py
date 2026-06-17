#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from scipy.stats import mannwhitneyu

from plot_rgc_marker_dotplot import add_merged_cluster_labels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find top marker genes for merged RGC clusters using Scanpy DEG."
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
    parser.add_argument("--method", default="wilcoxon", help="Scanpy rank_genes_groups method.")
    parser.add_argument("--top-n", type=int, default=20, help="Top markers to keep per cluster.")
    parser.add_argument("--pval-adj", type=float, default=0.05, help="Adjusted p-value cutoff.")
    parser.add_argument("--min-logfc", type=float, default=0.25, help="Minimum log fold-change.")
    parser.add_argument(
        "--min-pct-group",
        type=float,
        default=0.10,
        help="Minimum fraction of cells expressing the gene in the target cluster.",
    )
    parser.add_argument(
        "--pairwise-pval-adj",
        type=float,
        default=0.05,
        help="Adjusted p-value cutoff for strict pairwise target-vs-competing-cluster checks.",
    )
    parser.add_argument(
        "--competitor-tolerance",
        type=float,
        default=0.90,
        help=(
            "Pairwise-check clusters with mean or detection fraction at least this "
            "fraction of the target cluster's value."
        ),
    )
    parser.add_argument("--dpi", type=int, default=300, help="Output figure DPI.")
    return parser.parse_args()


def pct_group_column(df: pd.DataFrame) -> str | None:
    for column in ["pct_nz_group", "pts"]:
        if column in df.columns:
            return column
    return None


def filter_markers(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    filtered = df.copy()
    if "pvals_adj" in filtered.columns:
        filtered = filtered[filtered["pvals_adj"] <= args.pval_adj]
    if "logfoldchanges" in filtered.columns:
        filtered = filtered[filtered["logfoldchanges"] >= args.min_logfc]
    pct_col = pct_group_column(filtered)
    if pct_col is not None:
        filtered = filtered[filtered[pct_col] >= args.min_pct_group]
    return filtered


def add_specificity_columns(df: pd.DataFrame) -> pd.DataFrame:
    annotated = df.copy()
    if "pct_nz_group" in annotated.columns and "pct_nz_reference" in annotated.columns:
        annotated["pct_diff"] = annotated["pct_nz_group"] - annotated["pct_nz_reference"]
        annotated["pct_ratio"] = (annotated["pct_nz_group"] + 0.01) / (
            annotated["pct_nz_reference"] + 0.01
        )
    else:
        annotated["pct_diff"] = 0.0
        annotated["pct_ratio"] = 1.0
    if "logfoldchanges" in annotated.columns:
        annotated["specificity_score"] = annotated["logfoldchanges"] * annotated["pct_diff"].clip(lower=0)
    else:
        annotated["specificity_score"] = annotated["pct_diff"].clip(lower=0)
    return annotated


def top_per_cluster(df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    ranked = df.copy()
    ranked["group_sort"] = ranked["group"].astype(int)
    return (
        ranked.sort_values(["group_sort", "scores"], ascending=[True, False])
        .groupby("group", observed=False)
        .head(top_n)
        .drop(columns=["group_sort"])
        .reset_index(drop=True)
    )


def top_specific_per_cluster(df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    ranked = add_specificity_columns(df)
    ranked["group_sort"] = ranked["group"].astype(int)
    return (
        ranked.sort_values(
            ["group_sort", "specificity_score", "logfoldchanges", "pct_diff", "scores"],
            ascending=[True, False, False, False, False],
        )
        .groupby("group", observed=False)
        .head(top_n)
        .drop(columns=["group_sort"])
        .reset_index(drop=True)
    )


def top_pval_per_cluster(df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    ranked = add_specificity_columns(df)
    ranked["group_sort"] = ranked["group"].astype(int)
    sort_columns = ["group_sort", "pvals_adj", "logfoldchanges", "scores"]
    ascending = [True, True, False, False]
    if "pvals_adj" not in ranked.columns:
        sort_columns = ["group_sort", "scores"]
        ascending = [True, False]
    return (
        ranked.sort_values(sort_columns, ascending=ascending)
        .groupby("group", observed=False)
        .head(top_n)
        .drop(columns=["group_sort"])
        .reset_index(drop=True)
    )


def bh_adjust(pvals: list[float]) -> np.ndarray:
    pvals_array = np.asarray(pvals, dtype=float)
    adjusted = np.full(pvals_array.shape, np.nan, dtype=float)
    valid = ~np.isnan(pvals_array)
    valid_pvals = pvals_array[valid]
    if valid_pvals.size == 0:
        return adjusted
    order = np.argsort(valid_pvals)
    ranked = valid_pvals[order]
    n = ranked.size
    corrected = ranked * n / np.arange(1, n + 1)
    corrected = np.minimum.accumulate(corrected[::-1])[::-1]
    restored = np.empty_like(corrected)
    restored[order] = np.clip(corrected, 0, 1)
    adjusted[valid] = restored
    return adjusted


def expression_source(adata):
    if adata.raw is not None:
        return adata.raw.X, pd.Index(adata.raw.var_names)
    return adata.X, pd.Index(adata.var_names)


def matrix_column_vector(matrix, column_index: int) -> np.ndarray:
    column = matrix[:, column_index]
    if sparse.issparse(column):
        return column.toarray().ravel()
    return np.asarray(column).ravel()


def compute_cluster_expression_stats(adata, cluster_key: str, genes: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    matrix, var_names = expression_source(adata)
    present_genes = [gene for gene in genes if gene in var_names]
    gene_indices = var_names.get_indexer(present_genes)
    subset = matrix[:, gene_indices]
    cluster_labels = adata.obs[cluster_key].astype(str)
    clusters = sorted(cluster_labels.unique(), key=lambda value: int(value))

    mean_rows = []
    pct_rows = []
    for cluster in clusters:
        mask = (cluster_labels == cluster).to_numpy()
        cluster_matrix = subset[mask, :]
        if sparse.issparse(cluster_matrix):
            means = np.asarray(cluster_matrix.mean(axis=0)).ravel()
            pcts = cluster_matrix.getnnz(axis=0) / cluster_matrix.shape[0]
        else:
            means = np.asarray(cluster_matrix).mean(axis=0)
            pcts = (np.asarray(cluster_matrix) > 0).mean(axis=0)
        mean_rows.append(means)
        pct_rows.append(pcts)

    mean_df = pd.DataFrame(np.vstack(mean_rows).T, index=present_genes, columns=clusters)
    pct_df = pd.DataFrame(np.vstack(pct_rows).T, index=present_genes, columns=clusters)
    return mean_df, pct_df


def annotate_max_other_specificity(
    markers: pd.DataFrame,
    mean_df: pd.DataFrame,
    pct_df: pd.DataFrame,
) -> pd.DataFrame:
    annotated = add_specificity_columns(markers)
    clusters = list(mean_df.columns)

    rows = []
    for row in annotated.itertuples(index=False):
        gene = str(row.names)
        group = str(row.group)
        if gene not in mean_df.index or group not in clusters:
            rows.append({})
            continue
        other_clusters = [cluster for cluster in clusters if cluster != group]
        target_mean = float(mean_df.at[gene, group])
        target_pct = float(pct_df.at[gene, group])
        other_means = mean_df.loc[gene, other_clusters]
        other_pcts = pct_df.loc[gene, other_clusters]
        max_mean_cluster = str(other_means.idxmax())
        max_pct_cluster = str(other_pcts.idxmax())
        max_other_mean = float(other_means.loc[max_mean_cluster])
        max_other_pct = float(other_pcts.loc[max_pct_cluster])
        rows.append(
            {
                "mean_expr_group": target_mean,
                "max_other_mean_expr": max_other_mean,
                "max_other_mean_cluster": max_mean_cluster,
                "mean_expr_gap_vs_max_other": target_mean - max_other_mean,
                "pct_group_from_matrix": target_pct,
                "max_other_pct_nz": max_other_pct,
                "max_other_pct_cluster": max_pct_cluster,
                "pct_gap_vs_max_other": target_pct - max_other_pct,
            }
        )

    max_other = pd.DataFrame(rows)
    annotated = pd.concat([annotated.reset_index(drop=True), max_other], axis=1)
    pct_gap = annotated["pct_gap_vs_max_other"].fillna(0).clip(lower=0)
    mean_gap = annotated["mean_expr_gap_vs_max_other"].fillna(0).clip(lower=0)
    if "logfoldchanges" in annotated.columns:
        annotated["max_other_specificity_score"] = annotated["logfoldchanges"] * pct_gap * mean_gap
    else:
        annotated["max_other_specificity_score"] = pct_gap * mean_gap
    return annotated


def top_max_other_per_cluster(df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    ranked = df.copy()
    ranked["group_sort"] = ranked["group"].astype(int)
    return (
        ranked.sort_values(
            [
                "group_sort",
                "max_other_specificity_score",
                "pct_gap_vs_max_other",
                "mean_expr_gap_vs_max_other",
                "specificity_score",
                "scores",
            ],
            ascending=[True, False, False, False, False, False],
        )
        .groupby("group", observed=False)
        .head(top_n)
        .drop(columns=["group_sort"])
        .reset_index(drop=True)
    )


def pairwise_competitor_tests(
    adata,
    candidates: pd.DataFrame,
    mean_df: pd.DataFrame,
    pct_df: pd.DataFrame,
    cluster_key: str,
    competitor_tolerance: float,
    pval_adj_cutoff: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    matrix, var_names = expression_source(adata)
    cluster_labels = adata.obs[cluster_key].astype(str)
    clusters = list(mean_df.columns)
    cluster_masks = {
        cluster: (cluster_labels == cluster).to_numpy()
        for cluster in clusters
    }
    gene_to_index = {gene: idx for idx, gene in enumerate(var_names)}

    test_rows = []
    for row in candidates.itertuples(index=False):
        gene = str(row.names)
        group = str(row.group)
        if gene not in gene_to_index or gene not in mean_df.index or group not in clusters:
            continue
        target_mean = float(mean_df.at[gene, group])
        target_pct = float(pct_df.at[gene, group])
        other_clusters = [cluster for cluster in clusters if cluster != group]
        competitors = [
            cluster
            for cluster in other_clusters
            if float(mean_df.at[gene, cluster]) >= target_mean * competitor_tolerance
            or float(pct_df.at[gene, cluster]) >= target_pct * competitor_tolerance
        ]
        if not competitors:
            test_rows.append(
                {
                    "group": group,
                    "names": gene,
                    "competitor_cluster": "",
                    "competitor_reason": "none",
                    "target_mean_expr": target_mean,
                    "competitor_mean_expr": np.nan,
                    "target_pct_nz": target_pct,
                    "competitor_pct_nz": np.nan,
                    "pval": np.nan,
                }
            )
            continue

        expression = matrix_column_vector(matrix, gene_to_index[gene])
        target_values = expression[cluster_masks[group]]
        for competitor in competitors:
            competitor_values = expression[cluster_masks[competitor]]
            try:
                pval = mannwhitneyu(
                    target_values,
                    competitor_values,
                    alternative="greater",
                    method="asymptotic",
                ).pvalue
            except ValueError:
                pval = np.nan
            mean_competes = float(mean_df.at[gene, competitor]) >= target_mean * competitor_tolerance
            pct_competes = float(pct_df.at[gene, competitor]) >= target_pct * competitor_tolerance
            reasons = []
            if mean_competes:
                reasons.append("mean")
            if pct_competes:
                reasons.append("pct")
            test_rows.append(
                {
                    "group": group,
                    "names": gene,
                    "competitor_cluster": competitor,
                    "competitor_reason": "+".join(reasons),
                    "target_mean_expr": target_mean,
                    "competitor_mean_expr": float(mean_df.at[gene, competitor]),
                    "target_pct_nz": target_pct,
                    "competitor_pct_nz": float(pct_df.at[gene, competitor]),
                    "pval": pval,
                }
            )

    tests = pd.DataFrame(test_rows)
    if tests.empty:
        summary = candidates.copy()
        summary["pairwise_competing_clusters"] = ""
        summary["pairwise_n_competitors"] = 0
        summary["pairwise_worst_adj_p"] = np.nan
        summary["pairwise_passes_all_competitors"] = True
        return tests, summary

    tested = tests["competitor_cluster"].astype(str) != ""
    tests["pval_adj"] = np.nan
    tests.loc[tested, "pval_adj"] = bh_adjust(tests.loc[tested, "pval"].tolist())
    tests["pairwise_pass"] = ~tested | (tests["pval_adj"] <= pval_adj_cutoff)

    summary_rows = []
    for (group, gene), group_tests in tests.groupby(["group", "names"], observed=False):
        competitor_tests = group_tests[group_tests["competitor_cluster"].astype(str) != ""]
        summary_rows.append(
            {
                "group": group,
                "names": gene,
                "pairwise_competing_clusters": ";".join(
                    competitor_tests["competitor_cluster"].astype(str).tolist()
                ),
                "pairwise_n_competitors": int(len(competitor_tests)),
                "pairwise_worst_adj_p": (
                    float(competitor_tests["pval_adj"].max()) if not competitor_tests.empty else np.nan
                ),
                "pairwise_passes_all_competitors": bool(group_tests["pairwise_pass"].all()),
            }
        )
    summary = candidates.merge(pd.DataFrame(summary_rows), on=["group", "names"], how="left")
    summary["pairwise_competing_clusters"] = summary["pairwise_competing_clusters"].fillna("")
    summary["pairwise_n_competitors"] = summary["pairwise_n_competitors"].fillna(0).astype(int)
    summary["pairwise_passes_all_competitors"] = (
        summary["pairwise_passes_all_competitors"].fillna(True).astype(bool)
    )
    return tests, summary


def top_pairwise_validated_per_cluster(df: pd.DataFrame) -> pd.DataFrame:
    ranked = df.copy()
    ranked["group_sort"] = ranked["group"].astype(int)
    ranked = ranked.sort_values(
        [
            "group_sort",
            "pairwise_passes_all_competitors",
            "max_other_specificity_score",
            "pct_gap_vs_max_other",
            "mean_expr_gap_vs_max_other",
            "specificity_score",
            "scores",
        ],
        ascending=[True, False, False, False, False, False, False],
    )
    return (
        ranked.groupby("group", observed=False)
        .head(1)
        .drop(columns=["group_sort"])
        .reset_index(drop=True)
    )


def plot_top1_dotplot(adata, top1: pd.DataFrame, cluster_key: str, output_path: Path, dpi: int) -> None:
    genes = []
    for gene in top1["names"].astype(str):
        if gene not in genes:
            genes.append(gene)
    if not genes:
        return

    width = max(13.0, 0.42 * len(genes))
    height = max(7.5, 0.24 * adata.obs[cluster_key].nunique())
    dotplot = sc.pl.dotplot(
        adata,
        var_names=genes,
        groupby=cluster_key,
        use_raw=adata.raw is not None,
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
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    figures_dir = Path(args.figures_dir)
    tables_dir = Path(args.tables_dir)
    strict_figures_dir = figures_dir / "strict_marker_review"
    strict_tables_dir = tables_dir / "strict_marker_review"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    strict_figures_dir.mkdir(parents=True, exist_ok=True)
    strict_tables_dir.mkdir(parents=True, exist_ok=True)

    adata = sc.read_h5ad(args.h5ad)
    if args.cluster_key not in adata.obs:
        raise KeyError(f"Cluster key not found in adata.obs: {args.cluster_key}")
    cluster_key = "leiden_1_35_merged_renamed"
    add_merged_cluster_labels(adata, args.cluster_key, cluster_key)

    sc.tl.rank_genes_groups(
        adata,
        groupby=cluster_key,
        method=args.method,
        use_raw=adata.raw is not None,
        pts=True,
    )
    all_markers = sc.get.rank_genes_groups_df(adata, group=None)
    all_markers["group"] = all_markers["group"].astype(str)

    filtered = filter_markers(all_markers, args)
    top_markers = top_per_cluster(filtered, args.top_n)
    top1 = top_per_cluster(filtered, 1)
    specific_top_markers = top_specific_per_cluster(filtered, args.top_n)
    specific_top1 = top_specific_per_cluster(filtered, 1)
    pval_top_markers = top_pval_per_cluster(filtered, args.top_n)
    pval_top1 = top_pval_per_cluster(filtered, 1)
    filtered_genes = filtered["names"].astype(str).drop_duplicates().tolist()
    mean_df, pct_df = compute_cluster_expression_stats(adata, cluster_key, filtered_genes)
    max_other_markers = annotate_max_other_specificity(filtered, mean_df, pct_df)
    max_other_top_markers = top_max_other_per_cluster(max_other_markers, args.top_n)
    max_other_top1 = top_max_other_per_cluster(max_other_markers, 1)
    pairwise_tests, pairwise_top_markers = pairwise_competitor_tests(
        adata,
        max_other_top_markers,
        mean_df,
        pct_df,
        cluster_key,
        args.competitor_tolerance,
        args.pairwise_pval_adj,
    )
    pairwise_top1 = top_pairwise_validated_per_cluster(pairwise_top_markers)

    all_path = tables_dir / "rgc_merged_cluster_deg_markers_all.csv"
    filtered_path = tables_dir / "rgc_merged_cluster_deg_markers_filtered.csv"
    top_path = tables_dir / f"rgc_merged_cluster_deg_top{args.top_n}_markers.csv"
    top1_path = tables_dir / "rgc_merged_cluster_deg_top1_marker_per_cluster.csv"
    specific_top_path = tables_dir / f"rgc_merged_cluster_deg_specific_top{args.top_n}_markers.csv"
    specific_top1_path = tables_dir / "rgc_merged_cluster_deg_specific_top1_marker_per_cluster.csv"
    pval_top_path = tables_dir / f"rgc_merged_cluster_deg_pval_top{args.top_n}_markers.csv"
    pval_top1_path = tables_dir / "rgc_merged_cluster_deg_pval_top1_marker_per_cluster.csv"
    params_path = tables_dir / "rgc_merged_cluster_deg_parameters.csv"
    dotplot_path = figures_dir / "rgc_merged_cluster_deg_top1_marker_dotplot.png"
    specific_dotplot_path = figures_dir / "rgc_merged_cluster_deg_specific_top1_marker_dotplot.png"
    pval_dotplot_path = figures_dir / "rgc_merged_cluster_deg_pval_top1_marker_dotplot.png"
    max_other_markers_path = strict_tables_dir / "rgc_deg_max_other_annotated_filtered.csv"
    max_other_top_path = strict_tables_dir / f"rgc_deg_max_other_top{args.top_n}_markers.csv"
    max_other_top1_path = strict_tables_dir / "rgc_deg_max_other_top1_marker_per_cluster.csv"
    pairwise_tests_path = strict_tables_dir / f"rgc_deg_pairwise_competitor_tests_top{args.top_n}.csv"
    pairwise_top_path = strict_tables_dir / f"rgc_deg_pairwise_validated_top{args.top_n}_markers.csv"
    pairwise_top1_path = strict_tables_dir / "rgc_deg_pairwise_validated_top1_marker_per_cluster.csv"
    strict_params_path = strict_tables_dir / "rgc_deg_strict_marker_review_parameters.csv"
    max_other_dotplot_path = strict_figures_dir / "rgc_deg_max_other_top1_marker_dotplot.png"
    pairwise_dotplot_path = strict_figures_dir / "rgc_deg_pairwise_validated_top1_marker_dotplot.png"

    all_markers.to_csv(all_path, index=False)
    filtered.to_csv(filtered_path, index=False)
    top_markers.to_csv(top_path, index=False)
    top1.to_csv(top1_path, index=False)
    specific_top_markers.to_csv(specific_top_path, index=False)
    specific_top1.to_csv(specific_top1_path, index=False)
    pval_top_markers.to_csv(pval_top_path, index=False)
    pval_top1.to_csv(pval_top1_path, index=False)
    max_other_markers.to_csv(max_other_markers_path, index=False)
    max_other_top_markers.to_csv(max_other_top_path, index=False)
    max_other_top1.to_csv(max_other_top1_path, index=False)
    pairwise_tests.to_csv(pairwise_tests_path, index=False)
    pairwise_top_markers.to_csv(pairwise_top_path, index=False)
    pairwise_top1.to_csv(pairwise_top1_path, index=False)
    pd.DataFrame(
        [
            {
                "method": args.method,
                "use_raw": adata.raw is not None,
                "pval_adj": args.pval_adj,
                "min_logfc": args.min_logfc,
                "min_pct_group": args.min_pct_group,
                "top_n": args.top_n,
                "cluster_key": cluster_key,
            }
        ]
    ).to_csv(params_path, index=False)
    pd.DataFrame(
        [
            {
                "max_other_specificity_score": (
                    "logfoldchanges * max(pct_group_from_matrix - max_other_pct_nz, 0) "
                    "* max(mean_expr_group - max_other_mean_expr, 0)"
                ),
                "pairwise_test": (
                    "Mann-Whitney/Wilcoxon rank-sum, target cluster greater than each "
                    "competing cluster"
                ),
                "competing_cluster_rule": (
                    "other cluster mean_expr >= target mean_expr * competitor_tolerance "
                    "or other cluster pct_nz >= target pct_nz * competitor_tolerance"
                ),
                "competitor_tolerance": args.competitor_tolerance,
                "pairwise_pval_adj": args.pairwise_pval_adj,
                "pval_adj": args.pval_adj,
                "min_logfc": args.min_logfc,
                "min_pct_group": args.min_pct_group,
                "top_n": args.top_n,
                "cluster_key": cluster_key,
            }
        ]
    ).to_csv(strict_params_path, index=False)
    plot_top1_dotplot(adata, top1, cluster_key, dotplot_path, args.dpi)
    plot_top1_dotplot(adata, specific_top1, cluster_key, specific_dotplot_path, args.dpi)
    plot_top1_dotplot(adata, pval_top1, cluster_key, pval_dotplot_path, args.dpi)
    plot_top1_dotplot(adata, max_other_top1, cluster_key, max_other_dotplot_path, args.dpi)
    plot_top1_dotplot(adata, pairwise_top1, cluster_key, pairwise_dotplot_path, args.dpi)
    print(top_path)


if __name__ == "__main__":
    main()

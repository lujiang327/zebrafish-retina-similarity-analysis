#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse


H5AD = "results/h5ad/bc_9_sample_guca1b_gt2_mikiko_no_contam_26_28.h5ad"
OUTDIR = Path("results/tables/bc_cluster19_immaturity_review")
CLUSTER_KEY = "leiden"
TARGET_CLUSTER = "19"

MARKER_GENES = [
    "ascl1a",
    "atoh7",
    "ccnd1",
    "cdk1",
    "pcna",
    "mki67",
    "top2a",
    "hmgb2a",
    "hmgb2b",
    "stmn1a",
    "stmn1b",
    "marcksb",
    "marcksl1a",
    "marcksl1b",
    "id2a",
    "id3",
    "nfixa",
    "nfixb",
    "nfia",
    "sox2",
    "vsx2",
    "pax6a",
    "pax6b",
    "foxn4",
    "neurod1",
    "neurod4",
    "insm1a",
    "insm1b",
    "elavl3",
    "otx1",
    "otx2",
    "crx",
    "grm6a",
    "grm6b",
    "trpm1a",
    "trpm1b",
    "nyx",
    "prkcaa",
    "cabp5a",
    "cabp5b",
    "calb1",
    "calb2b",
    "grik1b",
    "grik4",
]


def expression_matrix_and_names(adata):
    if adata.raw is not None:
        return adata.raw.X, pd.Index(adata.raw.var_names), "raw"
    return adata.X, pd.Index(adata.var_names), "X"


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    adata = sc.read_h5ad(H5AD)
    if CLUSTER_KEY not in adata.obs:
        raise KeyError(f"Missing cluster key: {CLUSTER_KEY}")

    matrix, var_names, source = expression_matrix_and_names(adata)
    lookup = {str(gene).lower(): str(gene) for gene in var_names}
    matched_genes: list[str] = []
    records = []
    for requested in MARKER_GENES:
        matched = lookup.get(requested.lower())
        records.append(
            {
                "requested_gene": requested,
                "matched_gene": matched or "",
                "present": matched is not None,
                "expression_source": source,
            }
        )
        if matched is not None and matched not in matched_genes:
            matched_genes.append(matched)

    pd.DataFrame(records).to_csv(OUTDIR / "cluster19_marker_gene_match_report.csv", index=False)
    if not matched_genes:
        raise ValueError("No requested marker genes were found.")

    labels = adata.obs[CLUSTER_KEY].astype(str)
    clusters = sorted(labels.unique(), key=int)
    indices = var_names.get_indexer(matched_genes)
    subset = matrix[:, indices]

    rows = []
    for cluster in clusters:
        mask = (labels == cluster).to_numpy()
        cluster_matrix = subset[mask, :]
        if sparse.issparse(cluster_matrix):
            means = np.asarray(cluster_matrix.mean(axis=0)).ravel()
            pcts = cluster_matrix.getnnz(axis=0) / cluster_matrix.shape[0]
        else:
            array = np.asarray(cluster_matrix)
            means = array.mean(axis=0)
            pcts = (array > 0).mean(axis=0)
        for gene, mean, pct in zip(matched_genes, means, pcts):
            rows.append(
                {
                    "cluster": cluster,
                    "gene": gene,
                    "mean_expression": float(mean),
                    "fraction_expressing": float(pct),
                }
            )

    stats = pd.DataFrame(rows)
    stats["mean_rank_desc"] = (
        stats.groupby("gene")["mean_expression"].rank(method="min", ascending=False).astype(int)
    )
    stats["pct_rank_desc"] = (
        stats.groupby("gene")["fraction_expressing"].rank(method="min", ascending=False).astype(int)
    )
    stats.to_csv(OUTDIR / "cluster_marker_expression_all_clusters.csv", index=False)

    cluster19 = stats[stats["cluster"] == TARGET_CLUSTER].copy()
    cluster19 = cluster19.sort_values(
        ["mean_rank_desc", "mean_expression"],
        ascending=[True, False],
    )
    cluster19.to_csv(OUTDIR / "cluster19_marker_expression_summary.csv", index=False)

    source_dir = Path("results/tables/bc_guca1b_gt2_mikiko_no_contam_26_28_gene_umaps")
    for method in ["score", "specific", "max_other"]:
        csv_path = source_dir / f"bc_guca1b_gt2_deg_{method}_top100_markers.csv"
        if not csv_path.exists():
            continue
        markers = pd.read_csv(csv_path)
        markers[markers["group"].astype(str) == TARGET_CLUSTER].head(30).to_csv(
            OUTDIR / f"cluster19_{method}_top30_deg.csv",
            index=False,
        )

    n_cells = int((labels == TARGET_CLUSTER).sum())
    print(f"cluster {TARGET_CLUSTER} n_cells: {n_cells}")
    print(cluster19[["gene", "mean_expression", "fraction_expressing", "mean_rank_desc"]].head(30).to_string(index=False))
    print(OUTDIR)


if __name__ == "__main__":
    main()

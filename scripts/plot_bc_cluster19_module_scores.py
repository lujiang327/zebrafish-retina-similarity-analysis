#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc


H5AD = "results/h5ad/bc_9_sample_guca1b_gt2_mikiko_no_contam_26_28.h5ad"
OUTDIR = Path("results/figures/bc_cluster19_immaturity_review")
TABLE_DIR = Path("results/tables/bc_cluster19_immaturity_review")
CLUSTER_KEY = "leiden"
TARGET_CLUSTER = "19"

GENE_SETS = {
    "immature_transitional": [
        "stmn1b",
        "marcksl1b",
        "marcksl1a",
        "nfixa",
        "nfixb",
        "nfia",
        "id2a",
        "otx1",
        "hmgb2a",
        "hmgb2b",
        "neurod4",
    ],
    "immature_transitional_plus_rtn4a": [
        "stmn1b",
        "marcksl1b",
        "marcksl1a",
        "nfixa",
        "nfixb",
        "nfia",
        "id2a",
        "otx1",
        "hmgb2a",
        "hmgb2b",
        "neurod4",
        "rtn4a",
    ],
    "cycling_progenitor": ["mki67", "pcna", "top2a", "cdk1", "ccnd1", "ascl1a"],
    "mature_bc": ["cabp5a", "cabp5b", "grm6a", "grm6b", "trpm1a", "trpm1b", "nyx", "prkcaa"],
}


def resolve_genes(adata, genes: list[str]) -> list[str]:
    var_names = pd.Index(adata.raw.var_names if adata.raw is not None else adata.var_names)
    lookup = {str(gene).lower(): str(gene) for gene in var_names}
    return [lookup[gene.lower()] for gene in genes if gene.lower() in lookup]


def numeric_clusters(values) -> list[str]:
    return sorted([str(value) for value in values], key=lambda value: int(value))


def plot_cluster_score_distributions(adata, score_columns: list[str], output: Path) -> None:
    clusters = numeric_clusters(adata.obs[CLUSTER_KEY].astype(str).unique())
    fig, axes = plt.subplots(
        len(score_columns),
        1,
        figsize=(13, 3.25 * len(score_columns)),
        sharex=True,
    )
    axes = np.atleast_1d(axes)
    for ax, score_col in zip(axes, score_columns):
        data = [
            adata.obs.loc[adata.obs[CLUSTER_KEY].astype(str) == cluster, score_col].to_numpy()
            for cluster in clusters
        ]
        parts = ax.violinplot(
            data,
            positions=np.arange(len(clusters)),
            widths=0.8,
            showmeans=False,
            showmedians=True,
            showextrema=False,
        )
        for idx, body in enumerate(parts["bodies"]):
            cluster = clusters[idx]
            body.set_facecolor("#d97706" if cluster == TARGET_CLUSTER else "#9aa4b2")
            body.set_edgecolor("#334155")
            body.set_alpha(0.85 if cluster == TARGET_CLUSTER else 0.45)
        if "cmedians" in parts:
            parts["cmedians"].set_color("#111827")
            parts["cmedians"].set_linewidth(1.1)

        medians = [float(np.median(values)) for values in data]
        target_idx = clusters.index(TARGET_CLUSTER)
        ax.scatter(
            target_idx,
            medians[target_idx],
            s=70,
            color="#dc2626",
            edgecolor="white",
            zorder=4,
            label=f"cluster {TARGET_CLUSTER} median",
        )
        rank = int(pd.Series(medians, index=clusters).rank(ascending=False, method="min")[TARGET_CLUSTER])
        ax.set_title(f"{score_col.replace('_score', '').replace('_', ' ')} score; cluster 19 median rank {rank}")
        ax.set_ylabel("Module score")
        ax.axhline(0, color="#cbd5e1", linewidth=0.8)
        ax.grid(axis="y", color="#eef2f6", linewidth=0.6)
        ax.legend(frameon=False, loc="upper right")
    axes[-1].set_xticks(np.arange(len(clusters)))
    axes[-1].set_xticklabels(clusters, rotation=0, fontsize=8)
    axes[-1].set_xlabel("BC cluster")
    fig.suptitle("BC cluster 19 module-score review", fontsize=16, y=0.995)
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_umap_scores(adata, score_columns: list[str], output: Path) -> None:
    fig = sc.pl.umap(
        adata,
        color=score_columns,
        cmap="viridis",
        frameon=True,
        size=10,
        show=False,
        return_fig=True,
        ncols=3,
    )
    fig.set_size_inches(15, 4.8)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    adata = sc.read_h5ad(H5AD)
    if CLUSTER_KEY not in adata.obs:
        raise KeyError(f"Missing cluster key: {CLUSTER_KEY}")

    records = []
    score_columns = []
    for set_name, genes in GENE_SETS.items():
        matched = resolve_genes(adata, genes)
        records.append(
            {
                "module": set_name,
                "requested_genes": ";".join(genes),
                "matched_genes": ";".join(matched),
                "n_matched": len(matched),
            }
        )
        if not matched:
            continue
        score_col = f"{set_name}_score"
        sc.tl.score_genes(
            adata,
            gene_list=matched,
            score_name=score_col,
            use_raw=adata.raw is not None,
            random_state=0,
        )
        score_columns.append(score_col)

    pd.DataFrame(records).to_csv(TABLE_DIR / "cluster19_module_score_gene_sets.csv", index=False)

    clusters = numeric_clusters(adata.obs[CLUSTER_KEY].astype(str).unique())
    summary_rows = []
    for score_col in score_columns:
        medians = (
            adata.obs.groupby(adata.obs[CLUSTER_KEY].astype(str), observed=False)[score_col]
            .median()
            .reindex(clusters)
        )
        means = (
            adata.obs.groupby(adata.obs[CLUSTER_KEY].astype(str), observed=False)[score_col]
            .mean()
            .reindex(clusters)
        )
        ranks = medians.rank(ascending=False, method="min").astype(int)
        for cluster in clusters:
            summary_rows.append(
                {
                    "cluster": cluster,
                    "score": score_col,
                    "median": float(medians.loc[cluster]),
                    "mean": float(means.loc[cluster]),
                    "median_rank_desc": int(ranks.loc[cluster]),
                    "n_cells": int((adata.obs[CLUSTER_KEY].astype(str) == cluster).sum()),
                }
            )
    pd.DataFrame(summary_rows).to_csv(TABLE_DIR / "cluster19_module_score_by_cluster.csv", index=False)

    plot_cluster_score_distributions(
        adata,
        score_columns,
        OUTDIR / "bc_cluster19_module_score_violin.png",
    )
    plot_umap_scores(
        adata,
        score_columns,
        OUTDIR / "bc_cluster19_module_score_umaps.png",
    )
    adata.obs[[CLUSTER_KEY, *score_columns]].to_csv(
        TABLE_DIR / "cluster19_module_scores_per_cell.csv",
        index=True,
    )
    print(OUTDIR / "bc_cluster19_module_score_violin.png")
    print(OUTDIR / "bc_cluster19_module_score_umaps.png")


if __name__ == "__main__":
    main()

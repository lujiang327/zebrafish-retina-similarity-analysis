#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import scanpy as sc

from plot_bc_guca1b_gt2_deg_genes_ac_style import (
    expression_matrix_and_names,
    gene_lookup,
    plot_gene_all_conditions,
    plot_gene_by_condition,
)


DEFAULT_H5AD = "results/h5ad/bc_9_sample_guca1b_gt2_filtered.h5ad"
MANUAL_GENES = [
    "Aqp4",
    "Arr3a",
    "Ascl1a",
    "Atoh7",
    "Cabp5a",
    "Cabp5b",
    "Calb1",
    "Calb2a",
    "Calb2b",
    "Ccnd1",
    "Cdk1",
    "Crx",
    "Elavl3",
    "Gfap",
    "Gnat2",
    "Guca1b",
    "Isl2b",
    "Mbpa",
    "Mitfa",
    "Mpeg1.1",
    "Neurod1",
    "Nr2e3",
    "Nrl",
    "Ompa",
    "Pde6a",
    "Rbpms2b",
    "Tie1",
    "acta2",
    "Vsx1",
    "Vsx2",
    "vsx1",
    "vsx2",
    "gramd1b",
    "gram6a",
    "gram6b",
    "trpm1a",
    "trpm1b",
    "nyx",
    "rgs11",
    "prkca",
    "grik1",
    "gria1",
    "gria2",
    "cabp5",
    "calb1",
    "calb2",
    "cplx3a",
    "cplx3b",
    "sv2a",
    "sv2b",
    "syta1",
    "cadm1a",
    "cadm1b",
    "cadm2a",
    "cadm2b",
    "cadm3a",
    "cadm3b",
    "pcdh9",
    "pcdh17",
    "pcdh19",
    "nrxn1a",
    "nrcn1b",
    "nrxn2a",
    "nrxn3b",
    "nlgna",
    "nlgn1",
    "nlgn2a",
    "nlgn3b",
    "nlgn4a",
    "isl1",
    "foxn4",
    "pax6a",
    "pax6b",
]
ALIASES = {"gram6a": "grm6a", "gram6b": "grm6b", "nrcn1b": "nrxn1b"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot manual BC marker gene UMAPs for the guca1b > 2.0 filtered viewer."
    )
    parser.add_argument("--h5ad", default=DEFAULT_H5AD)
    parser.add_argument("--condition-key", default="renamed_samples")
    parser.add_argument("--sample-key", default="sample_split_zebra")
    parser.add_argument("--cluster-key", default="leiden")
    parser.add_argument("--figures-dir", default="results/figures/bc_guca1b_gt2_ac_style_gene_umaps")
    parser.add_argument("--tables-dir", default="results/tables/bc_guca1b_gt2_ac_style_gene_umaps")
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def ordered_counts(adata, key: str, preferred: list[str] | None = None) -> pd.DataFrame:
    values = adata.obs[key].astype(str)
    counts = values.value_counts()
    ordered = [name for name in (preferred or []) if name in counts.index]
    ordered.extend(sorted(name for name in counts.index if name not in ordered))
    return pd.DataFrame({"sample": ordered, "n_cells": [int(counts[name]) for name in ordered]})


def cluster_condition_counts(adata, cluster_key: str, condition_key: str) -> pd.DataFrame:
    table = pd.crosstab(
        adata.obs[cluster_key].astype(str),
        adata.obs[condition_key].astype(str),
    )
    for condition in ["Control", "LD", "NMDA"]:
        if condition not in table.columns:
            table[condition] = 0
    table = table[["Control", "LD", "NMDA"]]
    table["total"] = table.sum(axis=1)
    table = table.reset_index().rename(columns={cluster_key: "cluster"})
    table["cluster_sort"] = table["cluster"].astype(int)
    return table.sort_values("cluster_sort").drop(columns="cluster_sort")


def resolve_gene(lookup: dict[str, str], requested: str) -> tuple[str | None, str]:
    direct = lookup.get(requested.lower())
    if direct is not None:
        return direct, ""
    alias = ALIASES.get(requested.lower())
    if alias:
        return lookup.get(alias.lower()), alias
    return None, ""


def main() -> None:
    args = parse_args()
    figures_dir = Path(args.figures_dir)
    tables_dir = Path(args.tables_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    adata = sc.read_h5ad(args.h5ad)
    if "X_umap" not in adata.obsm:
        raise ValueError("Input h5ad does not contain adata.obsm['X_umap']")

    ordered_counts(
        adata,
        args.sample_key,
        ["TH115", "TH44", "TH54", "TH55", "TH56", "TH57", "TH71", "Zebra_LD", "Zebra_NMDA"],
    ).to_csv(tables_dir / "bc_guca1b_gt2_counts_by_9_sample.csv", index=False)
    ordered_counts(adata, args.condition_key, ["Control", "LD", "NMDA"]).to_csv(
        tables_dir / "bc_guca1b_gt2_counts_by_renamed_samples.csv",
        index=False,
    )
    cluster_condition_counts(adata, args.cluster_key, args.condition_key).to_csv(
        tables_dir / "bc_guca1b_gt2_cluster_counts_by_renamed_samples.csv",
        index=False,
    )

    lookup = gene_lookup(expression_matrix_and_names(adata)[1])
    records = []
    present_genes = []
    for requested in MANUAL_GENES:
        matched, alias = resolve_gene(lookup, requested)
        records.append(
            {
                "requested_gene": requested,
                "alias_used": alias,
                "matched_gene": matched or "",
                "present": matched is not None,
                "expression_source": "raw" if adata.raw is not None else "X",
            }
        )
        if matched is not None and matched not in present_genes:
            present_genes.append(matched)

    pd.DataFrame(records).to_csv(
        tables_dir / "bc_guca1b_gt2_manual_marker_gene_match_report.csv",
        index=False,
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

    print(tables_dir / "bc_guca1b_gt2_manual_marker_gene_match_report.csv")


if __name__ == "__main__":
    main()

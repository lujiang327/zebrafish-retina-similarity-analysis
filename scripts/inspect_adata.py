#!/usr/bin/env python
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import scanpy as sc

from fidelity_utils import (
    counts_table,
    ensure_output_dirs,
    load_config,
    resolve_metadata_columns,
    retinal_cell_type_mask,
    setup_logging,
    summarize_obs_columns,
)


LOGGER = logging.getLogger("retina_fidelity")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect a processed zebrafish retina AnnData object and save metadata summaries."
    )
    parser.add_argument("--config", default="config/config.yaml", help="Path to config YAML.")
    parser.add_argument("--h5ad", default=None, help="Override input .h5ad path from config.")
    parser.add_argument("--verbose", action="store_true", help="Print debug logging.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)
    config = load_config(args.config)
    tables_dir, _ = ensure_output_dirs(config)

    h5ad_path = Path(args.h5ad or config["data"]["h5ad_path"])
    LOGGER.info("Loading AnnData: %s", h5ad_path)
    adata = sc.read_h5ad(h5ad_path)

    LOGGER.info("AnnData shape: %s cells x %s genes", adata.n_obs, adata.n_vars)
    LOGGER.info("obs columns: %s", ", ".join(map(str, adata.obs.columns)))
    LOGGER.info("obsm keys: %s", ", ".join(map(str, adata.obsm.keys())))
    LOGGER.info("layers: %s", ", ".join(map(str, adata.layers.keys())) or "none")
    LOGGER.info("raw present: %s", adata.raw is not None)

    obs_summary = summarize_obs_columns(adata.obs)
    obs_summary_path = tables_dir / "obs_columns_summary.csv"
    obs_summary.to_csv(obs_summary_path, index=False)
    LOGGER.info("Saved obs column summary: %s", obs_summary_path)

    condition_column, cell_type_column = resolve_metadata_columns(adata, config)
    LOGGER.info("Using condition column: %s", condition_column)
    LOGGER.info("Using cell-type column: %s", cell_type_column)

    condition_order = config.get("metadata", {}).get("condition_order")
    all_counts = counts_table(
        adata.obs,
        condition_column=condition_column,
        cell_type_column=cell_type_column,
        condition_order=condition_order,
    )
    all_counts_path = tables_dir / "cell_counts_by_condition_cell_type.csv"
    all_counts.to_csv(all_counts_path, index=False)
    LOGGER.info("Saved all cell counts: %s", all_counts_path)

    retinal_mask = retinal_cell_type_mask(adata.obs, cell_type_column, config)
    retinal_counts = counts_table(
        adata.obs.loc[retinal_mask],
        condition_column=condition_column,
        cell_type_column=cell_type_column,
        condition_order=condition_order,
    )
    retinal_counts_path = tables_dir / "retinal_cell_counts_by_condition_cell_type.csv"
    retinal_counts.to_csv(retinal_counts_path, index=False)
    LOGGER.info("Saved retinal-focused cell counts: %s", retinal_counts_path)

    config_columns_path = tables_dir / "detected_metadata_columns.csv"
    with config_columns_path.open("w", encoding="utf-8") as handle:
        handle.write("role,column\n")
        handle.write(f"condition,{condition_column}\n")
        handle.write(f"cell_type,{cell_type_column}\n")
    LOGGER.info("Saved detected metadata columns: %s", config_columns_path)


if __name__ == "__main__":
    main()

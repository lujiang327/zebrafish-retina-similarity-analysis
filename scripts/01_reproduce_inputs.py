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
        description=(
            "First reproducibility check: load .h5ad, inspect obs columns, infer metadata, "
            "save cell-count summaries, and reproduce UMAPs."
        )
    )
    parser.add_argument("--config", default="config/config.yaml", help="Path to config YAML.")
    parser.add_argument("--h5ad", default=None, help="Override input .h5ad path from config.")
    parser.add_argument("--verbose", action="store_true", help="Print debug logging.")
    return parser.parse_args()


def save_umap(adata, color: str, output_path: Path, dpi: int) -> None:
    import matplotlib.pyplot as plt

    sc.pl.umap(adata, color=color, frameon=False, show=False)
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    LOGGER.info("Saved figure: %s", output_path)


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)
    config = load_config(args.config)
    tables_dir, figures_dir = ensure_output_dirs(config)
    output_cfg = config.get("outputs", {})
    figure_format = output_cfg.get("figure_format", "png")
    dpi = int(output_cfg.get("figure_dpi", 300))

    h5ad_path = Path(args.h5ad or config["data"]["h5ad_path"])
    LOGGER.info("Loading AnnData: %s", h5ad_path)
    adata = sc.read_h5ad(h5ad_path)
    LOGGER.info("Loaded shape: %s cells x %s genes", adata.n_obs, adata.n_vars)

    obs_summary_path = tables_dir / "obs_columns_summary.csv"
    summarize_obs_columns(adata.obs).to_csv(obs_summary_path, index=False)
    LOGGER.info("Saved obs summary: %s", obs_summary_path)

    condition_column, cell_type_column = resolve_metadata_columns(adata, config)
    LOGGER.info("Detected condition column: %s", condition_column)
    LOGGER.info("Detected cell-type column: %s", cell_type_column)

    condition_order = config.get("metadata", {}).get("condition_order")
    counts_path = tables_dir / "cell_counts_by_condition_cell_type.csv"
    counts_table(
        adata.obs,
        condition_column=condition_column,
        cell_type_column=cell_type_column,
        condition_order=condition_order,
    ).to_csv(counts_path, index=False)
    LOGGER.info("Saved counts: %s", counts_path)

    retinal_mask = retinal_cell_type_mask(adata.obs, cell_type_column, config)
    retinal_counts_path = tables_dir / "retinal_cell_counts_by_condition_cell_type.csv"
    counts_table(
        adata.obs.loc[retinal_mask],
        condition_column=condition_column,
        cell_type_column=cell_type_column,
        condition_order=condition_order,
    ).to_csv(retinal_counts_path, index=False)
    LOGGER.info("Saved retinal counts: %s", retinal_counts_path)

    detected_path = tables_dir / "detected_metadata_columns.csv"
    detected_path.write_text(
        f"role,column\ncondition,{condition_column}\ncell_type,{cell_type_column}\n",
        encoding="utf-8",
    )
    LOGGER.info("Saved detected columns: %s", detected_path)

    if "X_umap" not in adata.obsm:
        LOGGER.warning('No adata.obsm["X_umap"] found; skipping UMAP reproduction.')
        return

    save_umap(
        adata,
        condition_column,
        figures_dir / f"umap_condition.{figure_format}",
        dpi,
    )
    save_umap(
        adata,
        cell_type_column,
        figures_dir / f"umap_cell_type.{figure_format}",
        dpi,
    )

    if retinal_mask.sum() < adata.n_obs:
        adata_retinal = adata[retinal_mask].copy()
        save_umap(
            adata_retinal,
            condition_column,
            figures_dir / f"umap_condition_retinal.{figure_format}",
            dpi,
        )
        save_umap(
            adata_retinal,
            cell_type_column,
            figures_dir / f"umap_cell_type_retinal.{figure_format}",
            dpi,
        )


if __name__ == "__main__":
    main()

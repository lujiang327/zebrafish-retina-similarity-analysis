#!/usr/bin/env python
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import scanpy as sc

from fidelity_utils import (
    ensure_output_dirs,
    load_config,
    resolve_metadata_columns,
    retinal_cell_type_mask,
    setup_logging,
)


LOGGER = logging.getLogger("retina_fidelity")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce basic UMAP plots colored by condition and cell type."
    )
    parser.add_argument("--config", default="config/config.yaml", help="Path to config YAML.")
    parser.add_argument("--h5ad", default=None, help="Override input .h5ad path from config.")
    parser.add_argument(
        "--retinal-only",
        action="store_true",
        help="Plot only configured retinal cell types and exclude contaminant populations.",
    )
    parser.add_argument("--verbose", action="store_true", help="Print debug logging.")
    return parser.parse_args()


def save_umap(adata, color: str, output_prefix: Path, dpi: int, fmt: str) -> None:
    sc.pl.umap(
        adata,
        color=color,
        frameon=False,
        show=False,
        save=None,
    )
    import matplotlib.pyplot as plt

    output_path = output_prefix.with_suffix(f".{fmt}")
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    LOGGER.info("Saved UMAP: %s", output_path)


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)
    config = load_config(args.config)
    _, figures_dir = ensure_output_dirs(config)

    h5ad_path = Path(args.h5ad or config["data"]["h5ad_path"])
    LOGGER.info("Loading AnnData: %s", h5ad_path)
    adata = sc.read_h5ad(h5ad_path)

    if "X_umap" not in adata.obsm:
        raise KeyError(
            'No UMAP found in adata.obsm["X_umap"]. Run preprocessing or provide an object with UMAP coordinates.'
        )

    condition_column, cell_type_column = resolve_metadata_columns(adata, config)
    if args.retinal_only:
        mask = retinal_cell_type_mask(adata.obs, cell_type_column, config)
        LOGGER.info("Retinal-only subset: %s of %s cells", int(mask.sum()), adata.n_obs)
        adata = adata[mask].copy()

    output_cfg = config.get("outputs", {})
    dpi = int(output_cfg.get("figure_dpi", 300))
    fmt = output_cfg.get("figure_format", "png")
    suffix = "_retinal" if args.retinal_only else ""

    save_umap(adata, condition_column, figures_dir / f"umap_condition{suffix}", dpi, fmt)
    save_umap(adata, cell_type_column, figures_dir / f"umap_cell_type{suffix}", dpi, fmt)


if __name__ == "__main__":
    main()

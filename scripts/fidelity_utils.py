from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial.distance import cosine as cosine_distance
from scipy.stats import pearsonr, spearmanr
import yaml


LOGGER = logging.getLogger("retina_fidelity")


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Config file is empty or invalid: {path}")
    return config


def ensure_output_dirs(config: dict[str, Any]) -> tuple[Path, Path]:
    outputs = config.get("outputs", {})
    tables_dir = Path(outputs.get("tables_dir", "results/tables"))
    figures_dir = Path(outputs.get("figures_dir", "results/figures"))
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    return tables_dir, figures_dir


def choose_existing_column(
    obs: pd.DataFrame,
    configured: str | None,
    candidates: list[str],
    required_values: list[str] | None = None,
) -> str:
    if configured:
        if configured not in obs.columns:
            raise KeyError(f"Configured obs column not found: {configured}")
        return configured

    obs_columns = list(obs.columns)
    for column in candidates:
        if column in obs_columns:
            if required_values is None:
                return column
            values = set(obs[column].dropna().astype(str).unique())
            if set(required_values).intersection(values):
                return column

    if required_values is not None:
        required = set(required_values)
        for column in obs_columns:
            values = set(obs[column].dropna().astype(str).unique())
            if len(required.intersection(values)) >= 2:
                return column

    raise KeyError(
        "Could not infer required obs column. Set it explicitly in config/config.yaml."
    )


def resolve_metadata_columns(adata: Any, config: dict[str, Any]) -> tuple[str, str]:
    metadata = config.get("metadata", {})
    condition_order = metadata.get("condition_order", ["Control", "LD", "NMDA"])

    condition_column = choose_existing_column(
        adata.obs,
        metadata.get("condition_column"),
        metadata.get("condition_column_candidates", []),
        required_values=condition_order,
    )
    cell_type_column = choose_existing_column(
        adata.obs,
        metadata.get("cell_type_column"),
        metadata.get("cell_type_column_candidates", []),
    )
    return condition_column, cell_type_column


def summarize_obs_columns(obs: pd.DataFrame, max_values: int = 12) -> pd.DataFrame:
    rows = []
    for column in obs.columns:
        series = obs[column]
        values = series.dropna().astype(str)
        top_values = values.value_counts().head(max_values)
        rows.append(
            {
                "column": column,
                "dtype": str(series.dtype),
                "n_unique": int(series.nunique(dropna=True)),
                "n_missing": int(series.isna().sum()),
                "top_values": "; ".join(
                    f"{idx}: {count}" for idx, count in top_values.items()
                ),
            }
        )
    return pd.DataFrame(rows)


def normalize_labels(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip()


def retinal_cell_type_mask(
    obs: pd.DataFrame, cell_type_column: str, config: dict[str, Any]
) -> pd.Series:
    labels = normalize_labels(obs[cell_type_column])
    cell_cfg = config.get("cell_types", {})
    include = cell_cfg.get("include") or []
    exclude = cell_cfg.get("exclude") or []

    if include:
        include_lc = {x.lower() for x in include}
        mask = labels.str.lower().isin(include_lc)
    else:
        mask = pd.Series(True, index=obs.index)

    if exclude:
        exclude_terms = [x.lower() for x in exclude]
        excluded = labels.str.lower().apply(
            lambda value: any(term in value for term in exclude_terms)
        )
        mask = mask & ~excluded

    return mask


def counts_table(
    obs: pd.DataFrame,
    condition_column: str,
    cell_type_column: str,
    condition_order: list[str] | None = None,
) -> pd.DataFrame:
    table = (
        obs.assign(
            condition=normalize_labels(obs[condition_column]),
            cell_type=normalize_labels(obs[cell_type_column]),
        )
        .groupby(["cell_type", "condition"], observed=False)
        .size()
        .rename("n_cells")
        .reset_index()
    )
    if condition_order:
        table["condition"] = pd.Categorical(
            table["condition"], categories=condition_order, ordered=True
        )
        table = table.sort_values(["cell_type", "condition"])
    return table


def as_dense_vector(matrix: Any) -> np.ndarray:
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    return np.asarray(matrix).ravel()


def get_expression_source(adata: Any, config: dict[str, Any]) -> tuple[Any, np.ndarray, str]:
    analysis = config.get("analysis", {})
    layer = analysis.get("layer")
    use_raw = bool(analysis.get("use_raw_if_available", True))

    if layer:
        if layer not in adata.layers:
            raise KeyError(f"Configured layer not found in adata.layers: {layer}")
        return adata.layers[layer], adata.var_names.to_numpy(), f"layer:{layer}"

    if use_raw and adata.raw is not None:
        return adata.raw.X, adata.raw.var_names.to_numpy(), "raw.X"

    return adata.X, adata.var_names.to_numpy(), "X"


def mean_expression_by_mask(matrix: Any, mask: np.ndarray) -> np.ndarray:
    subset = matrix[mask]
    if subset.shape[0] == 0:
        raise ValueError("Cannot compute mean expression for an empty cell group.")
    if sparse.issparse(subset):
        return np.asarray(subset.mean(axis=0)).ravel()
    return np.asarray(subset).mean(axis=0).ravel()


def finite_similarity(x: np.ndarray, y: np.ndarray, metric: str) -> float:
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if x.size < 2:
        return np.nan

    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return np.nan

    if metric == "pearson":
        return float(pearsonr(x, y).statistic)
    if metric == "spearman":
        return float(spearmanr(x, y).statistic)
    if metric == "cosine":
        if np.linalg.norm(x) == 0 or np.linalg.norm(y) == 0:
            return np.nan
        return float(1 - cosine_distance(x, y))

    raise ValueError(f"Unsupported similarity metric: {metric}")

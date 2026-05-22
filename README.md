# Zebrafish Retina Cell Fidelity

Reproducible Scanpy pipeline for comparing native/control zebrafish retinal cell
types with regenerated/injury-condition retinal cells from LD and NMDA samples.

The key design is cell-type-specific fidelity. LD and NMDA are kept separate by
default, and comparisons are made within each annotated retinal class rather
than across whole samples globally.

## Reference Environment

The reference repository README reports `scanpy 1.9.3` in its debugging output.
This repo prioritizes matching that Scanpy workflow and pins a compatible
AnnData release:

- Python 3.11 recommended
- `scanpy==1.9.3`
- `anndata==0.8.0`

Note: the paper reports `AnnData 0.12.8`, but that version does not import with
`scanpy==1.9.3` because AnnData 0.12 removed APIs used by old Scanpy. If exact
paper-environment matching becomes necessary, use a modern Scanpy release
instead of 1.9.3.

Install:

```bash
conda env create -f environment.yml
conda activate retina-fidelity
```

Conda is recommended on macOS because `python-igraph` and `leidenalg` include
compiled components. If you use `pip`, make sure the active interpreter is
Python 3.11, not the system/default Python.

```bash
python --version
pip install -r requirements.txt
```

### macOS Install Troubleshooting

If the install log contains `cpython-313`, you are still using Python 3.13.
Do not continue with that environment.

Check that `python` and `pip` point to the same conda environment:

```bash
conda activate retina-fidelity
which python
python --version
python -m pip --version
```

Use `python -m pip`, not bare `pip`, when installing into a specific
environment. After `conda env create -f environment.yml`, you should not need
to run `pip install -r requirements.txt`; conda has already installed the
compiled packages such as `leidenalg` and `python-igraph`.

## Input Data

Place the processed `.h5ad` in the repo root or `data/`, then update
`config/config.yaml`.

Default expected file:

```text
annotated_clustered_corrected_doubletRemoved_Zebrafishes.h5ad
```

Large data files are ignored by git.

## First Reproducibility Check

Run:

```bash
python scripts/01_reproduce_inputs.py --config config/config.yaml
```

This script:

- loads the AnnData object
- inspects `adata.obs` columns
- infers or uses configured condition and cell-type columns
- saves cell counts by condition and cell type
- saves retinal-focused cell counts excluding likely contaminants
- reproduces UMAPs colored by condition and cell type if `X_umap` exists

Outputs:

```text
results/tables/obs_columns_summary.csv
results/tables/detected_metadata_columns.csv
results/tables/cell_counts_by_condition_cell_type.csv
results/tables/retinal_cell_counts_by_condition_cell_type.csv
results/figures/umap_condition.png
results/figures/umap_cell_type.png
results/figures/umap_condition_retinal.png
results/figures/umap_cell_type_retinal.png
```

You can also run the pieces separately:

```bash
python scripts/inspect_adata.py --config config/config.yaml
python scripts/plot_umap.py --config config/config.yaml
python scripts/plot_umap.py --config config/config.yaml --retinal-only
```

## Metadata Configuration

Do not hard-code metadata columns in scripts. Set these after inspection:

```yaml
metadata:
  condition_column: "renamed_samples"
  cell_type_column: "celltype"
```

If left as `null`, scripts try to infer the columns from configured candidates
and the expected condition labels `Control`, `LD`, and `NMDA`.

## Biological Scope

Primary retinal populations:

- Rod
- Cone / Cones
- Bipolar cell / BC
- Amacrine cell / AC
- Horizontal cell / HC
- Retinal ganglion cell / RGC
- Müller glia / MG
- Müller glia progenitor cell / MGPC
- PR precursor, if present

Likely contaminants are excluded from retinal-focused summaries:

- immune / microglia
- endothelial
- RPE
- pericyte
- melanocyte
- oligodendrocyte

## Planned Analysis Modules

Analysis 1: pseudobulk cell-type similarity

- For each retinal cell type, average expression within Control, LD, and NMDA.
- Compare Control-LD, Control-NMDA, and LD-NMDA.
- Test Pearson, Spearman, and cosine similarity.
- Repeat over HVGs, marker genes, and shared expressed genes.

Analysis 2: per-cell fidelity score

- Build a same-cell-type Control reference centroid.
- Score each Control, LD, and NMDA cell against that centroid.
- Include held-out Control-to-Control scores.
- Save per-cell CSVs and violin/box plots.

Analysis 3: model-based fidelity

- Use One-Class SVM, logistic regression or random forest when appropriate,
  and kNN reference mapping as secondary checks.
- Keep correlation/centroid scores as the main interpretable result.

Analysis 4: robustness

- Sweep HVG counts, gene sets, similarity metrics, minimum cell thresholds,
  and optional condition-balanced downsampling.
- Flag low-cell-count cell types and unexpectedly low Control-to-Control scores.

## Notes

The default config keeps `Control`, `LD`, and `NMDA` separate. Combining LD and
NMDA as a broader injury/regenerated group should be a later sensitivity
analysis, not the initial analysis.

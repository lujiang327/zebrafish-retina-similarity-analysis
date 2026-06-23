# Zebrafish Retina Cell Fidelity

Reproducible Scanpy pipeline for comparing native/control zebrafish retinal cell
types with regenerated/injury-condition retinal cells from LD and NMDA samples.

The key design is cell-type-specific fidelity. LD and NMDA are kept separate by
default, and comparisons are made within each annotated retinal class rather
than across whole samples globally.

## Reference Environment

Most recent analyses in this workspace were run in the conda environment
`zebrafish-singlecell-portal`. Key package versions:

```text
python        3.11.15
scanpy        1.10.4
anndata       0.11.3
numpy         2.1.3
pandas        2.2.3
scipy         1.14.1
matplotlib    3.10.9
seaborn       0.13.2
scikit-learn  1.9.0
numba         0.65.1
h5py          3.16.0
igraph        1.0.0
leidenalg     0.12.0
harmonypy     2.0.0
Pillow        12.2.0
```

When running Scanpy scripts on macOS in this environment, use writable cache
locations to avoid Matplotlib/Numba cache errors:

```bash
export MPLCONFIGDIR=/private/tmp/retina_mpl
export NUMBA_CACHE_DIR=/private/tmp/retina_numba_cache
export PYTHONPYCACHEPREFIX=/private/tmp/retina_pycache
```

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

## Analysis 1: Pseudobulk Cell-Type Similarity

Interactive notebook:

```text
notebooks/02_pseudobulk_similarity.ipynb
```

Scripted run:

```bash
python scripts/pseudobulk_similarity.py --config config/config.yaml
```

This analysis:

- subsets to retinal-focused cell types from `config/config.yaml`
- computes a pseudobulk mean expression profile for each cell type and condition
- keeps `Control`, `LD`, and `NMDA` separate
- compares `Control_vs_LD`, `Control_vs_NMDA`, and `LD_vs_NMDA`
- tests Pearson, Spearman, and cosine similarity
- repeats over shared expressed genes and configured HVG counts
- flags comparisons below the minimum cell-count threshold

Main outputs:

```text
results/tables/pseudobulk_similarity_scores.csv
results/tables/pseudobulk_group_metadata.csv
results/tables/pseudobulk_skipped_cell_types.csv
results/figures/pseudobulk_similarity_*.png
```

Panel-style cross-cell-type heatmaps, matching the layout of manuscript panels
A/B/C, can be generated with:

```bash
python scripts/celltype_similarity_heatmaps.py --config config/config.yaml
```

This produces one 3-panel figure for each metric and gene set:

```text
results/figures/celltype_similarity_panels_cosine_shared_expressed.png
results/figures/celltype_similarity_panels_cosine_hvg_1000.png
results/figures/celltype_similarity_panels_cosine_hvg_2000.png
results/figures/celltype_similarity_panels_pearson_shared_expressed.png
results/figures/celltype_similarity_panels_pearson_hvg_1000.png
results/figures/celltype_similarity_panels_pearson_hvg_2000.png
results/figures/celltype_similarity_panels_spearman_shared_expressed.png
results/figures/celltype_similarity_panels_spearman_hvg_1000.png
results/figures/celltype_similarity_panels_spearman_hvg_2000.png
```

## Analysis 2: Per-Cell Fidelity

Interactive notebook:

```text
notebooks/03_per_cell_fidelity.ipynb
```

Scripted run:

```bash
python scripts/per_cell_fidelity.py --config config/config.yaml
```

This analysis:

- works within each retinal cell type
- splits Control cells into a reference set and held-out Control cells
- builds a Control reference centroid from the reference split
- scores held-out Control, LD, and NMDA cells against that same-cell-type centroid
- outputs per-cell fidelity scores and condition summaries
- plots combined violin distributions, per-cell-type box plots, and a UMAP score overlay

Main outputs:

```text
results/tables/per_cell_fidelity_scores.csv
results/tables/per_cell_fidelity_summary.csv
results/tables/per_cell_fidelity_skipped_cell_types.csv
results/tables/per_cell_fidelity_gene_sets.csv
results/figures/per_cell_fidelity_*.png
results/figures/umap_per_cell_fidelity_*.png
```

### Gene-Set Experiment

The notebook [03_per_cell_fidelity.ipynb](notebooks/03_per_cell_fidelity.ipynb)
also contains an expressed-gene-set experiment. In addition to `hvg_1000` and
`hvg_2000`, it tests:

- `mean_gt_1p5_1k_2k`
- `mean_gt_1p5_frac_gt_10pct_1k_2k`

These sets are selected separately for each cell type from its Control reference
cells. Genes with mean expression greater than 1.5 are preferred, optionally
requiring expression in at least 10% of reference cells. The set is capped at
2,000 genes and filled toward 1,000 genes with the next-highest expressed genes
when the strict threshold returns fewer than 1,000.

The comparison run is saved under:

```text
results/tables/per_cell_fidelity_gene_set_experiment/
results/figures/per_cell_fidelity_gene_set_experiment/
```

The selected genes for each cell type and gene set are listed in:

```text
results/tables/per_cell_fidelity_gene_set_experiment/per_cell_fidelity_gene_sets.csv
```

## Analysis 3: Three-Group Similarity Views

Interactive notebook:

```text
notebooks/04_three_group_similarity.ipynb
```

Scripted run:

```bash
python scripts/three_group_similarity.py --config config/config.yaml
```

This analysis uses the same cell-type/condition pseudobulk centroids, but
visualizes Control, LD, and NMDA together:

- Control-referenced heatmaps with LD and NMDA shown side by side
- Delta heatmaps showing `Control_vs_NMDA - Control_vs_LD`
- Three-condition centroid PCA plots

Main outputs:

```text
results/tables/three_group_control_referenced_similarity.csv
results/tables/three_group_nmda_minus_ld_delta.csv
results/tables/three_group_centroids.csv
results/figures/three_group_control_referenced_*.png
results/figures/three_group_delta_nmda_minus_ld_*.png
results/figures/three_group_centroid_pca_*.png
```

## Subtype HTML Reviewers

Several subtype-focused HTML viewers were generated to review UMAPs, marker
genes, DEG-selected markers, dotplots, and sample/cluster counts.

### RGC

Input h5ad:

```text
/Users/hoanglab/Desktop/vscode_projects/zebrafish-singlecell-portal/corrected_RGC_annotated_clustered_corrected_doubletRemoved_Zebrafishes.h5ad
```

Main viewer:

```text
results/figures/rgc_subtype_gene_umaps/rgc_gene_umap_review.html
```

### AC

Input h5ad:

```text
/Users/hoanglab/Desktop/vscode_projects/zebrafish-singlecell-portal/AC_subtypes_reproduced.h5ad
```

The AC file matches the paper-level final counts:

```text
Control  10,035
LD          236
NMDA        867
```

Main viewer:

```text
results/figures/ac_subtype_gene_umaps/ac_gene_umap_review.html
```

### BC

The BC workflow starts from:

```text
/Users/hoanglab/Desktop/vscode_projects/zebrafish-singlecell-portal/BC_annotated_clustered_corrected_doubletRemoved_Zebrafishes.h5ad
```

The preferred BC preprocessing uses Harmony on a 9-sample label, splitting the
original `Zebra` sample into `Zebra_LD` and `Zebra_NMDA` before Harmony. The
reprocessed h5ad is:

```text
results/h5ad/bc_strict_harmony_by_9_samples_reprocessed.h5ad
```

After reviewing photoreceptor-like contamination markers, the current BC viewer
uses the object filtered by:

```text
guca1b > 2.0
```

Filtered h5ad:

```text
results/h5ad/bc_9_sample_guca1b_gt2_filtered.h5ad
```

Main AC-style BC viewer:

```text
results/figures/bc_guca1b_gt2_ac_style_gene_umaps/bc_guca1b_gt2_ac_style_gene_umap_review.html
```

This viewer includes:

- all-condition and split-condition gene UMAPs
- manually requested marker-gene UMAPs
- top1 DEG gene UMAPs
- Wilcoxon score, specificity, and max-other DEG dotplots
- cell counts by cluster
- cell counts by 9-sample label
- cell counts by `Control`, `LD`, and `NMDA`
- cell counts per cluster split by `Control`, `LD`, and `NMDA`

Useful tables:

```text
results/tables/bc_guca1b_gt2_ac_style_gene_umaps/bc_guca1b_gt2_counts_by_9_sample.csv
results/tables/bc_guca1b_gt2_ac_style_gene_umaps/bc_guca1b_gt2_counts_by_renamed_samples.csv
results/tables/bc_guca1b_gt2_ac_style_gene_umaps/bc_guca1b_gt2_cluster_counts_by_renamed_samples.csv
results/tables/bc_guca1b_gt2_ac_style_gene_umaps/bc_guca1b_gt2_manual_marker_gene_match_report.csv
```

The custom top1 marker dotplot renderer keeps one column per cluster-marker
choice. This avoids the default `scanpy.pl.dotplot` behavior that collapses
duplicate marker genes into one column.

Key scripts:

```text
scripts/reprocess_bc_strict_harmony.py
scripts/filter_bc_photoreceptor_like_cells.py
scripts/find_bc_guca1b_gt2_cluster_markers_ac_style.py
scripts/plot_bc_guca1b_gt2_deg_genes_ac_style.py
scripts/plot_bc_guca1b_gt2_manual_marker_genes.py
scripts/make_bc_guca1b_gt2_gene_umap_review_html_ac_style.py
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

## Notes

The default config keeps `Control`, `LD`, and `NMDA` separate. Combining LD and
NMDA as a broader injury/regenerated group should be a later sensitivity
analysis, not the initial analysis.

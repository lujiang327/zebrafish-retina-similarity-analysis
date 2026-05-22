# Data Directory

Place local input files here, or point `config/config.yaml` to another path.

Do not commit large single-cell data files. The repository `.gitignore` excludes
`.h5ad`, `.h5`, `.rds`, `.loom`, Matrix Market files, and compressed tables.

Expected primary input:

- A processed Scanpy/AnnData `.h5ad` with expression data.
- `adata.obs` columns describing condition/sample labels such as `Control`,
  `LD`, and `NMDA`.
- `adata.obs` cell-type annotations for retinal classes.
- UMAP coordinates in `adata.obsm["X_umap"]`, if reproducing UMAPs.

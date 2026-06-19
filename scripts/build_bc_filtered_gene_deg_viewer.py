#!/usr/bin/env python
from __future__ import annotations

import argparse
import base64
from datetime import datetime
import html
import mimetypes
import os
from pathlib import Path
import struct

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse


DEFAULT_H5AD = "results/h5ad/bc_9_sample_guca1b_gt2_filtered.h5ad"
MANUAL_GENES = [
    "vsx1", "vsx2", "gramd1b", "gram6a", "gram6b", "trpm1a", "trpm1b", "nyx",
    "rgs11", "prkca", "grik1", "gria1", "gria2", "cabp5", "calb1", "calb2",
    "cplx3a", "cplx3b", "sv2a", "sv2b", "syta1", "cadm1a", "cadm1b",
    "cadm2a", "cadm2b", "cadm3a", "cadm3b", "pcdh9", "pcdh17", "pcdh19",
    "nrxn1a", "nrcn1b", "nrxn2a", "nrxn3b", "nlgna", "nlgn1", "nlgn2a",
    "nlgn3b", "nlgn4a", "isl1", "foxn4", "pax6a", "pax6b",
]
ALIASES = {"gram6a": "grm6a", "gram6b": "grm6b", "nrcn1b": "nrxn1b"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build BC filtered gene UMAP and DEG dotplot viewer.")
    parser.add_argument("--h5ad", default=DEFAULT_H5AD)
    parser.add_argument("--cluster-key", default="leiden")
    parser.add_argument("--condition-key", default="renamed_samples")
    parser.add_argument("--sample-key", default="sample_split_zebra")
    parser.add_argument("--figures-dir", default="results/figures/bc_guca1b_gt2_gene_deg_viewer")
    parser.add_argument("--tables-dir", default="results/tables/bc_guca1b_gt2_gene_deg_viewer")
    parser.add_argument("--output", default=None)
    parser.add_argument("--method", default="wilcoxon")
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--pval-adj", type=float, default=0.05)
    parser.add_argument("--min-logfc", type=float, default=0.25)
    parser.add_argument("--min-pct-group", type=float, default=0.10)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def sort_key(value: str):
    return int(value) if str(value).isdigit() else str(value)


def expression_source(adata):
    if adata.raw is not None:
        return adata.raw.X, pd.Index(adata.raw.var_names)
    return adata.X, pd.Index(adata.var_names)


def dense_vector(matrix, column_index: int) -> np.ndarray:
    column = matrix[:, column_index]
    if sparse.issparse(column):
        return column.toarray().ravel()
    return np.asarray(column).ravel()


def expression_vector(adata, gene: str) -> np.ndarray:
    matrix, var_names = expression_source(adata)
    return dense_vector(matrix, int(var_names.get_loc(gene)))


def lookup_gene(lookup: dict[str, str], requested: str) -> tuple[str | None, str]:
    direct = lookup.get(requested.lower())
    if direct is not None:
        return direct, ""
    alias = ALIASES.get(requested.lower())
    if alias:
        match = lookup.get(alias.lower())
        if match is not None:
            return match, alias
    return None, alias or ""


def gene_match_report(adata, tables_dir: Path) -> list[str]:
    _, var_names = expression_source(adata)
    lookup = {str(gene).lower(): str(gene) for gene in var_names}
    records = []
    present = []
    seen = set()
    for requested in MANUAL_GENES:
        matched, alias = lookup_gene(lookup, requested)
        records.append(
            {
                "requested_gene": requested,
                "alias_used": alias,
                "matched_gene": matched or "",
                "present": matched is not None,
                "expression_source": "raw" if adata.raw is not None else "X",
            }
        )
        if matched and matched.lower() not in seen:
            present.append(matched)
            seen.add(matched.lower())
    pd.DataFrame(records).to_csv(tables_dir / "bc_manual_gene_match_report.csv", index=False)
    return present


def condition_panels(adata, condition_key: str) -> list[tuple[str, tuple[str, ...]]]:
    values = set(adata.obs[condition_key].astype(str))
    panels = [(v, (v,)) for v in ["Control", "LD", "NMDA"] if v in values]
    if {"LD", "NMDA"}.issubset(values):
        panels.append(("LD + NMDA", ("LD", "NMDA")))
    return panels


def save_overview_umap(adata, color: str, out: Path, dpi: int, title: str | None = None) -> None:
    fig = sc.pl.umap(
        adata,
        color=color,
        legend_loc="on data" if color == "leiden" or color.endswith("cluster") else None,
        title=title or color,
        show=False,
        return_fig=True,
        size=3,
        frameon=True,
    )
    fig.set_size_inches(9.6, 6.4)
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_sample_highlights(adata, sample_key: str, figures_dir: Path, dpi: int) -> None:
    coords = adata.obsm["X_umap"]
    samples = adata.obs[sample_key].astype(str).to_numpy()
    ordered = [s for s in ["TH115", "TH44", "TH54", "TH55", "TH56", "TH57", "TH71", "Zebra_LD", "Zebra_NMDA"] if s in set(samples)]
    ordered.extend(sorted(s for s in set(samples) if s not in ordered))
    colors = {
        "TH115": "#ff7f0e", "TH44": "#2ca02c", "TH54": "#d62728",
        "TH55": "#9467bd", "TH56": "#8c564b", "TH57": "#e377c2",
        "TH71": "#7f7f7f", "Zebra_LD": "#bcbd22", "Zebra_NMDA": "#17becf",
    }
    for sample in ordered:
        mask = samples == sample
        fig, ax = plt.subplots(figsize=(8.95, 6.0))
        ax.scatter(coords[:, 0], coords[:, 1], c="#d9d9d9", s=2, linewidths=0, alpha=0.25)
        ax.scatter(coords[mask, 0], coords[mask, 1], c=colors.get(sample, "#333333"), s=4, linewidths=0)
        ax.set_title(f"{sample_key}: {sample} ({int(mask.sum()):,})", fontsize=18)
        ax.set_xlabel("UMAP1")
        ax.set_ylabel("UMAP2")
        ax.set_xticks([])
        ax.set_yticks([])
        fig.savefig(figures_dir / f"bc_umap_by_{sample_key}_{sample}_highlight.png", dpi=dpi, bbox_inches="tight")
        plt.close(fig)


def plot_gene_all(adata, gene: str, out: Path, dpi: int) -> None:
    fig = sc.pl.umap(
        adata,
        color=gene,
        use_raw=adata.raw is not None,
        cmap="viridis",
        size=3,
        frameon=True,
        show=False,
        return_fig=True,
    )
    fig.set_size_inches(8.95, 6.0)
    ax = fig.axes[0]
    ax.set_title(gene, fontsize=24, pad=10)
    ax.set_xlabel("UMAP1", fontsize=18)
    ax.set_ylabel("UMAP2", fontsize=18)
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_gene_split(adata, gene: str, condition_key: str, out: Path, dpi: int) -> None:
    coords = adata.obsm["X_umap"]
    expr = expression_vector(adata, gene)
    finite = expr[np.isfinite(expr)]
    vmin = float(np.nanpercentile(finite, 1)) if finite.size else 0
    vmax = float(np.nanpercentile(finite, 99)) if finite.size else 1
    if not np.isfinite(vmax) or vmax <= vmin:
        vmax = float(np.nanmax(finite)) if finite.size else 1
    panels = condition_panels(adata, condition_key)
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 7.6))
    axes = axes.ravel()
    values = adata.obs[condition_key].astype(str).to_numpy()
    sca = None
    for ax, (label, allowed) in zip(axes, panels):
        mask = np.isin(values, allowed)
        ax.scatter(coords[:, 0], coords[:, 1], c="#d9d9d9", s=2, linewidths=0, alpha=0.22)
        sca = ax.scatter(coords[mask, 0], coords[mask, 1], c=expr[mask], s=4, linewidths=0, cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_title(f"{gene} | {label}", fontsize=12)
        ax.set_xlabel("UMAP1")
        ax.set_ylabel("UMAP2")
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes[len(panels):]:
        ax.axis("off")
    if sca is not None:
        fig.colorbar(sca, ax=axes.tolist(), label="Expression", fraction=0.025, pad=0.02)
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def pct_group_column(df: pd.DataFrame) -> str | None:
    for column in ["pct_nz_group", "pts"]:
        if column in df.columns:
            return column
    return None


def filter_markers(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    filtered = df.copy()
    if "pvals_adj" in filtered.columns:
        filtered = filtered[filtered["pvals_adj"] <= args.pval_adj]
    if "logfoldchanges" in filtered.columns:
        filtered = filtered[filtered["logfoldchanges"] >= args.min_logfc]
    pct_col = pct_group_column(filtered)
    if pct_col is not None:
        filtered = filtered[filtered[pct_col] >= args.min_pct_group]
    return filtered


def add_specificity_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "pct_nz_group" in out.columns and "pct_nz_reference" in out.columns:
        out["pct_diff"] = out["pct_nz_group"] - out["pct_nz_reference"]
    else:
        out["pct_diff"] = 0.0
    out["specificity_score"] = out.get("logfoldchanges", 1.0) * out["pct_diff"].clip(lower=0)
    return out


def top_per_cluster(df: pd.DataFrame, top_n: int, columns: list[str]) -> pd.DataFrame:
    ranked = df.copy()
    ranked["group_sort"] = ranked["group"].astype(str).map(sort_key)
    return ranked.sort_values(columns, ascending=[True] + [False] * (len(columns) - 1)).groupby("group", observed=False).head(top_n).drop(columns=["group_sort"]).reset_index(drop=True)


def compute_cluster_expression_stats(adata, cluster_key: str, genes: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    matrix, var_names = expression_source(adata)
    genes = [g for g in genes if g in var_names]
    subset = matrix[:, var_names.get_indexer(genes)]
    labels = adata.obs[cluster_key].astype(str)
    clusters = sorted(labels.unique(), key=sort_key)
    means, pcts = [], []
    for cluster in clusters:
        mask = (labels == cluster).to_numpy()
        m = subset[mask, :]
        if sparse.issparse(m):
            means.append(np.asarray(m.mean(axis=0)).ravel())
            pcts.append(m.getnnz(axis=0) / m.shape[0])
        else:
            arr = np.asarray(m)
            means.append(arr.mean(axis=0))
            pcts.append((arr > 0).mean(axis=0))
    return pd.DataFrame(np.vstack(means).T, index=genes, columns=clusters), pd.DataFrame(np.vstack(pcts).T, index=genes, columns=clusters)


def annotate_max_other(markers: pd.DataFrame, mean_df: pd.DataFrame, pct_df: pd.DataFrame) -> pd.DataFrame:
    out = add_specificity_columns(markers)
    rows = []
    for row in out.itertuples(index=False):
        gene, group = str(row.names), str(row.group)
        if gene not in mean_df.index or group not in mean_df.columns:
            rows.append({})
            continue
        other = [c for c in mean_df.columns if c != group]
        max_other_mean = float(mean_df.loc[gene, other].max())
        max_other_pct = float(pct_df.loc[gene, other].max())
        rows.append({
            "mean_expr_group": float(mean_df.at[gene, group]),
            "max_other_mean_expr": max_other_mean,
            "mean_expr_gap_vs_max_other": float(mean_df.at[gene, group]) - max_other_mean,
            "pct_group_from_matrix": float(pct_df.at[gene, group]),
            "max_other_pct_nz": max_other_pct,
            "pct_gap_vs_max_other": float(pct_df.at[gene, group]) - max_other_pct,
        })
    out = pd.concat([out.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
    out["max_other_specificity_score"] = out.get("logfoldchanges", 1.0) * out["pct_gap_vs_max_other"].fillna(0).clip(lower=0) * out["mean_expr_gap_vs_max_other"].fillna(0).clip(lower=0)
    return out


def plot_dotplot(adata, top1: pd.DataFrame, cluster_key: str, out: Path, dpi: int) -> None:
    genes = []
    for gene in top1["names"].astype(str):
        if gene not in genes:
            genes.append(gene)
    dp = sc.pl.dotplot(
        adata,
        var_names=genes,
        groupby=cluster_key,
        use_raw=adata.raw is not None,
        standard_scale="var",
        color_map="Reds",
        dendrogram=False,
        show=False,
        return_fig=True,
    )
    dp.style(cmap="Reds", dot_edge_color="#d0d5dd", dot_edge_lw=0.35, largest_dot=80, smallest_dot=0)
    dp.legend(width=1.2)
    dp.make_figure()
    dp.fig.set_size_inches(max(13, 0.45 * len(genes)), max(7.5, 0.24 * adata.obs[cluster_key].nunique()))
    dp.fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(dp.fig)


def run_deg(adata, args, figures_dir: Path, tables_dir: Path) -> None:
    sc.tl.rank_genes_groups(adata, groupby=args.cluster_key, method=args.method, use_raw=adata.raw is not None, pts=True)
    all_markers = sc.get.rank_genes_groups_df(adata, group=None)
    all_markers["group"] = all_markers["group"].astype(str)
    filtered = filter_markers(all_markers, args)
    top_score = top_per_cluster(filtered, 1, ["group_sort", "scores"])
    specific = add_specificity_columns(filtered)
    top_specific = top_per_cluster(specific, 1, ["group_sort", "specificity_score", "logfoldchanges", "pct_diff", "scores"])
    genes = filtered["names"].astype(str).drop_duplicates().tolist()
    mean_df, pct_df = compute_cluster_expression_stats(adata, args.cluster_key, genes)
    max_other = annotate_max_other(filtered, mean_df, pct_df)
    top_max_other = top_per_cluster(max_other, 1, ["group_sort", "max_other_specificity_score", "pct_gap_vs_max_other", "mean_expr_gap_vs_max_other", "specificity_score", "scores"])
    prefix = "bc_guca1b_gt2_deg"
    all_markers.to_csv(tables_dir / f"{prefix}_markers_all.csv", index=False)
    filtered.to_csv(tables_dir / f"{prefix}_markers_filtered.csv", index=False)
    top_score.to_csv(tables_dir / f"{prefix}_score_top1_marker_per_cluster.csv", index=False)
    top_specific.to_csv(tables_dir / f"{prefix}_specific_top1_marker_per_cluster.csv", index=False)
    max_other.to_csv(tables_dir / f"{prefix}_max_other_annotated_filtered.csv", index=False)
    top_max_other.to_csv(tables_dir / f"{prefix}_max_other_top1_marker_per_cluster.csv", index=False)
    pd.DataFrame([{
        "method": args.method,
        "use_raw": adata.raw is not None,
        "pval_adj": args.pval_adj,
        "min_logfc": args.min_logfc,
        "min_pct_group": args.min_pct_group,
        "cluster_key": args.cluster_key,
        "specificity_score": "logfoldchanges * max(pct_nz_group - pct_nz_reference, 0)",
        "max_other_specificity_score": "logfoldchanges * max(pct_group - max_other_pct, 0) * max(mean_group - max_other_mean, 0)",
    }]).to_csv(tables_dir / f"{prefix}_parameters.csv", index=False)
    plot_dotplot(adata, top_score, args.cluster_key, figures_dir / f"{prefix}_score_top1_marker_dotplot.png", args.dpi)
    plot_dotplot(adata, top_specific, args.cluster_key, figures_dir / f"{prefix}_specific_top1_marker_dotplot.png", args.dpi)
    plot_dotplot(adata, top_max_other, args.cluster_key, figures_dir / f"{prefix}_max_other_top1_marker_dotplot.png", args.dpi)


def data_uri(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def image_size_attrs(path: Path) -> str:
    try:
        header = path.read_bytes()[:24]
        if header[:8] == b"\x89PNG\r\n\x1a\n":
            w, h = struct.unpack(">II", header[16:24])
            return f' width="{w}" height="{h}"'
    except OSError:
        pass
    return ""


def rel(path: Path, output_dir: Path) -> str:
    return os.path.relpath(path, output_dir)


def panel(title: str, path: Path, output_dir: Path) -> str:
    if not path.exists():
        return f"<div class='panel missing'><h3>{html.escape(title)}</h3><p>Missing {html.escape(path.name)}</p></div>"
    href = html.escape(rel(path, output_dir))
    return f"""
    <div class="panel">
      <h3>{html.escape(title)}</h3>
      <a href="{href}" target="_blank"><img src="{data_uri(path)}" alt="{html.escape(title)}"{image_size_attrs(path)}></a>
      <div class="meta"><span>{html.escape(path.name)}</span><a href="{href}" download>Download PNG</a></div>
    </div>
    """


def table_html(path: Path, title: str) -> str:
    if not path.exists():
        return ""
    df = pd.read_csv(path).fillna("")
    cols = [c for c in ["group", "names", "scores", "logfoldchanges", "pvals_adj", "pct_nz_group", "pct_nz_reference", "specificity_score", "max_other_specificity_score"] if c in df.columns]
    rows = []
    for row in df[cols].itertuples(index=False):
        cells = []
        for col, value in zip(cols, row):
            if col not in {"group", "names"}:
                try:
                    value = f"{float(value):.3g}"
                except Exception:
                    pass
            cells.append(f"<td>{html.escape(str(value))}</td>")
        rows.append(f"<tr>{''.join(cells)}</tr>")
    return f"<div class='tablewrap'><h3>{html.escape(title)}</h3><table><thead><tr>{''.join(f'<th>{html.escape(c)}</th>' for c in cols)}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"


def build_html(figures_dir: Path, tables_dir: Path, output: Path) -> None:
    output_dir = output.parent
    match = pd.read_csv(tables_dir / "bc_manual_gene_match_report.csv")
    cards = []
    for row in match.itertuples(index=False):
        gene = str(row.matched_gene) if bool(row.present) else str(row.requested_gene)
        title = str(row.requested_gene)
        note = "" if not bool(row.present) else (f"matched as {html.escape(gene)}" if gene != title else "")
        if not bool(row.present):
            cards.append(f"<article class='card'><h2>{html.escape(title)}</h2><p class='missing'>Not found in this object.</p></article>")
            continue
        cards.append(f"""
        <article class="card">
          <div class="head"><h2>{html.escape(title)}</h2><span>{note}</span></div>
          <div class="grid">
            {panel("All conditions", figures_dir / f"bc_umap_gene_{gene}_all_conditions.png", output_dir)}
            {panel("Split by condition", figures_dir / f"bc_umap_gene_{gene}_split_by_condition.png", output_dir)}
          </div>
        </article>
        """)
    overview = "".join(panel(name, figures_dir / name, output_dir) for name in [
        "bc_umap_by_sample_split_zebra.png",
        "bc_umap_by_renamed_samples.png",
        "bc_umap_by_leiden.png",
    ] if (figures_dir / name).exists())
    prefix = "bc_guca1b_gt2_deg"
    deg = f"""
    <article class="card">
      <h2>BC DEG Top Markers</h2>
      <p class="note">Grouped by Leiden resolution 1.0 on the guca1b &gt; 2.0 filtered BC object.</p>
      <div class="deggrid">
        <section>{panel("Wilcoxon score top1 dotplot", figures_dir / f"{prefix}_score_top1_marker_dotplot.png", output_dir)}{table_html(tables_dir / f"{prefix}_score_top1_marker_per_cluster.csv", "Wilcoxon score top1")}</section>
        <section>{panel("Specificity-ranked top1 dotplot", figures_dir / f"{prefix}_specific_top1_marker_dotplot.png", output_dir)}{table_html(tables_dir / f"{prefix}_specific_top1_marker_per_cluster.csv", "Specificity top1")}</section>
        <section>{panel("Preferred max-other top1 dotplot", figures_dir / f"{prefix}_max_other_top1_marker_dotplot.png", output_dir)}{table_html(tables_dir / f"{prefix}_max_other_top1_marker_per_cluster.csv", "Max-other top1")}</section>
      </div>
    </article>
    """
    css = """
    body{margin:0;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;background:#f6f8fb;color:#172033}
    header{position:sticky;top:0;background:white;border-bottom:1px solid #d8e0ea;padding:10px 16px;z-index:5}
    h1{margin:0 0 8px;font-size:22px}.tabs{display:flex;gap:8px}.tabs button{padding:8px 10px;border:1px solid #d8e0ea;background:white;border-radius:6px;font-weight:700;cursor:pointer}.tabs button.active{background:#0b63ce;color:white}
    main{padding:14px}.tab{display:none}.tab.active{display:block}.card{background:white;border:1px solid #d8e0ea;border-radius:8px;padding:12px;margin-bottom:14px}.head{display:flex;justify-content:space-between;border-bottom:1px solid #e8edf3;margin-bottom:10px}.head h2{margin:0 0 8px}.head span{color:#667085}
    .grid{display:grid;grid-template-columns:repeat(2,minmax(300px,1fr));gap:12px}.deggrid{display:grid;grid-template-columns:repeat(3,minmax(300px,1fr));gap:14px}.panel{border:1px solid #e8edf3;border-radius:6px;padding:8px;background:white}.panel img{width:100%;height:auto;display:block}.panel h3{margin:0 0 8px;font-size:14px;color:#344054}.meta{display:flex;justify-content:space-between;gap:8px;font-size:12px;color:#667085;overflow-wrap:anywhere}.meta a{font-weight:700;color:#075ab8;text-decoration:none}.missing{color:#a0a7b3}.note{color:#344054}
    .tablewrap{max-height:420px;overflow:auto;border:1px solid #e8edf3;border-radius:6px;margin-top:8px}table{width:100%;border-collapse:collapse;font-size:12px}th,td{border-top:1px solid #eef2f6;padding:5px 7px;text-align:left;white-space:nowrap}
    @media(max-width:1000px){.grid,.deggrid{grid-template-columns:1fr}}
    """
    js = """
    document.querySelectorAll('.tabs button').forEach(b=>b.onclick=()=>{document.querySelectorAll('.tabs button').forEach(x=>x.classList.remove('active'));document.querySelectorAll('.tab').forEach(x=>x.classList.remove('active'));b.classList.add('active');document.getElementById(b.dataset.tab).classList.add('active')});
    """
    output.write_text(f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>BC guca1b > 2 Viewer</title><style>{css}</style></head><body><header><h1>BC guca1b &gt; 2.0 Filtered Viewer</h1><div class="tabs"><button class="active" data-tab="genes">Gene UMAPs</button><button data-tab="deg">DEG Top Markers</button></div></header><main><section id="genes" class="tab active"><article class="card"><h2>Embedding Overview</h2><div class="grid">{overview}</div></article>{''.join(cards)}</section><section id="deg" class="tab">{deg}</section></main><footer style="padding:10px 14px;color:#667085;font-size:12px">Generated {html.escape(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))}</footer><script>{js}</script></body></html>""", encoding="utf-8")


def main() -> None:
    args = parse_args()
    figures_dir = Path(args.figures_dir)
    tables_dir = Path(args.tables_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    output = Path(args.output) if args.output else figures_dir / "bc_guca1b_gt2_gene_deg_viewer.html"

    adata = sc.read_h5ad(args.h5ad)
    if args.sample_key not in adata.obs and args.sample_key == "sample_split_zebra":
        sample = adata.obs["sample"].astype(str).copy()
        cond = adata.obs[args.condition_key].astype(str)
        sample.loc[sample == "Zebra"] = "Zebra_" + cond.loc[sample == "Zebra"]
        adata.obs[args.sample_key] = sample.astype("category")
    genes = gene_match_report(adata, tables_dir)
    save_overview_umap(adata, args.sample_key, figures_dir / f"bc_umap_by_{args.sample_key}.png", args.dpi, "9 samples")
    save_overview_umap(adata, args.condition_key, figures_dir / f"bc_umap_by_{args.condition_key}.png", args.dpi, "Condition")
    save_overview_umap(adata, args.cluster_key, figures_dir / f"bc_umap_by_{args.cluster_key}.png", args.dpi, "Leiden clusters")
    save_sample_highlights(adata, args.sample_key, figures_dir, args.dpi)
    for gene in genes:
        plot_gene_all(adata, gene, figures_dir / f"bc_umap_gene_{gene}_all_conditions.png", args.dpi)
        plot_gene_split(adata, gene, args.condition_key, figures_dir / f"bc_umap_gene_{gene}_split_by_condition.png", args.dpi)
    run_deg(adata, args, figures_dir, tables_dir)
    build_html(figures_dir, tables_dir, output)
    print(output)


if __name__ == "__main__":
    main()

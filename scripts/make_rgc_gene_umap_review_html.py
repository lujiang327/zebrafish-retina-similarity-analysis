#!/usr/bin/env python
from __future__ import annotations

import argparse
import base64
from datetime import datetime
import html
import mimetypes
import os
from pathlib import Path
import re
import struct

import pandas as pd


REQUESTED_GROUPS = {
    "Spatial / Axon Guidance": [
        "ephb2b",
        "ephb3a",
        "ephnb4a",
        "epha4b",
        "efna2",
        "efnb2a",
        "efna5a",
        "efna5b",
        "cntn2",
        "efnb1",
    ],
    "Dorsal-Ventral / Nasal-Temporal": [
        "hmx1",
        "hmx4",
        "vax1",
        "vax2",
        "tbx5a",
        "zic2",
    ],
    "RGC Markers": [
        "isl2b",
        "rbpms2a",
        "rbpms2b",
        "mafaa",
        "eomesa",
        "tbr1b",
    ],
    "Neuronal / Regeneration": [
        "elavl3",
        "cdk1",
        "nr2e3",
        "guca1b",
        "gnat2",
        "cabp5a",
        "ompa",
    ],
    "Non-RGC / Contamination Checks": [
        "mpeg1.1",
        "mbpa",
        "rpe65a",
        "aqp4",
        "tie1",
        "acta2",
        "mitfa",
    ],
    "Other": [
        "efna2a",
        "efna2b",
        "zic2a",
        "zic2b",
        "gria3a",
        "crhb",
        "ccdc80",
        "tubb5",
        "six3a",
        "sgk1",
        "tacr2",
        "matn3b",
        "rasgef1ba",
        "tac1",
        "pcdh11",
        "sdk1a",
        "ptprt",
        "gpd1b",
        "rprma",
        "tbx3a",
        "si:dkey-1h6.8",
        "neurod2",
        "pgam2",
        "mfap4",
        "tox",
        "id3",
        "emid1",
        "ccka",
        "gulp1b",
        "plppr2",
        "pyyb",
        "dio1",
        "gapdh",
        "npb",
        "nmbb",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build interactive HTML review for RGC gene UMAPs.")
    parser.add_argument(
        "--figures-dir",
        default="results/figures/rgc_subtype_gene_umaps",
        help="Directory containing RGC gene UMAP PNG files.",
    )
    parser.add_argument(
        "--match-report",
        default="results/tables/rgc_subtype_gene_umaps/rgc_gene_match_report.csv",
        help="Gene match report from plot_rgc_subtype_genes.py.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output HTML path. Defaults to <figures-dir>/rgc_gene_umap_review.html.",
    )
    parser.add_argument(
        "--dotplot",
        default="results/figures/rgc_subtype_gene_umaps/rgc_marker_dotplot_leiden_1_35_merged.png",
        help="Merged-cluster marker dotplot PNG.",
    )
    parser.add_argument(
        "--merged-umap",
        default="results/figures/rgc_subtype_gene_umaps/rgc_umap_leiden_1_35_merged_renamed.png",
        help="UMAP colored by merged and renamed cluster labels.",
    )
    parser.add_argument(
        "--dotplot-pdf",
        default="results/figures/rgc_subtype_gene_umaps/rgc_marker_dotplot_leiden_1_35_merged.pdf",
        help="Merged-cluster marker dotplot PDF.",
    )
    parser.add_argument(
        "--dotplot-gene-report",
        default="results/tables/rgc_subtype_gene_umaps/rgc_dotplot_gene_match_report.csv",
        help="Dotplot gene match report CSV.",
    )
    parser.add_argument(
        "--dotplot-cluster-map",
        default="results/tables/rgc_subtype_gene_umaps/rgc_dotplot_cluster_1_35_merge_map.csv",
        help="Dotplot cluster merge map CSV.",
    )
    parser.add_argument(
        "--dotplot-gene-order",
        default="results/tables/rgc_subtype_gene_umaps/rgc_dotplot_gene_order_by_peak_cluster.csv",
        help="Dotplot gene order CSV.",
    )
    parser.add_argument(
        "--deg-dotplot",
        default="results/figures/rgc_subtype_gene_umaps/rgc_merged_cluster_deg_top1_marker_dotplot.png",
        help="Dotplot of DEG top marker per merged cluster.",
    )
    parser.add_argument(
        "--deg-top1",
        default="results/tables/rgc_subtype_gene_umaps/rgc_merged_cluster_deg_top1_marker_per_cluster.csv",
        help="Top DEG marker per merged cluster CSV.",
    )
    parser.add_argument(
        "--deg-top",
        default="results/tables/rgc_subtype_gene_umaps/rgc_merged_cluster_deg_top20_markers.csv",
        help="Top DEG markers per merged cluster CSV.",
    )
    parser.add_argument(
        "--deg-specific-dotplot",
        default="results/figures/rgc_subtype_gene_umaps/rgc_merged_cluster_deg_specific_top1_marker_dotplot.png",
        help="Dotplot of specificity-ranked DEG top marker per merged cluster.",
    )
    parser.add_argument(
        "--deg-specific-top1",
        default="results/tables/rgc_subtype_gene_umaps/rgc_merged_cluster_deg_specific_top1_marker_per_cluster.csv",
        help="Specificity-ranked top DEG marker per merged cluster CSV.",
    )
    parser.add_argument(
        "--deg-specific-top",
        default="results/tables/rgc_subtype_gene_umaps/rgc_merged_cluster_deg_specific_top20_markers.csv",
        help="Specificity-ranked top DEG markers per merged cluster CSV.",
    )
    parser.add_argument(
        "--deg-pval-dotplot",
        default="results/figures/rgc_subtype_gene_umaps/rgc_merged_cluster_deg_pval_top1_marker_dotplot.png",
        help="Dotplot of adjusted-p-value-ranked DEG top marker per merged cluster.",
    )
    parser.add_argument(
        "--deg-pval-top1",
        default="results/tables/rgc_subtype_gene_umaps/rgc_merged_cluster_deg_pval_top1_marker_per_cluster.csv",
        help="Adjusted-p-value-ranked top DEG marker per merged cluster CSV.",
    )
    parser.add_argument(
        "--deg-pval-top",
        default="results/tables/rgc_subtype_gene_umaps/rgc_merged_cluster_deg_pval_top20_markers.csv",
        help="Adjusted-p-value-ranked top DEG markers per merged cluster CSV.",
    )
    parser.add_argument(
        "--deg-max-other-dotplot",
        default="results/figures/rgc_subtype_gene_umaps/strict_marker_review/rgc_deg_max_other_top1_marker_dotplot.png",
        help="Dotplot of max-other strict DEG top marker per merged cluster.",
    )
    parser.add_argument(
        "--deg-max-other-top1",
        default="results/tables/rgc_subtype_gene_umaps/strict_marker_review/rgc_deg_max_other_top1_marker_per_cluster.csv",
        help="Max-other strict top DEG marker per merged cluster CSV.",
    )
    parser.add_argument(
        "--deg-max-other-top",
        default="results/tables/rgc_subtype_gene_umaps/strict_marker_review/rgc_deg_max_other_top20_markers.csv",
        help="Max-other strict top DEG markers per merged cluster CSV.",
    )
    parser.add_argument(
        "--deg-filtered",
        default="results/tables/rgc_subtype_gene_umaps/rgc_merged_cluster_deg_markers_filtered.csv",
        help="Filtered DEG markers CSV.",
    )
    parser.add_argument(
        "--deg-all",
        default="results/tables/rgc_subtype_gene_umaps/rgc_merged_cluster_deg_markers_all.csv",
        help="All DEG markers CSV.",
    )
    parser.add_argument(
        "--deg-params",
        default="results/tables/rgc_subtype_gene_umaps/rgc_merged_cluster_deg_parameters.csv",
        help="DEG parameters CSV.",
    )
    return parser.parse_args()


def make_anchor(text: str) -> str:
    anchor = re.sub(r"[^A-Za-z0-9_-]+", "_", text).strip("_")
    return f"gene-{anchor or 'unknown'}"


def image_data_uri(path: Path) -> str:
    mime_type = mimetypes.guess_type(path.name)[0] or "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def image_size_attrs(path: Path) -> str:
    if path.suffix.lower() != ".png":
        return ""
    try:
        with path.open("rb") as handle:
            header = handle.read(24)
        if header[:8] != b"\x89PNG\r\n\x1a\n":
            return ""
        width, height = struct.unpack(">II", header[16:24])
    except OSError:
        return ""
    return f' width="{width}" height="{height}"'


def href_for(path: Path, output_dir: Path) -> str:
    return os.path.relpath(path, output_dir)


def load_gene_records(match_report: Path) -> pd.DataFrame:
    df = pd.read_csv(match_report)
    for col in ["requested_gene", "matched_gene"]:
        if col in df:
            df[col] = df[col].fillna("").astype(str)
    if "present" in df:
        df["present"] = df["present"].astype(bool)
    return df


def deg_gene_groups(*paths: Path) -> dict[str, list[str]]:
    labels = [
        "DEG: Specificity-Ranked Top1",
        "DEG: Preferred Max-Other Top1",
    ]
    groups = {}
    for label, path in zip(labels, paths):
        if not path.exists():
            continue
        top_df = pd.read_csv(path)
        if "names" not in top_df.columns:
            continue
        genes = []
        for gene in top_df["names"].dropna().astype(str):
            if gene not in genes:
                genes.append(gene)
        if genes:
            groups[label] = genes
    return groups


def requested_gene_order(df: pd.DataFrame, extra_groups: dict[str, list[str]] | None = None) -> list[str]:
    ordered = []
    requested = set(df["requested_gene"].astype(str))
    for genes in REQUESTED_GROUPS.values():
        for gene in genes:
            if gene in requested and gene not in ordered:
                ordered.append(gene)
    for genes in (extra_groups or {}).values():
        for gene in genes:
            if gene in requested and gene not in ordered:
                ordered.append(gene)
    for gene in df["requested_gene"].astype(str):
        if gene not in ordered:
            ordered.append(gene)
    return ordered


def build_index(df: pd.DataFrame, extra_groups: dict[str, list[str]] | None = None) -> str:
    row_by_requested = {
        row["requested_gene"]: row for _, row in df.iterrows()
    }
    blocks = []
    used = set()
    grouped = dict(REQUESTED_GROUPS)
    grouped.update(extra_groups or {})
    for group_name, genes in grouped.items():
        links = []
        for gene in genes:
            if gene not in row_by_requested:
                continue
            used.add(gene)
            row = row_by_requested[gene]
            label = html.escape(gene)
            if bool(row["present"]):
                matched = html.escape(row["matched_gene"])
                anchor = html.escape(make_anchor(gene), quote=True)
                links.append(
                    f'<a href="#{anchor}" data-target="{anchor}" data-gene="{label}" '
                    f'title="matched: {matched}">{label}</a>'
                )
            else:
                links.append(f'<span class="missing" title="not found in var_names">{label}</span>')
        if links:
            blocks.append(
                f"""
                <section class="marker-group">
                  <h2>{html.escape(group_name)}</h2>
                  <div class="gene-links">{" ".join(links)}</div>
                </section>
                """
            )

    extras = [gene for gene in row_by_requested if gene not in used]
    if extras:
        links = []
        for gene in extras:
            row = row_by_requested[gene]
            label = html.escape(gene)
            if bool(row["present"]):
                anchor = html.escape(make_anchor(gene), quote=True)
                links.append(
                    f'<a href="#{anchor}" data-target="{anchor}" data-gene="{label}">{label}</a>'
                )
            else:
                links.append(f'<span class="missing">{label}</span>')
        blocks.append(
            f"""
            <section class="marker-group">
              <h2>Other</h2>
              <div class="gene-links">{" ".join(links)}</div>
            </section>
            """
        )
    return "\n".join(blocks)


def find_gene_images(figures_dir: Path, matched_gene: str) -> dict[str, Path]:
    return {
        "All conditions": figures_dir / f"rgc_umap_gene_{matched_gene}_all_conditions.png",
        "Split by condition": figures_dir / f"rgc_umap_gene_{matched_gene}_split_by_condition.png",
    }


def build_cards(
    df: pd.DataFrame,
    figures_dir: Path,
    extra_groups: dict[str, list[str]] | None = None,
) -> str:
    cards = []
    row_by_requested = {row["requested_gene"]: row for _, row in df.iterrows()}
    for requested_gene in requested_gene_order(df, extra_groups):
        row = row_by_requested[requested_gene]
        safe_requested = html.escape(requested_gene)
        anchor = html.escape(make_anchor(requested_gene), quote=True)
        if not bool(row["present"]):
            cards.append(
                f"""
                <article class="plot-card missing-card" id="{anchor}" data-gene="{safe_requested}">
                  <h2>{safe_requested}</h2>
                  <p class="missing-note">Not found in this AnnData object's var_names.</p>
                </article>
                """
            )
            continue

        matched_gene = row["matched_gene"]
        image_blocks = []
        for label, image_path in find_gene_images(figures_dir, matched_gene).items():
            if not image_path.exists():
                image_blocks.append(
                    f"""
                    <div class="image-panel absent-image">
                      <h3>{html.escape(label)}</h3>
                      <p>Missing image: {html.escape(image_path.name)}</p>
                    </div>
                    """
                )
                continue
            image_blocks.append(
                f"""
                <div class="image-panel">
                  <h3>{html.escape(label)}</h3>
                  <a href="{html.escape(image_path.name)}" target="_blank" rel="noopener">
                    <img src="{image_data_uri(image_path)}" alt="{safe_requested} {html.escape(label)}" loading="lazy"{image_size_attrs(image_path)}>
                  </a>
                  <div class="plot-meta">
                    <span>{html.escape(image_path.name)}</span>
                    <a href="{html.escape(image_path.name)}" download>Download PNG</a>
                  </div>
                </div>
                """
            )
        matched_note = "" if matched_gene == requested_gene else f"matched as {html.escape(matched_gene)}"
        cards.append(
            f"""
            <article class="plot-card" id="{anchor}" data-gene="{safe_requested} {html.escape(matched_gene)}">
              <div class="card-head">
                <h2>{safe_requested}</h2>
                <span>{matched_note}</span>
              </div>
              <div class="image-grid">{"".join(image_blocks)}</div>
            </article>
            """
        )
    return "\n".join(cards)


def build_overview(figures_dir: Path) -> str:
    overview_paths = sorted(
        p
        for p in figures_dir.glob("*_sHarmony*.png")
        if not any(sample in p.stem for sample in ["_Control", "_LD", "_NMDA"])
    )
    fallback_names = [
        "rgc_umap_by_renamed_samples_strict.png",
        "rgc_umap_by_renamed_samples_Control_highlight.png",
        "rgc_umap_by_renamed_samples_LD_highlight.png",
        "rgc_umap_by_renamed_samples_NMDA_highlight.png",
        "rgc_umap_by_rgc_leiden_strict.png",
    ]
    overview_paths.extend(
        figures_dir / name for name in fallback_names if (figures_dir / name).exists()
    )
    panels = []
    seen = set()
    for path in overview_paths:
        if path in seen:
            continue
        seen.add(path)
        if path.exists():
            panels.append(
                f"""
                <div class="image-panel">
                  <h3>{html.escape(path.name)}</h3>
                  <a href="{html.escape(path.name)}" target="_blank" rel="noopener">
                    <img src="{image_data_uri(path)}" alt="{html.escape(path.name)}" loading="lazy"{image_size_attrs(path)}>
                  </a>
                  <div class="plot-meta">
                    <span>{html.escape(path.name)}</span>
                    <a href="{html.escape(path.name)}" download>Download PNG</a>
                  </div>
                </div>
                """
            )
    if not panels:
        return ""
    return f"""
    <section class="overview" id="overview">
      <h2>RGC Subtype Embedding</h2>
      <div class="image-grid">{"".join(panels)}</div>
    </section>
    """


def build_missing_summary(df: pd.DataFrame) -> str:
    missing = df.loc[~df["present"], "requested_gene"].tolist()
    if not missing:
        return '<p class="note strong">All requested genes were found in var_names.</p>'
    chips = " ".join(f"<span>{html.escape(g)}</span>" for g in missing)
    return f"""
    <p class="note strong">Missing requested genes: {len(missing)}</p>
    <div class="missing-chips">{chips}</div>
    """


def build_dotplot_report(
    gene_report: Path,
    cluster_map: Path,
    gene_order: Path,
    output_dir: Path,
) -> str:
    blocks = []
    if gene_report.exists():
        df = pd.read_csv(gene_report).fillna("")
        missing = df.loc[~df["present"].astype(bool), "requested_gene"].astype(str).tolist()
        aliases = df.loc[df["alias_used"].astype(str) != "", ["requested_gene", "alias_used", "matched_gene"]]
        if not aliases.empty:
            rows = "\n".join(
                "<tr>"
                f"<td>{html.escape(str(row.requested_gene))}</td>"
                f"<td>{html.escape(str(row.alias_used))}</td>"
                f"<td>{html.escape(str(row.matched_gene))}</td>"
                "</tr>"
                for row in aliases.itertuples(index=False)
            )
            blocks.append(
                f"""
                <div class="small-table-wrap">
                  <h3>Gene name aliases</h3>
                  <table class="small-table">
                    <thead><tr><th>Requested</th><th>Used for lookup</th><th>Matched</th></tr></thead>
                    <tbody>{rows}</tbody>
                  </table>
                </div>
                """
            )
        if missing:
            chips = " ".join(f"<span>{html.escape(gene)}</span>" for gene in missing)
            blocks.append(
                f"""
                <p class="note strong">Dotplot genes not found: {len(missing)}</p>
                <div class="missing-chips">{chips}</div>
                """
            )
        else:
            blocks.append('<p class="note strong">All requested dotplot genes were found or mapped by alias.</p>')

    report_links = []
    if gene_report.exists():
        report_links.append(
            f'<a href="{html.escape(href_for(gene_report, output_dir))}" download>Download gene match CSV</a>'
        )
    if cluster_map.exists():
        report_links.append(
            f'<a href="{html.escape(href_for(cluster_map, output_dir))}" download>Download cluster merge CSV</a>'
        )
    if gene_order.exists():
        report_links.append(
            f'<a href="{html.escape(href_for(gene_order, output_dir))}" download>Download gene order CSV</a>'
        )
    if report_links:
        blocks.append(f'<div class="report-links">{" ".join(report_links)}</div>')
    return "\n".join(blocks)


def build_dotplot_tab(
    merged_umap: Path,
    dotplot_path: Path,
    dotplot_pdf: Path,
    gene_report: Path,
    cluster_map: Path,
    gene_order: Path,
    output_dir: Path,
) -> str:
    if not dotplot_path.exists():
        return """
        <article class="plot-card absent-image">
          <h2>Cluster Marker Dotplot</h2>
          <p>Dotplot image has not been generated yet.</p>
        </article>
        """
    umap_block = ""
    if merged_umap.exists():
        umap_href = html.escape(href_for(merged_umap, output_dir))
        umap_block = f"""
        <div class="image-panel merged-umap-panel">
          <h3>UMAP: clusters 1 and 35 merged, then renamed</h3>
          <a href="{umap_href}" target="_blank" rel="noopener">
            <img src="{image_data_uri(merged_umap)}" alt="UMAP colored by merged and renamed cluster labels" loading="lazy"{image_size_attrs(merged_umap)}>
          </a>
          <div class="plot-meta">
            <span>{html.escape(merged_umap.name)}</span>
            <a href="{umap_href}" download>Download PNG</a>
          </div>
        </div>
        """
    pdf_link = ""
    if dotplot_pdf.exists():
        pdf_link = f'<a href="{html.escape(href_for(dotplot_pdf, output_dir))}" download>Download PDF</a>'
    report = build_dotplot_report(gene_report, cluster_map, gene_order, output_dir)
    dotplot_href = html.escape(href_for(dotplot_path, output_dir))
    return f"""
    <article class="plot-card dotplot-card">
      <div class="card-head">
        <h2>Combined Clusters 1 and 35</h2>
        <span>clusters 1 and 35 merged, then all cluster groups renamed sequentially</span>
      </div>
      <div class="image-grid dotplot-grid">
        {umap_block}
      </div>
      <div class="image-panel">
        <h3>Marker dotplot</h3>
        <a href="{dotplot_href}" target="_blank" rel="noopener">
          <img src="{image_data_uri(dotplot_path)}" alt="RGC marker dotplot for merged clusters 1 and 35" loading="lazy"{image_size_attrs(dotplot_path)}>
        </a>
        <div class="plot-meta">
          <span>{html.escape(dotplot_path.name)}</span>
          <a href="{dotplot_href}" download>Download PNG</a>
          {pdf_link}
        </div>
      </div>
      {report}
    </article>
    """


def format_float(value, digits: int = 3) -> str:
    try:
        return f"{float(value):.{digits}g}"
    except (TypeError, ValueError):
        return ""


def build_deg_image_panel(title: str, image_path: Path, output_dir: Path) -> str:
    if not image_path.exists():
        return f"""
        <div class="image-panel absent-image">
          <h3>{html.escape(title)}</h3>
          <p>Missing image: {html.escape(image_path.name)}</p>
        </div>
        """
    image_href = html.escape(href_for(image_path, output_dir))
    return f"""
    <div class="image-panel">
      <h3>{html.escape(title)}</h3>
      <a href="{image_href}" target="_blank" rel="noopener">
        <img src="{image_data_uri(image_path)}" alt="{html.escape(title)}" loading="lazy"{image_size_attrs(image_path)}>
      </a>
      <div class="plot-meta">
        <span>{html.escape(image_path.name)}</span>
        <a href="{image_href}" download>Download PNG</a>
      </div>
    </div>
    """


def build_deg_top1_table(top1_path: Path, title: str) -> str:
    if not top1_path.exists():
        return ""
    df = pd.read_csv(top1_path).fillna("")
    columns = [
        "group",
        "names",
        "scores",
        "logfoldchanges",
        "pvals_adj",
        "pct_nz_group",
        "pct_nz_reference",
        "pct_diff",
        "pct_ratio",
        "specificity_score",
    ]
    available = [column for column in columns if column in df.columns]
    rows = []
    for row in df[available].itertuples(index=False):
        values = row._asdict()
        cells = []
        for column in available:
            value = values[column]
            if column not in {"group", "names"}:
                value = format_float(value)
            cells.append(f"<td>{html.escape(str(value))}</td>")
        rows.append(f"<tr>{''.join(cells)}</tr>")
    header_labels = {
        "group": "Cluster",
        "names": "Top marker",
        "scores": "Score",
        "logfoldchanges": "LogFC",
        "pvals_adj": "Adj. p",
        "pct_nz_group": "Pct cluster",
        "pct_nz_reference": "Pct rest",
        "pct_diff": "Pct diff",
        "pct_ratio": "Pct ratio",
        "specificity_score": "Specificity score",
    }
    headers = "".join(f"<th>{html.escape(header_labels.get(column, column))}</th>" for column in available)
    return f"""
    <div class="small-table-wrap deg-table-wrap">
      <h3>{html.escape(title)}</h3>
      <table class="small-table">
        <thead><tr>{headers}</tr></thead>
        <tbody>{''.join(rows)}</tbody>
      </table>
    </div>
    """


def build_deg_tab(
    deg_dotplot: Path,
    deg_top1: Path,
    deg_top: Path,
    deg_specific_dotplot: Path,
    deg_specific_top1: Path,
    deg_specific_top: Path,
    deg_pval_dotplot: Path,
    deg_pval_top1: Path,
    deg_pval_top: Path,
    deg_max_other_dotplot: Path,
    deg_max_other_top1: Path,
    deg_max_other_top: Path,
    deg_filtered: Path,
    deg_all: Path,
    deg_params: Path,
    output_dir: Path,
) -> str:
    specific_image = build_deg_image_panel(
        "Specificity-ranked top marker dotplot",
        deg_specific_dotplot,
        output_dir,
    )
    max_other_image = build_deg_image_panel(
        "Preferred max-other strict top marker dotplot",
        deg_max_other_dotplot,
        output_dir,
    )
    links = []
    for label, path in [
        ("Download specificity-ranked top1 CSV", deg_specific_top1),
        ("Download specificity-ranked top20 CSV", deg_specific_top),
        ("Download preferred max-other top1 CSV", deg_max_other_top1),
        ("Download preferred max-other top20 CSV", deg_max_other_top),
        ("Download filtered CSV", deg_filtered),
        ("Download all CSV", deg_all),
        ("Download parameters CSV", deg_params),
    ]:
        if path.exists():
            links.append(f'<a href="{html.escape(href_for(path, output_dir))}" download>{label}</a>')
    link_block = f'<div class="report-links">{" ".join(links)}</div>' if links else ""
    specific_table = build_deg_top1_table(
        deg_specific_top1,
        "Specificity-ranked top marker per merged cluster",
    )
    max_other_table = build_deg_top1_table(
        deg_max_other_top1,
        "Preferred max-other strict top marker per merged cluster",
    )
    return f"""
    <article class="plot-card dotplot-card deg-card">
      <div class="card-head">
        <h2>DEG Top Markers</h2>
        <span>Compare pooled-rest specificity with stricter max-other specificity on merged clusters</span>
      </div>
      <div class="method-note">
        <strong>Specificity ranking formula:</strong>
        specificity_score = logfoldchanges x max(pct_nz_group - pct_nz_reference, 0).
        <br>
        <strong>Preferred max-other formula:</strong>
        max_other_specificity_score = logfoldchanges x max(pct_group - max_other_pct, 0) x max(mean_group - max_other_mean, 0).
        This penalizes genes that are also highly expressed in any single other cluster.
      </div>
      <div class="deg-comparison-grid">
        <section class="deg-method-panel">
          {specific_image}
          {specific_table}
        </section>
        <section class="deg-method-panel">
          {max_other_image}
          {max_other_table}
        </section>
      </div>
      {link_block}
    </article>
    """


def main() -> None:
    args = parse_args()
    figures_dir = Path(args.figures_dir)
    match_report = Path(args.match_report)
    output = Path(args.output) if args.output else figures_dir / "rgc_gene_umap_review.html"
    output_dir = output.parent
    dotplot_path = Path(args.dotplot)
    merged_umap = Path(args.merged_umap)
    dotplot_pdf = Path(args.dotplot_pdf)
    dotplot_gene_report = Path(args.dotplot_gene_report)
    dotplot_cluster_map = Path(args.dotplot_cluster_map)
    dotplot_gene_order = Path(args.dotplot_gene_order)
    deg_dotplot = Path(args.deg_dotplot)
    deg_top1 = Path(args.deg_top1)
    deg_top = Path(args.deg_top)
    deg_specific_dotplot = Path(args.deg_specific_dotplot)
    deg_specific_top1 = Path(args.deg_specific_top1)
    deg_specific_top = Path(args.deg_specific_top)
    deg_pval_dotplot = Path(args.deg_pval_dotplot)
    deg_pval_top1 = Path(args.deg_pval_top1)
    deg_pval_top = Path(args.deg_pval_top)
    deg_max_other_dotplot = Path(args.deg_max_other_dotplot)
    deg_max_other_top1 = Path(args.deg_max_other_top1)
    deg_max_other_top = Path(args.deg_max_other_top)
    deg_filtered = Path(args.deg_filtered)
    deg_all = Path(args.deg_all)
    deg_params = Path(args.deg_params)

    if not figures_dir.exists():
        raise FileNotFoundError(f"Figure directory does not exist: {figures_dir}")
    if not match_report.exists():
        raise FileNotFoundError(f"Gene match report does not exist: {match_report}")

    df = load_gene_records(match_report)
    extra_gene_groups = deg_gene_groups(deg_specific_top1, deg_max_other_top1)
    gene_index = build_index(df, extra_gene_groups)
    cards = build_cards(df, figures_dir, extra_gene_groups)
    overview = build_overview(figures_dir)
    missing_summary = build_missing_summary(df)
    dotplot_tab = build_dotplot_tab(
        merged_umap,
        dotplot_path,
        dotplot_pdf,
        dotplot_gene_report,
        dotplot_cluster_map,
        dotplot_gene_order,
        output_dir,
    )
    deg_tab = build_deg_tab(
        deg_dotplot,
        deg_top1,
        deg_top,
        deg_specific_dotplot,
        deg_specific_top1,
        deg_specific_top,
        deg_pval_dotplot,
        deg_pval_top1,
        deg_pval_top,
        deg_max_other_dotplot,
        deg_max_other_top1,
        deg_max_other_top,
        deg_filtered,
        deg_all,
        deg_params,
        output_dir,
    )

    content = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>RGC Gene UMAP Review</title>
  <style>
    :root {{
      --bg: #f5f6f8;
      --panel: #ffffff;
      --text: #172033;
      --muted: #667085;
      --line: #d7dde7;
      --link: #155db2;
      --highlight: #ffd34d;
      --highlight-bg: #fff7d1;
      --bad: #b42318;
    }}

    html {{ scroll-behavior: smooth; }}
    body {{
      margin: 0;
      font-family: Arial, Helvetica, sans-serif;
      background: var(--bg);
      color: var(--text);
    }}
    header {{
      position: sticky;
      top: 0;
      z-index: 10;
      background: rgba(245, 246, 248, 0.96);
      border-bottom: 1px solid var(--line);
      padding: 12px 18px;
      backdrop-filter: blur(6px);
    }}
    h1 {{ margin: 0 0 4px 0; font-size: 22px; }}
    .note {{ margin: 0; color: var(--muted); font-size: 13px; }}
    .strong {{ color: var(--text); margin-top: 8px; }}
    .toolbar {{
      display: flex;
      gap: 8px;
      align-items: center;
      margin-top: 10px;
      flex-wrap: wrap;
    }}
    .toolbar input {{
      width: min(460px, 100%);
      padding: 8px 10px;
      border: 1px solid var(--line);
      border-radius: 6px;
      font-size: 13px;
      background: white;
    }}
    .tabs {{
      display: flex;
      gap: 6px;
      margin-top: 10px;
      flex-wrap: wrap;
    }}
    .tab-button {{
      border: 1px solid var(--line);
      background: white;
      color: var(--text);
      border-radius: 6px;
      padding: 7px 10px;
      font-size: 13px;
      cursor: pointer;
    }}
    .tab-button.active {{
      border-color: var(--link);
      background: #eef5ff;
      color: var(--link);
      font-weight: 700;
    }}
    .tab-panel {{ display: none; }}
    .tab-panel.active {{ display: block; }}
    .layout {{
      display: grid;
      grid-template-columns: minmax(250px, 330px) minmax(0, 1fr);
      gap: 14px;
      padding: 14px;
    }}
    .index-panel {{
      position: sticky;
      top: 98px;
      align-self: start;
      max-height: calc(100vh - 116px);
      overflow: auto;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px;
    }}
    .index-panel h2, .overview h2 {{
      margin: 0 0 8px 0;
      font-size: 15px;
    }}
    .marker-group {{
      border-top: 1px solid #edf0f5;
      padding: 8px 0;
    }}
    .marker-group:first-child {{ border-top: 0; padding-top: 0; }}
    .marker-group h2 {{
      margin: 0 0 5px 0;
      font-size: 12px;
      color: #344054;
      text-transform: uppercase;
      letter-spacing: 0.03em;
    }}
    .gene-links {{ line-height: 1.35; }}
    .gene-links a, .missing {{
      display: inline-block;
      margin: 0 5px 4px 0;
      font-size: 12px;
    }}
    .gene-links a {{
      color: var(--link);
      text-decoration: none;
      border-bottom: 1px solid transparent;
    }}
    .gene-links a:hover {{ border-bottom-color: var(--link); }}
    .gene-links a.active {{
      background: var(--highlight-bg);
      color: #111827;
      border-bottom-color: var(--highlight);
      border-radius: 4px;
      box-shadow: 0 0 0 3px var(--highlight-bg);
    }}
    .missing {{
      color: #a7acb5;
      text-decoration: line-through;
    }}
    .missing-chips span {{
      display: inline-block;
      margin: 0 5px 5px 0;
      padding: 3px 6px;
      border-radius: 999px;
      background: #fff1f0;
      color: var(--bad);
      font-size: 12px;
    }}
    .content {{
      min-width: 0;
    }}
    .overview, .plot-card {{
      scroll-margin-top: 112px;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px;
      margin-bottom: 12px;
      min-width: 0;
    }}
    .card-head {{
      display: flex;
      gap: 10px;
      align-items: baseline;
      margin-bottom: 8px;
    }}
    .card-head h2 {{
      display: inline-block;
      margin: 0;
      padding: 2px 5px;
      border-radius: 5px;
      font-size: 16px;
      line-height: 1.15;
    }}
    .card-head span {{
      color: var(--muted);
      font-size: 12px;
    }}
    .plot-card:target h2, .plot-card.selected h2 {{
      background: var(--highlight);
      color: #111827;
      box-shadow: 0 0 0 4px var(--highlight-bg);
    }}
    .plot-card.selected {{
      border-color: #f3c547;
      box-shadow: 0 0 0 3px var(--highlight-bg);
    }}
    .image-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(340px, 1fr));
      gap: 10px;
      align-items: start;
    }}
    .image-panel h3 {{
      margin: 0 0 6px 0;
      font-size: 12px;
      color: #344054;
    }}
    .dotplot-card {{
      margin: 14px;
    }}
    .dotplot-grid {{
      grid-template-columns: minmax(360px, 820px);
    }}
    .dotplot-card .image-panel img {{
      width: auto;
      max-width: 100%;
      max-height: 68vh;
      margin: 0 auto;
      object-fit: contain;
    }}
    .dotplot-card > .image-panel img {{
      max-height: 72vh;
    }}
    .deg-comparison-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(420px, 1fr));
      gap: 14px;
      align-items: start;
    }}
    .deg-method-panel {{
      min-width: 0;
    }}
    .deg-method-panel .image-panel img {{
      max-height: 58vh;
    }}
    .method-note {{
      margin: 0 0 12px 0;
      padding: 8px 10px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fbfcfe;
      color: #344054;
      font-size: 12px;
      line-height: 1.4;
    }}
    .image-panel img {{
      display: block;
      width: 100%;
      height: auto;
      border: 1px solid #edf0f5;
      border-radius: 6px;
      background: white;
    }}
    .plot-meta {{
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      align-items: baseline;
      margin-top: 5px;
      color: var(--muted);
      font-size: 11px;
      word-break: break-all;
    }}
    .plot-meta a {{
      color: var(--link);
      white-space: nowrap;
    }}
    .small-table-wrap {{
      margin-top: 12px;
      overflow-x: auto;
    }}
    .small-table-wrap h3 {{
      margin: 0 0 6px 0;
      font-size: 13px;
      color: #344054;
    }}
    .small-table {{
      border-collapse: collapse;
      font-size: 12px;
      min-width: 520px;
    }}
    .small-table th, .small-table td {{
      border: 1px solid var(--line);
      padding: 5px 7px;
      text-align: left;
      background: white;
    }}
    .report-links {{
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      margin-top: 10px;
      font-size: 12px;
    }}
    .report-links a {{ color: var(--link); }}
    .missing-card .missing-note, .absent-image p {{
      color: var(--bad);
      font-size: 13px;
      margin: 4px 0;
    }}
    .top-link {{
      position: fixed;
      right: 16px;
      bottom: 16px;
      z-index: 20;
      background: #202938;
      color: white;
      padding: 8px 11px;
      border-radius: 999px;
      text-decoration: none;
      font-size: 12px;
    }}
    .hidden {{ display: none; }}
    @media (max-width: 980px) {{
      .layout {{ display: block; }}
      .index-panel {{
        position: static;
        max-height: none;
        margin-bottom: 12px;
      }}
    }}
  </style>
</head>
<body id="top">
  <header>
    <h1>RGC Gene UMAP Review</h1>
    <p class="note">Generated {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}. Strict Harmony RGC embedding follows Sherine's batch_subset.py behavior. Gene plots use scaled AnnData expression with use_raw=False, matching Sherine's marker plotting behavior.</p>
    <div class="toolbar">
      <input id="search" type="search" placeholder="Filter genes, e.g. eph, rbpms, contaminant marker">
    </div>
    <nav class="tabs" aria-label="Reviewer tabs">
      <button class="tab-button active" type="button" data-tab="genes-tab">Gene UMAPs</button>
      <button class="tab-button" type="button" data-tab="dotplot-tab">Merged Cluster Dotplot</button>
      <button class="tab-button" type="button" data-tab="deg-tab">DEG Top Markers</button>
    </nav>
  </header>

  <a class="top-link" href="#top">Top</a>

  <main>
    <section class="tab-panel active" id="genes-tab">
      <div class="layout">
        <aside class="index-panel">
          <h2>Gene Index</h2>
          {gene_index}
          {missing_summary}
        </aside>

        <section class="content">
          {overview}
          {cards}
        </section>
      </div>
    </section>

    <section class="tab-panel" id="dotplot-tab">
      {dotplot_tab}
    </section>

    <section class="tab-panel" id="deg-tab">
      {deg_tab}
    </section>
  </main>

  <script>
    const search = document.getElementById('search');
    const cards = Array.from(document.querySelectorAll('.plot-card'));
    const geneLinks = Array.from(document.querySelectorAll('.gene-links a'));
    const tabButtons = Array.from(document.querySelectorAll('.tab-button'));
    const tabPanels = Array.from(document.querySelectorAll('.tab-panel'));

    function showTab(tabId) {{
      for (const panel of tabPanels) {{
        panel.classList.toggle('active', panel.id === tabId);
      }}
      for (const button of tabButtons) {{
        button.classList.toggle('active', button.dataset.tab === tabId);
      }}
    }}

    tabButtons.forEach((button) => {{
      button.addEventListener('click', () => {{
        showTab(button.dataset.tab);
        history.replaceState(null, '', `#${{button.dataset.tab}}`);
      }});
    }});

    function applyFilter() {{
      const query = search.value.trim().toLowerCase();
      for (const card of cards) {{
        const text = (card.dataset.gene || card.textContent).toLowerCase();
        card.classList.toggle('hidden', query && !text.includes(query));
      }}
    }}

    function markSelected(targetId, activeLink) {{
      for (const card of cards) {{
        card.classList.toggle('selected', card.id === targetId);
      }}
      for (const link of geneLinks) {{
        link.classList.toggle('active', link === activeLink);
      }}
    }}

    function scrollToCard(target) {{
      const header = document.querySelector('header');
      const headerOffset = (header ? header.getBoundingClientRect().height : 0) + 18;
      const top = target.getBoundingClientRect().top + window.scrollY - headerOffset;
      window.scrollTo({{ top, behavior: 'smooth' }});
    }}

    search.addEventListener('input', applyFilter);

    geneLinks.forEach((link) => {{
      link.addEventListener('click', (event) => {{
        const targetId = link.dataset.target;
        const target = document.getElementById(targetId);
        if (!target) {{
          return;
        }}
        event.preventDefault();
        search.value = '';
        applyFilter();
        markSelected(targetId, link);
        scrollToCard(target);
        window.setTimeout(() => scrollToCard(target), 350);
        history.replaceState(null, '', `#${{targetId}}`);
      }});
    }});

    window.addEventListener('load', () => {{
      const targetId = window.location.hash.slice(1);
      if (!targetId) {{
        return;
      }}
      if (targetId === 'genes-tab' || targetId === 'dotplot-tab' || targetId === 'deg-tab') {{
        showTab(targetId);
        return;
      }}
      const target = document.getElementById(targetId);
      if (!target) {{
        return;
      }}
      showTab('genes-tab');
      const activeLink = geneLinks.find((link) => link.dataset.target === targetId);
      markSelected(targetId, activeLink);
      scrollToCard(target);
    }});
  </script>
</body>
</html>
"""

    output.write_text(content, encoding="utf-8")
    print(f"Wrote RGC gene UMAP review HTML: {output}")


if __name__ == "__main__":
    main()

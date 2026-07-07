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
import shutil
import struct

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build BC guca1b > 2.0 DEG gene UMAP HTML review.")
    parser.add_argument(
        "--figures-dir",
        default="results/figures/bc_guca1b_gt2_ac_style_gene_umaps",
        help="Directory containing BC figures.",
    )
    parser.add_argument(
        "--tables-dir",
        default="results/tables/bc_guca1b_gt2_ac_style_gene_umaps",
        help="Directory containing BC tables.",
    )
    parser.add_argument(
        "--match-report",
        default="results/tables/bc_guca1b_gt2_ac_style_gene_umaps/bc_guca1b_gt2_deg_top1_gene_match_report.csv",
        help="Gene match report from plot_bc_guca1b_gt2_deg_genes.py.",
    )
    parser.add_argument(
        "--manual-match-report",
        default="results/tables/bc_guca1b_gt2_ac_style_gene_umaps/bc_guca1b_gt2_manual_marker_gene_match_report.csv",
        help="Optional manual marker gene match report.",
    )
    parser.add_argument("--page-title", default="BC guca1b > 2.0 DEG Gene UMAP Review")
    parser.add_argument("--cluster-label", default="BC guca1b > 2.0 Leiden clusters, grouped by leiden")
    parser.add_argument("--output", default=None, help="Output HTML path.")
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


def downloadable_copy(path: Path, output_dir: Path) -> Path:
    downloads_dir = output_dir / "downloads"
    downloads_dir.mkdir(parents=True, exist_ok=True)
    destination = downloads_dir / path.name
    if path.resolve() != destination.resolve():
        shutil.copy2(path, destination)
    return destination


def download_link(label: str, path: Path, output_dir: Path) -> str:
    local_path = downloadable_copy(path, output_dir)
    return f'<a href="{html.escape(href_for(local_path, output_dir))}" download>{html.escape(label)}</a>'


def load_gene_records(match_report: Path) -> pd.DataFrame:
    df = pd.read_csv(match_report)
    for column in ["requested_gene", "matched_gene"]:
        df[column] = df[column].fillna("").astype(str)
    df["present"] = df["present"].astype(bool)
    return df


def load_combined_gene_records(match_report: Path, manual_match_report: Path) -> pd.DataFrame:
    frames = []
    if manual_match_report.exists():
        manual = load_gene_records(manual_match_report)
        manual["source_group"] = "Manual Marker Genes"
        frames.append(manual)
    deg = load_gene_records(match_report)
    deg["source_group"] = "DEG Top1 Genes"
    frames.append(deg)
    combined = pd.concat(frames, ignore_index=True)
    return combined.drop_duplicates(subset=["requested_gene"], keep="first").reset_index(drop=True)


def top1_gene_groups(tables_dir: Path) -> dict[str, list[str]]:
    groups = {}
    for label, filename in [
        ("Preferred Max-Other Top1 DEG", "bc_guca1b_gt2_deg_max_other_top1_marker_per_cluster.csv"),
        ("Specificity-Ranked Top1 DEG", "bc_guca1b_gt2_deg_specific_top1_marker_per_cluster.csv"),
    ]:
        path = tables_dir / filename
        if not path.exists():
            continue
        df = pd.read_csv(path)
        genes = []
        for gene in df.get("names", pd.Series(dtype=str)).dropna().astype(str):
            if gene not in genes:
                genes.append(gene)
        if genes:
            groups[label] = genes
    return groups


def manual_gene_groups(df: pd.DataFrame) -> dict[str, list[str]]:
    if "source_group" not in df.columns:
        return {}
    genes = df.loc[df["source_group"] == "Manual Marker Genes", "requested_gene"].astype(str).tolist()
    return {"Manual Marker Genes": genes} if genes else {}


def build_index(df: pd.DataFrame, groups: dict[str, list[str]]) -> str:
    row_by_requested = {row["requested_gene"]: row for _, row in df.iterrows()}
    blocks = []
    used = set()
    for group_name, genes in groups.items():
        links = []
        for gene in genes:
            row = row_by_requested.get(gene)
            if row is None:
                continue
            used.add(gene)
            label = html.escape(gene)
            anchor = html.escape(make_anchor(gene), quote=True)
            if bool(row["present"]):
                links.append(f'<a href="#{anchor}" data-target="{anchor}" data-gene="{label}">{label}</a>')
            else:
                links.append(f'<span class="missing">{label}</span>')
        if links:
            blocks.append(
                f"""
                <section class="marker-group">
                  <h2>{html.escape(group_name)}</h2>
                  <div class="gene-links">{" ".join(links)}</div>
                </section>
                """
            )
    extras = [gene for gene in df["requested_gene"].astype(str) if gene not in used]
    if extras:
        links = []
        for gene in extras:
            row = row_by_requested[gene]
            label = html.escape(gene)
            anchor = html.escape(make_anchor(gene), quote=True)
            if bool(row["present"]):
                links.append(f'<a href="#{anchor}" data-target="{anchor}" data-gene="{label}">{label}</a>')
            else:
                links.append(f'<span class="missing">{label}</span>')
        blocks.append(
            f"""
            <section class="marker-group">
              <h2>Other DEG Top1 Genes</h2>
              <div class="gene-links">{" ".join(links)}</div>
            </section>
            """
        )
    return "\n".join(blocks)


def find_gene_images(figures_dir: Path, gene: str) -> dict[str, Path]:
    return {
        "All conditions": figures_dir / f"bc_umap_gene_{gene}_all_conditions.png",
        "Split by condition": figures_dir / f"bc_umap_gene_{gene}_split_by_condition.png",
    }


def image_panel(title: str, image_path: Path, output_dir: Path) -> str:
    if not image_path.exists():
        return f"""
        <div class="image-panel absent-image">
          <h3>{html.escape(title)}</h3>
          <p>Missing image: {html.escape(image_path.name)}</p>
        </div>
        """
    href = html.escape(href_for(image_path, output_dir))
    return f"""
    <div class="image-panel">
      <h3>{html.escape(title)}</h3>
      <a href="{href}" target="_blank" rel="noopener">
        <img src="{image_data_uri(image_path)}" alt="{html.escape(title)}" loading="lazy"{image_size_attrs(image_path)}>
      </a>
      <div class="plot-meta">
        <span>{html.escape(image_path.name)}</span>
        <a href="{href}" download>Download PNG</a>
      </div>
    </div>
    """


def gene_order(df: pd.DataFrame, groups: dict[str, list[str]]) -> list[str]:
    ordered = []
    requested = set(df["requested_gene"].astype(str))
    for genes in groups.values():
        for gene in genes:
            if gene in requested and gene not in ordered:
                ordered.append(gene)
    for gene in df["requested_gene"].astype(str):
        if gene not in ordered:
            ordered.append(gene)
    return ordered


def build_cards(df: pd.DataFrame, figures_dir: Path, groups: dict[str, list[str]], output_dir: Path) -> str:
    row_by_requested = {row["requested_gene"]: row for _, row in df.iterrows()}
    cards = []
    for requested_gene in gene_order(df, groups):
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
        panels = "".join(
            image_panel(label, image_path, output_dir)
            for label, image_path in find_gene_images(figures_dir, matched_gene).items()
        )
        matched_note = "" if matched_gene == requested_gene else f"matched as {html.escape(matched_gene)}"
        cards.append(
            f"""
            <article class="plot-card" id="{anchor}" data-gene="{safe_requested} {html.escape(matched_gene)}">
              <div class="card-head">
                <h2>{safe_requested}</h2>
                <span>{matched_note}</span>
              </div>
              <div class="image-grid">{panels}</div>
            </article>
            """
        )
    return "\n".join(cards)


def build_overview(figures_dir: Path, output_dir: Path) -> str:
    names = [
        "bc_umap_by_renamed_samples.png",
        "bc_umap_by_renamed_samples_Control_highlight.png",
        "bc_umap_by_renamed_samples_LD_highlight.png",
        "bc_umap_by_renamed_samples_NMDA_highlight.png",
        "bc_umap_by_leiden.png",
    ]
    panels = [
        image_panel(path.name, path, output_dir)
        for path in [figures_dir / name for name in names]
        if path.exists()
    ]
    return f"""
    <section class="overview" id="overview">
      <h2>BC guca1b > 2.0 Embedding</h2>
      <div class="image-grid">{"".join(panels)}</div>
    </section>
    """


def build_cluster_counts(tables_dir: Path) -> str:
    path = tables_dir / "bc_guca1b_gt2_deg_cluster_cell_counts.csv"
    if not path.exists():
        return ""
    df = pd.read_csv(path).fillna("")
    rows = []
    total = 0
    for row in df.itertuples(index=False):
        cluster = str(row.cluster)
        n_cells = int(row.n_cells)
        total += n_cells
        rows.append(
            f"""
            <tr>
              <td>{html.escape(cluster)}</td>
              <td>{n_cells:,}</td>
            </tr>
            """
        )
    return f"""
    <section class="overview" id="cluster-counts">
      <div class="card-head">
        <h2>Cells Per Cluster</h2>
        <span>Total {total:,} cells</span>
      </div>
      <div class="cluster-count-grid">
        <table class="small-table cluster-count-table">
          <thead><tr><th>Cluster</th><th>Cells</th></tr></thead>
          <tbody>{''.join(rows)}</tbody>
        </table>
      </div>
    </section>
    """


def build_count_table(path: Path, title: str, label: str = "Sample") -> str:
    if not path.exists():
        return ""
    df = pd.read_csv(path).fillna("")
    rows = []
    total = 0
    for row in df.itertuples(index=False):
        name = str(row.sample)
        n_cells = int(row.n_cells)
        total += n_cells
        rows.append(
            f"""
            <tr>
              <td>{html.escape(name)}</td>
              <td>{n_cells:,}</td>
            </tr>
            """
        )
    return f"""
    <section class="overview sample-count-card">
      <div class="card-head">
        <h2>{html.escape(title)}</h2>
        <span>Total {total:,} cells</span>
      </div>
      <div class="cluster-count-grid">
        <table class="small-table cluster-count-table">
          <thead><tr><th>{html.escape(label)}</th><th>Cells</th></tr></thead>
          <tbody>{''.join(rows)}</tbody>
        </table>
      </div>
    </section>
    """


def build_sample_counts(tables_dir: Path) -> str:
    return f"""
    <div class="count-card-grid">
      {build_count_table(tables_dir / "bc_guca1b_gt2_counts_by_9_sample.csv", "Cells Per 9-Sample Label", "Sample")}
      {build_count_table(tables_dir / "bc_guca1b_gt2_counts_by_renamed_samples.csv", "Cells Per Renamed Sample", "Condition")}
    </div>
    """


def build_cluster_condition_counts(tables_dir: Path) -> str:
    path = tables_dir / "bc_guca1b_gt2_cluster_counts_by_renamed_samples.csv"
    if not path.exists():
        return ""
    df = pd.read_csv(path).fillna(0)
    rows = []
    for row in df.itertuples(index=False):
        rows.append(
            f"""
            <tr>
              <td>{html.escape(str(row.cluster))}</td>
              <td>{int(row.Control):,}</td>
              <td>{int(row.LD):,}</td>
              <td>{int(row.NMDA):,}</td>
              <td>{int(row.total):,}</td>
            </tr>
            """
        )
    return f"""
    <section class="overview" id="cluster-condition-counts">
      <div class="card-head">
        <h2>Cells Per Cluster By Condition</h2>
        <span>Control, LD, NMDA</span>
      </div>
      <div class="cluster-count-grid">
        <table class="small-table cluster-count-table">
          <thead><tr><th>Cluster</th><th>Control</th><th>LD</th><th>NMDA</th><th>Total</th></tr></thead>
          <tbody>{''.join(rows)}</tbody>
        </table>
      </div>
    </section>
    """


def format_float(value, digits: int = 3) -> str:
    try:
        return f"{float(value):.{digits}g}"
    except (TypeError, ValueError):
        return ""


def build_top1_table(path: Path, title: str) -> str:
    if not path.exists():
        return ""
    df = pd.read_csv(path).fillna("")
    columns = [
        "group",
        "names",
        "scores",
        "logfoldchanges",
        "pvals_adj",
        "pct_nz_group",
        "pct_nz_reference",
        "specificity_score",
        "max_other_specificity_score",
        "max_other_mean_cluster",
        "max_other_pct_cluster",
    ]
    available = [column for column in columns if column in df.columns]
    header_labels = {
        "group": "Cluster",
        "names": "Top marker",
        "scores": "Score",
        "logfoldchanges": "LogFC",
        "pvals_adj": "Adj. p",
        "pct_nz_group": "Pct cluster",
        "pct_nz_reference": "Pct rest",
        "specificity_score": "Specificity",
        "max_other_specificity_score": "Max-other score",
        "max_other_mean_cluster": "Max mean other",
        "max_other_pct_cluster": "Max pct other",
    }
    rows = []
    for row in df[available].itertuples(index=False):
        values = row._asdict()
        cells = []
        for column in available:
            value = values[column]
            if column not in {"group", "names", "max_other_mean_cluster", "max_other_pct_cluster"}:
                value = format_float(value)
            cells.append(f"<td>{html.escape(str(value))}</td>")
        rows.append(f"<tr>{''.join(cells)}</tr>")
    headers = "".join(f"<th>{html.escape(header_labels.get(column, column))}</th>" for column in available)
    return f"""
    <div class="small-table-wrap">
      <h3>{html.escape(title)}</h3>
      <table class="small-table">
        <thead><tr>{headers}</tr></thead>
        <tbody>{''.join(rows)}</tbody>
      </table>
    </div>
    """


def build_deg_tab(figures_dir: Path, tables_dir: Path, output_dir: Path, cluster_label: str) -> str:
    score_dotplot = figures_dir / "bc_guca1b_gt2_deg_score_top1_marker_dotplot.png"
    specific_dotplot = figures_dir / "bc_guca1b_gt2_deg_specific_top1_marker_dotplot.png"
    max_other_dotplot = figures_dir / "bc_guca1b_gt2_deg_max_other_top1_marker_dotplot.png"
    score_top1 = tables_dir / "bc_guca1b_gt2_deg_score_top1_marker_per_cluster.csv"
    specific_top1 = tables_dir / "bc_guca1b_gt2_deg_specific_top1_marker_per_cluster.csv"
    max_other_top1 = tables_dir / "bc_guca1b_gt2_deg_max_other_top1_marker_per_cluster.csv"
    links = []
    for label, path in [
        ("Download Wilcoxon score top1 CSV", score_top1),
        ("Download Wilcoxon score top100/cluster CSV", tables_dir / "bc_guca1b_gt2_deg_score_top100_markers.csv"),
        ("Download specificity top1 CSV", specific_top1),
        ("Download specificity top100/cluster CSV", tables_dir / "bc_guca1b_gt2_deg_specific_top100_markers.csv"),
        ("Download max-other top1 CSV", max_other_top1),
        ("Download max-other top100/cluster CSV", tables_dir / "bc_guca1b_gt2_deg_max_other_top100_markers.csv"),
        ("Download filtered DEG CSV", tables_dir / "bc_guca1b_gt2_deg_markers_filtered.csv"),
        ("Download all DEG CSV", tables_dir / "bc_guca1b_gt2_deg_markers_all.csv"),
        ("Download DEG parameters CSV", tables_dir / "bc_guca1b_gt2_deg_parameters.csv"),
    ]:
        if path.exists():
            links.append(download_link(label, path, output_dir))
    link_block = f'<div class="report-links">{" ".join(links)}</div>' if links else ""
    return f"""
    <article class="plot-card deg-card">
      <div class="card-head">
        <h2>BC DEG Top Markers</h2>
        <span>{html.escape(cluster_label)}</span>
      </div>
      <div class="method-note">
        <strong>Wilcoxon score ranking:</strong>
        top marker is selected by the Scanpy Wilcoxon rank_genes_groups score after the same initial filters.
        <br>
        <strong>Specificity formula:</strong>
        specificity_score = logfoldchanges x max(pct_nz_group - pct_nz_reference, 0).
        <br>
        <strong>Preferred max-other formula:</strong>
        max_other_specificity_score = logfoldchanges x max(pct_group - max_other_pct, 0) x max(mean_group - max_other_mean, 0).
      </div>
      <div class="deg-comparison-grid">
        <section>
          {image_panel("Wilcoxon score top1 marker dotplot", score_dotplot, output_dir)}
          {build_top1_table(score_top1, "Wilcoxon score top1 per BC cluster")}
        </section>
        <section>
          {image_panel("Specificity-ranked top1 marker dotplot", specific_dotplot, output_dir)}
          {build_top1_table(specific_top1, "Specificity-ranked top1 per BC cluster")}
        </section>
        <section>
          {image_panel("Preferred max-other top1 marker dotplot", max_other_dotplot, output_dir)}
          {build_top1_table(max_other_top1, "Preferred max-other top1 per BC cluster")}
        </section>
      </div>
      {link_block}
    </article>
    """


def css() -> str:
    return """
    :root { --ink: #172033; --muted: #667085; --line: #d8e0ea; --panel: #fff; --accent: #0b63ce; --hi: #ffd84d; }
    * { box-sizing: border-box; }
    body { margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: var(--ink); background: #f6f8fb; }
    header { position: sticky; top: 0; z-index: 10; padding: 10px 16px; background: rgba(255,255,255,0.96); border-bottom: 1px solid var(--line); }
    h1 { margin: 0 0 8px; font-size: 22px; }
    .tabs { display: flex; gap: 8px; flex-wrap: wrap; }
    .tab-button { border: 1px solid var(--line); background: #fff; color: var(--ink); padding: 8px 10px; border-radius: 6px; cursor: pointer; font-weight: 700; }
    .tab-button.active { background: var(--accent); border-color: var(--accent); color: #fff; }
    main { padding: 14px; }
    .gene-layout { display: grid; grid-template-columns: 310px minmax(0, 1fr); gap: 14px; }
    aside { position: sticky; top: 72px; max-height: calc(100vh - 86px); overflow: auto; background: #fff; border: 1px solid var(--line); border-radius: 8px; padding: 12px; }
    .search { width: 100%; padding: 9px 10px; border: 1px solid var(--line); border-radius: 6px; font-size: 15px; }
    .marker-group { border-top: 1px solid #e8edf3; padding-top: 12px; margin-top: 12px; }
    .marker-group h2 { margin: 0 0 8px; font-size: 15px; color: #344054; text-transform: uppercase; }
    .gene-links { display: flex; flex-wrap: wrap; gap: 8px 12px; font-size: 16px; }
    .gene-links a { color: #075ab8; text-decoration: none; }
    .gene-links a.active { background: var(--hi); color: #111827; border-radius: 4px; padding: 1px 3px; }
    .missing { color: #a0a7b3; text-decoration: line-through; }
    .tab-panel { display: none; }
    .tab-panel.active { display: block; }
    .plot-card, .overview { background: #fff; border: 1px solid var(--line); border-radius: 8px; padding: 12px; margin-bottom: 14px; }
    .card-head { display: flex; align-items: baseline; justify-content: space-between; gap: 12px; border-bottom: 1px solid #e8edf3; padding-bottom: 8px; margin-bottom: 12px; }
    .card-head h2, .overview h2 { margin: 0; font-size: 20px; }
    .card-head span { color: var(--muted); font-size: 13px; }
    .image-grid { display: grid; grid-template-columns: repeat(2, minmax(280px, 1fr)); gap: 12px; }
    .image-panel { min-width: 0; border: 1px solid #e8edf3; border-radius: 6px; padding: 8px; background: #fff; }
    .image-panel h3 { margin: 0 0 8px; font-size: 14px; color: #344054; }
    .image-panel img { display: block; width: 100%; height: auto; object-fit: contain; }
    .plot-meta { display: flex; justify-content: space-between; gap: 8px; margin-top: 6px; color: var(--muted); font-size: 12px; overflow-wrap: anywhere; }
    .plot-meta a, .report-links a { color: #075ab8; font-weight: 700; text-decoration: none; }
    .method-note { border: 1px solid #e8edf3; border-radius: 6px; background: #fbfcfe; padding: 10px; margin-bottom: 12px; color: #344054; }
    .deg-comparison-grid { display: grid; grid-template-columns: repeat(3, minmax(520px, 1fr)); gap: 14px; align-items: start; overflow-x: auto; padding-bottom: 4px; }
    .deg-card .image-panel { overflow-x: auto; }
    .deg-card .image-panel img { min-width: 500px; }
    .cluster-count-grid { max-height: 360px; overflow: auto; border: 1px solid #e8edf3; border-radius: 6px; }
    .cluster-count-table { font-size: 13px; }
    .count-card-grid { display: grid; grid-template-columns: repeat(2, minmax(260px, 1fr)); gap: 12px; margin-bottom: 14px; }
    .sample-count-card { margin-bottom: 0; }
    .small-table-wrap { overflow: auto; max-height: 480px; border: 1px solid #e8edf3; border-radius: 6px; margin-top: 10px; }
    .small-table-wrap h3 { margin: 8px; font-size: 14px; }
    .small-table { width: 100%; border-collapse: collapse; font-size: 12px; }
    .small-table th, .small-table td { border-top: 1px solid #eef2f6; padding: 5px 7px; text-align: left; white-space: nowrap; }
    .report-links { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 12px; }
    .highlight { outline: 3px solid var(--hi); outline-offset: 2px; }
    @media (max-width: 1000px) {
      .gene-layout { grid-template-columns: 1fr; }
      aside { position: static; max-height: none; }
      .image-grid, .deg-comparison-grid, .count-card-grid { grid-template-columns: 1fr; }
      .deg-card .image-panel img { min-width: 0; }
    }
    """


def javascript() -> str:
    return """
    const buttons = document.querySelectorAll('.tab-button');
    const panels = document.querySelectorAll('.tab-panel');
    buttons.forEach(button => {
      button.addEventListener('click', () => {
        buttons.forEach(b => b.classList.remove('active'));
        panels.forEach(p => p.classList.remove('active'));
        button.classList.add('active');
        document.getElementById(button.dataset.tab).classList.add('active');
      });
    });

    const links = document.querySelectorAll('.gene-links a');
    links.forEach(link => {
      link.addEventListener('click', event => {
        event.preventDefault();
        const target = document.getElementById(link.dataset.target);
        if (!target) return;
        links.forEach(item => item.classList.remove('active'));
        link.classList.add('active');
        target.scrollIntoView({ behavior: 'smooth', block: 'start' });
        target.classList.add('highlight');
        setTimeout(() => target.classList.remove('highlight'), 1600);
        history.replaceState(null, '', '#' + link.dataset.target);
      });
    });

    const search = document.getElementById('gene-search');
    if (search) {
      search.addEventListener('input', () => {
        const query = search.value.trim().toLowerCase();
        document.querySelectorAll('.plot-card[data-gene]').forEach(card => {
          card.style.display = card.dataset.gene.toLowerCase().includes(query) ? '' : 'none';
        });
      });
    }
    """


def main() -> None:
    args = parse_args()
    figures_dir = Path(args.figures_dir)
    tables_dir = Path(args.tables_dir)
    output = Path(args.output) if args.output else figures_dir / "bc_guca1b_gt2_ac_style_gene_umap_review.html"
    output_dir = output.parent
    match_report = Path(args.match_report)
    manual_match_report = Path(args.manual_match_report)
    df = load_combined_gene_records(match_report, manual_match_report)
    groups = {**manual_gene_groups(df), **top1_gene_groups(tables_dir)}

    page_title = args.page_title
    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(page_title)}</title>
  <style>{css()}</style>
</head>
<body>
  <header>
    <h1>{html.escape(page_title)}</h1>
    <nav class="tabs">
      <button class="tab-button active" data-tab="gene-tab">Gene UMAPs</button>
      <button class="tab-button" data-tab="deg-tab">DEG Top Markers</button>
    </nav>
  </header>
  <main>
    <section>
      <div id="gene-tab" class="tab-panel active">
        <div class="gene-layout">
          <aside>
            <input id="gene-search" class="search" type="search" placeholder="Search genes">
            {build_index(df, groups)}
          </aside>
          <section>
            {build_overview(figures_dir, output_dir)}
            {build_cluster_counts(tables_dir)}
            {build_sample_counts(tables_dir)}
            {build_cluster_condition_counts(tables_dir)}
            {build_cards(df, figures_dir, groups, output_dir)}
          </section>
        </div>
      </div>
      <div id="deg-tab" class="tab-panel">
        {build_deg_tab(figures_dir, tables_dir, output_dir, args.cluster_label)}
      </div>
    </section>
  </main>
  <footer style="padding: 10px 14px; color: #667085; font-size: 12px;">
    Generated {html.escape(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))}
  </footer>
  <script>{javascript()}</script>
</body>
</html>
"""
    output.write_text(document, encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()

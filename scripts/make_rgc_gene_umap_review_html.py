#!/usr/bin/env python
from __future__ import annotations

import argparse
import base64
from datetime import datetime
import html
import mimetypes
from pathlib import Path
import re

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
    return parser.parse_args()


def make_anchor(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text)


def image_data_uri(path: Path) -> str:
    mime_type = mimetypes.guess_type(path.name)[0] or "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def load_gene_records(match_report: Path) -> pd.DataFrame:
    df = pd.read_csv(match_report)
    for col in ["requested_gene", "matched_gene"]:
        if col in df:
            df[col] = df[col].fillna("").astype(str)
    if "present" in df:
        df["present"] = df["present"].astype(bool)
    return df


def requested_gene_order(df: pd.DataFrame) -> list[str]:
    ordered = []
    requested = set(df["requested_gene"].astype(str))
    for genes in REQUESTED_GROUPS.values():
        for gene in genes:
            if gene in requested and gene not in ordered:
                ordered.append(gene)
    for gene in df["requested_gene"].astype(str):
        if gene not in ordered:
            ordered.append(gene)
    return ordered


def build_index(df: pd.DataFrame) -> str:
    row_by_requested = {
        row["requested_gene"]: row for _, row in df.iterrows()
    }
    blocks = []
    used = set()
    for group_name, genes in REQUESTED_GROUPS.items():
        links = []
        for gene in genes:
            if gene not in row_by_requested:
                continue
            used.add(gene)
            row = row_by_requested[gene]
            label = html.escape(gene)
            if bool(row["present"]):
                matched = html.escape(row["matched_gene"])
                links.append(
                    f'<a href="#{make_anchor(gene)}" data-gene="{label}" '
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
                links.append(f'<a href="#{make_anchor(gene)}" data-gene="{label}">{label}</a>')
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


def build_cards(df: pd.DataFrame, figures_dir: Path) -> str:
    cards = []
    row_by_requested = {row["requested_gene"]: row for _, row in df.iterrows()}
    for requested_gene in requested_gene_order(df):
        row = row_by_requested[requested_gene]
        safe_requested = html.escape(requested_gene)
        anchor = make_anchor(requested_gene)
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
                    <img src="{image_data_uri(image_path)}" alt="{safe_requested} {html.escape(label)}" loading="lazy">
                  </a>
                  <div class="plot-meta">{html.escape(image_path.name)}</div>
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
    fallback_names = ["rgc_umap_by_renamed_samples_strict.png", "rgc_umap_by_rgc_leiden_strict.png"]
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
                    <img src="{image_data_uri(path)}" alt="{html.escape(path.name)}" loading="lazy">
                  </a>
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


def main() -> None:
    args = parse_args()
    figures_dir = Path(args.figures_dir)
    match_report = Path(args.match_report)
    output = Path(args.output) if args.output else figures_dir / "rgc_gene_umap_review.html"

    if not figures_dir.exists():
        raise FileNotFoundError(f"Figure directory does not exist: {figures_dir}")
    if not match_report.exists():
        raise FileNotFoundError(f"Gene match report does not exist: {match_report}")

    df = load_gene_records(match_report)
    gene_index = build_index(df)
    cards = build_cards(df, figures_dir)
    overview = build_overview(figures_dir)
    missing_summary = build_missing_summary(df)

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
    .plot-card:target h2 {{
      background: var(--highlight);
      color: #111827;
      box-shadow: 0 0 0 4px var(--highlight-bg);
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
    .image-panel img {{
      display: block;
      width: 100%;
      height: auto;
      border: 1px solid #edf0f5;
      border-radius: 6px;
      background: white;
    }}
    .plot-meta {{
      margin-top: 5px;
      color: var(--muted);
      font-size: 11px;
      word-break: break-all;
    }}
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
  </header>

  <a class="top-link" href="#top">Top</a>

  <main class="layout">
    <aside class="index-panel">
      <h2>Gene Index</h2>
      {gene_index}
      {missing_summary}
    </aside>

    <section class="content">
      {overview}
      {cards}
    </section>
  </main>

  <script>
    const search = document.getElementById('search');
    const cards = Array.from(document.querySelectorAll('.plot-card'));
    search.addEventListener('input', () => {{
      const query = search.value.trim().toLowerCase();
      for (const card of cards) {{
        const text = (card.dataset.gene || card.textContent).toLowerCase();
        card.classList.toggle('hidden', query && !text.includes(query));
      }}
    }});
  </script>
</body>
</html>
"""

    output.write_text(content, encoding="utf-8")
    print(f"Wrote RGC gene UMAP review HTML: {output}")


if __name__ == "__main__":
    main()

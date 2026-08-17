#!/usr/bin/env python3
"""
Build a self-contained HTML deck from Cosmos Transfer inference outputs.

For each clip shows: input | edge-rainy | depth-rainy | multi-rainy | edge-snowy | depth-snowy | multi-snowy
Prompts shown once at top. All video paths are relative so the deck is portable.

Usage:
    python3.10 05_build_html_deck.py [--out tartandrive/outputs/deck]
"""

import argparse
import shutil
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
DATA_DIR   = SCRIPT_DIR / "data"
SPECS_DIR  = SCRIPT_DIR / "specs" / "chunks"
PROMPTS_DIR = SCRIPT_DIR / "prompts"

EDGE_DIR  = SCRIPT_DIR / "outputs" / "chunks" / "edge"
DEPTH_DIR = SCRIPT_DIR / "outputs" / "chunks" / "depth"
MULTI_DIR = SCRIPT_DIR / "outputs" / "chunks" / "multi"

SEQUENCES = [
    "figure_8_2023-09-13-17-24-26",
    "2023-11-14-15-02-21_figure_8",
    "turnpike_2023-09-12-12-53-32",
    "turnpike_2023-09-13-18-15-03",
    "2023-10-26-14-42-35_turnpike_afternoon_fall",
    "turnpike_flat_2023-09-12-13-42-26",
    "turnpike_warehouse_2023-09-13-19-03-05",
    "gupta_2023-09-14-11-12-19",
    "2023-11-14-14-52-22_gupta",
    "meadows_2023-09-14-12-08-57",
    "sam_loop_2023-09-27-13-04-54",
    "slag_heap_skydio_2023-09-14-12-32-13",
]

COLS = [
    ("input",       "Input",          "b-input"),
    ("edge_rainy",  "Edge / Rainy",   "b-edge b-rainy"),
    ("depth_rainy", "Depth / Rainy",  "b-depth b-rainy"),
    ("multi_rainy", "Multi / Rainy",  "b-multi b-rainy"),
    ("edge_snowy",  "Edge / Snowy",   "b-edge b-snowy"),
    ("depth_snowy", "Depth / Snowy",  "b-depth b-snowy"),
    ("multi_snowy", "Multi / Snowy",  "b-multi b-snowy"),
]


def find_video(key: str, seq: str, chunk_idx: int) -> Path | None:
    chunk = f"chunk_{chunk_idx:03d}"
    if key == "input":
        p = DATA_DIR / seq / "chunks" / f"{chunk}_input.mp4"
    elif key == "edge_rainy":
        p = EDGE_DIR / f"{seq}_{chunk}_edge_rainy.mp4"
    elif key == "edge_snowy":
        p = EDGE_DIR / f"{seq}_{chunk}_edge_snowy.mp4"
    elif key == "depth_rainy":
        p = DEPTH_DIR / f"{seq}_{chunk}_depth_rainy.mp4"
    elif key == "depth_snowy":
        p = DEPTH_DIR / f"{seq}_{chunk}_depth_snowy.mp4"
    elif key == "multi_rainy":
        p = MULTI_DIR / f"{seq}_{chunk}_multi_rainy.mp4"
    elif key == "multi_snowy":
        p = MULTI_DIR / f"{seq}_{chunk}_multi_snowy.mp4"
    else:
        return None
    return p if p.exists() else None


def video_tag(rel_path: str) -> str:
    if rel_path is None:
        return '<div class="missing">not generated yet</div>'
    return f'<video autoplay loop muted playsinline><source src="{rel_path}" type="video/mp4"></video>'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="tartandrive/outputs/deck",
                        help="Output directory for the deck (default: tartandrive/outputs/deck)")
    args = parser.parse_args()

    out_dir = Path(args.out)
    videos_dir = out_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    rainy_prompt = (PROMPTS_DIR / "rainy.txt").read_text().strip()
    snowy_prompt = (PROMPTS_DIR / "snowy.txt").read_text().strip()

    # Collect all clips and copy videos
    clips = []  # list of dicts: {seq, chunk_idx, videos: {key: rel_path}}

    for seq in SEQUENCES:
        chunks_dir = DATA_DIR / seq / "chunks"
        if not chunks_dir.exists():
            continue
        input_chunks = sorted(chunks_dir.glob("chunk_*_input.mp4"))
        for inp in input_chunks:
            chunk_idx = int(inp.stem.split("_")[1])
            # First check which videos exist (without copying)
            sources = {}
            for key, _, _ in COLS:
                sources[key] = find_video(key, seq, chunk_idx)

            # Skip clips with no outputs at all
            has_output = any(
                sources.get(k) for k, _, _ in COLS if k != "input"
            )
            if not has_output:
                continue

            # Now copy and record paths
            row = {"seq": seq, "chunk": chunk_idx, "videos": {}}
            for key, _, _ in COLS:
                src = sources[key]
                if src:
                    dst_name = f"{seq}_chunk{chunk_idx:03d}_{key}.mp4"
                    dst = videos_dir / dst_name
                    if not dst.exists():
                        shutil.copy2(src, dst)
                    row["videos"][key] = f"videos/{dst_name}"
                else:
                    row["videos"][key] = None

            clips.append(row)

    print(f"Found {len(clips)} clips, building deck...")

    # Group clips by sequence for section headers
    by_seq = {}
    for clip in clips:
        by_seq.setdefault(clip["seq"], []).append(clip)

    # Build HTML
    col_headers = "".join(f'<th>{label}</th>' for _, label, _ in COLS)

    rows_html = ""
    for seq, seq_clips in by_seq.items():
        seq_label = seq.replace("_", " ")
        rows_html += f'<tr class="seq-header"><td colspan="{len(COLS)}">{seq_label}</td></tr>\n'
        for clip in seq_clips:
            cells = ""
            for key, _, badge_cls in COLS:
                rel = clip["videos"].get(key)
                cells += f'<td class="vcell">{video_tag(rel)}</td>'
            rows_html += f"<tr>{cells}</tr>\n"

    n_complete = sum(1 for c in clips for key, _, _ in COLS if key != "input" and c["videos"].get(key))
    n_total = len(clips) * (len(COLS) - 1)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Cosmos Transfer 2.5 — TartanDrive2 Weather Augmentation</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}

  body {{
    background: #0c0c0c; color: #ddd;
    font-family: 'Segoe UI', system-ui, sans-serif;
    padding: 28px 20px;
  }}

  header {{ margin-bottom: 24px; border-bottom: 1px solid #1e1e1e; padding-bottom: 16px; }}
  header h1 {{ font-size: 1.3rem; font-weight: 600; color: #76b900; }}
  header p  {{ margin-top: 4px; font-size: 0.78rem; color: #555; }}

  .prompts {{
    display: grid; grid-template-columns: 1fr 1fr; gap: 12px;
    margin-bottom: 28px;
  }}
  .prompt-box {{
    background: #111; border: 1px solid #1e1e1e;
    border-radius: 6px; padding: 12px 14px;
    font-size: 0.82rem; line-height: 1.6; color: #999;
  }}
  .prompt-box .label {{
    font-size: 0.64rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.1em; margin-bottom: 6px;
  }}
  .label-rainy {{ color: #6aadff; }}
  .label-snowy {{ color: #aaddff; }}

  .stats {{ font-size: 0.75rem; color: #555; margin-bottom: 20px; }}

  table {{
    width: 100%; border-collapse: collapse;
    table-layout: fixed;
  }}

  th {{
    padding: 6px 4px; font-size: 0.63rem; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.08em;
    color: #555; border-bottom: 1px solid #1a1a1a;
    background: #0c0c0c; position: sticky; top: 0; z-index: 10;
    white-space: nowrap;
  }}

  tr.seq-header td {{
    background: #111; color: #444; font-size: 0.68rem;
    font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em;
    padding: 8px 6px; border-top: 2px solid #1a1a1a;
    border-bottom: 1px solid #1a1a1a;
  }}

  td.vcell {{
    padding: 3px; vertical-align: top;
    border-bottom: 1px solid #141414;
  }}

  td.vcell video {{
    width: 100%; display: block; background: #000;
    border-radius: 3px;
  }}

  .missing {{
    background: #0e0e0e; color: #333; font-size: 0.65rem;
    text-align: center; padding: 20px 4px; border-radius: 3px;
    font-style: italic;
  }}
</style>
</head>
<body>

<header>
  <h1>Cosmos Transfer 2.5 &mdash; TartanDrive2 Weather Augmentation</h1>
  <p>Sequences: turnpike_flat (5) + sam_loop (2) &nbsp;&bull;&nbsp; 86 clips &nbsp;&bull;&nbsp;
     Edge / Depth / Multicontrol &nbsp;&bull;&nbsp; Rainy + Snowy</p>
</header>

<div class="prompts">
  <div class="prompt-box">
    <div class="label label-rainy">Rainy Prompt</div>
    {rainy_prompt}
  </div>
  <div class="prompt-box">
    <div class="label label-snowy">Snowy Prompt</div>
    {snowy_prompt}
  </div>
</div>

<p class="stats">{n_complete} / {n_total} outputs generated &nbsp;&bull;&nbsp; {len(clips)} clips total</p>

<table>
  <thead><tr>{col_headers}</tr></thead>
  <tbody>
{rows_html}
  </tbody>
</table>

</body>
</html>"""

    out_html = out_dir / "index.html"
    out_html.write_text(html)
    print(f"Deck written to: {out_html}")
    print(f"Videos copied to: {videos_dir}")
    print(f"\nTo package for download:")
    print(f"  tar -czf cosmos_tartandrive_deck.tar.gz -C {out_dir.parent} {out_dir.name}/")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Extract 20s clips aligned to Alpamayo val keyframes from input/edge/depth videos,
then generate multicontrol rainy inference specs for each keyframe.

For each keyframe (clip_id, t0_us):
  - start = t0_us/1e6 - 1.6s  (1.6s history before prediction point)
  - extract 200 frames (20s @ 10fps) from input, edge, depth
  - write one multicontrol rainy spec

Output:
  tartandrive/data/{clip_id}/kf_chunks/kf_{idx:03d}_{input|edge|depth}.mp4
  tartandrive/specs/kf_multi/kf_{clip_id}_{idx:03d}_multi_rainy.json
  tartandrive/specs/kf_multi/batch_gpu{0|1|2}.txt

Usage:
  python3.10 06_extract_kf_clips_and_specs.py [--n-gpus 3]
"""

import argparse
import json
import subprocess
from pathlib import Path

SCRIPT_DIR   = Path(__file__).parent
DATA_DIR     = SCRIPT_DIR / "data"
PROMPTS_DIR  = SCRIPT_DIR / "prompts"
SPECS_DIR    = SCRIPT_DIR / "specs" / "kf_multi"
VAL_JSON     = Path("/raid/home/ruchaa/projects/alpamayo-coc-autolabeler/data/td2_sft/val_examples.json")

FPS          = 10
HISTORY_SEC  = 1.6   # seconds of history before t0
CLIP_FRAMES  = 200   # 20s @ 10fps

SKIP_TERRAINS = {"sam_loop", "slag_heap"}  # excluded from generation


def extract_clip(src: Path, out_path: Path, start_sec: float, n_frames: int) -> None:
    if out_path.exists():
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "warning",
            "-ss", f"{start_sec:.6f}",
            "-i", str(src),
            "-frames:v", str(n_frames),
            "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            "-pix_fmt", "yuv420p",
            str(out_path),
        ],
        check=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-gpus", type=int, default=3)
    args = parser.parse_args()

    SPECS_DIR.mkdir(parents=True, exist_ok=True)

    with open(VAL_JSON) as f:
        keyframes = json.load(f)

    print(f"Processing {len(keyframes)} keyframes across {len(set(e['clip_id'] for e in keyframes))} clips")

    specs = []
    skipped = 0

    for idx, kf in enumerate(keyframes):
        clip_id = kf["clip_id"]

        # Skip excluded terrains
        terrain = next((t for t in SKIP_TERRAINS if t in clip_id), None)
        if terrain:
            continue

        t0_sec  = kf["t0_us"] / 1e6
        start_sec = max(0.0, t0_sec - HISTORY_SEC)

        seq_dir   = DATA_DIR / clip_id
        kf_dir    = seq_dir / "kf_chunks"
        label     = f"kf_{idx:03d}"

        inp_src   = seq_dir / "input.mp4"
        edg_src   = seq_dir / "edge.mp4"
        dep_src   = seq_dir / "depth.mp4"

        if not inp_src.exists() or not edg_src.exists() or not dep_src.exists():
            print(f"[SKIP] {clip_id} kf{idx:03d} — missing source videos")
            skipped += 1
            continue

        inp_out = kf_dir / f"{label}_input.mp4"
        edg_out = kf_dir / f"{label}_edge.mp4"
        dep_out = kf_dir / f"{label}_depth.mp4"

        extract_clip(inp_src, inp_out, start_sec, CLIP_FRAMES)
        extract_clip(edg_src, edg_out, start_sec, CLIP_FRAMES)
        extract_clip(dep_src, dep_out, start_sec, CLIP_FRAMES)

        spec_name = f"{clip_id}_{label}_multi_rainy"
        spec_path = SPECS_DIR / f"{spec_name}.json"
        spec = {
            "name": spec_name,
            "video_path": str(inp_out.resolve()),
            "prompt_path": str((PROMPTS_DIR / "rainy.txt").resolve()),
            "guidance": 5.0,
            "edge":  {"control_path": str(edg_out.resolve()), "control_weight": 1.0},
            "depth": {"control_path": str(dep_out.resolve()), "control_weight": 1.0},
        }
        with open(spec_path, "w") as f:
            json.dump(spec, f, indent=4)
        specs.append(spec_path)

        if idx % 50 == 0:
            print(f"  [{idx}/{len(keyframes)}] {clip_id} t0={t0_sec:.1f}s")

    print(f"\n{len(specs)} specs written ({skipped} skipped)")

    # Split specs across GPUs
    n = len(specs)
    size = (n + args.n_gpus - 1) // args.n_gpus
    for gpu in range(args.n_gpus):
        chunk = specs[gpu * size : (gpu + 1) * size]
        batch_path = SPECS_DIR / f"batch_gpu{gpu}.txt"
        with open(batch_path, "w") as f:
            for p in chunk:
                f.write(str(p.resolve()) + "\n")
        print(f"  GPU{gpu}: {len(chunk)} specs → {batch_path}")


if __name__ == "__main__":
    main()

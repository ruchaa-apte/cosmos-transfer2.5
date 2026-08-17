#!/usr/bin/env python3
"""
Split input.mp4, edge.mp4, and depth.mp4 (if present) for each TartanDrive2 sequence
into ≤15s chunks, then generate inference spec JSONs for every chunk × control × weather.

Control modalities produced:
  - edge  (always, from 02_extract_edge_control.py)
  - depth (when depth.mp4 exists, from 02b_extract_depth_control.py)

Usage:
    python3.10 04_split_and_regen_specs.py [--max-seconds 15] [--guidance 5] [--control-weight 1.0]

Output:
    tartandrive/data/{sequence}/chunks/chunk_NNN_{input|edge|depth}.mp4
    tartandrive/specs/chunks/{sequence}_chunk_NNN_{edge|depth}_{rainy|snowy}.json
    tartandrive/specs/chunks/batch_edge.txt
    tartandrive/specs/chunks/batch_depth.txt
    tartandrive/specs/chunks/batch_all.txt
"""

import argparse
import json
import subprocess
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / "data"
SPECS_DIR = SCRIPT_DIR / "specs" / "chunks"
PROMPTS_DIR = SCRIPT_DIR / "prompts"

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

WEATHER_CONDITIONS = ["rainy", "snowy"]


def get_duration(video_path: Path) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)],
        capture_output=True, text=True, check=True,
    )
    return float(result.stdout.strip())


def get_frame_count(video_path: Path) -> int:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=nb_frames",
         "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)],
        capture_output=True, text=True, check=True,
    )
    return int(result.stdout.strip())


def get_fps(video_path: Path) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=r_frame_rate",
         "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)],
        capture_output=True, text=True, check=True,
    )
    num, den = result.stdout.strip().split("/")
    return int(num) / int(den)


def split_video(src: Path, out_dir: Path, stem: str, max_frames: int, fps: float) -> list[Path]:
    """Split src into exact max_frames chunks. Tail chunk shorter than max_frames is dropped."""
    out_dir.mkdir(parents=True, exist_ok=True)

    existing = sorted(out_dir.glob(f"chunk_*_{stem}.mp4"))
    if existing:
        return existing

    total = get_frame_count(src)
    chunk_paths = []

    for i, start_frame in enumerate(range(0, total, max_frames)):
        n_frames = total - start_frame
        if n_frames < max_frames:
            break  # drop tail clip shorter than min duration
        out_path = out_dir / f"chunk_{i:03d}_{stem}.mp4"
        subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "warning",
                "-ss", f"{start_frame / fps:.6f}",
                "-i", str(src),
                "-frames:v", str(max_frames),
                "-c:v", "libx264", "-preset", "fast", "-crf", "18",
                "-pix_fmt", "yuv420p",
                str(out_path),
            ],
            check=True,
        )
        chunk_paths.append(out_path)

    return chunk_paths


def write_batch_file(path: Path, spec_paths: list[Path]) -> None:
    with open(path, "w") as f:
        for p in spec_paths:
            f.write(str(p.resolve()) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-seconds", type=int, default=20,
                        help="Clip length in seconds — clips shorter than this are dropped (default: 20)")
    parser.add_argument("--guidance", type=float, default=5.0,
                        help="CFG guidance scale (default: 5.0)")
    parser.add_argument("--control-weight", type=float, default=1.0,
                        help="Control signal weight for both edge and depth (default: 1.0)")
    args = parser.parse_args()

    SPECS_DIR.mkdir(parents=True, exist_ok=True)

    edge_specs, depth_specs, multi_specs = [], [], []
    skipped = []

    for seq in SEQUENCES:
        seq_dir = DATA_DIR / seq
        input_mp4 = seq_dir / "input.mp4"
        edge_mp4 = seq_dir / "edge.mp4"
        depth_mp4 = seq_dir / "depth.mp4"
        chunks_dir = seq_dir / "chunks"

        if not input_mp4.exists() or not edge_mp4.exists():
            missing = [m for m, p in [("input.mp4", input_mp4), ("edge.mp4", edge_mp4)] if not p.exists()]
            print(f"[SKIP] {seq} — missing: {', '.join(missing)}")
            skipped.append(seq)
            continue

        has_depth = depth_mp4.exists()
        dur = get_duration(input_mp4)
        fps = get_fps(input_mp4)
        max_frames = int(args.max_seconds * fps)
        modalities = "edge" + (" + depth" if has_depth else " (no depth.mp4 yet)")
        print(f"[INFO] {seq} — {dur:.1f}s @ {fps:.0f}fps → ≤{args.max_seconds}s chunks | {modalities}")

        input_chunks = split_video(input_mp4, chunks_dir, "input", max_frames, fps)
        edge_chunks = split_video(edge_mp4, chunks_dir, "edge", max_frames, fps)

        if len(input_chunks) != len(edge_chunks):
            print(f"[WARN] {seq} — chunk count mismatch ({len(input_chunks)} input vs {len(edge_chunks)} edge), skipping")
            skipped.append(seq)
            continue

        depth_chunks = []
        if has_depth:
            depth_chunks = split_video(depth_mp4, chunks_dir, "depth", max_frames, fps)
            if len(depth_chunks) != len(input_chunks):
                print(f"[WARN] {seq} — depth chunk count mismatch ({len(depth_chunks)} vs {len(input_chunks)} input), skipping depth")
                depth_chunks = []

        print(f"        → {len(input_chunks)} chunks"
              + (f" (edge + depth + multi)" if depth_chunks else " (edge only)"))

        for i, (inp_chunk, edg_chunk) in enumerate(zip(input_chunks, edge_chunks)):
            for weather in WEATHER_CONDITIONS:
                # --- edge spec ---
                spec_name = f"{seq}_chunk_{i:03d}_edge_{weather}"
                spec_path = SPECS_DIR / f"{spec_name}.json"
                spec = {
                    "name": spec_name,
                    "video_path": str(inp_chunk.resolve()),
                    "prompt_path": str((PROMPTS_DIR / f"{weather}.txt").resolve()),
                    "guidance": args.guidance,
                    "edge": {
                        "control_path": str(edg_chunk.resolve()),
                        "control_weight": args.control_weight,
                    },
                }
                with open(spec_path, "w") as f:
                    json.dump(spec, f, indent=4)
                edge_specs.append(spec_path)

                # --- depth spec (if available) ---
                if depth_chunks:
                    dep_chunk = depth_chunks[i]
                    spec_name = f"{seq}_chunk_{i:03d}_depth_{weather}"
                    spec_path = SPECS_DIR / f"{spec_name}.json"
                    spec = {
                        "name": spec_name,
                        "video_path": str(inp_chunk.resolve()),
                        "prompt_path": str((PROMPTS_DIR / f"{weather}.txt").resolve()),
                        "guidance": args.guidance,
                        "depth": {
                            "control_path": str(dep_chunk.resolve()),
                            "control_weight": args.control_weight,
                        },
                    }
                    with open(spec_path, "w") as f:
                        json.dump(spec, f, indent=4)
                    depth_specs.append(spec_path)

                    # --- multicontrol spec (edge + depth combined) ---
                    spec_name = f"{seq}_chunk_{i:03d}_multi_{weather}"
                    spec_path = SPECS_DIR / f"{spec_name}.json"
                    spec = {
                        "name": spec_name,
                        "video_path": str(inp_chunk.resolve()),
                        "prompt_path": str((PROMPTS_DIR / f"{weather}.txt").resolve()),
                        "guidance": args.guidance,
                        "edge": {
                            "control_path": str(edg_chunk.resolve()),
                            "control_weight": args.control_weight,
                        },
                        "depth": {
                            "control_path": str(dep_chunk.resolve()),
                            "control_weight": args.control_weight,
                        },
                    }
                    with open(spec_path, "w") as f:
                        json.dump(spec, f, indent=4)
                    multi_specs.append(spec_path)

    all_specs = edge_specs + depth_specs + multi_specs

    if edge_specs:
        write_batch_file(SPECS_DIR / "batch_edge.txt", edge_specs)
    if depth_specs:
        write_batch_file(SPECS_DIR / "batch_depth.txt", depth_specs)
    if multi_specs:
        write_batch_file(SPECS_DIR / "batch_multi.txt", multi_specs)
    if all_specs:
        write_batch_file(SPECS_DIR / "batch_all.txt", all_specs)

    print(f"\n--- Summary ---")
    print(f"Edge specs:       {len(edge_specs)}")
    print(f"Depth specs:      {len(depth_specs)}")
    print(f"Multicontrol specs: {len(multi_specs)}")
    print(f"Total:            {len(all_specs)}")
    print(f"Output dir:       {SPECS_DIR}")
    print(f"\nRun edge inference:")
    print(f"  python examples/inference.py $(xargs -a {SPECS_DIR}/batch_edge.txt -I{{}} echo -i {{}}) -o tartandrive/outputs/chunks/edge/")
    print(f"\nRun depth inference:")
    print(f"  python examples/inference.py $(xargs -a {SPECS_DIR}/batch_depth.txt -I{{}} echo -i {{}}) -o tartandrive/outputs/chunks/depth/")
    print(f"\nRun multicontrol inference (edge+depth):")
    print(f"  python examples/inference.py $(xargs -a {SPECS_DIR}/batch_multi.txt -I{{}} echo -i {{}}) -o tartandrive/outputs/chunks/multi/")

    if skipped:
        print(f"\n[WARNING] Skipped sequences: {skipped}")


if __name__ == "__main__":
    main()

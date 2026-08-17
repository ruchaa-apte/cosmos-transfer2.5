#!/usr/bin/env python3
"""
Extract Canny edge control videos from TartanDrive2 image sequences.
Reads PNGs directly (avoids re-encoding artifacts from step 1).

Usage:
    python3.10 02_extract_edge_control.py [--low 100] [--high 200] [--fps 10]
"""

import argparse
import os
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np

TARTANDRIVE_ROOT = Path("/raid/home/ruchaa/projects/alpamayo-coc-autolabeler/data/tartandrive")
OUT_ROOT = Path(__file__).parent / "data"

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


def extract_edge(frame_bgr: np.ndarray, low: int, high: int) -> np.ndarray:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    # Slight Gaussian blur reduces noise before Canny
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, low, high)
    # Return 3-channel so ffmpeg can encode as color video
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)


def frames_to_mp4(frame_dir: Path, out_path: Path, fps: int) -> None:
    """Use ffmpeg to assemble PNGs from a temp dir into an MP4."""
    cmd = [
        "ffmpeg", "-y", "-loglevel", "warning",
        "-framerate", str(fps),
        "-i", str(frame_dir / "%06d.png"),
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-pix_fmt", "yuv420p",
        str(out_path),
    ]
    subprocess.run(cmd, check=True)


def process_sequence(seq: str, low: int, high: int, fps: int) -> None:
    src_dir = TARTANDRIVE_ROOT / seq / "image_left_color"
    out_dir = OUT_ROOT / seq
    out_mp4 = out_dir / "edge.mp4"

    if not src_dir.exists():
        print(f"[SKIP] {seq} — source not found")
        return

    if out_mp4.exists():
        print(f"[SKIP] {seq} — edge.mp4 already exists")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    frames = sorted(src_dir.glob("*.png"))
    print(f"[INFO] {seq} — {len(frames)} frames, Canny({low},{high})")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        for idx, frame_path in enumerate(frames):
            bgr = cv2.imread(str(frame_path))
            if bgr is None:
                raise RuntimeError(f"Failed to read {frame_path}")
            edge = extract_edge(bgr, low, high)
            cv2.imwrite(str(tmp_path / f"{idx:06d}.png"), edge)

        frames_to_mp4(tmp_path, out_mp4, fps)

    print(f"[DONE] {seq} → {out_mp4}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--low", type=int, default=100, help="Canny low threshold (default: 100)")
    parser.add_argument("--high", type=int, default=200, help="Canny high threshold (default: 200)")
    parser.add_argument("--fps", type=int, default=10, help="Output video framerate (default: 10)")
    args = parser.parse_args()

    for seq in SEQUENCES:
        process_sequence(seq, args.low, args.high, args.fps)

    print(f"\nEdge control videos written to: {OUT_ROOT}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Extract monocular depth control videos from TartanDrive2 image sequences
using Depth Anything V2 Small. Output is a normalized grayscale MP4 (3-channel)
matching the format expected by Cosmos Transfer's depth control model.

Usage:
    python3.10 02b_extract_depth_control.py [--model small|base|large] [--fps 10] [--batch-size 8]
"""

import argparse
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

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

MODEL_IDS = {
    "small": "depth-anything/Depth-Anything-V2-Small-hf",
    "base":  "depth-anything/Depth-Anything-V2-Base-hf",
    "large": "depth-anything/Depth-Anything-V2-Large-hf",
}


def load_model(model_size: str, device: torch.device):
    model_id = MODEL_IDS[model_size]
    print(f"[INFO] Loading {model_id} on {device}")
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForDepthEstimation.from_pretrained(model_id, torch_dtype=torch.float16)
    model = model.to(device).eval()
    return processor, model


@torch.inference_mode()
def predict_depth_batch(
    frames_bgr: list[np.ndarray],
    processor,
    model,
    device: torch.device,
) -> list[np.ndarray]:
    """Run depth estimation on a batch of BGR frames. Returns list of uint8 grayscale arrays."""
    pil_images = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames_bgr]
    inputs = processor(images=pil_images, return_tensors="pt").to(device)
    # Cast pixel_values to fp16 to match model dtype
    inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)
    outputs = model(**inputs)
    # predicted_depth: (B, H, W) — relative depth, larger = farther
    depths = outputs.predicted_depth.float().cpu().numpy()

    result = []
    for depth in depths:
        # Cosmos Transfer convention: near=bright, far/sky=dark.
        # Depth Anything V2 predicted_depth: larger value = closer.
        # Clip to 2nd–98th percentile to prevent sky's uniform low values
        # from collapsing the useful range.
        lo, hi = np.percentile(depth, 2), np.percentile(depth, 98)
        if hi > lo:
            norm = (depth - lo) / (hi - lo)
        else:
            norm = np.zeros_like(depth)
        norm = np.clip(norm, 0.0, 1.0)
        gray = (norm * 255).astype(np.uint8)
        # 3-channel grayscale (matches reference car_depth.mp4 format)
        result.append(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))
    return result


def frames_to_mp4(frame_dir: Path, out_path: Path, fps: int) -> None:
    cmd = [
        "ffmpeg", "-y", "-loglevel", "warning",
        "-framerate", str(fps),
        "-i", str(frame_dir / "%06d.png"),
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-pix_fmt", "yuv420p",
        str(out_path),
    ]
    subprocess.run(cmd, check=True)


def process_sequence(seq: str, processor, model, device: torch.device, fps: int, batch_size: int) -> None:
    src_dir = TARTANDRIVE_ROOT / seq / "image_left_color"
    out_dir = OUT_ROOT / seq
    out_mp4 = out_dir / "depth.mp4"

    if not src_dir.exists():
        print(f"[SKIP] {seq} — source not found")
        return
    if out_mp4.exists():
        print(f"[SKIP] {seq} — depth.mp4 already exists")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    frames = sorted(src_dir.glob("*.png"))
    print(f"[INFO] {seq} — {len(frames)} frames")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        idx = 0
        for batch_start in range(0, len(frames), batch_size):
            batch_paths = frames[batch_start : batch_start + batch_size]
            batch_bgr = [cv2.imread(str(p)) for p in batch_paths]
            depth_frames = predict_depth_batch(batch_bgr, processor, model, device)
            for depth_frame in depth_frames:
                cv2.imwrite(str(tmp_path / f"{idx:06d}.png"), depth_frame)
                idx += 1

            if (batch_start // batch_size) % 10 == 0:
                print(f"  {idx}/{len(frames)} frames", end="\r", flush=True)

        print(f"  {len(frames)}/{len(frames)} frames")
        frames_to_mp4(tmp_path, out_mp4, fps)

    print(f"[DONE] {seq} → {out_mp4}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["small", "base", "large"], default="small",
                        help="Depth Anything V2 model size (default: small)")
    parser.add_argument("--fps", type=int, default=10, help="Output video framerate (default: 10)")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Frames per GPU batch (default: 8, reduce if OOM)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor, model = load_model(args.model, device)

    for seq in SEQUENCES:
        process_sequence(seq, processor, model, device, args.fps, args.batch_size)

    print(f"\nDepth control videos written to: {OUT_ROOT}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Cosmos Reason quality critic for weather-augmented videos.
Watches an output directory for new MP4s and scores each with Cosmos-Reason1-7B.

Can run in two modes:
  --watch   : live watcher — scores videos as they appear (for parallel use during generation)
  --batch   : batch mode  — score all existing videos in the output dir

Usage:
    # Live (parallel with generation, tries GPU 4)
    CUDA_VISIBLE_DEVICES=4 python3.10 07_cosmos_reason_critic.py --watch \
        --video-dir tartandrive/outputs/kf_multi \
        --out tartandrive/outputs/kf_multi/quality_log.jsonl

    # Batch (after generation is done)
    python3.10 07_cosmos_reason_critic.py --batch \
        --video-dir tartandrive/outputs/kf_multi \
        --out tartandrive/outputs/kf_multi/quality_log.jsonl
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "/raid/home/ruchaa/projects/cosmos-transfer2.5/packages/cosmos-oss")
from vqa.cosmos_reason_inference import CosmosReasonModel

QUESTIONS = [
    {
        "id": "rain_realism",
        "q": "Does this video show realistic rain effects on an off-road trail? "
             "Look for wet ground, puddles, rain streaks, overcast sky, and reduced visibility. "
             "Answer yes or no and briefly explain.",
    },
    {
        "id": "geometry_preservation",
        "q": "Is the trail, road, or terrain geometry preserved naturally in this video? "
             "Does the path shape, vegetation placement, and overall scene layout match "
             "what you would expect for this type of off-road environment? "
             "Answer yes or no and briefly explain.",
    },
    {
        "id": "artifacts",
        "q": "Are there any obvious visual artifacts in this video such as flickering, "
             "unnatural color patches, blurry regions, grid patterns, or geometric distortions? "
             "Answer yes or no and briefly explain.",
    },
    {
        "id": "photorealism",
        "q": "How photorealistic does this weather-augmented video look? "
             "Could this footage be mistaken for real rainy weather captured by an onboard robot camera? "
             "Rate on a scale of 1 (clearly fake) to 10 (indistinguishable from real).",
    },
    {
        "id": "overall_quality",
        "q": "Give an overall quality assessment of this weather-augmented video. "
             "Consider realism of rain effects, preservation of scene structure, "
             "temporal consistency, and absence of artifacts. "
             "Rate 1-10 and provide a one-sentence summary.",
    },
]


def score_video(model: CosmosReasonModel, video_path: Path) -> dict:
    result = {"video": str(video_path.name), "scores": {}}
    for item in QUESTIONS:
        try:
            answer = model.infer(video_path=str(video_path), question=item["q"], verbose=False)
            result["scores"][item["id"]] = answer
        except Exception as e:
            result["scores"][item["id"]] = f"ERROR: {e}"
    return result


def batch_mode(model: CosmosReasonModel, video_dir: Path, out_path: Path, done: set) -> set:
    videos = sorted(video_dir.glob("*_multi_rainy.mp4"))
    new_videos = [v for v in videos if v.name not in done]
    print(f"[BATCH] {len(new_videos)} new videos to score")
    for video in new_videos:
        print(f"  Scoring: {video.name}")
        result = score_video(model, video)
        with open(out_path, "a") as f:
            f.write(json.dumps(result) + "\n")
        done.add(video.name)
        print(f"  Done: photorealism={result['scores'].get('photorealism', '?')[:60]}")
    return done


def watch_mode(model: CosmosReasonModel, video_dir: Path, out_path: Path):
    done = set()
    # Load already-scored videos
    if out_path.exists():
        with open(out_path) as f:
            for line in f:
                try:
                    done.add(json.loads(line)["video"])
                except Exception:
                    pass
    print(f"[WATCH] Monitoring {video_dir} (already scored: {len(done)})")
    while True:
        done = batch_mode(model, video_dir, out_path, done)
        time.sleep(120)  # check every 2 min


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--watch", action="store_true", help="Live watch mode")
    parser.add_argument("--batch", action="store_true", help="Batch mode (score all existing)")
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    print("[CRITIC] Loading Cosmos-Reason1-7B...")
    try:
        model = CosmosReasonModel()
        print("[CRITIC] Model loaded successfully")
    except Exception as e:
        print(f"[CRITIC] Failed to load model: {e}")
        sys.exit(1)

    done = set()
    if args.out.exists():
        with open(args.out) as f:
            for line in f:
                try:
                    done.add(json.loads(line)["video"])
                except Exception:
                    pass
    print(f"[CRITIC] Already scored: {len(done)} videos")

    if args.watch:
        watch_mode(model, args.video_dir, args.out)
    elif args.batch:
        batch_mode(model, args.video_dir, args.out, done)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

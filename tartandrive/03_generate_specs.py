#!/usr/bin/env python3
"""
Generate Cosmos Transfer inference spec JSONs for TartanDrive2 weather augmentation.
Produces one JSON per sequence × weather condition (rainy, snowy).

Usage:
    python3.10 03_generate_specs.py [--guidance 5] [--control-weight 1.0]

Output:
    tartandrive/specs/{sequence}_{weather}.json
    tartandrive/specs/batch_all.txt   (lists all spec paths, for use with -i)
"""

import argparse
import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / "data"
SPECS_DIR = SCRIPT_DIR / "specs"
PROMPTS_DIR = SCRIPT_DIR / "prompts"

SEQUENCES = [
    "sam_loop_2023-09-27-12-42-17",
    "sam_loop_2023-09-27-13-04-54",
    "turnpike_flat_2023-09-12-13-08-55",
    "turnpike_flat_2023-09-12-13-11-44",
    "turnpike_flat_2023-09-12-13-42-26",
    "turnpike_flat_2023-09-12-14-16-06",
    "turnpike_flat_2023-09-12-14-21-03",
]

WEATHER_CONDITIONS = ["rainy", "snowy"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--guidance", type=float, default=5.0,
                        help="CFG guidance scale (default: 5.0). Higher = closer to input structure.")
    parser.add_argument("--control-weight", type=float, default=1.0,
                        help="Edge control weight (default: 1.0)")
    args = parser.parse_args()

    SPECS_DIR.mkdir(parents=True, exist_ok=True)

    generated = []
    skipped = []

    for seq in SEQUENCES:
        input_mp4 = DATA_DIR / seq / "input.mp4"
        edge_mp4 = DATA_DIR / seq / "edge.mp4"

        missing = []
        if not input_mp4.exists():
            missing.append(f"input.mp4 (run 01_images_to_mp4.sh first)")
        if not edge_mp4.exists():
            missing.append(f"edge.mp4 (run 02_extract_edge_control.py first)")

        if missing:
            print(f"[SKIP] {seq} — missing: {', '.join(missing)}")
            skipped.append(seq)
            continue

        for weather in WEATHER_CONDITIONS:
            prompt_path = PROMPTS_DIR / f"{weather}.txt"
            spec_path = SPECS_DIR / f"{seq}_{weather}.json"

            spec = {
                "name": f"{seq}_{weather}",
                "video_path": str(input_mp4.resolve()),
                "prompt_path": str(prompt_path.resolve()),
                "guidance": args.guidance,
                "edge": {
                    "control_path": str(edge_mp4.resolve()),
                    "control_weight": args.control_weight,
                },
            }

            with open(spec_path, "w") as f:
                json.dump(spec, f, indent=4)

            generated.append(spec_path)
            print(f"[DONE] {spec_path.name}")

    # Write a batch file listing all spec paths (one per line)
    batch_file = SPECS_DIR / "batch_all.txt"
    if generated:
        with open(batch_file, "w") as f:
            for p in generated:
                f.write(str(p.resolve()) + "\n")
        print(f"\n{len(generated)} specs written to: {SPECS_DIR}")
        print(f"Batch list: {batch_file}")
        print(f"\nTo run all specs in one model-load:")
        spec_args = " ".join(f"-i {p}" for p in generated)
        print(f"  python examples/inference.py {spec_args} -o tartandrive/outputs/")
        print(f"\nOr use xargs with the batch list:")
        print(f"  xargs -a {batch_file} -I{{}} echo -i {{}} | xargs python examples/inference.py -o tartandrive/outputs/")

    if skipped:
        print(f"\n[WARNING] {len(skipped)} sequences skipped due to missing files: {skipped}")


if __name__ == "__main__":
    main()

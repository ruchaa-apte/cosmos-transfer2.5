#!/usr/bin/env python3
"""
Build new batch_gpu{0,1,2}.txt files excluding the 113 already-completed clips.
Run: python3.10 make_remaining_batches.py
"""
from pathlib import Path

SPECS_DIR = Path("/raid/home/ruchaa/projects/cosmos-transfer2.5/tartandrive/specs/kf_rainy")

COMPLETED = {
    # turnpike_afternoon_fall kf_000..035
    *[f"2023-10-26-14-42-35_turnpike_afternoon_fall_kf_{i:03d}_rainy" for i in range(36)],
    # figure_8 kf_157..198
    *[f"figure_8_2023-09-13-17-24-26_kf_{i:03d}_rainy" for i in range(157, 199)],
    # turnpike kf_344..378
    *[f"turnpike_2023-09-12-12-53-32_kf_{i:03d}_rainy" for i in range(344, 379)],
}

# Read all 3 original batch files in order to preserve original ordering
all_specs = []
for gpu in range(3):
    batch = SPECS_DIR / f"batch_gpu{gpu}.txt"
    with open(batch) as f:
        for line in f:
            p = line.strip()
            if not p:
                continue
            name = Path(p).stem  # e.g. "2023-10-26..._kf_000_rainy"
            if name not in COMPLETED:
                all_specs.append(p)

print(f"Remaining specs: {len(all_specs)} (excluded {len(COMPLETED)} completed)")

# Split evenly across 3 GPUs
n = len(all_specs)
size = (n + 2) // 3
for gpu in range(3):
    chunk = all_specs[gpu * size : (gpu + 1) * size]
    out = SPECS_DIR / f"batch_remaining_gpu{gpu}.txt"
    with open(out, "w") as f:
        f.write("\n".join(chunk) + "\n")
    print(f"  GPU{gpu}: {len(chunk)} specs → {out}")

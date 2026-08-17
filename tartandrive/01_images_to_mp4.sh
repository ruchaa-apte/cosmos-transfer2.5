#!/usr/bin/env bash
# Convert TartanDrive2 image sequences (PNG frames) to MP4 for Cosmos Transfer inference.
# Usage: bash 01_images_to_mp4.sh [FPS]   (default FPS=10)

set -euo pipefail

TARTANDRIVE_ROOT="/raid/home/ruchaa/projects/alpamayo-coc-autolabeler/data/tartandrive"
OUT_ROOT="$(dirname "$0")/data"
FPS="${1:-10}"

SEQUENCES=(
    figure_8_2023-09-13-17-24-26
    2023-11-14-15-02-21_figure_8
    turnpike_2023-09-12-12-53-32
    turnpike_2023-09-13-18-15-03
    2023-10-26-14-42-35_turnpike_afternoon_fall
    turnpike_flat_2023-09-12-13-42-26
    turnpike_warehouse_2023-09-13-19-03-05
    gupta_2023-09-14-11-12-19
    2023-11-14-14-52-22_gupta
    meadows_2023-09-14-12-08-57
    sam_loop_2023-09-27-13-04-54
    slag_heap_skydio_2023-09-14-12-32-13
)

for SEQ in "${SEQUENCES[@]}"; do
    SRC="${TARTANDRIVE_ROOT}/${SEQ}/image_left_color"
    OUT_DIR="${OUT_ROOT}/${SEQ}"
    OUT_MP4="${OUT_DIR}/input.mp4"

    if [ ! -d "$SRC" ]; then
        echo "[SKIP] $SEQ — source not found: $SRC"
        continue
    fi

    mkdir -p "$OUT_DIR"

    if [ -f "$OUT_MP4" ]; then
        echo "[SKIP] $SEQ — already exists: $OUT_MP4"
        continue
    fi

    NFRAMES=$(ls "$SRC"/*.png 2>/dev/null | wc -l)
    echo "[INFO] $SEQ — $NFRAMES frames @ ${FPS}fps → $OUT_MP4"

    ffmpeg -y -loglevel warning \
        -framerate "$FPS" \
        -i "${SRC}/%06d.png" \
        -c:v libx264 -preset fast -crf 18 \
        -pix_fmt yuv420p \
        "$OUT_MP4"

    echo "[DONE] $SEQ"
done

echo ""
echo "All input MP4s written to: $OUT_ROOT"

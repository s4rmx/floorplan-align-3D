#!/usr/bin/env bash
# SAM3 roof-mask smoke: 20 equirect frames, all 6 cubemap crops + overlays.
# Usage: 00_smoke_sam3_roof.sh [video] [out_dir] [n_frames]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VIDEO="${1:-$ROOT/outputs/run2-stella/equirect_1920x960.mp4}"
OUT_DIR="${2:-$ROOT/outputs/_smoke_sam3_roof}"
N_FRAMES="${3:-20}"

mkdir -p "$OUT_DIR"
echo "[smoke] SAM3 roof mask -> $OUT_DIR  n=$N_FRAMES"
bash "$(dirname "$0")/00_run_sam3_roof.sh" "$VIDEO" "$OUT_DIR" "$N_FRAMES" --debug-faces
echo "[smoke] open $OUT_DIR/index.html"

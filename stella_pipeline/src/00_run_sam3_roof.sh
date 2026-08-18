#!/usr/bin/env bash
# Run SAM3 roof-mask builder. Uses sam3-pipeline image + host checkpoint.
# Usage: 00_run_sam3_roof.sh <video> <out_dir> [n_frames]
set -euo pipefail

VIDEO="${1:?video required}"
OUT_DIR="${2:?out_dir required}"
N_FRAMES="${3:-20}"
SRC_DIR="$(cd "$(dirname "$0")" && pwd)"
SAM3_IMAGE="${SAM3_IMAGE:-sam3-pipeline:latest}"
SAM3_PY="${SAM3_PY:-/opt/conda/envs/sam3_env/bin/python}"
SAM3_CKPT="${SAM3_CKPT:-/home/kodifly/Desktop/work-road-asst-segm/spatial-ai-sam3/local-weights/sam3.pt}"

VIDEO="$(readlink -f "$VIDEO")"
OUT_DIR="$(mkdir -p "$OUT_DIR" && readlink -f "$OUT_DIR")"
VIDEO_DIR="$(dirname "$VIDEO")"
VIDEO_BASE="$(basename "$VIDEO")"

if [[ ! -f "$SAM3_CKPT" ]]; then
  echo "[roof] missing checkpoint: $SAM3_CKPT" >&2
  exit 1
fi

echo "[roof] SAM3 via $SAM3_IMAGE  n_frames=$N_FRAMES"
echo "[roof] ckpt $SAM3_CKPT"
docker run --rm --gpus all \
  --entrypoint "$SAM3_PY" \
  -e NVIDIA_DISABLE_REQUIRE=true \
  --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "$SRC_DIR:/scripts:ro" \
  -v "$OUT_DIR:/out" \
  -v "$VIDEO_DIR:/video:ro" \
  -v "$SAM3_CKPT:/opt/sam3/sam3.pt:ro" \
  "$SAM3_IMAGE" \
  /scripts/00_sam3_roof_mask.py \
    --video "/video/$VIDEO_BASE" \
    --out-dir /out \
    --n-frames "$N_FRAMES" \
    --ckpt /opt/sam3/sam3.pt \
    "${@:4}"

echo "[roof] wrote $OUT_DIR/roof_mask.png"

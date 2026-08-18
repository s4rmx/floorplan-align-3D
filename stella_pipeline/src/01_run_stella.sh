#!/usr/bin/env bash
# Run StellaVSLAM dense on an Insta360 equirectangular MP4.
# Usage: 01_run_stella.sh <run_dir> <equirect_src> [stella_data] [stella_root]
set -euo pipefail

RUN_DIR="${1:?run_dir required}"
EQUIRECT_SRC="${2:?equirect mp4 required}"
STELLA_DATA="${3:-/home/kodifly/Desktop/stella-vslam-dense/data}"
STELLA_ROOT="${4:-/home/kodifly/Desktop/stella-vslam-dense/stella_vslam_dense}"
STELLA_IMAGE="${STELLA_IMAGE:-stella_vslam_dense}"
FRAME_STEP="${FRAME_STEP:-2}"
ROOF_MASK="${ROOF_MASK:-}"

mkdir -p "$RUN_DIR/traj" "$RUN_DIR/keyframes"

RESIZED="$RUN_DIR/equirect_1920x960.mp4"
if [[ ! -f "$RESIZED" ]] || [[ "$EQUIRECT_SRC" -nt "$RESIZED" ]]; then
  echo "[stella] resizing to 1920x960 -> $RESIZED"
  for VCODEC in libx264 mpeg4 libx265; do
    if ffmpeg -y -hide_banner -loglevel error -i "$EQUIRECT_SRC" \
      -vf scale=1920:960 -c:v "$VCODEC" -preset fast \
      "$RESIZED" 2>/dev/null; then
      break
    fi
  done
  if [[ ! -f "$RESIZED" ]]; then
    echo "[stella] ffmpeg resize failed (no suitable encoder)" >&2
    exit 1
  fi
fi

echo "[stella] running dense SLAM (frame-step=$FRAME_STEP)"
MASK_ARGS=()
if [[ -n "$ROOF_MASK" && -f "$ROOF_MASK" ]]; then
  echo "[stella] using roof mask $ROOF_MASK"
  MASK_ARGS+=(--mask /pipeline_out/roof_mask.png)
  if [[ "$(readlink -f "$ROOF_MASK")" != "$(readlink -f "$RUN_DIR/roof_mask.png")" ]]; then
    cp -f "$ROOF_MASK" "$RUN_DIR/roof_mask.png"
  fi
fi

docker run --rm --gpus all \
  -e NVIDIA_DISABLE_REQUIRE=true \
  --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "$STELLA_DATA:/data:ro" \
  -v "$STELLA_ROOT:/stella:ro" \
  -v "$RUN_DIR:/pipeline_out" \
  "$STELLA_IMAGE" \
  python3 -u /stella/tools/run_video_slam.py \
    -v /data/orb_vocab.fbow \
    -c /stella/example/dense/dense_batch_1920.yaml \
    -m /pipeline_out/equirect_1920x960.mp4 \
    --frame-step "$FRAME_STEP" \
    "${MASK_ARGS[@]}" \
    -o /pipeline_out/out.db \
    -p /pipeline_out/out.ply \
    -k /pipeline_out/keyframes/ \
    --eval-log-dir /pipeline_out/traj \
    --auto-term --disable-viewer

echo "[stella] done -> $RUN_DIR/out.ply"

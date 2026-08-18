"""MASt3R sparse global alignment odometry for Insta360 frame sequences."""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
MAST3R_ROOT = ROOT / "third_party" / "mast3r"


def ensure_mast3r_importable() -> Path:
    if not MAST3R_ROOT.is_dir():
        raise RuntimeError(
            "MASt3R not installed. Run once:\n  bash scripts/setup_mast3r.sh"
        )
    for p in (MAST3R_ROOT, MAST3R_ROOT / "dust3r"):
        ps = str(p)
        if ps not in sys.path:
            sys.path.insert(0, ps)
    return MAST3R_ROOT


def extract_frames(
    video: Path,
    out_dir: Path,
    fps: float = 2.0,
    max_frames: int = 50,
) -> tuple[list[Path], list[float]]:
    # Always start clean: leftover JPGs from a previous sampling scheme
    # would keep the same filenames and poison MASt3R's path-keyed cache.
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video}")
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration_s = total_frames / video_fps if total_frames > 0 else 0.0

    # Uniform timestamps across the full walk (needed for late landmarks).
    if duration_s > 0:
        n = min(max_frames, max(int(duration_s * fps), 2))
        times = [i * duration_s / max(n - 1, 1) for i in range(n)]
        indices = sorted({min(int(round(t * video_fps)), total_frames - 1) for t in times})
    else:
        stride = max(int(round(video_fps / fps)), 1)
        indices = list(range(0, max_frames * stride, stride))

    paths: list[Path] = []
    times_out: list[float] = []
    for frame_idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            continue
        t = frame_idx / video_fps
        h, w = frame.shape[:2]
        scale = min(1.0, 960 / max(h, w))
        if scale < 1.0:
            frame = cv2.resize(
                frame,
                (int(w * scale), int(h * scale)),
                interpolation=cv2.INTER_AREA,
            )
        path = out_dir / f"frame_{len(paths):04d}_t{t:07.2f}s.jpg"
        cv2.imwrite(str(path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        paths.append(path)
        times_out.append(t)
        if len(paths) >= max_frames:
            break
    cap.release()
    if len(paths) < 2:
        raise RuntimeError("Need at least 2 frames for MASt3R")
    return paths, times_out


def _poses_to_xy_yaw(cams2world: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Project camera centers to a ground plane (Y-up) and estimate yaw."""
    n = len(cams2world)
    pos = cams2world[:, :3, 3].copy()
    # Ground plane: X right, Z forward (Y is up in dust3r/opengl-ish coords).
    xy = pos[:, [0, 2]]
    xy -= xy[0]

    yaws = np.zeros(n, dtype=np.float64)
    for i in range(n):
        r = cams2world[i, :3, :3]
        # Camera forward (+Z cam) in world.
        fwd = r @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
        yaws[i] = float(np.arctan2(fwd[0], fwd[2]))
    yaws -= yaws[0]
    return xy, yaws


def run_mast3r_sga(
    frame_paths: list[Path],
    cache_dir: Path,
    device: str = "cuda",
    image_size: int = 512,
    winsize: int = 5,
    niter1: int = 200,
    niter2: int = 100,
) -> np.ndarray:
    ensure_mast3r_importable()
    import mast3r.utils.path_to_dust3r  # noqa: F401
    from dust3r.utils.image import load_images
    from mast3r.cloud_opt.sparse_ga import sparse_global_alignment
    from mast3r.image_pairs import make_pairs
    from mast3r.model import AsymmetricMASt3R

    filelist = [str(p) for p in frame_paths]
    cache_dir.mkdir(parents=True, exist_ok=True)

    model_name = "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
    model = AsymmetricMASt3R.from_pretrained(model_name).to(device)

    imgs = load_images(filelist, size=image_size, verbose=True)
    # Sliding window pairs along the walk (good for video).
    pairs = make_pairs(
        imgs,
        scene_graph=f"swin-{winsize}-noncyclic",
        prefilter=None,
        symmetrize=True,
    )

    scene = sparse_global_alignment(
        filelist,
        pairs,
        str(cache_dir),
        model,
        device=device,
        lr1=0.07,
        niter1=niter1,
        lr2=0.01,
        niter2=niter2,
        matching_conf_thr=0.0,
        shared_intrinsics=False,
    )

    cams2world = scene.get_im_poses().detach().cpu().numpy()
    del model, scene
    torch.cuda.empty_cache()
    return cams2world


def reconstruct_mast3r(
    video: Path,
    out_dir: Path,
    fps: float = 2.0,
    max_frames: int = 50,
    winsize: int = 5,
    keep_frames: bool = True,
) -> dict:
    frames_dir = out_dir / "mast3r_frames"
    # Isolate cache by sampling so 25-frame vs 50-frame runs cannot reuse pairs.
    cache_dir = out_dir / f"mast3r_cache_n{max_frames}_fps{fps:g}_win{winsize}"
    # MASt3R caches by file path, not pixels. Wipe this cache dir if the
    # caller re-extracts into new filenames; stale index-named caches from
    # older runs live under mast3r_cache/ and are simply ignored.
    frame_paths, times = extract_frames(video, frames_dir, fps=fps, max_frames=max_frames)

    print(
        f"MASt3R: {len(frame_paths)} frames spanning "
        f"{times[0]:.1f}s–{times[-1]:.1f}s @ ~{fps} fps, winsize={winsize}"
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("Warning: MASt3R on CPU will be very slow.")

    cams2world = run_mast3r_sga(
        frame_paths,
        cache_dir,
        device=device,
        winsize=winsize,
    )
    if len(cams2world) != len(times):
        n = min(len(cams2world), len(times))
        cams2world = cams2world[:n]
        times = times[:n]

    xy, yaw = _poses_to_xy_yaw(cams2world)
    t_s = np.asarray(times, dtype=np.float64)
    path_len = float(np.sum(np.linalg.norm(np.diff(xy, axis=0), axis=1)))

    if not keep_frames:
        shutil.rmtree(frames_dir, ignore_errors=True)

    return {
        "t_s": t_s,
        "xy_m": xy,
        "yaw_rad": yaw,
        "cams2world": cams2world,
        "meta": {
            "backend": "mast3r",
            "n_frames": int(len(t_s)),
            "mast3r_fps": float(fps),
            "max_frames": int(max_frames),
            "winsize": int(winsize),
            "path_length_m": path_len,
            "net_displacement_m": float(np.linalg.norm(xy[-1] - xy[0])),
            "device": device,
            "frames_dir": str(frames_dir),
        },
    }

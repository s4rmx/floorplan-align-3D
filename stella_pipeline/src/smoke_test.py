#!/usr/bin/env python3
"""Smoke test: synthetic mini PLY+traj through BEV/align/overlay (no Stella)."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))

from common import import_parent_src, load_config, run_dir, write_json  # noqa: E402


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def write_synthetic_ply(path: Path, n: int = 800) -> None:
    """L-shaped wall cloud in Y-up Stella coords."""
    rng = np.random.default_rng(0)
    pts = []
    for x in np.linspace(-2, 2, n // 4):
        for y in np.linspace(0.5, 2.0, 4):
            pts.append([x + rng.normal(0, 0.02), y, 0 + rng.normal(0, 0.02)])
    for z in np.linspace(0, 4, n // 4):
        for y in np.linspace(0.5, 2.0, 4):
            pts.append([-2 + rng.normal(0, 0.02), y, z + rng.normal(0, 0.02)])
    pts = np.asarray(pts)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(pts)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for p in pts:
            f.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f} 128 128 128\n")


def write_synthetic_traj(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for i, z in enumerate(np.linspace(0, 3.5, 40)):
        t = i * 0.1
        x = -1.8 + 0.01 * i
        y = 1.5
        lines.append(f"{t:.6f} {x:.6f} {y:.6f} {z:.6f} 0 0 0 1\n")
    path.write_text("".join(lines))


def main() -> None:
    cfg = load_config()
    run_name = "_smoke"
    out = run_dir(run_name)

    # 1) config + floorplan walls
    floorplan = Path(cfg["floorplan_default"])
    assert floorplan.exists(), floorplan
    import_parent_src(cfg["parent_src"])
    from floorplan import walls_from_floorplan

    fp = walls_from_floorplan(floorplan, out_dir=out)
    assert fp["walls"].sum() > 100

    # 2) synthetic stella outputs
    write_synthetic_ply(out / "out.ply")
    write_synthetic_traj(out / "traj" / "frame_trajectory.txt")

    imu = Path(cfg["imu_default"])
    if not imu.exists():
        raise SystemExit(f"IMU csv missing for smoke test: {imu}")

    bev = _load("bev", SRC / "02_bev_extract.py")
    align = _load("align", SRC / "03_auto_align.py")
    overlay = _load("overlay", SRC / "04_overlay.py")

    bev_meta = bev.run_bev_extract(
        out / "out.ply",
        out / "traj" / "frame_trajectory.txt",
        imu,
        out,
        res_m=float(cfg["bev_res_m"]),
    )
    assert bev_meta["n_lines"] >= 0

    align_summary = align.run_auto_align(
        out,
        floorplan,
        ransac_iters=200,
        confidence_min=0.0,
    )
    overlay.run_overlay(out, floorplan)

    roof = _load("roof", SRC / "00_sam3_roof_mask.py")
    dummy = np.zeros((960, 1920, 3), dtype=np.uint8)
    dummy[:200] = 200
    stella_mask, roof_meta, _face_masks = roof.build_roof_mask(
        [dummy],
        face_size=64,
        prompts=["ceiling"],
        conf=0.35,
        ckpt=None,
        min_elevation_deg=40.0,
        zenith_band=0.10,
        dilate_px=3,
        skip_sam3=True,
        debug_dir=None,
    )
    assert stella_mask.shape == (960, 1920)
    assert stella_mask[0, 0] == 0
    assert stella_mask[800, 960] == 255

    # 3) ffmpeg resize smoke (first 2s only, no docker)
    equirect = Path(cfg["equirect_src"])
    clip = out / "equirect_smoke_2s.mp4"
    resize_ok = False
    if equirect.exists():
        for vcodec in ("libx264", "mpeg4", "libx265"):
            try:
                subprocess.run(
                    [
                        "ffmpeg",
                        "-y",
                        "-hide_banner",
                        "-loglevel",
                        "error",
                        "-i",
                        str(equirect),
                        "-t",
                        "2",
                        "-vf",
                        "scale=1920:960",
                        "-c:v",
                        vcodec,
                        str(clip),
                    ],
                    check=True,
                )
                resize_ok = clip.stat().st_size > 10_000
                break
            except subprocess.CalledProcessError:
                continue

    result = {
        "status": "ok",
        "run_dir": str(out),
        "bev": bev_meta,
        "align": align_summary,
        "equirect_resize_smoke": resize_ok,
        "roof_mask_zenith_smoke": roof_meta,
        "artifacts": sorted(p.name for p in out.iterdir()),
    }
    write_json(out / "smoke_result.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

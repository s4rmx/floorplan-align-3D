#!/usr/bin/env python3
"""End-to-end: equirect -> Stella dense -> BEV -> auto-align -> overlay."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))

from common import load_config, read_json, run_dir, write_json  # noqa: E402

import importlib.util


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def ensure_resized_equirect(src: Path, out: Path, size: str = "1920x960") -> Path:
    dest = out / "equirect_1920x960.mp4"
    if dest.exists() and dest.stat().st_mtime >= src.stat().st_mtime:
        return dest
    w, h = size.lower().split("x")
    print(f"[pipeline] resizing {src} -> {dest} ({size})")
    last_err = None
    for vcodec in ("libx264", "mpeg4", "libx265"):
        try:
            subprocess.run(
                [
                    "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                    "-i", str(src), "-vf", f"scale={w}:{h}",
                    "-c:v", vcodec, "-preset", "fast", str(dest),
                ],
                check=True,
            )
            return dest
        except subprocess.CalledProcessError as exc:
            last_err = exc
    raise SystemExit(f"ffmpeg resize failed: {last_err}")


def run_roof_mask(out: Path, video: Path, cfg: dict) -> Path:
    script = SRC / "00_run_sam3_roof.sh"
    n_frames = str(int(cfg.get("sam3_roof_frames", 20)))
    env = os.environ.copy()
    env["FARM_IMAGE"] = str(cfg.get("farm_image", "farm-e2e-farm:latest"))
    env["SAM3_IMAGE"] = str(cfg.get("sam3_image", "sam3-pipeline:latest"))
    if cfg.get("sam3_ckpt"):
        env["SAM3_CKPT"] = str(cfg["sam3_ckpt"])
    subprocess.run(
        ["bash", str(script), str(video), str(out), n_frames],
        check=True,
        env=env,
    )
    mask = out / "roof_mask.png"
    if not mask.exists():
        raise SystemExit("SAM3 roof mask was not written")
    return mask


def run_stella(out: Path, equirect: Path, cfg: dict, roof_mask: Path | None = None) -> None:
    script = SRC / "01_run_stella.sh"
    env = os.environ.copy()
    env["FRAME_STEP"] = str(cfg["frame_step"])
    if roof_mask is not None and roof_mask.exists():
        env["ROOF_MASK"] = str(roof_mask)
    subprocess.run(
        [
            "bash",
            str(script),
            str(out),
            str(equirect),
            cfg["stella_data"],
            cfg["stella_root"],
        ],
        check=True,
        env=env,
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True, help="Output subfolder under pipeline/outputs/")
    p.add_argument("--equirect", type=Path, default=None)
    p.add_argument("--imu", type=Path, default=None)
    p.add_argument("--floorplan", type=Path, default=None)
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--skip-stella", action="store_true")
    p.add_argument("--skip-bev", action="store_true")
    p.add_argument("--skip-align", action="store_true")
    p.add_argument("--skip-overlay", action="store_true")
    p.add_argument(
        "--skip-roof-mask",
        action="store_true",
        help="Do not run SAM3; Stella uses existing roof_mask.png if present",
    )
    p.add_argument("--force-roof-mask", action="store_true", help="Rebuild SAM3 roof mask")
    p.add_argument("--force-align", action="store_true", help="Re-run align even if sim2 exists")
    p.add_argument(
        "--click-align",
        action="store_true",
        help="Open BEV+floorplan click UI and fit Sim(2) from 2+ correspondences",
    )
    p.add_argument(
        "--correspondences",
        type=Path,
        default=None,
        help="Existing correspondences.json from a previous click session",
    )
    args = p.parse_args()

    cfg = load_config(args.config)
    out = run_dir(args.run)
    equirect = args.equirect or Path(cfg["equirect_src"])
    imu = args.imu or Path(cfg["imu_default"])
    floorplan = args.floorplan or Path(cfg["floorplan_default"])

    write_json(
        out / "run_meta.json",
        {
            "run": args.run,
            "equirect": str(equirect.resolve()),
            "imu": str(imu.resolve()),
            "floorplan": str(floorplan.resolve()),
            "out_dir": str(out.resolve()),
        },
    )

    if not args.skip_stella:
        if not equirect.exists():
            raise SystemExit(f"Missing equirect video: {equirect}")
        resized = ensure_resized_equirect(equirect, out, str(cfg.get("equirect_size", "1920x960")))
        roof_mask = out / "roof_mask.png"
        want_roof = bool(cfg.get("sam3_roof", True)) and not args.skip_roof_mask
        if want_roof and (args.force_roof_mask or not roof_mask.exists()):
            roof_mask = run_roof_mask(out, resized, cfg)
        elif roof_mask.exists():
            print(f"[pipeline] reusing {roof_mask}")
        else:
            roof_mask = None
        run_stella(out, equirect, cfg, roof_mask=roof_mask)
    elif not (out / "out.ply").exists():
        raise SystemExit("--skip-stella but out.ply missing")

    bev_mod = _load_module("bev", SRC / "02_bev_extract.py")
    align_mod = _load_module("align", SRC / "03_auto_align.py")
    overlay_mod = _load_module("overlay", SRC / "04_overlay.py")

    if not args.skip_bev:
        bev_mod.run_bev_extract(
            out / "out.ply",
            out / "traj" / "frame_trajectory.txt",
            imu,
            out,
            res_m=float(cfg["bev_res_m"]),
            height_min_m=float(cfg["bev_height_min_m"]),
            height_max_m=float(cfg["bev_height_max_m"]),
            flip_x=bool(cfg.get("bev_flip_x", True)),
        )

    sim2_path = out / "sim2_transform.json"
    if args.click_align or args.correspondences:
        click_mod = _load_module("click", SRC / "05_click_align.py")
        if args.correspondences and args.correspondences.exists():
            pairs = read_json(args.correspondences).get("pairs", [])
        else:
            pairs = click_mod.collect_correspondences(out, floorplan)
        if len(pairs) < 2:
            raise SystemExit("Need at least 2 BEV–floorplan clicks")
        align_mod.run_click_align(out, floorplan, pairs)
    elif not args.skip_align and (args.force_align or not sim2_path.exists()):
        align_mod.run_auto_align(
            out,
            floorplan,
            ransac_iters=int(cfg["ransac_iters"]),
            confidence_min=float(cfg["confidence_min"]),
        )

    if not args.skip_overlay:
        if not sim2_path.exists():
            raise SystemExit("No sim2_transform.json — run align first or add --skip-align false")
        overlay_mod.run_overlay(out, floorplan)

    summary = {
        "run": args.run,
        "out_dir": str(out),
        "artifacts": sorted(p.name for p in out.iterdir()),
    }
    write_json(out / "pipeline_summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

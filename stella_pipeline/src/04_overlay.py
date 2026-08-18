"""Apply Sim(2) to camera trajectory and overlay on floorplan."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from common import load_config, read_json


def load_sim2(path: Path) -> tuple[float, np.ndarray, np.ndarray]:
    d = read_json(path)
    R = np.asarray(d["R"], dtype=np.float64)
    t = np.asarray(d["translation_px"], dtype=np.float64)
    return float(d["scale_px_per_m"]), R, t


def apply_sim2(xy_m: np.ndarray, scale: float, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return (scale * (xy_m @ R.T)) + t


def overlay_trajectory(
    floorplan: Path,
    traj_px: np.ndarray,
    bev_grid: np.ndarray | None,
    bev_meta: dict | None,
    sim2: dict,
    out_png: Path,
    out_csv: Path,
    t_s: np.ndarray,
) -> None:
    img = np.array(Image.open(floorplan).convert("RGBA"))
    h, w = img.shape[:2]
    fig, ax = plt.subplots(figsize=(10, 8.2), dpi=120)
    ax.imshow(img)
    # Keep the camera on the floorplan. BEV clouds in Stella units can be
    # huge after scale and would otherwise zoom the drawing off-screen.
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)

    if bev_grid is not None and bev_meta is not None:
        origin = np.asarray(bev_meta["origin_xy"], dtype=np.float64)
        res = float(bev_meta["res_m"])
        ys, xs = np.nonzero(bev_grid)
        if len(xs):
            pts_m = np.stack([xs * res + origin[0], ys * res + origin[1]], axis=1)
            scale, R, t = load_sim2(Path(out_csv.parent / "sim2_transform.json"))
            pts_px = apply_sim2(pts_m, scale, R, t)
            inb = (pts_px[:, 0] >= 0) & (pts_px[:, 0] < w) & (pts_px[:, 1] >= 0) & (pts_px[:, 1] < h)
            pts_px = pts_px[inb]
            if len(pts_px) > 8000:
                rng = np.random.default_rng(0)
                pts_px = pts_px[rng.choice(len(pts_px), size=8000, replace=False)]
            if len(pts_px):
                ax.scatter(
                    pts_px[:, 0],
                    pts_px[:, 1],
                    s=1,
                    c="#888888",
                    alpha=0.18,
                    label="BEV walls",
                    zorder=2,
                )

    ax.plot(traj_px[:, 0], traj_px[:, 1], color="#e10600", lw=2.0, alpha=0.95, label="camera path", zorder=3)
    ax.scatter([traj_px[0, 0]], [traj_px[0, 1]], c="#2ecc71", s=40, zorder=4, label="start")
    ax.scatter([traj_px[-1, 0]], [traj_px[-1, 1]], c="#3498db", s=40, zorder=4, label="end")
    conf = sim2.get("confidence", sim2.get("wall_hit_score", 0))
    src = sim2.get("source", "auto")
    ax.set_title(f"Stella {src}-align  confidence={conf:.2f}")
    ax.set_axis_off()
    ax.legend(loc="lower right", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_s", "x_px", "y_px"])
        for i in range(len(t_s)):
            w.writerow([f"{t_s[i]:.4f}", f"{traj_px[i, 0]:.2f}", f"{traj_px[i, 1]:.2f}"])


def run_overlay(run_dir: Path, floorplan: Path) -> None:
    run_dir = Path(run_dir)
    sim2_path = run_dir / "sim2_transform.json"
    sim2 = read_json(sim2_path)
    scale, R, t = load_sim2(sim2_path)

    traj_xy = np.load(run_dir / "traj_xy_m.npy")
    t_s = np.load(run_dir / "traj_t_s.npy")
    traj_px = apply_sim2(traj_xy, scale, R, t)

    bev_grid = np.load(run_dir / "bev.npy") if (run_dir / "bev.npy").exists() else None
    bev_meta = read_json(run_dir / "bev_meta.json") if (run_dir / "bev_meta.json").exists() else None

    overlay_trajectory(
        floorplan,
        traj_px,
        bev_grid,
        bev_meta,
        sim2,
        run_dir / "trajectory_aligned.png",
        run_dir / "trajectory_aligned.csv",
        t_s,
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--floorplan", type=Path, default=None)
    p.add_argument("--config", type=Path, default=None)
    args = p.parse_args()
    cfg = load_config(args.config)
    floorplan = args.floorplan or Path(cfg["floorplan_default"])
    run_overlay(args.run_dir, floorplan)
    print(f"wrote {args.run_dir / 'trajectory_aligned.png'}")


if __name__ == "__main__":
    main()

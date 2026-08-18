#!/usr/bin/env python3
"""Start-pinned Sim(2): fix entrance on floorplan, search scale + rotation.

Uses BEV occupancy near the walk and/or bev_lines.npy (wall segments).
No hull matching. Reads Stella outputs from an existing run (e.g. roof-mask run).

Writes to a separate output folder; does not modify the source run.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))

from common import import_parent_src, load_config, read_json, write_json  # noqa: E402
import importlib.util


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


align = _load("align03", SRC / "03_auto_align.py")
overlay = _load("overlay04", SRC / "04_overlay.py")
Sim2 = align.Sim2


def _rotation(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def angle_of(R: np.ndarray) -> float:
    return float(np.arctan2(R[1, 0], R[0, 0]))


def angle_diff_deg(a: float, b: float) -> float:
    return float((a - b + 180.0) % 360.0 - 180.0)


def strip_page_border(walls: np.ndarray, margin: int = 4) -> np.ndarray:
    out = walls.copy()
    m = int(margin)
    out[:m] = False
    out[-m:] = False
    out[:, :m] = False
    out[:, -m:] = False
    return out


def dist_to_walls(walls: np.ndarray) -> np.ndarray:
    free = np.where(walls, 0, 255).astype(np.uint8)
    return cv2.distanceTransform(free, cv2.DIST_L2, 5)


def pinned_sim(
    scale: float,
    theta: float,
    reflect: bool,
    plan_start: np.ndarray,
    anchor_world: np.ndarray,
) -> Sim2:
    R = _rotation(theta)
    if reflect:
        R = R @ np.diag([1.0, -1.0])
    t = plan_start - scale * (R @ anchor_world)
    return Sim2(scale=scale, R=R, t=t, reflected=reflect)


def wall_points_near_path(
    grid: np.ndarray,
    meta: dict,
    traj_xy: np.ndarray,
    band_m: float = 2.5,
    max_pts: int = 4000,
) -> np.ndarray:
    origin = np.asarray(meta["origin_xy"], dtype=np.float64)
    res = float(meta["res_m"])
    band_px = max(6, int(round(band_m / res)))
    traj_img = np.zeros(grid.shape, np.uint8)
    cells = np.floor((traj_xy - origin) / res).astype(int)
    for i in range(len(cells) - 1):
        p1 = (int(cells[i, 0]), int(cells[i, 1]))
        p2 = (int(cells[i + 1, 0]), int(cells[i + 1, 1]))
        cv2.line(traj_img, p1, p2, 255, thickness=band_px)
    ys, xs = np.nonzero(grid & (traj_img > 0))
    if len(xs) == 0:
        ys, xs = np.nonzero(grid)
    pts = np.stack([xs * res + origin[0], ys * res + origin[1]], axis=1)
    if len(pts) > max_pts:
        rng = np.random.default_rng(0)
        pts = pts[rng.choice(len(pts), size=max_pts, replace=False)]
    return pts


def sample_line_points(lines_world: np.ndarray, n_per_seg: int = 12) -> np.ndarray:
    if len(lines_world) == 0:
        return np.empty((0, 2))
    pts: list[np.ndarray] = []
    for x1, y1, x2, y2 in lines_world:
        for u in np.linspace(0.0, 1.0, n_per_seg):
            pts.append([x1 + u * (x2 - x1), y1 + u * (y2 - y1)])
    return np.asarray(pts, dtype=np.float64)


def mean_wall_distance(pix: np.ndarray, dist: np.ndarray, walls_shape: tuple[int, int]) -> tuple[float, float]:
    h, w = walls_shape
    inb = (pix[:, 0] >= 0) & (pix[:, 0] < w) & (pix[:, 1] >= 0) & (pix[:, 1] < h)
    if not np.any(inb):
        return 1e9, 0.0
    xi = np.clip(np.round(pix[inb, 0]).astype(int), 0, w - 1)
    yi = np.clip(np.round(pix[inb, 1]).astype(int), 0, h - 1)
    d = dist[yi, xi]
    mean_d = float(d.mean())
    chamfer = float(math.exp(-mean_d / 6.0) * inb.mean())
    return mean_d, chamfer


def score_pose(
    sim: Sim2,
    wall_pts: np.ndarray,
    line_pts: np.ndarray,
    traj_m: np.ndarray,
    walls: np.ndarray,
    dist: np.ndarray,
    *,
    use_points: bool,
    use_lines: bool,
    w_pts: float,
    w_lines: float,
    w_free: float,
    scale_center: float | None = None,
    w_scale_prior: float = 0.18,
) -> dict:
    h, w = walls.shape
    pt_ch, ln_ch = 0.0, 0.0
    mean_d = 0.0
    n_terms = 0
    if use_points and len(wall_pts):
        pix = sim.apply(wall_pts)
        mean_d, pt_ch = mean_wall_distance(pix, dist, (h, w))
        n_terms += 1
    if use_lines and len(line_pts):
        lpix = sim.apply(line_pts)
        mean_dl, ln_ch = mean_wall_distance(lpix, dist, (h, w))
        if not use_points:
            mean_d = mean_dl
        n_terms += 1
    free = align.traj_free_score(traj_m, sim, walls) if len(traj_m) else 0.0
    w_sum = (w_pts if use_points and len(wall_pts) else 0) + (
        w_lines if use_lines and len(line_pts) else 0
    ) + w_free
    if w_sum <= 0:
        geom = 0.0
    else:
        geom = 0.0
        if use_points and len(wall_pts):
            geom += w_pts * pt_ch
        if use_lines and len(line_pts):
            geom += w_lines * ln_ch
        geom += w_free * free
        geom /= w_sum
    scale_prior = 1.0
    if scale_center is not None and scale_center > 0:
        scale_prior = float(math.exp(-((sim.scale - scale_center) / 5.5) ** 2))
    score = (1.0 - w_scale_prior) * geom + w_scale_prior * scale_prior
    return {
        "score": score,
        "geom_score": geom,
        "scale_prior": scale_prior,
        "point_chamfer": pt_ch,
        "line_chamfer": ln_ch,
        "mean_dt_px": mean_d,
        "free": free,
    }


def grid_search(
    plan_start: np.ndarray,
    anchor_world: np.ndarray,
    wall_pts: np.ndarray,
    line_pts: np.ndarray,
    traj_m: np.ndarray,
    walls: np.ndarray,
    dist: np.ndarray,
    *,
    theta_center_deg: float,
    scale_center: float,
    use_points: bool,
    use_lines: bool,
) -> tuple[Sim2, dict]:
    best: Sim2 | None = None
    best_m: dict = {"score": -1.0}
    scales = np.linspace(30.0, 46.0, 33)
    theta_deg_sets = [theta_center_deg + np.linspace(-18, 18, 13)]
    for base in (0.0, 90.0, 180.0, 270.0):
        theta_deg_sets.append(base + np.linspace(-18, 18, 9))
    thetas = np.deg2rad(np.unique(np.concatenate(theta_deg_sets)))
    for reflect in (False, True):
        for scale in scales:
            for theta in thetas:
                sim = pinned_sim(float(scale), float(theta), reflect, plan_start, anchor_world)
                m = score_pose(
                    sim,
                    wall_pts,
                    line_pts,
                    traj_m,
                    walls,
                    dist,
                    use_points=use_points,
                    use_lines=use_lines,
                    w_pts=0.40,
                    w_lines=0.50,
                    w_free=0.10,
                    scale_center=scale_center,
                )
                if m["score"] > best_m["score"]:
                    best, best_m = sim, m
    assert best is not None
    return best, best_m


def local_refine(
    sim: Sim2,
    plan_start: np.ndarray,
    anchor_world: np.ndarray,
    wall_pts: np.ndarray,
    line_pts: np.ndarray,
    traj_m: np.ndarray,
    walls: np.ndarray,
    dist: np.ndarray,
    *,
    scale_center: float,
    use_points: bool,
    use_lines: bool,
) -> tuple[Sim2, dict]:
    best = sim
    best_m = score_pose(
        best,
        wall_pts,
        line_pts,
        traj_m,
        walls,
        dist,
        use_points=use_points,
        use_lines=use_lines,
        w_pts=0.40,
        w_lines=0.50,
        w_free=0.10,
        scale_center=scale_center,
    )
    base_ang = angle_of(best.R)
    base_scale = best.scale
    for ds in np.linspace(-0.06, 0.06, 7):
        for dth in np.linspace(-0.06, 0.06, 7):
            cand = pinned_sim(
                base_scale * (1.0 + ds),
                base_ang + dth,
                best.reflected,
                plan_start,
                anchor_world,
            )
            m = score_pose(
                cand,
                wall_pts,
                line_pts,
                traj_m,
                walls,
                dist,
                use_points=use_points,
                use_lines=use_lines,
                w_pts=0.40,
                w_lines=0.50,
                w_free=0.10,
                scale_center=scale_center,
            )
            if m["score"] > best_m["score"]:
                best, best_m = cand, m
    best.score = best_m["score"]
    return best, best_m


def overlay_lines(
    floorplan: Path,
    lines_world: np.ndarray,
    sim: Sim2,
    out_png: Path,
    title: str,
) -> None:
    img = np.array(Image.open(floorplan).convert("RGBA"))
    h, w = img.shape[:2]
    fig, ax = plt.subplots(figsize=(10, 8.2), dpi=120)
    ax.imshow(img)
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    for x1, y1, x2, y2 in lines_world:
        p1 = sim.apply(np.array([x1, y1]))
        p2 = sim.apply(np.array([x2, y2]))
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color="#00bcd4", lw=1.2, alpha=0.85)
    ax.set_title(title)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def compare_to_click(sim: Sim2, click: dict) -> dict:
    click_R = np.asarray(click["R"], dtype=np.float64)
    click_t = np.asarray(click["translation_px"], dtype=np.float64)
    return {
        "scale_px_per_m": sim.scale,
        "click_scale_px_per_m": click["scale_px_per_m"],
        "scale_ratio": sim.scale / float(click["scale_px_per_m"]),
        "rotation_deg": float(np.degrees(angle_of(sim.R))),
        "click_rotation_deg": click["rotation_deg"],
        "rotation_err_deg": angle_diff_deg(np.degrees(angle_of(sim.R)), float(click["rotation_deg"])),
        "translation_px": sim.t.tolist(),
        "click_translation_px": click_t.tolist(),
        "translation_err_px": float(np.linalg.norm(sim.t - click_t)),
        "reflected": sim.reflected,
        "click_reflected": bool(click.get("reflected", False)),
    }


def resolve_start_anchor(
    src_run: Path,
    correspondences: Path | None,
    plan_px: tuple[float, float] | None,
    anchor_world: tuple[float, float] | None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    meta = read_json(src_run / "bev_meta.json")
    if correspondences and correspondences.exists():
        pairs = read_json(correspondences).get("pairs", [])
        if pairs:
            p = pairs[0]
            plan = np.asarray(p["plan_px"], dtype=np.float64)
            world = np.asarray(p["world_xy"], dtype=np.float64)
            return plan, world, {"source": str(correspondences), "pair": p.get("label", "pair-1")}
    if plan_px is not None and anchor_world is not None:
        return (
            np.asarray(plan_px, dtype=np.float64),
            np.asarray(anchor_world, dtype=np.float64),
            {"source": "cli"},
        )
    traj = np.load(src_run / "traj_xy_m.npy")
    raise SystemExit(
        "Need --correspondences (pair-1 = start) or --start-plan-px and --anchor-world"
    )


def run(
    src_run: Path,
    out_dir: Path,
    floorplan: Path,
    *,
    mode: str,
    correspondences: Path | None,
    plan_px: tuple[float, float] | None,
    anchor_world: tuple[float, float] | None,
    theta_center_deg: float | None,
    scale_center: float | None,
) -> dict:
    src_run = Path(src_run)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config()
    import_parent_src(cfg["parent_src"])
    from floorplan import walls_from_floorplan

    fp = walls_from_floorplan(floorplan, out_dir=out_dir)
    walls = strip_page_border(fp["walls"])
    dist = dist_to_walls(walls)

    grid = np.load(src_run / "bev.npy")
    meta = read_json(src_run / "bev_meta.json")
    traj_m = np.load(src_run / "traj_xy_m.npy")
    t_s = np.load(src_run / "traj_t_s.npy")
    lines_world = np.load(src_run / "bev_lines.npy")

    plan_start, anchor_world, anchor_meta = resolve_start_anchor(
        src_run, correspondences, plan_px, anchor_world
    )
    write_json(
        out_dir / "start_anchor.json",
        {
            "plan_px": plan_start.tolist(),
            "anchor_world": anchor_world.tolist(),
            **anchor_meta,
        },
    )

    wall_pts = wall_points_near_path(grid, meta, traj_m)
    line_pts = sample_line_points(lines_world)
    np.save(out_dir / "wall_pts_near_path.npy", wall_pts)
    np.save(out_dir / "line_sample_pts.npy", line_pts)

    use_points = mode in ("occupancy", "both")
    use_lines = mode in ("lines", "both")

    click_ref = src_run / "sim2_transform.json"
    click = read_json(click_ref) if click_ref.exists() else {}
    th_c = theta_center_deg if theta_center_deg is not None else float(click.get("rotation_deg", 0.0))
    sc_c = scale_center if scale_center is not None else float(click.get("scale_px_per_m", 38.0))

    sim, coarse_m = grid_search(
        plan_start,
        anchor_world,
        wall_pts,
        line_pts,
        traj_m,
        walls,
        dist,
        theta_center_deg=th_c,
        scale_center=sc_c,
        use_points=use_points,
        use_lines=use_lines,
    )
    sim, best_m = local_refine(
        sim,
        plan_start,
        anchor_world,
        wall_pts,
        line_pts,
        traj_m,
        walls,
        dist,
        scale_center=sc_c,
        use_points=use_points,
        use_lines=use_lines,
    )

    conf = best_m["score"]
    out_sim = align.sim2_to_dict(sim, conf)
    out_sim.update(
        {
            "source": f"start_fit_{mode}",
            "free_space_score": best_m["free"],
            "mean_dt_px": best_m["mean_dt_px"],
            "point_chamfer": best_m["point_chamfer"],
            "line_chamfer": best_m["line_chamfer"],
            "mode": mode,
        }
    )
    write_json(out_dir / "sim2_transform.json", out_sim)

    overlay.overlay_trajectory(
        floorplan,
        sim.apply(traj_m),
        grid,
        meta,
        out_sim,
        out_dir / "trajectory_aligned.png",
        out_dir / "trajectory_aligned.csv",
        t_s,
    )

    if use_lines and len(lines_world):
        overlay_lines(
            floorplan,
            lines_world,
            sim,
            out_dir / "lines_overlay.png",
            f"start-fit ({mode})  score={best_m['score']:.3f}",
        )

    summary = {
        "source": f"start_fit_{mode}",
        "src_run": str(src_run.resolve()),
        "out_dir": str(out_dir.resolve()),
        "mode": mode,
        "scale_px_per_m": sim.scale,
        "rotation_deg": float(np.degrees(angle_of(sim.R))),
        "translation_px": sim.t.tolist(),
        "reflected": sim.reflected,
        "score": best_m["score"],
        "mean_dt_px": best_m["mean_dt_px"],
        "point_chamfer": best_m["point_chamfer"],
        "line_chamfer": best_m["line_chamfer"],
        "geom_score": best_m.get("geom_score"),
        "scale_prior": best_m.get("scale_prior"),
        "free_space_score": best_m["free"],
        "n_wall_pts": int(len(wall_pts)),
        "n_line_pts": int(len(line_pts)),
        "n_line_segments": int(len(lines_world)),
        "theta_center_deg": th_c,
        "scale_center": sc_c,
    }
    if click_ref.exists() and click.get("source") == "click":
        summary["vs_click"] = compare_to_click(sim, click)
        overlay.overlay_trajectory(
            floorplan,
            Sim2(
                float(click["scale_px_per_m"]),
                np.asarray(click["R"], dtype=np.float64),
                np.asarray(click["translation_px"], dtype=np.float64),
                bool(click.get("reflected", False)),
            ).apply(traj_m),
            grid,
            meta,
            click,
            out_dir / "overlay_click_reference.png",
            out_dir / "trajectory_aligned_click_ref.csv",
            t_s,
        )

    write_json(out_dir / "summary.json", summary)
    write_json(
        out_dir / "run_meta.json",
        {
            "src_run": str(src_run.resolve()),
            "floorplan": str(floorplan.resolve()),
            "mode": mode,
        },
    )
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--src-run",
        type=Path,
        required=True,
        help="Existing Stella run with bev.npy / traj (e.g. run2-stella-roofmask)",
    )
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--floorplan", type=Path, default=None)
    p.add_argument(
        "--correspondences",
        type=Path,
        default=None,
        help="Use pair-1 plan_px + world_xy as start anchor (from click session)",
    )
    p.add_argument("--start-plan-px", type=float, nargs=2, metavar=("X", "Y"))
    p.add_argument("--anchor-world", type=float, nargs=2, metavar=("X", "Y"))
    p.add_argument(
        "--mode",
        choices=("occupancy", "lines", "both"),
        default="both",
        help="BEV signal: occupancy near path, wall line segments, or both",
    )
    p.add_argument("--theta-center-deg", type=float, default=None)
    p.add_argument("--scale-center", type=float, default=None)
    p.add_argument("--config", type=Path, default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    floorplan = args.floorplan or Path(cfg["floorplan_default"])
    corr = args.correspondences
    if corr is None and (args.src_run / "correspondences.json").exists():
        corr = args.src_run / "correspondences.json"

    summary = run(
        args.src_run,
        args.out_dir,
        floorplan,
        mode=args.mode,
        correspondences=corr,
        plan_px=tuple(args.start_plan_px) if args.start_plan_px else None,
        anchor_world=tuple(args.anchor_world) if args.anchor_world else None,
        theta_center_deg=args.theta_center_deg,
        scale_center=args.scale_center,
    )
    print(json.dumps(summary, indent=2))
    print(f"[start-fit] wrote {args.out_dir}")


if __name__ == "__main__":
    main()

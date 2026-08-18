#!/usr/bin/env python3
"""Hull-lock + distance-transform scan match: BEV walls -> floorplan.

Writes everything under <run-dir>/auto-align/ and does not touch click-align files.
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
    d = (a - b + 180.0) % 360.0 - 180.0
    return float(d)


def flood_from_border(free_u8: np.ndarray) -> np.ndarray:
    """True = reachable from the image border through free pixels."""
    h, w = free_u8.shape
    canvas = np.where(free_u8 > 0, 255, 0).astype(np.uint8)
    ff_mask = np.zeros((h + 2, w + 2), np.uint8)
    seeds = [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)]
    for x in range(0, w, max(1, w // 8)):
        seeds.append((x, 0))
        seeds.append((x, h - 1))
    for y in range(0, h, max(1, h // 8)):
        seeds.append((0, y))
        seeds.append((w - 1, y))
    for x, y in seeds:
        if canvas[y, x] == 0:
            continue
        cv2.floodFill(canvas, ff_mask, (int(x), int(y)), 128)
    return canvas == 128


def largest_contour(mask: np.ndarray) -> np.ndarray | None:
    u8 = (mask.astype(np.uint8) * 255) if mask.dtype != np.uint8 else mask
    if u8.max() <= 1:
        u8 = (u8 > 0).astype(np.uint8) * 255
    cnts, _ = cv2.findContours(u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None
    cnt = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(cnt) < 50:
        return None
    return cnt.reshape(-1, 2).astype(np.float64)


def strip_page_border(walls: np.ndarray, margin: int = 4) -> np.ndarray:
    """Floorplan PNGs often have a 1px ink frame that ruins convex hulls."""
    out = walls.copy()
    m = int(margin)
    out[:m] = False
    out[-m:] = False
    out[:, :m] = False
    out[:, -m:] = False
    return out


def extract_plan_hull(walls: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Outer envelope of the drawing: convex hull of wall ink."""
    walls = strip_page_border(walls)
    ys, xs = np.nonzero(walls)
    if len(xs) < 20:
        raise RuntimeError("Could not extract floorplan building hull")
    pts = np.stack([xs, ys], axis=1).astype(np.float32)
    hull_idx = cv2.convexHull(pts).reshape(-1, 2).astype(np.float64)
    keep = np.zeros(walls.shape, np.uint8)
    cv2.fillConvexPoly(keep, hull_idx.astype(np.int32), 1)
    return hull_idx, keep.astype(bool)


def extract_bev_hull(
    grid: np.ndarray,
    meta: dict,
    traj_xy: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Hull of BEV occupancy that sits near the camera path (drops site clutter)."""
    origin = np.asarray(meta["origin_xy"], dtype=np.float64)
    res = float(meta["res_m"])
    band_px = max(8, int(round(2.4 / res)))
    traj_img = np.zeros(grid.shape, np.uint8)
    cells = np.floor((traj_xy - origin) / res).astype(int)
    for i in range(len(cells) - 1):
        p1 = (int(cells[i, 0]), int(cells[i, 1]))
        p2 = (int(cells[i + 1, 0]), int(cells[i + 1, 1]))
        cv2.line(traj_img, p1, p2, 255, thickness=band_px)
    near = traj_img > 0
    keep = (grid.astype(bool) & near)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    keep_u8 = cv2.morphologyEx(keep.astype(np.uint8) * 255, cv2.MORPH_CLOSE, k, iterations=2)
    ys, xs = np.nonzero(keep_u8)
    if len(xs) < 20:
        ys, xs = np.nonzero(grid)
    pts = np.stack([xs, ys], axis=1).astype(np.float32)
    hull_px = cv2.convexHull(pts).reshape(-1, 2).astype(np.float64)
    building = np.zeros(grid.shape, np.uint8)
    cv2.fillConvexPoly(building, hull_px.astype(np.int32), 1)
    hull_world = np.stack(
        [hull_px[:, 0] * res + origin[0], hull_px[:, 1] * res + origin[1]],
        axis=1,
    )
    return hull_world, hull_px, building.astype(bool)


def rot90_frame(u: np.ndarray, v: np.ndarray, w: float, h: float, k: int):
    for _ in range(k % 4):
        u, v, w, h = v.copy(), -u.copy(), h, w
    return u, v, w, h


def orthonormalize(R: np.ndarray, allow_reflect: bool) -> np.ndarray:
    U, _, Vt = np.linalg.svd(R)
    R2 = U @ Vt
    det = np.linalg.det(R2)
    if det < 0 and not allow_reflect:
        U[:, 1] *= -1
        R2 = U @ Vt
    if allow_reflect and det > 0:
        U[:, 1] *= -1
        R2 = U @ Vt
    return R2


def hull_candidates(src_world: np.ndarray, dst_px: np.ndarray) -> list[Sim2]:
    """Sim(2) hypotheses from min-area rectangles of the two hulls."""
    src_rect = cv2.minAreaRect(src_world.astype(np.float32))
    dst_rect = cv2.minAreaRect(dst_px.astype(np.float32))
    (csx, csy), (sw, sh), sang = src_rect
    (cdx, cdy), (dw, dh), dang = dst_rect
    c_s = np.array([csx, csy], dtype=np.float64)
    c_d = np.array([cdx, cdy], dtype=np.float64)
    th_s = np.deg2rad(float(sang))
    th_d = np.deg2rad(float(dang))
    u_s = np.array([np.cos(th_s), np.sin(th_s)])
    v_s = np.array([-np.sin(th_s), np.cos(th_s)])
    u_d = np.array([np.cos(th_d), np.sin(th_d)])
    v_d = np.array([-np.sin(th_d), np.cos(th_d)])

    cands: list[Sim2] = []
    for k in range(4):
        for reflect in (False, True):
            us, vs, w, h = rot90_frame(u_s, v_s, float(sw), float(sh), k)
            if reflect:
                vs = -vs
            scale_w = float(dw) / max(w, 1e-6)
            scale_h = float(dh) / max(h, 1e-6)
            ratio = max(scale_w, scale_h) / max(min(scale_w, scale_h), 1e-6)
            if ratio > 2.2:
                continue
            for sf in (0.80, 0.90, 1.0, 1.10, 1.25, 1.50, 1.80, 2.10):
                scale = 0.5 * (scale_w + scale_h) * sf
                S = np.column_stack([us, vs])
                D = np.column_stack([u_d, v_d])
                try:
                    R = D @ np.linalg.inv(S)
                except np.linalg.LinAlgError:
                    continue
                R = orthonormalize(R, reflect)
                t = c_d - scale * (R @ c_s)
                cands.append(Sim2(scale=scale, R=R, t=t, reflected=reflect))
    # Dedup near-identical poses.
    uniq: list[Sim2] = []
    for c in cands:
        ang = angle_of(c.R)
        rec = True
        for u in uniq:
            if (
                abs(c.scale - u.scale) / max(u.scale, 1) < 0.02
                and abs(angle_diff_deg(np.degrees(ang), np.degrees(angle_of(u.R)))) < 3
                and float(np.linalg.norm(c.t - u.t)) < 8
            ):
                rec = False
                break
        if rec:
            uniq.append(c)
    return uniq


def dist_to_walls(walls: np.ndarray) -> np.ndarray:
    free = np.where(walls, 0, 255).astype(np.uint8)
    return cv2.distanceTransform(free, cv2.DIST_L2, 5)


def score_sim(
    sim: Sim2,
    wall_m: np.ndarray,
    traj_m: np.ndarray,
    walls: np.ndarray,
    dist: np.ndarray,
    hull_world: np.ndarray | None,
    plan_hull: np.ndarray | None,
    plan_building: np.ndarray | None,
) -> dict:
    h, w = walls.shape
    pix = sim.apply(wall_m)
    inb = (pix[:, 0] >= 0) & (pix[:, 0] < w) & (pix[:, 1] >= 0) & (pix[:, 1] < h)
    inb_frac = float(inb.mean()) if len(pix) else 0.0
    if inb_frac < 0.15:
        return {"score": -1.0, "chamfer": 0.0, "mean_dt": 1e9, "free": 0.0, "iou": 0.0, "inb": inb_frac}
    xi = np.clip(np.round(pix[inb, 0]).astype(int), 0, w - 1)
    yi = np.clip(np.round(pix[inb, 1]).astype(int), 0, h - 1)
    d = dist[yi, xi]
    mean_dt = float(d.mean())
    chamfer = math.exp(-mean_dt / 10.0) * inb_frac
    free = align.traj_free_score(traj_m, sim, walls) if len(traj_m) else 0.0
    iou = 0.0
    if hull_world is not None and plan_building is not None:
        pred = np.zeros((h, w), np.uint8)
        hp = np.round(sim.apply(hull_world)).astype(np.int32)
        cv2.fillPoly(pred, [hp], 1)
        inter = int(np.logical_and(pred.astype(bool), plan_building).sum())
        union = int(np.logical_or(pred.astype(bool), plan_building).sum())
        iou = float(inter / union) if union else 0.0
    score = 0.50 * chamfer + 0.30 * free + 0.20 * iou
    return {
        "score": score,
        "chamfer": chamfer,
        "mean_dt": mean_dt,
        "free": free,
        "iou": iou,
        "inb": inb_frac,
    }


def scan_match(
    seed: Sim2,
    wall_m: np.ndarray,
    traj_m: np.ndarray,
    walls: np.ndarray,
    dist: np.ndarray,
    hull_world: np.ndarray,
    plan_building: np.ndarray,
) -> tuple[Sim2, dict]:
    best = seed
    best_m = score_sim(seed, wall_m, traj_m, walls, dist, hull_world, None, plan_building)
    base_ang = angle_of(seed.R)
    for scale in np.linspace(seed.scale * 0.90, seed.scale * 1.10, 7):
        for dth in np.linspace(np.deg2rad(-14), np.deg2rad(14), 11):
            R = _rotation(base_ang + dth)
            if seed.reflected:
                R = R @ np.diag([1.0, -1.0])
            for dx in range(-56, 57, 8):
                for dy in range(-56, 57, 8):
                    cand = Sim2(
                        scale=float(scale),
                        R=R,
                        t=seed.t + np.array([dx, dy], dtype=np.float64),
                        reflected=seed.reflected,
                    )
                    m = score_sim(cand, wall_m, traj_m, walls, dist, None, None, None)
                    # skip IoU in inner loop; add it for the winners
                    if m["score"] > best_m["score"]:
                        best, best_m = cand, m
    best_m = score_sim(best, wall_m, traj_m, walls, dist, hull_world, None, plan_building)
    best.score = best_m["score"]
    return best, best_m


def draw_mask_overlay(bg_bgr: np.ndarray, mask: np.ndarray, hull: np.ndarray, color, path: Path) -> None:
    vis = bg_bgr.copy()
    if vis.ndim == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
    tint = vis.astype(np.float32)
    tint[mask] = tint[mask] * 0.45 + np.array(color, dtype=np.float32) * 0.55
    vis = tint.clip(0, 255).astype(np.uint8)
    pts = hull.astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(vis, [pts], True, (0, 255, 255), 2)
    rect = cv2.minAreaRect(hull.astype(np.float32))
    box = cv2.boxPoints(rect).astype(np.int32)
    cv2.polylines(vis, [box], True, (0, 165, 255), 2)
    cv2.imwrite(str(path), vis)


def overlay_on_plan(
    floorplan: Path,
    sim: Sim2,
    wall_m: np.ndarray,
    traj_m: np.ndarray,
    hull_world: np.ndarray,
    title: str,
    out_png: Path,
) -> None:
    img = np.array(Image.open(floorplan).convert("RGBA"))
    h, w = img.shape[:2]
    fig, ax = plt.subplots(figsize=(10, 8.2), dpi=120)
    ax.imshow(img)
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    wp = sim.apply(wall_m)
    inb = (wp[:, 0] >= -20) & (wp[:, 0] < w + 20) & (wp[:, 1] >= -20) & (wp[:, 1] < h + 20)
    wp = wp[inb]
    if len(wp) > 8000:
        rng = np.random.default_rng(0)
        wp = wp[rng.choice(len(wp), size=8000, replace=False)]
    if len(wp):
        ax.scatter(wp[:, 0], wp[:, 1], s=1, c="#888888", alpha=0.22, label="BEV walls", zorder=2)
    hp = sim.apply(hull_world)
    ax.plot(np.r_[hp[:, 0], hp[0, 0]], np.r_[hp[:, 1], hp[0, 1]], color="#00bcd4", lw=2.0, label="BEV hull", zorder=3)
    tp = sim.apply(traj_m)
    ax.plot(tp[:, 0], tp[:, 1], color="#e10600", lw=2.0, alpha=0.95, label="camera path", zorder=4)
    ax.scatter([tp[0, 0]], [tp[0, 1]], c="#2ecc71", s=40, zorder=5, label="start")
    ax.scatter([tp[-1, 0]], [tp[-1, 1]], c="#3498db", s=40, zorder=5, label="end")
    ax.set_title(title)
    ax.set_axis_off()
    ax.legend(loc="lower right", framealpha=0.9)
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


def run(run_dir: Path, floorplan: Path, out_dir: Path) -> dict:
    run_dir = Path(run_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config()
    import_parent_src(cfg["parent_src"])
    from floorplan import walls_from_floorplan

    fp = walls_from_floorplan(floorplan, out_dir=out_dir)
    walls = strip_page_border(fp["walls"])
    dist = dist_to_walls(walls)
    cv2.imwrite(str(out_dir / "walls.png"), (walls.astype(np.uint8) * 255))
    dt_vis = np.clip(dist * 4, 0, 255).astype(np.uint8)
    cv2.imwrite(str(out_dir / "wall_distance.png"), cv2.applyColorMap(dt_vis, cv2.COLORMAP_TURBO))

    grid = np.load(run_dir / "bev.npy")
    meta = read_json(run_dir / "bev_meta.json")
    traj_m = np.load(run_dir / "traj_xy_m.npy")
    wall_m = align.bev_points_world(run_dir, max_pts=8000)

    plan_hull, plan_building = extract_plan_hull(walls)
    bev_hull_m, bev_hull_px, bev_building = extract_bev_hull(grid, meta, traj_m)

    origin = np.asarray(meta["origin_xy"], dtype=np.float64)
    res = float(meta["res_m"])
    cells = np.floor((wall_m - origin) / res).astype(int)
    h_b, w_b = bev_building.shape
    valid = (
        (cells[:, 0] >= 0)
        & (cells[:, 0] < w_b)
        & (cells[:, 1] >= 0)
        & (cells[:, 1] < h_b)
    )
    inside = np.zeros(len(wall_m), dtype=bool)
    inside[valid] = bev_building[cells[valid, 1], cells[valid, 0]]
    if int(inside.sum()) >= 400:
        wall_m = wall_m[inside]

    np.save(out_dir / "plan_hull_px.npy", plan_hull)
    np.save(out_dir / "bev_hull_world.npy", bev_hull_m)
    np.save(out_dir / "bev_hull_px.npy", bev_hull_px)

    bev_vis = cv2.imread(str(run_dir / "bev.png"))
    if bev_vis is None:
        bev_vis = np.full((*grid.shape, 3), 30, np.uint8)
        bev_vis[grid] = (200, 200, 200)
    draw_mask_overlay(bev_vis, bev_building, bev_hull_px, (0, 180, 80), out_dir / "bev_hull.png")

    plan_bg = cv2.cvtColor(np.array(Image.open(floorplan).convert("RGB")), cv2.COLOR_RGB2BGR)
    draw_mask_overlay(plan_bg, plan_building, plan_hull, (0, 180, 80), out_dir / "plan_hull.png")

    seeds = hull_candidates(bev_hull_m, plan_hull)
    if not seeds:
        raise RuntimeError("No hull Sim(2) candidates (aspect mismatch?)")

    ranked = []
    for sim in seeds:
        m = score_sim(sim, wall_m, traj_m, walls, dist, bev_hull_m, plan_hull, plan_building)
        sim.score = m["score"]
        ranked.append((sim, m))
    ranked.sort(key=lambda x: x[1]["score"], reverse=True)

    write_json(
        out_dir / "coarse_candidates.json",
        [
            {
                "rank": i,
                "scale_px_per_m": s.scale,
                "rotation_deg": float(np.degrees(angle_of(s.R))),
                "translation_px": s.t.tolist(),
                "reflected": s.reflected,
                **{k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in m.items()},
            }
            for i, (s, m) in enumerate(ranked[:12])
        ],
    )

    coarse, coarse_m = ranked[0]
    overlay_on_plan(
        floorplan,
        coarse,
        wall_m,
        traj_m,
        bev_hull_m,
        f"coarse hull  score={coarse_m['score']:.3f}  dt={coarse_m['mean_dt']:.1f}px  free={coarse_m['free']:.2f}",
        out_dir / "overlay_coarse.png",
    )

    refined: list[tuple[Sim2, dict]] = []
    for sim, _m in ranked[:6]:
        s2, m2 = scan_match(sim, wall_m, traj_m, walls, dist, bev_hull_m, plan_building)
        refined.append((s2, m2))
    refined.sort(key=lambda x: x[1]["score"], reverse=True)
    best, best_m = refined[0]
    best.score = best_m["score"]

    overlay_on_plan(
        floorplan,
        best,
        wall_m,
        traj_m,
        bev_hull_m,
        f"scan-match  score={best_m['score']:.3f}  dt={best_m['mean_dt']:.1f}px  free={best_m['free']:.2f}  iou={best_m['iou']:.2f}",
        out_dir / "overlay_fine.png",
    )

    sim_dict = align.sim2_to_dict(best, best_m["score"])
    sim_dict.update(
        {
            "source": "hull_scan",
            "free_space_score": best_m["free"],
            "mean_dt_px": best_m["mean_dt"],
            "chamfer": best_m["chamfer"],
            "hull_iou": best_m["iou"],
            "in_bounds": best_m["inb"],
            "n_coarse_candidates": len(seeds),
        }
    )
    write_json(out_dir / "sim2_transform.json", sim_dict)

    t_s = np.load(run_dir / "traj_t_s.npy")
    overlay.overlay_trajectory(
        floorplan,
        best.apply(traj_m),
        grid,
        meta,
        sim_dict,
        out_dir / "trajectory_aligned.png",
        out_dir / "trajectory_aligned.csv",
        t_s,
    )

    summary = {
        "source": "hull_scan",
        "out_dir": str(out_dir),
        "scale_px_per_m": best.scale,
        "rotation_deg": float(np.degrees(angle_of(best.R))),
        "translation_px": best.t.tolist(),
        "reflected": best.reflected,
        "score": best_m["score"],
        "mean_dt_px": best_m["mean_dt"],
        "free_space_score": best_m["free"],
        "hull_iou": best_m["iou"],
        "chamfer": best_m["chamfer"],
        "n_wall_samples": int(len(wall_m)),
        "n_traj": int(len(traj_m)),
        "bev_hull_area_px": float(cv2.contourArea(bev_hull_px.astype(np.float32))),
        "plan_hull_area_px": float(cv2.contourArea(plan_hull.astype(np.float32))),
        "coarse_scale_px_per_m": coarse.scale,
        "coarse_rotation_deg": float(np.degrees(angle_of(coarse.R))),
        "coarse_score": coarse_m["score"],
    }

    click_path = run_dir / "sim2_transform.json"
    if click_path.exists():
        click = read_json(click_path)
        if click.get("source") == "click":
            summary["vs_click"] = compare_to_click(best, click)
            overlay_on_plan(
                floorplan,
                Sim2(
                    scale=float(click["scale_px_per_m"]),
                    R=np.asarray(click["R"], dtype=np.float64),
                    t=np.asarray(click["translation_px"], dtype=np.float64),
                    reflected=bool(click.get("reflected", False)),
                ),
                wall_m,
                traj_m,
                bev_hull_m,
                "click-align (reference)",
                out_dir / "overlay_click_reference.png",
            )

    write_json(out_dir / "summary.json", summary)
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--floorplan", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--config", type=Path, default=None)
    args = p.parse_args()
    cfg = load_config(args.config)
    floorplan = args.floorplan or Path(cfg["floorplan_default"])
    run_dir = Path(args.run_dir)
    out_dir = args.out_dir or (run_dir / "auto-align")
    summary = run(run_dir, floorplan, out_dir)
    print(json.dumps(summary, indent=2))
    print(f"[hull-scan] wrote {out_dir}")


if __name__ == "__main__":
    main()

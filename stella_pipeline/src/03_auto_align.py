"""RANSAC Sim(2) alignment: BEV wall structure -> floorplan walls."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from common import import_parent_src, load_config, read_json, write_json


@dataclass
class Sim2:
    scale: float
    R: np.ndarray
    t: np.ndarray
    reflected: bool
    score: float = 0.0

    def apply(self, xy: np.ndarray) -> np.ndarray:
        return (self.scale * (xy @ self.R.T)) + self.t


def _rotation(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def _reflect_matrix() -> np.ndarray:
    return np.diag([1.0, -1.0])


def bev_points_world(run_dir: Path, max_pts: int = 4000) -> np.ndarray:
    grid = np.load(run_dir / "bev.npy")
    meta = read_json(run_dir / "bev_meta.json")
    origin = np.asarray(meta["origin_xy"], dtype=np.float64)
    res = float(meta["res_m"])
    ys, xs = np.nonzero(grid)
    if len(xs) == 0:
        return np.empty((0, 2))
    pts = np.stack([xs * res + origin[0], ys * res + origin[1]], axis=1)
    if len(pts) > max_pts:
        rng = np.random.default_rng(0)
        pts = pts[rng.choice(len(pts), size=max_pts, replace=False)]
    return pts


def plan_wall_points(walls: np.ndarray, step: int = 4) -> np.ndarray:
    ys, xs = np.nonzero(walls)
    pts = np.stack([xs[::step], ys[::step]], axis=1).astype(np.float64)
    return pts


def wall_hit_score(src_world: np.ndarray, sim: Sim2, plan_walls: np.ndarray) -> float:
    if len(src_world) == 0:
        return 0.0
    pix = sim.apply(src_world)
    h, w = plan_walls.shape
    xi = np.clip(np.round(pix[:, 0]).astype(int), 0, w - 1)
    yi = np.clip(np.round(pix[:, 1]).astype(int), 0, h - 1)
    inb = (pix[:, 0] >= 0) & (pix[:, 0] < w) & (pix[:, 1] >= 0) & (pix[:, 1] < h)
    if not np.any(inb):
        return 0.0
    hits = plan_walls[yi[inb], xi[inb]]
    return float(hits.mean() * inb.mean())


def traj_free_score(traj_xy: np.ndarray, sim: Sim2, plan_walls: np.ndarray) -> float:
    import_parent_src(load_config()["parent_src"])
    from floorplan import free_space_score

    pix = sim.apply(traj_xy)
    return free_space_score(pix, plan_walls)


def two_point_sim(src: np.ndarray, dst: np.ndarray, reflect: bool) -> tuple[float, np.ndarray, np.ndarray]:
    a, b = src[0], src[1]
    p, q = dst[0], dst[1]
    va = b - a
    vb = q - p
    n1 = np.linalg.norm(va)
    n2 = np.linalg.norm(vb)
    scale = 1.0 if n1 < 1e-9 else float(n2 / n1)
    if reflect:
        va_f = np.array([va[0], -va[1]])
        dang = np.arctan2(vb[1], vb[0]) - np.arctan2(va_f[1], va_f[0])
        c, s = np.cos(dang), np.sin(dang)
        R = np.array([[c, -s], [s, c]]) @ np.array([[1.0, 0.0], [0.0, -1.0]])
    else:
        dang = np.arctan2(vb[1], vb[0]) - np.arctan2(va[1], va[0])
        c, s = np.cos(dang), np.sin(dang)
        R = np.array([[c, -s], [s, c]])
    t = p - scale * (R @ a)
    return scale, R, t


def umeyama_2d(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    n = src.shape[0]
    mu_s = src.mean(axis=0)
    mu_d = dst.mean(axis=0)
    src_c = src - mu_s
    dst_c = dst - mu_d
    var_s = float(np.sum(src_c**2) / n)
    cov = (dst_c.T @ src_c) / n
    U, d, Vt = np.linalg.svd(cov)
    S = np.eye(2)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[1, 1] = -1
    R = U @ S @ Vt
    scale = 1.0 if var_s < 1e-12 else float(np.trace(np.diag(d) @ S) / var_s)
    t = mu_d - scale * (R @ mu_s)
    return scale, R, t


def coarse_search(
    src: np.ndarray,
    dst: np.ndarray,
    traj: np.ndarray,
    walls: np.ndarray,
    *,
    n_iter: int,
) -> Sim2:
    rng = np.random.default_rng(42)
    h, w = walls.shape
    dst_c = dst.mean(axis=0)
    src_c = src.mean(axis=0)
    src_span = max(float(np.ptp(src[:, 0])), float(np.ptp(src[:, 1])), 1e-3)
    plan_span = max(float(np.ptp(dst[:, 0])), float(np.ptp(dst[:, 1])), 1.0)
    base_scale = plan_span / src_span

    best = Sim2(scale=base_scale, R=_rotation(0.0), t=dst_c - base_scale * src_c, reflected=False, score=-1.0)

    # Deterministic seeds: 4 rotations × mirror × scale sweep
    seeds: list[Sim2] = []
    for deg in (0, 90, 180, 270):
        th = np.radians(deg)
        for refl in (False, True):
            R = _rotation(th)
            if refl:
                R = R @ _reflect_matrix()
            for sf in (0.6, 0.8, 1.0, 1.2, 1.5):
                s = base_scale * sf
                t = dst_c - s * (src_c @ R.T)
                seeds.append(Sim2(scale=s, R=R, t=t, reflected=refl))

    candidates = seeds.copy()
    n_src, n_dst = len(src), len(dst)
    for _ in range(n_iter):
        if n_src < 3 or n_dst < 3:
            break
        idx_s = rng.choice(n_src, size=3, replace=False)
        idx_d = rng.choice(n_dst, size=3, replace=False)
        try:
            s, R, t = umeyama_2d(src[idx_s], dst[idx_d])
        except Exception:
            continue
        candidates.append(Sim2(scale=s, R=R, t=t, reflected=False))

    for cand in candidates:
        wall_s = wall_hit_score(src, cand, walls)
        free_s = traj_free_score(traj, cand, walls) if len(traj) else 0.0
        cand.score = 0.65 * wall_s + 0.35 * free_s
        if cand.score > best.score:
            best = cand

    return best


def refine_sim2(src: np.ndarray, sim: Sim2, walls: np.ndarray, steps: int = 30) -> Sim2:
    """Small local search around best coarse hypothesis."""
    best = Sim2(sim.scale, sim.R.copy(), sim.t.copy(), sim.reflected, sim.score)
    base_t = sim.t.copy()
    for _ in range(steps):
        for ds in (-0.05, 0.0, 0.05):
            for dth in np.linspace(-0.08, 0.08, 5):
                for dx, dy in ((0, 0), (8, 0), (-8, 0), (0, 8), (0, -8)):
                    cand = Sim2(
                        scale=best.scale * (1 + ds),
                        R=_rotation(np.arctan2(best.R[1, 0], best.R[0, 0]) + dth),
                        t=base_t + np.array([dx, dy]),
                        reflected=best.reflected,
                    )
                    cand.score = wall_hit_score(src, cand, walls)
                    if cand.score > best.score:
                        best = cand
    return best


def sim2_to_dict(sim: Sim2, confidence: float) -> dict:
    ang = float(np.arctan2(sim.R[1, 0], sim.R[0, 0]))
    return {
        "scale_px_per_m": sim.scale,
        "rotation_deg": float(np.degrees(ang)),
        "translation_px": sim.t.tolist(),
        "reflected": sim.reflected,
        "confidence": confidence,
        "wall_hit_score": sim.score,
        "R": sim.R.tolist(),
    }


def run_auto_align(
    run_dir: Path,
    floorplan: Path,
    *,
    ransac_iters: int = 2000,
    confidence_min: float = 0.30,
) -> dict:
    run_dir = Path(run_dir)
    import_parent_src(load_config()["parent_src"])
    from floorplan import walls_from_floorplan

    fp = walls_from_floorplan(floorplan, out_dir=run_dir)
    walls = fp["walls"]

    src = bev_points_world(run_dir)
    dst = plan_wall_points(walls)
    traj = np.load(run_dir / "traj_xy_m.npy")

    if len(src) < 20 or len(dst) < 20:
        raise RuntimeError("Not enough BEV or floorplan wall samples for alignment")

    sim = coarse_search(src, dst, traj, walls, n_iter=ransac_iters)
    sim = refine_sim2(src, sim, walls)

    free_s = traj_free_score(traj, sim, walls)
    confidence = 0.65 * sim.score + 0.35 * free_s

    out = sim2_to_dict(sim, confidence)
    out["free_space_score"] = free_s
    write_json(run_dir / "sim2_transform.json", out)

    summary = {
        "confidence": confidence,
        "confidence_min": confidence_min,
        "low_confidence": confidence < confidence_min,
        "n_bev_points": int(len(src)),
        "n_plan_wall_points": int(len(dst)),
    }
    write_json(run_dir / "align_summary.json", summary)

    if confidence < confidence_min:
        print(
            f"WARNING: alignment confidence {confidence:.3f} < {confidence_min}; "
            "manual landmarks recommended."
        )
    return summary


def run_click_align(
    run_dir: Path,
    floorplan: Path,
    correspondences: list[dict],
) -> dict:
    """Fit Sim(2) from BEV-world <-> floorplan pixel pairs (2+ clicks)."""
    run_dir = Path(run_dir)
    if len(correspondences) < 2:
        raise ValueError("Need at least 2 BEV–floorplan correspondences")

    src = np.stack([[c["world_xy"][0], c["world_xy"][1]] for c in correspondences], axis=0)
    dst = np.stack([[c["plan_px"][0], c["plan_px"][1]] for c in correspondences], axis=0)

    import_parent_src(load_config()["parent_src"])
    from floorplan import walls_from_floorplan

    walls = walls_from_floorplan(floorplan, out_dir=run_dir)["walls"]
    traj = np.load(run_dir / "traj_xy_m.npy")
    bev_pts = bev_points_world(run_dir)

    best: Sim2 | None = None
    for reflect in (False, True):
        if len(src) == 2:
            s, R, t = two_point_sim(src, dst, reflect)
        else:
            s, R, t = umeyama_2d(src, dst)
            if reflect:
                R = R @ _reflect_matrix()
                t = dst.mean(axis=0) - s * (src.mean(axis=0) @ R.T)
        sim = Sim2(scale=s, R=R, t=t, reflected=reflect)
        pred = sim.apply(src)
        rmse = float(np.sqrt(np.mean(np.sum((pred - dst) ** 2, axis=1))))
        free_s = traj_free_score(traj, sim, walls) if len(traj) else 0.0
        wall_s = wall_hit_score(bev_pts, sim, walls)
        sim.score = -rmse + 0.01 * free_s + 0.01 * wall_s
        if best is None or sim.score > best.score:
            best = sim
            best._rmse = rmse  # type: ignore[attr-defined]
            best._free = free_s  # type: ignore[attr-defined]
            best._wall = wall_s  # type: ignore[attr-defined]

    assert best is not None
    rmse = float(getattr(best, "_rmse", 0.0))
    free_s = float(getattr(best, "_free", 0.0))
    wall_s = float(getattr(best, "_wall", 0.0))
    confidence = max(0.0, 1.0 - rmse / 80.0)

    out = sim2_to_dict(best, confidence)
    out["free_space_score"] = free_s
    out["wall_hit_score"] = wall_s
    out["landmark_rmse_px"] = rmse
    out["source"] = "click"
    out["n_correspondences"] = len(correspondences)
    write_json(run_dir / "sim2_transform.json", out)
    write_json(run_dir / "correspondences.json", {"pairs": correspondences})

    summary = {
        "confidence": confidence,
        "source": "click",
        "n_correspondences": len(correspondences),
        "landmark_rmse_px": rmse,
        "scale_px_per_m": best.scale,
        "rotation_deg": float(np.degrees(np.arctan2(best.R[1, 0], best.R[0, 0]))),
        "low_confidence": False,
    }
    write_json(run_dir / "align_summary.json", summary)
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--floorplan", type=Path, default=None)
    p.add_argument("--config", type=Path, default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    floorplan = args.floorplan or Path(cfg["floorplan_default"])
    summary = run_auto_align(
        args.run_dir,
        floorplan,
        ransac_iters=int(cfg["ransac_iters"]),
        confidence_min=float(cfg["confidence_min"]),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

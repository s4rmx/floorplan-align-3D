"""Extract BEV wall occupancy and line segments from Stella dense PLY + TUM poses."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from common import load_config, write_json


def load_ply_xyz(path: Path, max_points: int = 500_000) -> np.ndarray:
    """Load ASCII PLY xyz; subsample if huge."""
    path = Path(path)
    with path.open() as f:
        header_lines: list[str] = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"Invalid PLY header: {path}")
            header_lines.append(line)
            if line.strip() == "end_header":
                break
        n_vertex = 0
        for line in header_lines:
            if line.startswith("element vertex"):
                n_vertex = int(line.split()[-1])
                break
        if n_vertex == 0:
            raise ValueError(f"No vertex count in {path}")

        stride = max(1, n_vertex // max_points)
        pts: list[list[float]] = []
        for i, line in enumerate(f):
            if i % stride != 0:
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            pts.append([float(parts[0]), float(parts[1]), float(parts[2])])
            if len(pts) >= max_points:
                break
    if not pts:
        raise ValueError(f"No points read from {path}")
    return np.asarray(pts, dtype=np.float64)


def rotation_align_vector_to_axis(v: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return 3x3 rotation R with R @ v_hat ~= target_hat."""
    v = v / (np.linalg.norm(v) + 1e-9)
    target = target / (np.linalg.norm(target) + 1e-9)
    c = float(np.dot(v, target))
    if c > 0.999:
        return np.eye(3)
    if c < -0.999:
        return np.diag([1.0, -1.0, -1.0])
    axis = np.cross(v, target)
    axis = axis / (np.linalg.norm(axis) + 1e-9)
    K = np.array(
        [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]],
        dtype=np.float64,
    )
    return np.eye(3) + K + K @ K * (1 / (1 + c))


def fit_floor_plane(points: np.ndarray) -> tuple[np.ndarray, float]:
    """RANSAC floor plane; prefer near-horizontal normals (Stella Y-up world)."""
    n = len(points)
    if n < 100:
        return np.array([0.0, 1.0, 0.0]), float(np.percentile(points[:, 1], 2))

    rng = np.random.default_rng(0)
    idx = rng.choice(n, size=min(n, 12000), replace=False)
    sample = points[idx]

    best_inl = None
    best_n = np.array([0.0, 1.0, 0.0])
    best_d = 0.0
    best_score = -1.0

    for _ in range(600):
        i, j, k = rng.choice(len(sample), size=3, replace=False)
        p1, p2, p3 = sample[i], sample[j], sample[k]
        nvec = np.cross(p2 - p1, p3 - p1)
        norm = np.linalg.norm(nvec)
        if norm < 1e-6:
            continue
        nvec = nvec / norm
        if abs(nvec[1]) < 0.65:
            continue
        d = -float(np.dot(nvec, p1))
        dist = np.abs(sample @ nvec + d)
        inl = dist < 0.12
        score = float(inl.sum()) * abs(nvec[1])
        if score > best_score:
            best_score = score
            best_inl = inl
            best_n, best_d = nvec, d

    if best_n[1] < 0:
        best_n, best_d = -best_n, -best_d
    return best_n, best_d


def align_to_floor(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Stella dense cloud is already Y-up. Align detected floor plane to y=0;
    only rotate when the floor normal is noticeably tilted.
    """
    floor_n, _floor_d = fit_floor_plane(points)
    R = rotation_align_vector_to_axis(floor_n, np.array([0.0, 1.0, 0.0]))
    aligned = (R @ points.T).T
    floor_y = float(np.percentile(aligned[:, 1], 1))
    aligned[:, 1] -= floor_y
    return aligned, R, floor_y


def load_tum_positions(path: Path) -> tuple[np.ndarray, np.ndarray]:
    t, pos = [], []
    with path.open() as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 8:
                continue
            t.append(float(parts[0]))
            pos.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return np.asarray(t), np.asarray(pos, dtype=np.float64)


def transform_positions(
    pos: np.ndarray, R: np.ndarray, floor_y: float
) -> np.ndarray:
    out = (R @ pos.T).T
    out[:, 1] -= floor_y
    return out


def extract_lines(bev_u8: np.ndarray) -> np.ndarray:
    """Return Nx4 line segments in pixel indices (col, row)."""
    lines: list[list[float]] = []
    if hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "createFastLineDetector"):
        fld = cv2.ximgproc.createFastLineDetector(length_threshold=12)
        detected = fld.detect(bev_u8)
        if detected is not None:
            for seg in detected.reshape(-1, 4):
                lines.append(seg.tolist())
    if not lines:
        detected = cv2.HoughLinesP(
            bev_u8, 1, np.pi / 180, threshold=25, minLineLength=15, maxLineGap=6
        )
        if detected is not None:
            for seg in detected.reshape(-1, 4):
                lines.append(seg.tolist())
    if not lines:
        return np.empty((0, 4), dtype=np.float64)
    return np.asarray(lines, dtype=np.float64)


def run_bev_extract(
    ply: Path,
    traj: Path,
    imu_csv: Path | None,
    out_dir: Path,
    *,
    res_m: float = 0.05,
    height_min_m: float = 0.3,
    height_max_m: float = 2.5,
    max_points: int = 500_000,
    flip_x: bool = True,
) -> dict:
    del imu_csv  # Stella world frame is Y-up; IMU body frame must not rotate the cloud.
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pts = load_ply_xyz(ply, max_points=max_points)
    pts, R_floor, floor_y = align_to_floor(pts)

    t_s, cam_pos = load_tum_positions(traj)
    cam_pos = transform_positions(cam_pos, R_floor, floor_y)
    traj_xy = cam_pos[:, [0, 2]]
    if flip_x:
        # Mirror left/right so BEV matches walking view / floorplan handedness.
        pts[:, 0] *= -1
        traj_xy[:, 0] *= -1

    band = (pts[:, 1] >= height_min_m) & (pts[:, 1] <= height_max_m)
    wall_pts = pts[band][:, [0, 2]]

    if len(wall_pts) < 50:
        low = pts[:, 1] < height_max_m
        wall_pts = pts[low][:, [0, 2]]

    all_xy = np.vstack([wall_pts, traj_xy]) if len(wall_pts) else traj_xy
    pad = 1.0
    mn = all_xy.min(axis=0) - pad
    mx = all_xy.max(axis=0) + pad
    wh = np.ceil((mx - mn) / res_m).astype(int) + 1
    wh = np.clip(wh, 16, 2000)
    grid = np.zeros((int(wh[1]), int(wh[0])), dtype=np.uint8)

    if len(wall_pts):
        cells = np.floor((wall_pts - mn) / res_m).astype(int)
        for x, y in cells:
            if 0 <= x < grid.shape[1] and 0 <= y < grid.shape[0]:
                grid[y, x] = 1

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    grid = cv2.dilate(grid, k, iterations=1)

    lines_px = extract_lines((grid * 255).astype(np.uint8))
    lines_world = lines_px.copy()
    if len(lines_world):
        lines_world[:, [0, 2]] = lines_world[:, [0, 2]] * res_m + mn[0]
        lines_world[:, [1, 3]] = lines_world[:, [1, 3]] * res_m + mn[1]

    np.save(out_dir / "bev.npy", grid.astype(bool))
    write_json(
        out_dir / "bev_meta.json",
        {
            "origin_xy": mn.tolist(),
            "res_m": res_m,
            "shape": list(grid.shape),
            "floor_y_m": floor_y,
            "R_floor_align": R_floor.tolist(),
            "coord_note": "Stella Y-up; BEV uses X-Z; X flipped so left/right match the walk",
            "flip_x": flip_x,
        },
    )
    np.save(out_dir / "bev_lines.npy", lines_world)
    np.save(out_dir / "traj_xy_m.npy", traj_xy)
    np.save(out_dir / "traj_t_s.npy", t_s)

    vis = np.full((*grid.shape, 3), 30, np.uint8)
    vis[grid > 0] = (200, 200, 200)
    for x1, y1, x2, y2 in lines_px.astype(int):
        cv2.line(vis, (x1, y1), (x2, y2), (0, 80, 255), 1)

    traj_cells = np.floor((traj_xy - mn) / res_m).astype(int)
    for i in range(len(traj_cells) - 1):
        p1 = (int(traj_cells[i, 0]), int(traj_cells[i, 1]))
        p2 = (int(traj_cells[i + 1, 0]), int(traj_cells[i + 1, 1]))
        if 0 <= p1[0] < grid.shape[1] and 0 <= p1[1] < grid.shape[0]:
            cv2.circle(vis, p1, 2, (40, 200, 40), -1)
        cv2.line(vis, p1, p2, (40, 220, 40), 1)

    cv2.imwrite(str(out_dir / "bev.png"), vis)

    meta = {
        "n_points": int(len(pts)),
        "n_wall_band": int(band.sum()),
        "n_lines": int(len(lines_world)),
        "traj_samples": int(len(t_s)),
        "floor_tilt_deg": float(
            np.degrees(np.arccos(np.clip(R_floor[1, 1], -1, 1)))
        ),
    }
    write_json(out_dir / "bev_summary.json", meta)
    return meta


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--ply", type=Path, default=None)
    p.add_argument("--traj", type=Path, default=None)
    p.add_argument("--imu", type=Path, default=None)
    p.add_argument("--config", type=Path, default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    run = Path(args.run_dir)
    ply = args.ply or run / "out.ply"
    traj = args.traj or run / "traj" / "frame_trajectory.txt"
    imu = args.imu or Path(cfg.get("imu_default", ""))

    meta = run_bev_extract(
        ply,
        traj,
        imu if imu.exists() else None,
        run,
        res_m=float(cfg["bev_res_m"]),
        height_min_m=float(cfg["bev_height_min_m"]),
        height_max_m=float(cfg["bev_height_max_m"]),
        flip_x=bool(cfg.get("bev_flip_x", True)),
    )
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()

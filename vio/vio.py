"""Classical visual-inertial odometry: IMU heading + visual speed + ground occupancy."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from backend import OccupancyGrid, Reconstruction
from pdr import detect_steps, heading_from_gyro

LK = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
)
FEATURE = dict(maxCorners=350, qualityLevel=0.02, minDistance=8, blockSize=7)


def _preprocess(frame: np.ndarray, size: int = 640) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def _interp_yaw(t_query: np.ndarray, t_imu: np.ndarray, yaw: np.ndarray) -> np.ndarray:
    return np.interp(t_query, t_imu, np.unwrap(yaw))


def _visual_speed_px(prev: np.ndarray, cur: np.ndarray, pts: np.ndarray) -> tuple[float, np.ndarray]:
    nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev, cur, pts, None, **LK)
    if nxt is None:
        return 0.0, np.empty((0, 2), np.float32)
    good_old = pts[st.ravel() == 1]
    good_new = nxt[st.ravel() == 1]
    if len(good_old) < 8:
        return 0.0, good_new
    flow = good_new - good_old
    mag = np.linalg.norm(flow, axis=1)
    return float(np.median(mag)), good_new


def _project_ground_points(
    pts: np.ndarray,
    yaw: float,
    cam_xy: np.ndarray,
    img_size: int,
    cam_height_m: float = 1.70,
    fx_frac: float = 0.38,
) -> np.ndarray:
    """Back-project lower-half features onto a ground plane at z=0."""
    if pts is None or len(pts) == 0:
        return np.empty((0, 2))
    cx = cy = img_size / 2.0
    fx = fy = img_size * fx_frac
    p = pts.reshape(-1, 2)
    u, v = p[:, 0], p[:, 1]
    below = v > cy + 8
    p = p[below]
    if len(p) == 0:
        return np.empty((0, 2))
    u, v = p[:, 0], p[:, 1]
    ang_down = np.arctan((v - cy) / fy)
    ang_down = np.clip(ang_down, 0.08, 1.2)
    depth = cam_height_m / np.tan(ang_down)
    x_cam = (u - cx) / fx * depth
    z_cam = depth
    c, s = np.cos(yaw), np.sin(yaw)
    x_w = cam_xy[0] + z_cam * c - x_cam * s
    y_w = cam_xy[1] + z_cam * s + x_cam * c
    keep = depth < 12.0
    return np.stack([x_w[keep], y_w[keep]], axis=1)


def _occupancy_from_points(points: np.ndarray, traj: np.ndarray, res_m: float = 0.25) -> OccupancyGrid:
    pts = np.vstack([points, traj]) if len(points) else traj
    pad = 2.0
    mn = pts.min(axis=0) - pad
    mx = pts.max(axis=0) + pad
    wh = np.ceil((mx - mn) / res_m).astype(int) + 1
    wh = np.clip(wh, 8, 800)
    grid = np.zeros((int(wh[1]), int(wh[0])), dtype=bool)
    cells = np.floor((points - mn) / res_m).astype(int) if len(points) else np.empty((0, 2), int)
    for x, y in cells:
        if 0 <= x < grid.shape[1] and 0 <= y < grid.shape[0]:
            grid[y, x] = True
    if grid.any():
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        grid = cv2.dilate(grid.astype(np.uint8), k, iterations=1).astype(bool)
    return OccupancyGrid(grid=grid, origin_xy=mn, res_m=res_m)


class ClassicalVIOBackend:
    name = "classical_vio"

    def reconstruct(
        self,
        video: Path,
        imu: dict,
        out_dir: Path,
        fps: float = 8.0,
        step_length_m: float = 0.70,
        **_kwargs,
    ) -> Reconstruction:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        yaw_imu = heading_from_gyro(imu["acc"], imu["gyro"], imu["t_s"])
        steps = detect_steps(imu["acc"], imu["t_s"])
        step_times = imu["t_s"][steps] if len(steps) else np.array([])

        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open {video}")
        video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        stride = max(int(round(video_fps / fps)), 1)
        dt_nom = stride / video_fps

        prev_gray = None
        prev_pts = None
        prev_t = None
        xs: list[float] = [0.0]
        ys: list[float] = [0.0]
        yaws: list[float] = []
        times: list[float] = []
        ground_pts: list[np.ndarray] = []

        idx = 0
        used = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % stride != 0:
                idx += 1
                continue
            t = idx / video_fps
            gray = _preprocess(frame)
            yaw = float(np.interp(t, imu["t_s"], np.unwrap(yaw_imu)))
            if not yaws:
                yaws.append(yaw)
                times.append(t)
                idx += 1
                prev_gray = gray
                prev_pts = cv2.goodFeaturesToTrack(gray, **FEATURE)
                prev_t = t
                continue

            dt = max(t - prev_t, 1e-3)
            dyaw = yaw - yaws[-1]
            h, w = gray.shape
            fx = 0.38 * w
            rot_px = abs(dyaw) * fx

            vis_px = 0.0
            tracked = np.empty((0, 2), np.float32)
            if prev_pts is not None and len(prev_pts):
                vis_px, tracked = _visual_speed_px(prev_gray, gray, prev_pts)
            trans_px = max(vis_px - 0.6 * rot_px, 0.0)
            v_vis = float(np.clip((trans_px / dt) * 0.045, 0.0, 2.4))

            n_steps = int(np.sum((step_times > prev_t) & (step_times <= t))) if len(step_times) else 0
            v_step = (n_steps * step_length_m) / dt
            if n_steps > 0:
                speed = 0.35 * v_vis + 0.65 * v_step
            elif v_vis > 0.28:
                speed = v_vis
            else:
                speed = 0.0
            speed = float(np.clip(speed, 0.0, 2.5))

            xs.append(xs[-1] + speed * dt * np.cos(yaw))
            ys.append(ys[-1] + speed * dt * np.sin(yaw))
            yaws.append(yaw)
            times.append(t)

            cam = np.array([xs[-1], ys[-1]])
            gpts = _project_ground_points(tracked, yaw, cam, gray.shape[0])
            if len(gpts):
                ground_pts.append(gpts)

            prev_gray = gray
            prev_pts = cv2.goodFeaturesToTrack(gray, **FEATURE)
            prev_t = t
            used += 1
            idx += 1
        cap.release()

        t_s = np.asarray(times, dtype=np.float64)
        xy = np.stack([xs[: len(t_s)], ys[: len(t_s)]], axis=1)
        yaw = np.asarray(yaws, dtype=np.float64)
        gstack = np.vstack(ground_pts) if ground_pts else np.empty((0, 2))
        occ = _occupancy_from_points(gstack, xy)

        rec = Reconstruction(
            t_s=t_s,
            xy_m=xy,
            yaw_rad=yaw,
            occupancy=occ,
            meta={
                "backend": self.name,
                "n_frames": int(len(t_s)),
                "video_fps": float(video_fps),
                "stride": int(stride),
                "path_length_m": float(np.sum(np.linalg.norm(np.diff(xy, axis=0), axis=1))),
                "n_ground_points": int(len(gstack)),
                "processed_pairs": int(used),
            },
        )
        _save_vio_artifacts(out_dir, rec)
        return rec


def _save_vio_artifacts(out_dir: Path, rec: Reconstruction) -> None:
    import csv

    import matplotlib.pyplot as plt

    with (out_dir / "trajectory_vio.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_s", "x_m", "y_m", "yaw_rad"])
        for i in range(len(rec.t_s)):
            w.writerow(
                [
                    f"{rec.t_s[i]:.4f}",
                    f"{rec.xy_m[i, 0]:.4f}",
                    f"{rec.xy_m[i, 1]:.4f}",
                    f"{rec.yaw_rad[i]:.6f}",
                ]
            )
    if rec.occupancy is not None:
        np.save(out_dir / "occupancy.npy", rec.occupancy.grid)
        img = np.zeros((*rec.occupancy.grid.shape, 3), np.uint8)
        img[rec.occupancy.grid] = (220, 220, 220)
        cv2.imwrite(str(out_dir / "occupancy.png"), img)
        (out_dir / "occupancy_meta.json").write_text(
            __import__("json").dumps(
                {
                    "origin_xy": rec.occupancy.origin_xy.tolist(),
                    "res_m": rec.occupancy.res_m,
                    "shape": list(rec.occupancy.grid.shape),
                },
                indent=2,
            )
        )
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(rec.xy_m[:, 0], rec.xy_m[:, 1], color="#e10600", lw=1.5)
    ax.scatter(rec.xy_m[0, 0], rec.xy_m[0, 1], c="#2ecc71", s=30, zorder=3, label="start")
    ax.scatter(rec.xy_m[-1, 0], rec.xy_m[-1, 1], c="#3498db", s=30, zorder=3, label="end")
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(f"VIO path  {rec.meta.get('path_length_m', 0):.1f} m")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "trajectory_vio.png", dpi=120)
    plt.close(fig)
    (out_dir / "vio_meta.json").write_text(__import__("json").dumps(rec.meta, indent=2))


def load_reconstruction(out_dir: Path) -> Reconstruction:
    import csv

    t, x, y, yaw = [], [], [], []
    with (out_dir / "trajectory_vio.csv").open() as f:
        for row in csv.DictReader(f):
            t.append(float(row["t_s"]))
            x.append(float(row["x_m"]))
            y.append(float(row["y_m"]))
            yaw.append(float(row["yaw_rad"]))
    occ = None
    grid_path = out_dir / "occupancy.npy"
    meta_path = out_dir / "occupancy_meta.json"
    if grid_path.exists() and meta_path.exists():
        meta = __import__("json").loads(meta_path.read_text())
        occ = OccupancyGrid(
            grid=np.load(grid_path),
            origin_xy=np.asarray(meta["origin_xy"], dtype=np.float64),
            res_m=float(meta["res_m"]),
        )
    return Reconstruction(
        t_s=np.asarray(t),
        xy_m=np.stack([x, y], axis=1),
        yaw_rad=np.asarray(yaw),
        occupancy=occ,
        meta=__import__("json").loads((out_dir / "vio_meta.json").read_text())
        if (out_dir / "vio_meta.json").exists()
        else {},
    )

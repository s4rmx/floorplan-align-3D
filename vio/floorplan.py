"""Turn a raster floorplan into a wall occupancy mask."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def walls_from_floorplan(
    path: Path,
    out_dir: Path | None = None,
    dilate_px: int = 1,
) -> dict:
    """
    Walls are dark ink (black/grey lines). Free space is the light paper.
    Returns bool mask True = wall / blocked.
    """
    path = Path(path)
    rgba = np.array(Image.open(path).convert("RGBA"))
    rgb = rgba[:, :, :3].astype(np.float32)
    alpha = rgba[:, :, 3]
    gray = cv2.cvtColor(rgba[:, :, :3], cv2.COLOR_RGB2GRAY)

    # Ink is dark; ignore near-white paper and fully transparent pixels.
    dark = gray < 170
    not_white = rgb.min(axis=2) < 230
    opaque = alpha > 16
    walls = dark & not_white & opaque

    # Thin lines → slightly thicker so trajectory tests are conservative.
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    walls_u8 = walls.astype(np.uint8)
    if dilate_px > 0:
        walls_u8 = cv2.dilate(walls_u8, k, iterations=dilate_px)

    walls_u8 = cv2.morphologyEx(walls_u8, cv2.MORPH_OPEN, k, iterations=1)

    # Drop letter-sized blobs so room labels are not treated as walls.
    nlab, labels, stats, _ = cv2.connectedComponentsWithStats(walls_u8, connectivity=8)
    keep = np.zeros_like(walls_u8)
    for i in range(1, nlab):
        area = int(stats[i, cv2.CC_STAT_AREA])
        ww = int(stats[i, cv2.CC_STAT_WIDTH])
        hh = int(stats[i, cv2.CC_STAT_HEIGHT])
        if area >= 100 or max(ww, hh) >= 36:
            keep[labels == i] = 1
    walls_u8 = keep
    walls = walls_u8.astype(bool)

    free = (~walls).astype(np.uint8)
    dist_free = cv2.distanceTransform(free, cv2.DIST_L2, 5)

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / "walls.npy", walls)
        vis = np.repeat(gray[:, :, None], 3, axis=2)
        vis[walls] = (40, 40, 200)
        cv2.imwrite(str(out_dir / "walls.png"), vis)
        free_img = np.full((*walls.shape, 3), 255, np.uint8)
        free_img[walls] = (20, 20, 20)
        cv2.imwrite(str(out_dir / "free_space.png"), free_img)

    return {
        "walls": walls,
        "gray": gray,
        "dist_free": dist_free,
        "size": (int(walls.shape[1]), int(walls.shape[0])),
    }


def load_walls(out_dir: Path) -> np.ndarray:
    return np.load(Path(out_dir) / "walls.npy")


def free_space_score(xy_px: np.ndarray, walls: np.ndarray) -> float:
    """Fraction of path samples that land on free pixels (in-bounds)."""
    h, w = walls.shape
    x = np.clip(np.round(xy_px[:, 0]).astype(int), 0, w - 1)
    y = np.clip(np.round(xy_px[:, 1]).astype(int), 0, h - 1)
    inb = (
        (xy_px[:, 0] >= 0)
        & (xy_px[:, 0] < w)
        & (xy_px[:, 1] >= 0)
        & (xy_px[:, 1] < h)
    )
    if not np.any(inb):
        return 0.0
    free = ~walls[y[inb], x[inb]]
    return float(free.mean() * inb.mean())

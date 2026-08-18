"""Nudge an aligned pixel path so it stays in free space without inventing a new walk."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt


def refine_off_walls(
    xy_px: np.ndarray,
    walls: np.ndarray,
    landmark_px: np.ndarray | None = None,
    max_iter: int = 25,
    smooth: float = 0.35,
    max_shift_px: float = 18.0,
) -> np.ndarray:
    """
    Snap samples that sit on walls to the nearest free pixel, then lightly
    smooth. Landmark pixels (if given) stay pinned.
    """
    h, w = walls.shape
    orig = xy_px.astype(np.float64).copy()
    pts = orig.copy()
    _, inds = distance_transform_edt(walls, return_indices=True)

    pin_idx: list[int] = []
    if landmark_px is not None and len(landmark_px):
        for lp in landmark_px:
            pin_idx.append(int(np.argmin(np.linalg.norm(orig - lp[None, :], axis=1))))

    def _clamp(p):
        p[:, 0] = np.clip(p[:, 0], 0, w - 1)
        p[:, 1] = np.clip(p[:, 1], 0, h - 1)
        return p

    def _pin(p):
        if landmark_px is None:
            return p
        for i, lp in zip(pin_idx, landmark_px):
            p[i] = lp
        return p

    pts = _clamp(pts)
    for _ in range(max_iter):
        xi = np.clip(np.round(pts[:, 0]).astype(int), 0, w - 1)
        yi = np.clip(np.round(pts[:, 1]).astype(int), 0, h - 1)
        on_wall = walls[yi, xi]
        if np.any(on_wall):
            ny = inds[0, yi[on_wall], xi[on_wall]]
            nx = inds[1, yi[on_wall], xi[on_wall]]
            pts[on_wall, 0] = nx
            pts[on_wall, 1] = ny
        pts = _pin(pts)

        sm = pts.copy()
        sm[1:-1] = (1 - smooth) * pts[1:-1] + 0.5 * smooth * (pts[:-2] + pts[2:])
        shift = sm - orig
        nrm = np.linalg.norm(shift, axis=1, keepdims=True)
        nrm = np.maximum(nrm, 1e-6)
        too = nrm.ravel() > max_shift_px
        sm[too] = orig[too] + shift[too] * (max_shift_px / nrm[too])
        pts = _clamp(_pin(sm))

    return pts

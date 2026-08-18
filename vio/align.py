"""SE(2)+scale alignment of a VIO path onto floorplan pixels."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from floorplan import free_space_score


@dataclass
class Similarity:
    scale: float
    R: np.ndarray  # 2x2
    t: np.ndarray  # (2,)
    reflected: bool
    score: float = 0.0

    def apply(self, xy: np.ndarray) -> np.ndarray:
        return (self.scale * (self.R @ xy.T).T) + self.t


def interpolate_world(t_s: np.ndarray, xy: np.ndarray, t_query: float) -> np.ndarray:
    t_query = float(np.clip(t_query, t_s[0], t_s[-1]))
    x = np.interp(t_query, t_s, xy[:, 0])
    y = np.interp(t_query, t_s, xy[:, 1])
    return np.array([x, y], dtype=np.float64)


def load_landmarks(path: Path) -> list[dict]:
    data = json.loads(Path(path).read_text())
    if isinstance(data, dict):
        data = data.get("landmarks", [])
    out = []
    for row in data:
        out.append(
            {
                "t_s": float(row["t_s"]),
                "px": float(row["px"]),
                "py": float(row["py"]),
                "label": row.get("label", ""),
            }
        )
    if len(out) < 2:
        raise ValueError("Need at least 2 landmarks (t_s, px, py)")
    return out


def save_landmarks(path: Path, landmarks: list[dict]) -> None:
    Path(path).write_text(json.dumps({"landmarks": landmarks}, indent=2))


def umeyama(src: np.ndarray, dst: np.ndarray, allow_reflection: bool) -> tuple[float, np.ndarray, np.ndarray]:
    """Least-squares 2D similarity mapping src -> dst."""
    n = src.shape[0]
    mu_s = src.mean(axis=0)
    mu_d = dst.mean(axis=0)
    src_c = src - mu_s
    dst_c = dst - mu_d
    var_s = float(np.sum(src_c**2) / n)
    cov = (dst_c.T @ src_c) / n
    U, d, Vt = np.linalg.svd(cov)
    S = np.eye(2)
    if not allow_reflection and np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[1, 1] = -1
    R = U @ S @ Vt
    scale = 1.0 if var_s < 1e-12 else float(np.trace(np.diag(d) @ S) / var_s)
    t = mu_d - scale * (R @ mu_s)
    return scale, R, t


def _two_point_similarity(src: np.ndarray, dst: np.ndarray, reflect: bool) -> tuple[float, np.ndarray, np.ndarray]:
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


def fit_similarity(src: np.ndarray, dst: np.ndarray, reflect: bool) -> tuple[float, np.ndarray, np.ndarray]:
    if len(src) == 2:
        return _two_point_similarity(src, dst, reflect)
    return umeyama(src, dst, allow_reflection=reflect)


def align_with_landmarks(
    t_s: np.ndarray,
    xy_m: np.ndarray,
    landmarks: list[dict],
    walls: np.ndarray | None = None,
) -> tuple[Similarity, np.ndarray]:
    src = np.stack([interpolate_world(t_s, xy_m, lm["t_s"]) for lm in landmarks])
    dst = np.stack([[lm["px"], lm["py"]] for lm in landmarks])

    candidates: list[Similarity] = []
    for reflect in (False, True):
        scale, R, t = fit_similarity(src, dst, reflect)
        sim = Similarity(scale=scale, R=R, t=t, reflected=reflect)
        pred = sim.apply(src)
        rmse = float(np.sqrt(np.mean(np.sum((pred - dst) ** 2, axis=1))))
        if walls is not None:
            pix = sim.apply(xy_m)
            sim.score = free_space_score(pix, walls) - 0.0005 * rmse
        else:
            sim.score = -rmse
        candidates.append(sim)

    best = max(candidates, key=lambda s: s.score)
    return best, best.apply(xy_m)


def similarity_to_dict(sim: Similarity) -> dict:
    ang = float(np.arctan2(sim.R[1, 0], sim.R[0, 0]))
    return {
        "scale_px_per_m": sim.scale,
        "rotation_deg": float(np.degrees(ang)),
        "translation_px": sim.t.tolist(),
        "reflected": sim.reflected,
        "free_space_score": sim.score,
        "R": sim.R.tolist(),
    }

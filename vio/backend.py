"""Reconstruction backends: classical VIO now, MASt3R later."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np


@dataclass
class OccupancyGrid:
    """Top-down occupancy in the VIO/world frame. True = occupied (wall/structure)."""

    grid: np.ndarray  # (H, W) bool
    origin_xy: np.ndarray  # world xy of grid[0, 0] (min x, min y)
    res_m: float

    def world_to_cell(self, xy: np.ndarray) -> np.ndarray:
        cells = np.floor((xy - self.origin_xy) / self.res_m).astype(np.int32)
        return cells

    def cell_to_world(self, cells: np.ndarray) -> np.ndarray:
        return (cells.astype(np.float64) + 0.5) * self.res_m + self.origin_xy


@dataclass
class Reconstruction:
    t_s: np.ndarray
    xy_m: np.ndarray
    yaw_rad: np.ndarray
    occupancy: OccupancyGrid | None = None
    meta: dict = field(default_factory=dict)


class ReconstructionBackend(Protocol):
    name: str

    def reconstruct(self, video: Path, imu: dict, out_dir: Path, **kwargs) -> Reconstruction:
        """Build a metric-ish 2D trajectory (and optional occupancy) from video + IMU."""
        ...


def get_backend(name: str) -> ReconstructionBackend:
    key = name.lower().strip()
    if key in {"classical", "vio", "classical_vio"}:
        from vio import ClassicalVIOBackend

        return ClassicalVIOBackend()
    if key in {"mast3r", "mast3r_slam"}:
        from mast3r_backend import Mast3rBackend

        return Mast3rBackend()
    raise ValueError(f"Unknown reconstruction backend: {name}")

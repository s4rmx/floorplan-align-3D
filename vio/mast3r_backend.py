"""MASt3R sparse-global-alignment reconstruction backend."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from backend import Reconstruction
from mast3r_odom import reconstruct_mast3r
from vio import _occupancy_from_points, _save_vio_artifacts


class Mast3rBackend:
    name = "mast3r"

    def reconstruct(self, video: Path, imu: dict, out_dir: Path, **kwargs) -> Reconstruction:
        fps = float(kwargs.get("mast3r_fps", kwargs.get("fps", 2.0)))
        max_frames = int(kwargs.get("max_frames", 50))
        winsize = int(kwargs.get("winsize", 5))

        result = reconstruct_mast3r(
            Path(video),
            Path(out_dir),
            fps=fps,
            max_frames=max_frames,
            winsize=winsize,
            keep_frames=True,
        )

        xy = result["xy_m"]
        occ = _occupancy_from_points(np.empty((0, 2)), xy)
        rec = Reconstruction(
            t_s=result["t_s"],
            xy_m=xy,
            yaw_rad=result["yaw_rad"],
            occupancy=occ,
            meta=result["meta"],
        )
        _save_vio_artifacts(out_dir, rec)
        np.save(out_dir / "mast3r_cams2world.npy", result["cams2world"])
        return rec

"""Pedestrian dead reckoning from Insta360 IMU (accel + gyro)."""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks


def _butter(data: np.ndarray, cutoff_hz: float, fs: float, btype: str, order: int = 2):
    nyq = 0.5 * fs
    wn = min(cutoff_hz / nyq, 0.99)
    b, a = butter(order, wn, btype=btype)
    return filtfilt(b, a, data, axis=0)


def estimate_gyro_bias(gyro: np.ndarray, t_s: np.ndarray, window_s: float = 1.0) -> np.ndarray:
    """Use the lowest-variance window as a standing-still bias estimate."""
    dt = np.median(np.diff(t_s))
    w = max(int(window_s / dt), 8)
    mag = np.linalg.norm(gyro, axis=1)
    best = None
    best_var = np.inf
    for i in range(0, max(len(mag) - w, 1), max(w // 4, 1)):
        var = float(np.var(mag[i : i + w]))
        if var < best_var:
            best_var = var
            best = gyro[i : i + w].mean(axis=0)
    return best if best is not None else gyro[:w].mean(axis=0)


def heading_from_gyro(acc: np.ndarray, gyro: np.ndarray, t_s: np.ndarray) -> np.ndarray:
    """Yaw by integrating gyro around gravity (low-passed accel)."""
    dt = np.diff(t_s, prepend=t_s[0])
    dt[0] = dt[1] if len(dt) > 1 else 0.001
    fs = 1.0 / np.median(dt[1:])
    gyro = gyro - estimate_gyro_bias(gyro, t_s)
    gvec = _butter(acc, cutoff_hz=0.5, fs=fs, btype="low")
    gnorm = np.linalg.norm(gvec, axis=1, keepdims=True)
    gnorm = np.clip(gnorm, 1e-6, None)
    ghat = gvec / gnorm
    yaw_rate = np.sum(gyro * ghat, axis=1)
    yaw = np.cumsum(yaw_rate * dt)
    yaw -= yaw[0]
    return yaw


def detect_steps(acc: np.ndarray, t_s: np.ndarray, min_step_s: float = 0.30) -> np.ndarray:
    dt = np.median(np.diff(t_s))
    fs = 1.0 / dt
    mag = np.linalg.norm(acc, axis=1)
    mag = mag - np.mean(mag[: min(len(mag), int(fs))])
    band = _butter(mag, cutoff_hz=3.0, fs=fs, btype="low")
    distance = max(int(min_step_s * fs), 1)
    height = max(0.04, 0.35 * np.std(band))
    peaks, _ = find_peaks(band, distance=distance, height=height)
    return peaks


def dead_reckon(
    t_s: np.ndarray,
    acc: np.ndarray,
    gyro: np.ndarray,
    step_length_m: float = 0.70,
) -> dict:
    yaw = heading_from_gyro(acc, gyro, t_s)
    steps = detect_steps(acc, t_s)
    xy = np.zeros((len(t_s), 2), dtype=np.float64)
    x = y = 0.0
    step_i = 0
    for i in range(len(t_s)):
        if step_i < len(steps) and i == steps[step_i]:
            heading = yaw[i]
            x += step_length_m * np.cos(heading)
            y += step_length_m * np.sin(heading)
            step_i += 1
        xy[i] = (x, y)
    return {
        "yaw_rad": yaw,
        "steps": steps,
        "xy_m": xy,
        "step_length_m": step_length_m,
        "n_steps": int(len(steps)),
        "path_length_m": float(len(steps) * step_length_m),
    }

"""Extract IMU/exposure from an INSV, run PDR, overlay on a floorplan."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from insv import dump_records_summary, parse_exposure, parse_imu_20, parse_maker_notes, read_trailer
from pdr import dead_reckon

ROOT = Path(__file__).resolve().parents[1]


def write_imu_csv(path: Path, imu: dict) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_s", "ts_us", "acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"])
        for i in range(len(imu["t_s"])):
            ax, ay, az = imu["acc"][i]
            gx, gy, gz = imu["gyro"][i]
            w.writerow(
                [
                    f"{imu['t_s'][i]:.6f}",
                    int(imu["ts_us"][i]),
                    f"{ax:.6f}",
                    f"{ay:.6f}",
                    f"{az:.6f}",
                    f"{gx:.6f}",
                    f"{gy:.6f}",
                    f"{gz:.6f}",
                ]
            )


def write_traj_csv(path: Path, imu: dict, traj: dict) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_s", "x_m", "y_m", "yaw_rad", "is_step"])
        step_set = set(int(i) for i in traj["steps"])
        for i in range(len(imu["t_s"])):
            w.writerow(
                [
                    f"{imu['t_s'][i]:.6f}",
                    f"{traj['xy_m'][i, 0]:.4f}",
                    f"{traj['xy_m'][i, 1]:.4f}",
                    f"{traj['yaw_rad'][i]:.6f}",
                    int(i in step_set),
                ]
            )


def extract_keyframes(video: Path, out_dir: Path, every_s: float = 2.0) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(int(round(fps * every_s)), 1)
    saved = []
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % step == 0:
            t = i / fps
            name = f"kf_{t:06.2f}s.jpg"
            dest = out_dir / name
            cv2.imwrite(str(dest), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            saved.append(name)
        i += 1
    cap.release()
    return saved


def overlay_floorplan(floorplan: Path, xy_m: np.ndarray, out_path: Path) -> None:
    img = np.array(Image.open(floorplan).convert("RGBA"))
    h, w = img.shape[:2]
    xs, ys = xy_m[:, 0], xy_m[:, 1]
    span = max(np.ptp(xs), np.ptp(ys), 1.0)
    margin = 40
    scale = (min(w, h) - 2 * margin) / span
    # Floorplan y grows down; world y grows up.
    px = (xs - xs.min()) * scale + (w - np.ptp(xs) * scale) / 2
    py = h - ((ys - ys.min()) * scale + (h - np.ptp(ys) * scale) / 2)

    fig, ax = plt.subplots(figsize=(10, 8.2), dpi=120)
    ax.imshow(img)
    ax.plot(px, py, color="#e10600", linewidth=2.0, alpha=0.9, label="PDR walk")
    ax.scatter([px[0]], [py[0]], c="#2ecc71", s=40, zorder=3, label="start")
    ax.scatter([px[-1]], [py[-1]], c="#3498db", s=40, zorder=3, label="end")
    ax.set_axis_off()
    ax.legend(loc="lower right", framealpha=0.9)
    ax.set_title("IMU pedestrian path on floorplan (unaligned similarity fit)")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_imu(imu: dict, traj: dict, out_path: Path) -> None:
    t = imu["t_s"]
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    axes[0].plot(t, imu["acc"])
    axes[0].set_ylabel("acc (g)")
    axes[0].legend(["x", "y", "z"], loc="upper right")
    axes[1].plot(t, imu["gyro"])
    axes[1].set_ylabel("gyro (rad/s)")
    axes[1].legend(["x", "y", "z"], loc="upper right")
    axes[2].plot(t, np.degrees(traj["yaw_rad"]), color="purple")
    axes[2].set_ylabel("yaw (deg)")
    if len(traj["steps"]):
        axes[2].scatter(t[traj["steps"]], np.degrees(traj["yaw_rad"][traj["steps"]]), s=8, c="k")
    axes[3].plot(t, traj["xy_m"][:, 0], label="x")
    axes[3].plot(t, traj["xy_m"][:, 1], label="y")
    axes[3].set_ylabel("pos (m)")
    axes[3].set_xlabel("t (s)")
    axes[3].legend(loc="upper right")
    fig.suptitle(f"IMU / PDR  steps={traj['n_steps']}  path={traj['path_length_m']:.1f} m")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--insv", type=Path, required=True)
    p.add_argument("--lrv", type=Path, default=None)
    p.add_argument("--floorplan", type=Path, default=ROOT / "floorplan-gf-maaksons.png")
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--step-length", type=float, default=0.70)
    p.add_argument("--kf-every", type=float, default=2.0)
    args = p.parse_args()

    out = args.out or (ROOT / "outputs" / args.insv.stem)
    out.mkdir(parents=True, exist_ok=True)

    trailer_len, records = read_trailer(args.insv)
    summary = dump_records_summary(records)
    meta = {"trailer_bytes": trailer_len, "records": summary}

    imu_rec = records.get(0x3) or records.get(0x300)
    if imu_rec is None:
        raise SystemExit("No IMU record (0x03 / 0x300) in trailer")
    imu = parse_imu_20(imu_rec.payload)

    exp_rec = records.get(0x4) or records.get(0x400)
    if exp_rec is not None:
        exp = parse_exposure(exp_rec.payload)
        with (out / "exposure.csv").open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t_s", "ts_us", "exposure_s"])
            for i in range(len(exp["t_s"])):
                w.writerow([f"{exp['t_s'][i]:.6f}", int(exp["ts_us"][i]), f"{exp['exp_s'][i]:.8f}"])
        meta["n_exposure"] = int(len(exp["t_s"]))

    notes_rec = records.get(0x101)
    if notes_rec is not None:
        meta["maker_notes"] = parse_maker_notes(notes_rec.payload)

    write_imu_csv(out / "imu.csv", imu)
    traj = dead_reckon(imu["t_s"], imu["acc"], imu["gyro"], step_length_m=args.step_length)
    write_traj_csv(out / "trajectory.csv", imu, traj)
    plot_imu(imu, traj, out / "imu_pdr.png")
    overlay_floorplan(args.floorplan, traj["xy_m"], out / "trajectory_on_floorplan.png")

    meta["imu"] = {
        "n_samples": int(len(imu["t_s"])),
        "duration_s": float(imu["t_s"][-1]),
        "rate_hz": float((len(imu["t_s"]) - 1) / imu["t_s"][-1]),
        "acc_mean": imu["acc"].mean(axis=0).tolist(),
        "gyro_mean": imu["gyro"].mean(axis=0).tolist(),
    }
    meta["pdr"] = {
        "n_steps": traj["n_steps"],
        "step_length_m": traj["step_length_m"],
        "path_length_m": traj["path_length_m"],
        "end_xy_m": traj["xy_m"][-1].tolist(),
        "yaw_end_deg": float(np.degrees(traj["yaw_rad"][-1])),
    }

    if args.lrv and args.lrv.exists():
        kfs = extract_keyframes(args.lrv, out / "keyframes", every_s=args.kf_every)
        meta["keyframes"] = {"n": len(kfs), "every_s": args.kf_every, "source": str(args.lrv)}

    (out / "summary.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps({k: meta[k] for k in ("imu", "pdr") if k in meta}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

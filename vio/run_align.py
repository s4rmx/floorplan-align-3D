"""Extract IMU if needed, run VIO, align to floorplan with landmarks, refine off walls."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from align import align_with_landmarks, load_landmarks, similarity_to_dict
from backend import get_backend
from floorplan import walls_from_floorplan
from insv import parse_imu_20, read_trailer
from optimize import refine_off_walls
from run_walk import extract_keyframes
from vio import load_reconstruction

ROOT = Path(__file__).resolve().parents[1]


def resolve_out_dir(insv: Path, out: Path | None, run: str | None) -> Path:
    if out is not None:
        return out
    if run:
        return ROOT / "outputs" / run
    return ROOT / "outputs" / insv.stem


def load_imu_csv(path: Path) -> dict:
    t, acc, gyro, ts = [], [], [], []
    with path.open() as f:
        for row in csv.DictReader(f):
            t.append(float(row["t_s"]))
            ts.append(int(row["ts_us"]))
            acc.append([float(row["acc_x"]), float(row["acc_y"]), float(row["acc_z"])])
            gyro.append([float(row["gyro_x"]), float(row["gyro_y"]), float(row["gyro_z"])])
    return {
        "t_s": np.asarray(t),
        "ts_us": np.asarray(ts),
        "acc": np.asarray(acc),
        "gyro": np.asarray(gyro),
    }


def ensure_imu(insv: Path, out: Path) -> dict:
    csv_path = out / "imu.csv"
    if csv_path.exists():
        return load_imu_csv(csv_path)
    _trailer, records = read_trailer(insv)
    rec = records.get(0x3) or records.get(0x300)
    if rec is None:
        raise SystemExit("No IMU record in INSV")
    imu = parse_imu_20(rec.payload)
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_s", "ts_us", "acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"])
        for i in range(len(imu["t_s"])):
            ax, ay, az = imu["acc"][i]
            gx, gy, gz = imu["gyro"][i]
            w.writerow(
                [f"{imu['t_s'][i]:.6f}", int(imu["ts_us"][i]), ax, ay, az, gx, gy, gz]
            )
    return imu


def parse_landmark_arg(raw: str) -> dict:
    parts = raw.split(",")
    if len(parts) < 3:
        raise argparse.ArgumentTypeError("landmark must be t,px,py[,label]")
    label = parts[3] if len(parts) > 3 else f"t={parts[0]}s"
    return {
        "t_s": float(parts[0]),
        "px": float(parts[1]),
        "py": float(parts[2]),
        "label": label,
    }


def overlay(
    floorplan: Path,
    paths: list[tuple[np.ndarray, str, str]],
    landmarks: list[dict] | None,
    out_path: Path,
    title: str,
) -> None:
    img = np.array(Image.open(floorplan).convert("RGBA"))
    fig, ax = plt.subplots(figsize=(10, 8.2), dpi=120)
    ax.imshow(img)
    for xy, color, label in paths:
        ax.plot(xy[:, 0], xy[:, 1], color=color, lw=2.0, alpha=0.9, label=label)
        ax.scatter([xy[0, 0]], [xy[0, 1]], c="#2ecc71", s=36, zorder=4)
        ax.scatter([xy[-1, 0]], [xy[-1, 1]], c="#3498db", s=36, zorder=4)
    if landmarks:
        for lm in landmarks:
            ax.scatter([lm["px"]], [lm["py"]], c="#f1c40f", s=48, zorder=5, edgecolors="k")
            ax.annotate(lm.get("label", ""), (lm["px"], lm["py"]), fontsize=7, color="#333")
    ax.set_axis_off()
    ax.legend(loc="lower right", framealpha=0.9)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--insv", type=Path, required=True)
    p.add_argument("--lrv", type=Path, default=None, help="Preview video for keyframe extraction (click UI)")
    p.add_argument("--floorplan", type=Path, default=ROOT / "floorplan-gf-maaksons.png")
    p.add_argument("--out", type=Path, default=None, help="Output dir (overrides --run)")
    p.add_argument(
        "--run",
        type=str,
        default=None,
        help="Output subfolder under outputs/, e.g. run2-capture -> outputs/run2-capture/",
    )
    p.add_argument("--backend", default="classical", help="classical | mast3r")
    p.add_argument("--fps", type=float, default=8.0)
    p.add_argument("--landmarks", type=Path, default=None)
    p.add_argument(
        "--landmark",
        action="append",
        default=[],
        help="Repeatable t,px,py[,label]  (pixel coords on the floorplan PNG)",
    )
    p.add_argument("--reuse-vio", action="store_true", help="Skip VIO if trajectory_vio.csv exists")
    p.add_argument(
        "--wall-refine",
        action="store_true",
        help="Nudge path off wall pixels (off by default; fine for unfinished sites)",
    )
    p.add_argument(
        "--use-walls-for-flip",
        action="store_true",
        help="Pick mirrored vs normal alignment using wall free-space score (off by default)",
    )
    p.add_argument("--kf-every", type=float, default=2.0, help="Keyframe interval (seconds)")
    p.add_argument("--mast3r-fps", type=float, default=2.0, help="Frame rate for MASt3R (lower=faster)")
    p.add_argument("--mast3r-max-frames", type=int, default=50, help="Max frames sent to MASt3R")
    p.add_argument("--mast3r-winsize", type=int, default=5, help="MASt3R sliding-window pair size")
    args = p.parse_args()

    out = resolve_out_dir(args.insv, args.out, args.run)
    out.mkdir(parents=True, exist_ok=True)

    run_meta = {
        "run": args.run or args.insv.stem,
        "insv": str(args.insv.resolve()),
        "lrv": str(args.lrv.resolve()) if args.lrv else None,
        "floorplan": str(args.floorplan.resolve()),
        "out_dir": str(out.resolve()),
    }
    (out / "run_meta.json").write_text(json.dumps(run_meta, indent=2))

    imu = ensure_imu(args.insv, out)
    backend = get_backend(args.backend)

    vio_csv = out / "trajectory_vio.csv"
    if args.reuse_vio and vio_csv.exists():
        rec = load_reconstruction(out)
    else:
        rec = backend.reconstruct(
            args.insv,
            imu,
            out,
            fps=args.fps,
            mast3r_fps=args.mast3r_fps,
            max_frames=args.mast3r_max_frames,
            winsize=args.mast3r_winsize,
        )

    fp = walls_from_floorplan(args.floorplan, out_dir=out)

    if args.lrv and args.lrv.exists():
        kf_dir = out / "keyframes"
        kfs = extract_keyframes(args.lrv, kf_dir, every_s=args.kf_every)
        summary_kf = {"n": len(kfs), "every_s": args.kf_every, "dir": str(kf_dir)}
        run_meta["keyframes"] = summary_kf
        (out / "run_meta.json").write_text(json.dumps(run_meta, indent=2))

    landmarks = []
    if args.landmarks and args.landmarks.exists():
        landmarks = load_landmarks(args.landmarks)
    elif args.landmark:
        landmarks = [parse_landmark_arg(s) for s in args.landmark]
        if len(landmarks) < 2:
            raise SystemExit("Need at least two --landmark t,px,py")

    summary = {
        "backend": rec.meta,
        "floorplan": str(args.floorplan),
        "walls_px": int(fp["walls"].sum()),
    }

    if len(landmarks) < 2:
        # Unaligned stretch-to-fit so there is still a picture.
        xs, ys = rec.xy_m[:, 0], rec.xy_m[:, 1]
        h, w = fp["walls"].shape
        span = max(np.ptp(xs), np.ptp(ys), 1.0)
        margin = 40
        scale = (min(w, h) - 2 * margin) / span
        px = (xs - xs.min()) * scale + (w - np.ptp(xs) * scale) / 2
        py = h - ((ys - ys.min()) * scale + (h - np.ptp(ys) * scale) / 2)
        pix = np.stack([px, py], axis=1)
        overlay(
            args.floorplan,
            [(pix, "#e10600", "VIO (unaligned)")],
            None,
            out / "trajectory_aligned.png",
            "VIO on floorplan — unaligned (add --landmarks)",
        )
        summary["aligned"] = False
        summary["run"] = args.run or args.insv.stem
        summary["needs_landmarks"] = True
        (out / "align_summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary, indent=2))
        lm_path = out / "landmarks.json"
        kf_path = out / "keyframes"
        print(
            f"\nLandmarks required for THIS walk (each capture needs its own clicks).\n"
            f"  python src/click_align.py --keyframes {kf_path} --out {lm_path}\n"
            f"  python src/run_align.py --run {args.run or 'YOUR_RUN'} --reuse-vio "
            f"--insv {args.insv} --landmarks {lm_path}"
        )
        print(f"wrote {out}")
        return

    sim, pix = align_with_landmarks(
        rec.t_s,
        rec.xy_m,
        landmarks,
        fp["walls"] if args.use_walls_for_flip else None,
    )
    lm_px = np.stack([[lm["px"], lm["py"]] for lm in landmarks])
    pix_out = refine_off_walls(pix, fp["walls"], landmark_px=lm_px) if args.wall_refine else pix

    paths = [(pix_out, "#e10600", "VIO + clicks")]
    if args.wall_refine:
        paths.insert(0, (pix, "#e67e22", "before wall refine"))

    overlay(
        args.floorplan,
        paths,
        landmarks,
        out / "trajectory_aligned.png",
        "VIO aligned to floorplan (clicks)"
        + (" + wall refine" if args.wall_refine else ""),
    )
    np.save(out / "trajectory_aligned_px.npy", pix_out)
    with (out / "trajectory_aligned.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_s", "x_px", "y_px"])
        for i in range(len(rec.t_s)):
            w.writerow([f"{rec.t_s[i]:.4f}", f"{pix_out[i, 0]:.2f}", f"{pix_out[i, 1]:.2f}"])
    summary["aligned"] = True
    summary["wall_refine"] = bool(args.wall_refine)
    summary["use_walls_for_flip"] = bool(args.use_walls_for_flip)
    summary["similarity"] = similarity_to_dict(sim)
    summary["landmarks"] = landmarks
    (out / "align_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"wrote {out / 'trajectory_aligned.png'}")


if __name__ == "__main__":
    main()

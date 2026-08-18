"""Click 2–3 walk times onto the floorplan. Saves landmarks.json.

Uses matplotlib widgets (slider + buttons) — no OpenCV GUI required.

Controls
  Slider or Prev/Next buttons — pick keyframe time
  Click floorplan (left) — mark where you stood at that time
  Undo / Save / Quit buttons at the bottom
  Keys still work if the window has focus: n p z s q
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, Slider
from PIL import Image

from align import save_landmarks

ROOT = Path(__file__).resolve().parents[1]


def _list_keyframes(kf_dir: Path) -> list[tuple[float, Path]]:
    rows = []
    for p in sorted(kf_dir.glob("kf_*.jpg")):
        stem = p.stem
        t = float(stem.replace("kf_", "").replace("s", ""))
        rows.append((t, p))
    if not rows:
        raise SystemExit(f"No keyframes in {kf_dir}")
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--floorplan", type=Path, default=ROOT / "floorplan-gf-maaksons.png")
    p.add_argument("--keyframes", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True, help="landmarks.json path")
    args = p.parse_args()

    kfs = _list_keyframes(args.keyframes)
    plan = np.array(Image.open(args.floorplan).convert("RGB"))
    kf_images = [np.array(Image.open(path)) for _, path in kfs]

    idx = 0
    landmarks: list[dict] = []
    done = False
    _updating_slider = False

    fig = plt.figure(figsize=(14, 8.5))
    fig.canvas.manager.set_window_title("click_align — floorplan + keyframe")

    ax_plan = fig.add_axes([0.04, 0.22, 0.44, 0.72])
    ax_kf = fig.add_axes([0.52, 0.22, 0.44, 0.72])
    ax_slider = fig.add_axes([0.12, 0.13, 0.76, 0.028])
    ax_prev = fig.add_axes([0.12, 0.04, 0.09, 0.06])
    ax_next = fig.add_axes([0.22, 0.04, 0.09, 0.06])
    ax_undo = fig.add_axes([0.40, 0.04, 0.09, 0.06])
    ax_save = fig.add_axes([0.58, 0.04, 0.09, 0.06])
    ax_quit = fig.add_axes([0.68, 0.04, 0.09, 0.06])

    ax_plan.imshow(plan)
    kf_im = ax_kf.imshow(kf_images[0])
    ax_plan.set_title("Floorplan — click where you stood")
    for ax in (ax_plan, ax_kf):
        ax.set_axis_off()

    scatter = ax_plan.scatter([], [], c="#e74c3c", s=70, zorder=5, edgecolors="k")
    texts: list = []
    status = fig.text(0.5, 0.185, "", ha="center", fontsize=10)

    slider = Slider(
        ax_slider,
        "frame",
        0,
        len(kfs) - 1,
        valinit=0,
        valstep=1,
        valfmt="%0.0f",
    )
    btn_prev = Button(ax_prev, "◀ Prev")
    btn_next = Button(ax_next, "Next ▶")
    btn_undo = Button(ax_undo, "Undo")
    btn_save = Button(ax_save, "Save")
    btn_quit = Button(ax_quit, "Quit")

    def refresh():
        t, kpath = kfs[idx]
        kf_im.set_data(kf_images[idx])
        ax_kf.set_title(f"t = {t:.2f} s   ({idx + 1}/{len(kfs)})   {kpath.name}", fontsize=10)
        if landmarks:
            scatter.set_offsets(np.column_stack([[lm["px"] for lm in landmarks], [lm["py"] for lm in landmarks]]))
        else:
            scatter.set_offsets(np.empty((0, 2)))
        for t_art in texts:
            t_art.remove()
        texts.clear()
        for i, lm in enumerate(landmarks):
            texts.append(
                ax_plan.text(
                    lm["px"] + 6,
                    lm["py"] - 6,
                    f"{i + 1}: {lm['t_s']:.1f}s",
                    fontsize=8,
                    color="#c0392b",
                )
            )
        status.set_text(
            f"landmarks: {len(landmarks)}  |  slider or Prev/Next to change time  |  click floorplan to add"
        )
        fig.canvas.draw_idle()

    def set_idx(new_idx: int):
        nonlocal idx, _updating_slider
        idx = int(np.clip(new_idx, 0, len(kfs) - 1))
        _updating_slider = True
        slider.set_val(idx)
        _updating_slider = False
        refresh()

    def on_slider(val):
        nonlocal idx, _updating_slider
        if _updating_slider:
            return
        idx = int(val)
        refresh()

    def on_click(event):
        if event.inaxes is not ax_plan or event.xdata is None or event.ydata is None:
            return
        t, _ = kfs[idx]
        x, y = float(event.xdata), float(event.ydata)
        landmarks.append({"t_s": float(t), "px": x, "py": y, "label": f"t={t:.1f}s"})
        print(f"landmark {len(landmarks)}: t={t:.2f}s -> ({x:.1f}, {y:.1f})")
        refresh()

    def do_save():
        nonlocal done
        if len(landmarks) < 2:
            print("Need at least 2 landmarks before saving")
            return
        args.out.parent.mkdir(parents=True, exist_ok=True)
        save_landmarks(args.out, landmarks)
        print(f"saved {args.out}")
        done = True
        plt.close(fig)

    def on_key(event):
        nonlocal done
        key = (event.key or "").lower()
        if key in ("q", "escape"):
            done = True
            plt.close(fig)
        elif key in ("n", "right"):
            set_idx(idx + 1)
        elif key in ("p", "left"):
            set_idx(idx - 1)
        elif key == "z" and landmarks:
            landmarks.pop()
            print("undid last landmark")
            refresh()
        elif key == "s":
            do_save()

    def do_undo():
        if landmarks:
            landmarks.pop()
            print("undid last landmark")
            refresh()

    def do_quit():
        nonlocal done
        done = True
        plt.close(fig)

    slider.on_changed(on_slider)
    btn_prev.on_clicked(lambda _e: set_idx(idx - 1))
    btn_next.on_clicked(lambda _e: set_idx(idx + 1))
    btn_undo.on_clicked(lambda _e: do_undo())
    btn_save.on_clicked(lambda _e: do_save())
    btn_quit.on_clicked(lambda _e: do_quit())

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)

    refresh()
    print(f"Loaded {len(kfs)} keyframes. Use slider/buttons to change time, click floorplan to mark.")
    plt.show()

    if not done and len(landmarks) >= 2:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        save_landmarks(args.out, landmarks)
        print(f"saved {args.out} on exit")


if __name__ == "__main__":
    main()

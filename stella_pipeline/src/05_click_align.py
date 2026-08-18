"""Click BEV ↔ floorplan correspondences, then fit Sim(2).

Left = BEV (Stella top-down). Right = floorplan.
Click a distinctive point on the BEV, then the same place on the floorplan.
Need at least 2 pairs (3 is better).

Navigate
  Scroll           zoom under cursor
  Middle-drag      pan that panel
  Shift + drag     pan that panel
  Toolbar pan/zoom also works; left-click only adds a pair when toolbar is idle

Keys: z undo  s save+fit  q quit
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button
from PIL import Image

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))

from common import load_config, read_json  # noqa: E402


def _load_align():
    import importlib.util

    path = SRC / "03_auto_align.py"
    spec = importlib.util.spec_from_file_location("align_mod", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["align_mod"] = mod
    spec.loader.exec_module(mod)
    return mod


def bev_px_to_world(col: float, row: float, meta: dict) -> list[float]:
    origin = meta["origin_xy"]
    res = float(meta["res_m"])
    return [origin[0] + col * res, origin[1] + row * res]


def _toolbar_busy(fig) -> bool:
    tb = getattr(fig.canvas, "toolbar", None)
    mode = (getattr(tb, "mode", "") or "").lower()
    return bool(mode)


def _attach_pan_zoom(ax, fig):
    """Scroll-zoom and middle/shift-drag pan on one axes."""
    state = {"press": None}

    def on_scroll(event):
        if event.inaxes is not ax or event.xdata is None or event.ydata is None:
            return
        scale = 0.8 if event.button == "up" else 1.25
        x, y = event.xdata, event.ydata
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        ax.set_xlim(x - (x - x0) * scale, x + (x1 - x) * scale)
        ax.set_ylim(y - (y - y0) * scale, y + (y1 - y) * scale)
        fig.canvas.draw_idle()

    def on_press(event):
        if event.inaxes is not ax or event.xdata is None:
            return
        if event.button == 2 or (event.button == 1 and event.key == "shift"):
            state["press"] = (event.xdata, event.ydata, ax.get_xlim(), ax.get_ylim())

    def on_move(event):
        if state["press"] is None or event.inaxes is not ax or event.xdata is None:
            return
        x0, y0, xlim, ylim = state["press"]
        dx = event.xdata - x0
        dy = event.ydata - y0
        ax.set_xlim(xlim[0] - dx, xlim[1] - dx)
        ax.set_ylim(ylim[0] - dy, ylim[1] - dy)
        fig.canvas.draw_idle()

    def on_release(_event):
        state["press"] = None

    fig.canvas.mpl_connect("scroll_event", on_scroll)
    fig.canvas.mpl_connect("button_press_event", on_press)
    fig.canvas.mpl_connect("motion_notify_event", on_move)
    fig.canvas.mpl_connect("button_release_event", on_release)


def collect_correspondences(run_dir: Path, floorplan: Path) -> list[dict]:
    run_dir = Path(run_dir)
    bev_img = np.array(Image.open(run_dir / "bev.png").convert("RGB"))
    plan = np.array(Image.open(floorplan).convert("RGB"))
    meta = read_json(run_dir / "bev_meta.json")

    pairs: list[dict] = []
    pending_bev: tuple[float, float] | None = None

    plt.rcParams["toolbar"] = "toolbar2"
    fig = plt.figure(figsize=(14, 8.2))
    fig.canvas.manager.set_window_title("click BEV then matching floorplan point")
    ax_bev = fig.add_axes([0.04, 0.16, 0.44, 0.78])
    ax_plan = fig.add_axes([0.52, 0.16, 0.44, 0.78])
    ax_undo = fig.add_axes([0.28, 0.04, 0.12, 0.07])
    ax_save = fig.add_axes([0.44, 0.04, 0.12, 0.07])
    ax_quit = fig.add_axes([0.60, 0.04, 0.12, 0.07])

    ax_bev.imshow(bev_img)
    ax_plan.imshow(plan)
    ax_bev.set_title("BEV — scroll/zoom, middle-drag pan, left-click landmark")
    ax_plan.set_title("Floorplan — scroll/zoom, middle-drag pan, left-click match")
    ax_bev.set_xticks([])
    ax_bev.set_yticks([])
    ax_plan.set_xticks([])
    ax_plan.set_yticks([])

    _attach_pan_zoom(ax_bev, fig)
    _attach_pan_zoom(ax_plan, fig)

    sc_bev = ax_bev.scatter([], [], c="#e74c3c", s=70, zorder=5, edgecolors="k")
    sc_plan = ax_plan.scatter([], [], c="#e74c3c", s=70, zorder=5, edgecolors="k")
    texts: list = []
    status = fig.text(0.5, 0.125, "", ha="center", fontsize=10)

    btn_undo = Button(ax_undo, "Undo")
    btn_save = Button(ax_save, "Save + fit")
    btn_quit = Button(ax_quit, "Quit")
    done = {"ok": False}

    def refresh():
        if pairs:
            sc_bev.set_offsets(np.array([p["bev_px"] for p in pairs]))
            sc_plan.set_offsets(np.array([p["plan_px"] for p in pairs]))
        else:
            sc_bev.set_offsets(np.empty((0, 2)))
            sc_plan.set_offsets(np.empty((0, 2)))
        for t in texts:
            t.remove()
        texts.clear()
        for i, p in enumerate(pairs):
            texts.append(
                ax_bev.text(p["bev_px"][0] + 4, p["bev_px"][1] - 4, str(i + 1), color="#c0392b", fontsize=9)
            )
            texts.append(
                ax_plan.text(p["plan_px"][0] + 4, p["plan_px"][1] - 4, str(i + 1), color="#c0392b", fontsize=9)
            )
        waiting = "click BEV first" if pending_bev is None else "now click the matching floorplan pixel"
        status.set_text(
            f"pairs={len(pairs)}  |  {waiting}  |  scroll zoom  middle-drag pan  |  z undo  s save  q quit"
        )
        fig.canvas.draw_idle()

    def on_click(event):
        nonlocal pending_bev
        if event.inaxes not in (ax_bev, ax_plan):
            return
        if event.button != 1 or event.key == "shift":
            return
        if _toolbar_busy(fig):
            return
        if event.xdata is None or event.ydata is None:
            return
        x, y = float(event.xdata), float(event.ydata)
        if event.inaxes is ax_bev:
            pending_bev = (x, y)
            print(f"BEV click ({x:.1f}, {y:.1f}) — now click floorplan")
            refresh()
        elif event.inaxes is ax_plan:
            if pending_bev is None:
                print("Click BEV first, then the matching floorplan point")
                return
            bx, by = pending_bev
            world = bev_px_to_world(bx, by, meta)
            pairs.append(
                {
                    "bev_px": [bx, by],
                    "world_xy": world,
                    "plan_px": [x, y],
                    "label": f"pair-{len(pairs)+1}",
                }
            )
            print(f"pair {len(pairs)}: BEV ({bx:.1f},{by:.1f}) world {world} -> plan ({x:.1f},{y:.1f})")
            pending_bev = None
            refresh()

    def do_undo(_e=None):
        nonlocal pending_bev
        if pending_bev is not None:
            pending_bev = None
        elif pairs:
            pairs.pop()
            print("undid last pair")
        refresh()

    def do_save(_e=None):
        if len(pairs) < 2:
            print("Need at least 2 pairs")
            return
        done["ok"] = True
        plt.close(fig)

    def do_quit(_e=None):
        done["ok"] = False
        plt.close(fig)

    def on_key(event):
        key = (event.key or "").lower()
        if key in ("q", "escape"):
            do_quit()
        elif key == "z":
            do_undo()
        elif key == "s":
            do_save()

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)
    btn_undo.on_clicked(do_undo)
    btn_save.on_clicked(do_save)
    btn_quit.on_clicked(do_quit)
    refresh()
    plt.show()
    return pairs if done["ok"] or len(pairs) >= 2 else pairs


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--floorplan", type=Path, default=None)
    p.add_argument("--config", type=Path, default=None)
    args = p.parse_args()

    cfg = load_config(args.config)
    floorplan = args.floorplan or Path(cfg["floorplan_default"])
    pairs = collect_correspondences(args.run_dir, floorplan)
    if len(pairs) < 2:
        raise SystemExit("Need at least 2 correspondences")

    align = _load_align()
    import importlib.util

    spec = importlib.util.spec_from_file_location("overlay_mod", SRC / "04_overlay.py")
    overlay_mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["overlay_mod"] = overlay_mod
    spec.loader.exec_module(overlay_mod)

    summary = align.run_click_align(args.run_dir, floorplan, pairs)
    overlay_mod.run_overlay(args.run_dir, floorplan)
    print(json.dumps(summary, indent=2))
    print(f"wrote {args.run_dir / 'trajectory_aligned.png'}")


if __name__ == "__main__":
    main()

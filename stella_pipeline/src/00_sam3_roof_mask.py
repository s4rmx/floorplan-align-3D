#!/usr/bin/env python3
"""Build a static Stella roof mask from 360 equirect frames via SAM3.

SAM3 is prompted on each pinhole cubemap crop (front/right/back/left/up),
not the raw equirect and not a geometric row cut. Horizon faces use
ceiling/roof/metal roof/overhead beams; the up face also uses sky. Masks are
warped to equirect, unioned across frames, OR'd with a small zenith band,
then written as a 1920x960 PNG: 0 = ignore (roof), 255 = keep.

This script is meant to run inside sam3-pipeline:latest with sam3.pt mounted.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np


DEFAULT_PROMPTS = ("ceiling", "roof", "sky")


def face_K(face_size: int, hfov_deg: float = 90.0) -> np.ndarray:
    focal = face_size / (2.0 * np.tan(np.deg2rad(hfov_deg) / 2.0))
    c = face_size / 2.0
    return np.array([[focal, 0, c], [0, focal, c], [0, 0, 1]], dtype=np.float64)


def up_face_rotation() -> np.ndarray:
    """cam_from_pano looking at +Y (zenith). Farm/COLMAP pitch_deg=-90."""
    pitch = np.deg2rad(90.0)
    cp, sp = np.cos(pitch), np.sin(pitch)
    return np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]], dtype=np.float64)


def horizon_rotation(yaw_deg: float) -> np.ndarray:
    """cam_from_pano for a 90° horizon face. yaw 0 = +Z (front)."""
    yaw = np.deg2rad(-float(yaw_deg))
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float64)


def down_face_rotation() -> np.ndarray:
    """cam_from_pano looking at -Y (nadir)."""
    pitch = np.deg2rad(-90.0)
    cp, sp = np.cos(pitch), np.sin(pitch)
    return np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]], dtype=np.float64)


FACE_ROTATIONS: dict[str, np.ndarray] = {
    "front": horizon_rotation(0.0),
    "right": horizon_rotation(90.0),
    "back": horizon_rotation(180.0),
    "left": horizon_rotation(270.0),
    "up": up_face_rotation(),
    "down": down_face_rotation(),
}
FACE_ORDER = ("front", "right", "back", "left", "up", "down")
# Roof is visible on horizon faces, not only zenith. Skip nadir (operator/ground).
SAM3_FACES = ("front", "right", "back", "left", "up")
HORIZON_PROMPTS = ("corrugated roof", "metal roof", "roof")
HORIZON_NEGATIVE = ("sky", "tree")
UP_PROMPTS = ("ceiling", "roof")


def build_equirect_to_face_maps(
    equirect_hw: tuple[int, int],
    face_size: int,
    R_cam_from_pano: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Remap maps: perspective-face pixel -> equirect pixel. Geographic y-up pano."""
    H_eq, W_eq = equirect_hw
    K = face_K(face_size)
    us = np.arange(face_size, dtype=np.float64)
    vs = np.arange(face_size, dtype=np.float64)
    uu, vv = np.meshgrid(us, vs)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    dx = (uu - cx) / fx
    dy = -(vv - cy) / fy
    dz = np.ones_like(dx)
    D = np.stack([dx, dy, dz], axis=-1)
    D_pano = D @ R_cam_from_pano
    D_pano = D_pano / np.maximum(np.linalg.norm(D_pano, axis=-1, keepdims=True), 1e-9)
    lon = np.arctan2(D_pano[..., 0], D_pano[..., 2])
    lat = np.arcsin(np.clip(D_pano[..., 1], -1.0, 1.0))
    map_x = ((lon / (2 * np.pi)) + 0.5) * W_eq
    map_y = (0.5 - lat / np.pi) * H_eq
    return map_x.astype(np.float32), map_y.astype(np.float32)


def build_equirect_to_up_maps(
    equirect_hw: tuple[int, int],
    face_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    return build_equirect_to_face_maps(equirect_hw, face_size, up_face_rotation())


def extract_face(equirect_bgr: np.ndarray, map_x: np.ndarray, map_y: np.ndarray) -> np.ndarray:
    return cv2.remap(
        equirect_bgr,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_WRAP,
    )


def extract_up_face(equirect_bgr: np.ndarray, map_x: np.ndarray, map_y: np.ndarray) -> np.ndarray:
    return extract_face(equirect_bgr, map_x, map_y)


def face_mask_to_equirect(
    face_mask: np.ndarray,
    equirect_hw: tuple[int, int],
    face_size: int,
    R_cam_from_pano: np.ndarray,
) -> np.ndarray:
    """Project a perspective-face mask onto the equirect (float 0-1)."""
    H, W = equirect_hw
    K = face_K(face_size)
    us = (np.arange(W, dtype=np.float64) + 0.5) / W
    vs = (np.arange(H, dtype=np.float64) + 0.5) / H
    uu, vv = np.meshgrid(us, vs)
    lon = (uu * 2.0 - 1.0) * np.pi
    lat = (0.5 - vv) * np.pi
    cp, sp = np.cos(lat), np.sin(lat)
    dpx = np.sin(lon) * cp
    dpy = sp
    dpz = np.cos(lon) * cp
    D_pano = np.stack([dpx, dpy, dpz], axis=-1).reshape(-1, 3)
    D = (R_cam_from_pano @ D_pano.T).T
    dz = D[:, 2]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    fh, fw = face_mask.shape[:2]
    u = fx * (D[:, 0] / np.maximum(dz, 1e-8)) + cx
    v = cy - fy * (D[:, 1] / np.maximum(dz, 1e-8))
    valid = (dz > 1e-6) & (u >= 0) & (u < fw - 1e-6) & (v >= 0) & (v < fh - 1e-6)
    fm = face_mask.astype(np.float32)
    if fm.size and float(fm.max()) > 1.5:
        fm = fm / 255.0
    mapped = cv2.remap(
        fm,
        u.reshape(H, W).astype(np.float32),
        v.reshape(H, W).astype(np.float32),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    mapped[~valid.reshape(H, W)] = 0
    return mapped.astype(np.float32)


def up_mask_to_equirect(
    face_mask: np.ndarray,
    equirect_hw: tuple[int, int],
    face_size: int,
) -> np.ndarray:
    return face_mask_to_equirect(face_mask, equirect_hw, face_size, up_face_rotation())


def elevation_gate(equirect_hw: tuple[int, int], min_elevation_deg: float) -> np.ndarray:
    H, W = equirect_hw
    vs = (np.arange(H, dtype=np.float64) + 0.5) / H
    lat_deg = (0.5 - vs) * 180.0
    return (lat_deg[:, None] >= min_elevation_deg).astype(np.float32)


def zenith_band_mask(equirect_hw: tuple[int, int], band: float) -> np.ndarray:
    H, W = equirect_hw
    rows = int(round(H * band))
    m = np.zeros((H, W), dtype=np.float32)
    m[: max(rows, 1), :] = 1.0
    return m


def sample_frame_indices(n_video: int, n_keep: int) -> list[int]:
    if n_video <= 0:
        return []
    n_keep = min(n_keep, n_video)
    if n_keep == 1:
        return [0]
    return [int(round(i * (n_video - 1) / (n_keep - 1))) for i in range(n_keep)]


def read_sampled_frames(video: Path, n_frames: int) -> tuple[list[np.ndarray], list[int]]:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    idxs = sample_frame_indices(max(total, 1), n_frames)
    frames: list[np.ndarray] = []
    used: list[int] = []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if ok and frame is not None:
            frames.append(frame)
            used.append(idx)
    cap.release()
    if len(frames) < 1:
        raise RuntimeError(f"No frames read from {video}")
    return frames, used


class Sam3Session:
    def __init__(self, ckpt: Path, conf: float) -> None:
        import torch
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.conf = conf
        print(f"[roof] loading SAM3 {ckpt} on {self.device}", file=sys.stderr)
        self.model = build_sam3_image_model(
            checkpoint_path=str(ckpt),
            load_from_HF=False,
            device=self.device,
            eval_mode=True,
        )
        self.processor = Sam3Processor(
            self.model, device=self.device, confidence_threshold=conf
        )

    def segment_face(
        self,
        face_bgr: np.ndarray,
        prompts: list[str],
        *,
        is_horizon: bool = False,
        debug_dir: Path | None = None,
        tag: str = "",
    ) -> np.ndarray:
        from PIL import Image

        torch = self.torch
        rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        h, w = face_bgr.shape[:2]
        acc = np.zeros((h, w), dtype=np.float32)
        ctx = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if self.device == "cuda"
            else _nullctx()
        )
        if debug_dir is not None:
            debug_dir.mkdir(parents=True, exist_ok=True)
        with torch.inference_mode(), ctx:
            state = self.processor.set_image(image)
            for prompt in prompts:
                try:
                    self.processor.reset_all_prompts(state)
                except Exception:
                    pass
                out = self.processor.set_text_prompt(prompt=prompt, state=state)
                if isinstance(out, dict):
                    state = out
                masks = out.get("masks") if isinstance(out, dict) else None
                scores = out.get("scores") if isinstance(out, dict) else None
                if masks is None:
                    continue
                if not isinstance(masks, torch.Tensor):
                    masks = torch.as_tensor(masks)
                if masks.numel() == 0:
                    continue
                if masks.ndim == 4:
                    masks = masks.squeeze(1)
                if masks.ndim == 2:
                    masks = masks.unsqueeze(0)
                if scores is not None and not isinstance(scores, torch.Tensor):
                    scores = torch.as_tensor(scores)
                prompt_m = np.zeros((h, w), dtype=np.float32)
                n_keep = 0
                for mi in range(int(masks.shape[0])):
                    score = 1.0
                    if scores is not None:
                        score = float(scores.reshape(-1)[mi].detach().cpu())
                        if score < self.conf:
                            continue
                    m = masks[mi].float().detach().cpu().numpy()
                    if m.ndim == 3:
                        m = m.max(axis=0)
                    if m.shape != (h, w):
                        m = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
                    frac = float((m >= 0.5).mean())
                    if frac < 0.002:
                        continue
                    if is_horizon:
                        ys = np.where(m >= 0.5)[0]
                        if ys.size and float(ys.mean()) / h > 0.55:
                            continue
                    prompt_m = np.maximum(prompt_m, m)
                    n_keep += 1
                if debug_dir is not None:
                    slug = prompt.replace(" ", "_")
                    vis = overlay_preview(
                        face_bgr, np.where(prompt_m >= 0.5, 0, 255).astype(np.uint8)
                    )
                    cv2.imwrite(str(debug_dir / f"{slug}.jpg"), vis)
                    print(
                        f"[roof] {tag} '{prompt}' keep={n_keep} frac={float((prompt_m >= 0.5).mean()):.3f}",
                        file=sys.stderr,
                    )
                acc = np.maximum(acc, prompt_m)
            try:
                self.processor.reset_all_prompts(state)
            except Exception:
                pass
        return acc


class _nullctx:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def build_roof_mask(
    frames: list[np.ndarray],
    *,
    face_size: int,
    prompts: list[str],
    conf: float,
    ckpt: Path | None,
    min_elevation_deg: float,
    zenith_band: float,
    dilate_px: int,
    skip_sam3: bool,
    debug_dir: Path | None,
    horizon_top_frac: float = 0.0,
) -> tuple[np.ndarray, dict, list[dict[str, np.ndarray]]]:
    h, w = frames[0].shape[:2]
    maps = {
        name: build_equirect_to_face_maps((h, w), face_size, FACE_ROTATIONS[name])
        for name in SAM3_FACES
    }
    gate = elevation_gate((h, w), min_elevation_deg) if min_elevation_deg > -89 else None
    zenith = zenith_band_mask((h, w), zenith_band)
    union = np.zeros((h, w), dtype=np.float32)
    n_sam3 = 0
    hits_by_face = {name: 0 for name in SAM3_FACES}
    per_frame_faces: list[dict[str, np.ndarray]] = []
    session: Sam3Session | None = None
    if not skip_sam3:
        if ckpt is None or not Path(ckpt).is_file():
            raise FileNotFoundError(f"SAM3 checkpoint missing: {ckpt}")
        session = Sam3Session(Path(ckpt), conf)

    dilate_k = None
    if dilate_px > 0:
        dilate_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px, dilate_px))

    for i, frame in enumerate(frames):
        if frame.shape[0] != h or frame.shape[1] != w:
            frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
        frame_hit = False
        face_store: dict[str, np.ndarray] = {}
        for name in SAM3_FACES:
            mx, my = maps[name]
            crop = extract_face(frame, mx, my)
            if debug_dir is not None and i == 0 and name == "up":
                debug_dir.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(debug_dir / "roof_up_face.jpg"), crop)
            face_m = np.zeros(crop.shape[:2], dtype=np.float32)
            if name != "up" and horizon_top_frac > 0:
                cut = int(round(crop.shape[0] * horizon_top_frac))
                face_m[:cut] = 1.0
            if session is not None:
                face_prompts = list(UP_PROMPTS) if name == "up" else list(HORIZON_PROMPTS)
                prompt_dbg = None
                if debug_dir is not None and i == 0:
                    prompt_dbg = debug_dir / "sam3_prompts" / name
                sam = session.segment_face(
                    crop,
                    face_prompts,
                    is_horizon=(name != "up"),
                    debug_dir=prompt_dbg,
                    tag=f"f{i:02d}/{name}",
                )
                if name != "up":
                    neg = session.segment_face(
                        crop,
                        list(HORIZON_NEGATIVE),
                        is_horizon=False,
                        debug_dir=prompt_dbg,
                        tag=f"f{i:02d}/{name}/neg",
                    )
                    sam = np.where(neg >= 0.5, 0.0, sam)
                if float(sam.max()) > 0:
                    hits_by_face[name] += 1
                    frame_hit = True
                    face_m = np.maximum(face_m, sam)
            if dilate_k is not None and float(face_m.max()) > 0:
                face_m = cv2.dilate((face_m >= 0.5).astype(np.uint8), dilate_k).astype(np.float32)
            face_store[name] = face_m
            if debug_dir is not None and i == 0:
                sam3_dir = debug_dir / "sam3_faces"
                sam3_dir.mkdir(parents=True, exist_ok=True)
                vis = overlay_preview(crop, np.where(face_m >= 0.5, 0, 255).astype(np.uint8))
                cv2.imwrite(str(sam3_dir / f"{name}_sam3.jpg"), vis)
            eq = face_mask_to_equirect(face_m, (h, w), face_size, FACE_ROTATIONS[name])
            if gate is not None:
                eq = eq * gate
            union = np.maximum(union, eq)
        per_frame_faces.append(face_store)
        if frame_hit:
            n_sam3 += 1

    roof = np.clip(np.maximum(union, zenith), 0.0, 1.0)
    roof_u8 = (roof >= 0.5).astype(np.uint8) * 255

    stella = np.where(roof_u8 > 0, 0, 255).astype(np.uint8)
    meta = {
        "n_frames": len(frames),
        "n_frames_sam3_hit": n_sam3,
        "sam3_hits_by_face": hits_by_face,
        "sam3_faces": list(SAM3_FACES),
        "skip_sam3": skip_sam3,
        "prompts": prompts,
        "horizon_prompts": list(HORIZON_PROMPTS),
        "horizon_negative": list(HORIZON_NEGATIVE),
        "up_prompts": list(UP_PROMPTS),
        "horizon_top_frac": horizon_top_frac,
        "min_elevation_deg": min_elevation_deg,
        "zenith_band": zenith_band,
        "dilate_px": dilate_px,
        "face_size": face_size,
        "shape": [int(h), int(w)],
        "roof_fraction": float((stella == 0).mean()),
        "mask_source": "sam3_per_face",
    }
    return stella, meta, per_frame_faces


def overlay_preview(frame: np.ndarray, stella_mask: np.ndarray) -> np.ndarray:
    vis = frame.copy()
    if stella_mask.shape[:2] != vis.shape[:2]:
        stella_mask = cv2.resize(stella_mask, (vis.shape[1], vis.shape[0]), interpolation=cv2.INTER_NEAREST)
    roof = stella_mask == 0
    vis[roof] = (vis[roof] * 0.35 + np.array([0, 0, 220], dtype=np.float32)).clip(0, 255).astype(
        np.uint8
    )
    return vis


def _label(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.rectangle(out, (0, 0), (min(out.shape[1], 8 + 9 * len(text)), 28), (0, 0, 0), -1)
    cv2.putText(out, text, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def _hstack(imgs: list[np.ndarray], gap: int = 4) -> np.ndarray:
    h = max(im.shape[0] for im in imgs)
    rows = []
    for im in imgs:
        if im.shape[0] != h:
            im = cv2.resize(im, (int(im.shape[1] * h / im.shape[0]), h))
        rows.append(im)
    pad = np.full((h, gap, 3), 20, dtype=np.uint8)
    out = rows[0]
    for im in rows[1:]:
        out = np.hstack([out, pad, im])
    return out


def _vstack(imgs: list[np.ndarray], gap: int = 4) -> np.ndarray:
    w = max(im.shape[1] for im in imgs)
    rows = []
    for im in imgs:
        if im.shape[1] != w:
            im = cv2.resize(im, (w, int(im.shape[0] * w / im.shape[1])))
        rows.append(im)
    pad = np.full((gap, w, 3), 20, dtype=np.uint8)
    out = rows[0]
    for im in rows[1:]:
        out = np.vstack([out, pad, im])
    return out


def write_cubemap_debug(
    frames: list[np.ndarray],
    frame_indices: list[int],
    stella_mask: np.ndarray,
    out_dir: Path,
    face_size: int,
    face_masks: list[dict[str, np.ndarray]] | None = None,
) -> Path:
    """Save raw + SAM3-masked cubemap crops for every sampled equirect frame."""
    h, w = frames[0].shape[:2]
    maps = {
        name: build_equirect_to_face_maps((h, w), face_size, R)
        for name, R in FACE_ROTATIONS.items()
    }
    frames_root = out_dir / "frames"
    frames_root.mkdir(parents=True, exist_ok=True)
    montage_thumbs: list[np.ndarray] = []

    for i, (frame, fidx) in enumerate(zip(frames, frame_indices)):
        if frame.shape[0] != h or frame.shape[1] != w:
            frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
        folder = frames_root / f"frame_{i:02d}"
        folder.mkdir(parents=True, exist_ok=True)
        stored = face_masks[i] if face_masks is not None and i < len(face_masks) else {}
        frame_eq = np.zeros((h, w), dtype=np.float32)
        for name, fm in stored.items():
            frame_eq = np.maximum(
                frame_eq,
                face_mask_to_equirect(fm, (h, w), face_size, FACE_ROTATIONS[name]),
            )
        frame_mask_u8 = np.where(frame_eq >= 0.5, 0, 255).astype(np.uint8)
        eq_overlay = overlay_preview(frame, frame_mask_u8)
        cv2.imwrite(str(folder / "equirect.jpg"), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        cv2.imwrite(str(folder / "equirect_overlay.jpg"), eq_overlay, [int(cv2.IMWRITE_JPEG_QUALITY), 85])

        raw_faces: list[np.ndarray] = []
        ov_faces: list[np.ndarray] = []
        for name in FACE_ORDER:
            mx, my = maps[name]
            crop = extract_face(frame, mx, my)
            if name in stored:
                fm = stored[name]
                crop_mask_u8 = np.where(fm >= 0.5, 0, 255).astype(np.uint8)
            else:
                crop_mask_u8 = np.full(crop.shape[:2], 255, dtype=np.uint8)
            overlay = overlay_preview(crop, crop_mask_u8)
            cv2.imwrite(str(folder / f"{name}.jpg"), crop, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            cv2.imwrite(str(folder / f"{name}_overlay.jpg"), overlay, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            cv2.imwrite(str(folder / f"{name}_mask.png"), crop_mask_u8)
            raw_faces.append(_label(crop, name))
            ov_faces.append(_label(overlay, f"{name} SAM3"))

        eq_small = cv2.resize(eq_overlay, (face_size * 3 + 8, face_size // 2))
        eq_small = _label(eq_small, f"frame {i:02d}  video#{fidx}  equirect overlay")
        montage = _vstack(
            [
                eq_small,
                _hstack(raw_faces),
                _hstack(ov_faces),
            ]
        )
        cv2.imwrite(str(folder / "montage.jpg"), montage, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        thumb = cv2.resize(montage, (min(1600, montage.shape[1]), int(montage.shape[0] * min(1600, montage.shape[1]) / montage.shape[1])))
        montage_thumbs.append(_label(thumb, f"{i:02d}"))

    contact = _vstack(montage_thumbs, gap=8)
    contact_path = out_dir / "cubemap_contact_sheet.jpg"
    cv2.imwrite(str(contact_path), contact, [int(cv2.IMWRITE_JPEG_QUALITY), 80])

    html = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'><title>SAM3 roof mask smoke</title>",
        "<style>body{font-family:sans-serif;background:#111;color:#eee;margin:24px}",
        "img{max-width:100%;height:auto;background:#000} h2{margin-top:2.5rem}",
        ".faces{display:grid;grid-template-columns:repeat(6,1fr);gap:8px}",
        ".faces img{width:100%}</style></head><body>",
        "<h1>SAM3 roof-mask smoke (cubemap crops)</h1>",
        "<p>Red on each crop = SAM3 prompts for that face: corrugated/metal/roof, minus sky/tree on side faces. Equirect overlay on each frame is that frame's warped SAM3 only. The union PNG at the top is what Stella gets (all sampled frames OR'd).</p>",
        "<p><a href='roof_mask.png'>roof_mask.png</a> · <a href='roof_mask_overlay.jpg'>equirect overlay</a> · <a href='cubemap_contact_sheet.jpg'>contact sheet</a> · <a href='roof_mask_meta.json'>meta</a></p>",
        "<h2>Stella union mask (all frames)</h2>",
        "<img src='roof_mask_overlay.jpg' alt='equirect overlay'>",
    ]
    for i, fidx in enumerate(frame_indices):
        rel = f"frames/frame_{i:02d}"
        html.append(f"<h2>Frame {i:02d} (video index {fidx})</h2>")
        html.append(f"<p><img src='{rel}/montage.jpg' alt='montage {i}'></p>")
        html.append("<div class='faces'>")
        for name in FACE_ORDER:
            html.append(
                f"<div><div>{name}</div>"
                f"<img src='{rel}/{name}.jpg' alt='{name}'>"
                f"<img src='{rel}/{name}_overlay.jpg' alt='{name} overlay'></div>"
            )
        html.append("</div>")
    html.append("</body></html>")
    (out_dir / "index.html").write_text("\n".join(html), encoding="utf-8")
    return contact_path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--n-frames", type=int, default=20)
    p.add_argument("--face-size", type=int, default=512)
    p.add_argument("--prompts", nargs="+", default=list(DEFAULT_PROMPTS))
    p.add_argument("--conf", type=float, default=0.40)
    p.add_argument("--ckpt", type=Path, default=Path("/opt/sam3/sam3.pt"))
    p.add_argument(
        "--min-elevation-deg",
        type=float,
        default=-90.0,
        help="Drop warped SAM3 below this elevation. -90 disables the gate.",
    )
    p.add_argument(
        "--horizon-top-frac",
        type=float,
        default=0.0,
        help="Optional geometric extra: mask this top fraction of side faces (0=off, SAM3 only).",
    )
    p.add_argument("--zenith-band", type=float, default=0.06)
    p.add_argument("--dilate-px", type=int, default=7)
    p.add_argument("--skip-sam3", action="store_true", help="Zenith band only (no GPU)")
    p.add_argument(
        "--debug-faces",
        action="store_true",
        help="Write all 6 cubemap crops + overlays for every sampled frame",
    )
    args = p.parse_args()

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    frames, frame_indices = read_sampled_frames(args.video, args.n_frames)
    print(
        f"[roof] sampled {len(frames)} frames (idx {frame_indices[0]}..{frame_indices[-1]}) from {args.video}",
        file=sys.stderr,
    )

    skip = args.skip_sam3
    ckpt = args.ckpt
    if not skip and not ckpt.is_file():
        print(f"[roof] checkpoint missing at {ckpt}, zenith-band only", file=sys.stderr)
        skip = True

    stella, meta, face_masks = build_roof_mask(
        frames,
        face_size=args.face_size,
        prompts=list(args.prompts),
        conf=args.conf,
        ckpt=ckpt,
        min_elevation_deg=args.min_elevation_deg,
        zenith_band=args.zenith_band,
        dilate_px=args.dilate_px,
        skip_sam3=skip,
        debug_dir=out,
        horizon_top_frac=args.horizon_top_frac,
    )
    meta["frame_indices"] = frame_indices
    mask_path = out / "roof_mask.png"
    cv2.imwrite(str(mask_path), stella)
    cv2.imwrite(str(out / "roof_mask_overlay.jpg"), overlay_preview(frames[0], stella))
    if args.debug_faces:
        write_cubemap_debug(
            frames, frame_indices, stella, out, args.face_size, face_masks=face_masks
        )
        meta["debug_faces"] = True
        meta["index_html"] = str(out / "index.html")
    (out / "roof_mask_meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps({"mask": str(mask_path), **meta}, indent=2))


if __name__ == "__main__":
    main()

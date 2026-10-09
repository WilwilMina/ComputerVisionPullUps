"""Measure the wrist-to-bar offset from hand-read frames. Measurement only: the result is NOT applied anywhere.

The pipeline's bar line is the wrist line, but the hands wrap over the bar, so the real bar sits a little above
the wrist landmark. This tool picks frames where both wrists are confident and gripping, writes zoomed crops with a
pixel ruler centred on the wrist line, and (after a person or agent reads where the bar is against the ruler)
turns the readings into body-scale units, the unit BAR_OFFSET_BODY_SCALES uses.

    python -m pullup_rom.bar_offset crops  VIDEO KEYPOINTS.npz OUT_DIR [--n 12]
    python -m pullup_rom.bar_offset summarize MANIFEST.json READINGS.json [READINGS2.json ...]

READINGS.json: {"<crop file name>": {"center": px, "top": px, "bottom": px} | null}, all in original pixels
measured up from the wrist line (positive = the bar is above the wrist line); null = bar not visible.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from . import config
from .metrics import MetricsParams, Signals, compute_signals
from .pose import PoseResult, load_pose

CROP_HALF_WIDTH_SCALES = 3.5  # crop extends this many body scales left and right of the wrists
CROP_ABOVE_SCALES = 1.2  # ... above the wrist line
CROP_BELOW_SCALES = 0.8  # ... below it
CROP_TARGET_HEIGHT_PX = 460  # crops are enlarged to about this height so the ruler is readable


def pick_frames(pose: PoseResult, signals: Signals, n: int) -> list[int]:
    """n frames, evenly spread, where both wrists are confident, hands are on the bar, and the scale is known."""
    kp, m = pose.keypoints, config.MIN_LANDMARK_CONFIDENCE
    ok = (
        (kp[:, config.LEFT_WRIST, 2] >= m) & (kp[:, config.RIGHT_WRIST, 2] >= m)
        & np.isfinite(kp[:, config.LEFT_WRIST, :2]).all(axis=1) & np.isfinite(kp[:, config.RIGHT_WRIST, :2]).all(axis=1)
        & signals.gripping & signals.head_valid & np.isfinite(signals.body_scale_px)
    )
    idx = np.flatnonzero(ok)
    if idx.size == 0:
        return []
    return [int(i) for i in idx[np.unique(np.linspace(0, idx.size - 1, min(n, idx.size)).round().astype(int))]]


def frame_geometry(pose: PoseResult, signals: Signals, frame: int) -> dict:
    kp = pose.keypoints[frame]
    lw, rw = kp[config.LEFT_WRIST, :2], kp[config.RIGHT_WRIST, :2]
    return {
        "frame": frame,
        "time_s": round(frame / pose.fps, 3),
        "wrist_y_px": float((lw[1] + rw[1]) / 2),  # image pixels, y grows downward
        "wrist_x_mid_px": float((lw[0] + rw[0]) / 2),
        "wrists_px": [[float(lw[0]), float(lw[1])], [float(rw[0]), float(rw[1])]],
        "body_scale_px": float(signals.body_scale_px[frame]),
    }


def tick_step_px(body_scale_px: float) -> int:
    """Ruler tick spacing in original pixels: fine enough to read, sparse enough to stay legible."""
    return 2 if body_scale_px < 60 else 5 if body_scale_px < 200 else 10


def write_crops(video_path: str | Path, pose: PoseResult, signals: Signals, out_dir: str | Path, n: int) -> list[dict]:
    import cv2

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"cannot read video: {video_path}")
    manifest: list[dict] = []
    for frame in pick_frames(pose, signals, n):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ok, img = cap.read()
        if not ok:
            continue
        g = frame_geometry(pose, signals, frame)
        sc, wy, wx = g["body_scale_px"], g["wrist_y_px"], g["wrist_x_mid_px"]
        x0, x1 = int(max(0, wx - CROP_HALF_WIDTH_SCALES * sc)), int(min(img.shape[1], wx + CROP_HALF_WIDTH_SCALES * sc))
        y0, y1 = int(max(0, wy - CROP_ABOVE_SCALES * sc)), int(min(img.shape[0], wy + CROP_BELOW_SCALES * sc))
        if x1 <= x0 or y1 <= y0:
            continue  # the wrists are outside the image: nothing to crop
        zoom = CROP_TARGET_HEIGHT_PX / max(1, (y1 - y0))
        crop = cv2.resize(img[y0:y1, x0:x1], None, fx=zoom, fy=zoom, interpolation=cv2.INTER_CUBIC)
        zy = (wy - y0) * zoom  # wrist-line row in the enlarged crop
        step = tick_step_px(sc)
        k_max = int(max(wy - y0, y1 - wy) / step) + 1
        for k in range(-k_max, k_max + 1):
            y = int(round(zy - k * step * zoom))  # k > 0 is above the wrist line
            if not 0 <= y < crop.shape[0]:
                continue
            major = k % 4 == 0
            cv2.line(crop, (0, y), (18 if major else 9, y), (0, 255, 255), 1)
            cv2.line(crop, (crop.shape[1] - (18 if major else 9), y), (crop.shape[1] - 1, y), (0, 255, 255), 1)
            if major:
                label = f"{k * step:+d}"
                cv2.putText(crop, label, (22, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(crop, label, (crop.shape[1] - 62, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.line(crop, (0, int(round(zy))), (crop.shape[1] - 1, int(round(zy))), (255, 255, 0), 1, cv2.LINE_AA)  # the wrist line
        for wxp, wyp in g["wrists_px"]:
            cv2.circle(crop, (int((wxp - x0) * zoom), int((wyp - y0) * zoom)), 5, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.putText(crop, "cyan line = wrist line (0). ruler: original px, + above / - below", (24, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 0), 1, cv2.LINE_AA)
        name = f"frame_{frame:05d}.png"
        cv2.imwrite(str(out / name), crop)
        manifest.append({"file": name, "ruler_step_px": step, **g})
    cap.release()
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def offsets_in_body_scales(manifest: Sequence[dict], readings: dict) -> list[dict]:
    """Per read frame: the bar centre above the wrist line, in body scales (positive = bar above the wrist)."""
    rows = []
    for m in manifest:
        r = readings.get(m["file"])
        if not r or r.get("center") is None:
            continue
        sc = m.get("body_scale_px")
        if sc is None or not np.isfinite(sc) or sc <= 0:
            continue  # a hand-edited manifest without a usable body scale
        thickness = None if r.get("top") is None or r.get("bottom") is None else (r["top"] - r["bottom"]) / sc
        rows.append({"file": m["file"], "time_s": m["time_s"], "center_px": r["center"], "center_scales": r["center"] / sc, "thickness_scales": thickness})
    return rows


def summarize(rows: Sequence[dict]) -> Optional[dict]:
    if not rows:
        return None
    v = np.array([r["center_scales"] for r in rows])
    t = np.array([r["thickness_scales"] for r in rows if r["thickness_scales"] is not None])
    return {
        "n": int(v.size),
        "mean": float(v.mean()),
        "std": float(v.std(ddof=1)) if v.size > 1 else 0.0,
        "min": float(v.min()),
        "max": float(v.max()),
        "median": float(np.median(v)),
        "thickness_mean": float(t.mean()) if t.size else None,
    }


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Measure the wrist-to-bar offset from hand-read frames (measurement only).")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("crops", help="write ruler crops for reading")
    c.add_argument("video"), c.add_argument("keypoints"), c.add_argument("out_dir")
    c.add_argument("--n", type=int, default=12)
    s = sub.add_parser("summarize", help="turn readings into body-scale offsets")
    s.add_argument("manifest"), s.add_argument("readings", nargs="+")
    args = ap.parse_args(argv)

    if args.cmd == "crops":
        pose = load_pose(args.keypoints)
        signals = compute_signals(pose, MetricsParams(elbow_angle_source="image"))  # the bar offset does not use the elbow angle
        manifest = write_crops(args.video, pose, signals, args.out_dir, args.n)
        print(f"wrote {len(manifest)} crops and manifest.json to {args.out_dir}")
        return
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    readings: dict = {}
    for path in args.readings:
        readings.update(json.loads(Path(path).read_text(encoding="utf-8")))
    rows = offsets_in_body_scales(manifest, readings)
    for r in rows:
        print(f"  {r['file']}  t={r['time_s']:6.2f}s  bar centre {r['center_px']:+5.1f} px = {r['center_scales']:+.3f} body scales")
    print(json.dumps(summarize(rows), indent=2))


if __name__ == "__main__":
    main()

"""Run slices 1-2 over a folder of clips and print one line per clip.

Tracking-quality diagnostics only. There is no ground truth here, so nothing
printed is an accuracy figure, and nothing here feeds rep logic or sets a threshold.

Needs the pose model and real videos, so it is NOT part of pytest:
    python -m pullup_rom.eval_clips data/external

Per-clip keypoints are cached in config.EVAL_CACHE_DIR (delete a file to re-run the model).
Per-clip debug plot and contact sheet go to config.EVAL_PLOT_DIR.
Clips under a directory named "rejected" are skipped.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from . import config
from .debug_plot import plot_signals
from .metrics import ARM_LEFT, ARM_RIGHT, MetricsParams, Signals, compute_signals
from .pose import MediaPipePoseBackend, PoseResult, load_pose, save_pose
from .rep_report import analyze, format_reps

_JOINT_GROUPS = {
    "head": list(config.HEAD_LANDMARKS),
    "shoulders": [config.LEFT_SHOULDER, config.RIGHT_SHOULDER],
    "elbows": [config.LEFT_ELBOW, config.RIGHT_ELBOW],
    "wrists": [config.LEFT_WRIST, config.RIGHT_WRIST],
}
_SHEET_FRAMES = 8
_SHEET_TILE_H = 360


def find_clips(root: Path) -> list[Path]:
    return sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in config.EVAL_VIDEO_SUFFIXES and "rejected" not in p.parts
    )


def clip_stem(clip: Path, root: Path) -> str:
    return "_".join(clip.relative_to(root).with_suffix("").parts)


def cache_path(clip: Path, root: Path) -> Path:
    return Path(config.EVAL_CACHE_DIR) / (clip_stem(clip, root) + ".npz")


def load_or_extract(clip: Path, root: Path, variant: str) -> PoseResult:
    cp = cache_path(clip, root)
    if cp.exists() and cp.stat().st_mtime >= clip.stat().st_mtime:
        return load_pose(cp)
    result = MediaPipePoseBackend(variant).extract(clip)
    save_pose(cp, result)
    return result


def head_spread(pose: PoseResult, s: Signals) -> Optional[tuple[float, float, float]]:
    """How much the head landmarks (nose, eyes, ears) disagree on height, in body scales.

    Per head_valid frame, using only landmarks the model was confident about and needing at least two:
    (highest - lowest landmark height) / body scale. Returns (median, 95th percentile, share of
    head_valid frames that had >= 2 confident landmarks). Small spread means the landmarks agree; it does
    NOT prove they are right, since from behind the face points are guessed (consistently or not).
    """
    idx = np.flatnonzero(s.head_valid)
    if idx.size == 0:
        return None
    lm = list(config.HEAD_AGREEMENT_LANDMARKS)
    conf_ok = pose.keypoints[idx][:, lm, 2] >= config.MIN_LANDMARK_CONFIDENCE
    height = pose.frame_size[1] - pose.keypoints[idx][:, lm, 1]
    height = np.where(conf_ok, height, np.nan)
    enough = conf_ok.sum(axis=1) >= 2
    if not enough.any():
        return None
    spread = (np.nanmax(height[enough], axis=1) - np.nanmin(height[enough], axis=1)) / s.body_scale_px[idx][enough]
    return float(np.median(spread)), float(np.percentile(spread, 95)), float(enough.mean())


def bar_stability(s: Signals) -> tuple[Optional[float], Optional[float]]:
    """(drift, scatter), both in body scales over head_valid frames after the first EVAL_DRIFT_SKIP_S seconds.

    drift   = max - min of the bar line: how far the bar moves in the image (camera motion, or a bad estimate).
    scatter = 95th percentile of |confident wrist height - bar line|: how closely the wrists follow the
              estimated bar. Large scatter means 'wrists stay on the bar' does not hold here.
    """
    idx = np.flatnonzero(s.head_valid & (s.t_s >= config.EVAL_DRIFT_SKIP_S))
    if idx.size == 0:
        return None, None
    scale = np.median(s.body_scale_px[idx])
    drift = float((s.bar_height_px[idx].max() - s.bar_height_px[idx].min()) / scale)
    w = idx[np.isfinite(s.wrist_height_px[idx])]
    scatter = float(np.percentile(np.abs(s.wrist_height_px[w] - s.bar_height_px[w]), 95) / scale) if w.size else None
    return drift, scatter


def elbow_range(angle: np.ndarray, valid: np.ndarray) -> Optional[tuple[float, float]]:
    a = angle[valid & np.isfinite(angle)]
    return (float(a.min()), float(a.max())) if a.size else None


def median_conf(pose: PoseResult, joints: list[int]) -> float:
    det = pose.detected
    if not det.any():
        return float("nan")
    return float(np.median(pose.keypoints[det][:, joints, 2]))


def _fmt_range(r: Optional[tuple[float, float]]) -> str:
    return "n/a" if r is None else f"{r[0]:.0f}-{r[1]:.0f}"


def _fmt(x: Optional[float], spec: str) -> str:
    return "n/a" if x is None else format(x, spec)


def report_line(name: str, pose: PoseResult, s: Signals) -> str:
    n = pose.num_frames
    conf = " ".join(f"{k[:4]} {median_conf(pose, j):.2f}" for k, j in _JOINT_GROUPS.items())
    drift, scatter = bar_stability(s)
    spread = head_spread(pose, s)
    spread_txt = "n/a" if spread is None else f"med {spread[0]:.2f} p95 {spread[1]:.2f}"
    one_arm = s.elbow_valid & (s.elbow_arms_used != (ARM_LEFT | ARM_RIGHT))
    carried = 100 * (s.bar_carried & s.head_valid).sum() / max(1, s.head_valid.sum())
    both_invalid = ~(s.head_valid & s.elbow_valid)
    return (
        f"{name:<30} {pose.frame_size[0]}x{pose.frame_size[1]:<5} {pose.fps:4.1f}fps {n / pose.fps:4.1f}s | "
        f"person {100 * pose.detected.mean():5.1f}% | "
        f"invalid head {100 * (~s.head_valid).mean():5.1f}% elbow {100 * (~s.elbow_valid).mean():5.1f}% either {100 * both_invalid.mean():5.1f}% | "
        f"bar carried {carried:4.1f}% of head-valid | "
        f"conf {conf} | head spread {spread_txt} (body scales) | "
        f"bar drift {_fmt(drift, '.2f')} scatter {_fmt(scatter, '.2f')} (body scales, after {config.EVAL_DRIFT_SKIP_S:g}s) | "
        f"elbow deg 3D {_fmt_range(elbow_range(s.elbow_world_deg, s.elbow_valid))} 2D {_fmt_range(elbow_range(s.elbow_image_deg, s.elbow_valid))} | "
        f"one-arm {100 * one_arm.sum() / max(1, s.elbow_valid.sum()):4.1f}% of elbow-valid"
    )


def save_contact_sheet(clip: Path, pose: PoseResult, out_path: Path) -> None:
    """8 evenly spaced frames with the skeleton drawn, 2 rows x 4, for eyeballing tracking."""
    import cv2

    cap = cv2.VideoCapture(str(clip))
    idx = np.linspace(0, pose.num_frames - 1, _SHEET_FRAMES).astype(int)
    tiles = []
    for i in idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = cap.read()
        if not ok:
            frame = np.zeros((pose.frame_size[1], pose.frame_size[0], 3), dtype=np.uint8)
        kp = pose.keypoints[i]
        thick = max(2, frame.shape[0] // 300)
        for a, b in config.PREVIEW_BONES:
            if np.isfinite(kp[[a, b], :2]).all():
                cv2.line(frame, tuple(int(v) for v in kp[a, :2]), tuple(int(v) for v in kp[b, :2]), (0, 255, 0), thick)
        for j in {k for bone in config.PREVIEW_BONES for k in bone}:
            if np.isfinite(kp[j, :2]).all():
                cv2.circle(frame, (int(kp[j, 0]), int(kp[j, 1])), thick * 2, (0, 0, 255), -1)
        scale = _SHEET_TILE_H / frame.shape[0]
        tile = cv2.resize(frame, (int(frame.shape[1] * scale), _SHEET_TILE_H))
        cv2.putText(tile, f"{i / pose.fps:.1f}s", (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        tiles.append(tile)
    cap.release()
    half = _SHEET_FRAMES // 2
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), np.vstack([np.hstack(tiles[:half]), np.hstack(tiles[half:])]))


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Run slices 1-2 over every clip in a folder.")
    ap.add_argument("root", help="folder of clips, searched recursively")
    ap.add_argument("--variant", default=config.POSE_MODEL_VARIANT, choices=["lite", "full", "heavy"])
    ap.add_argument("--no-plots", action="store_true", help="skip the per-clip plot and contact sheet")
    ap.add_argument("--reps", action="store_true", help="print each clip's detected reps with their bottom elbow angles")
    ap.add_argument("--cached-only", action="store_true", help="never run the pose model; list clips without a cache as not evaluated")
    args = ap.parse_args(argv)

    root = Path(args.root)
    clips = find_clips(root)
    if not clips:
        raise SystemExit(f"error: no video files under {root}")
    print(f"{len(clips)} clip(s); all settings PROVISIONAL (see config.py); tracking diagnostics, not accuracy", flush=True)
    for clip in clips:
        name = clip_stem(clip, root)
        cp = cache_path(clip, root)
        if args.cached_only and not (cp.exists() and cp.stat().st_mtime >= clip.stat().st_mtime):
            print(f"{name:<30} NOT EVALUATED (no keypoint cache; --cached-only)", flush=True)
            continue
        try:
            pose = load_or_extract(clip, root, args.variant)
            s = compute_signals(pose, MetricsParams())
            print(report_line(name, pose, s), flush=True)
            if args.reps:
                _, analysis = analyze(pose)
                for line in format_reps(analysis, config.ELBOW_ANGLE_SOURCE):
                    print("    " + line, flush=True)
            if not args.no_plots:
                plot_signals(s, Path(config.EVAL_PLOT_DIR) / f"{name}.png", title=f"{name} (provisional; grey = not valid, orange = bar carried from neighbours)")
                save_contact_sheet(clip, pose, Path(config.EVAL_PLOT_DIR) / f"{name}_sheet.jpg")
        except Exception as e:  # keep going: one bad clip must not hide the others
            print(f"{name:<30} FAILED: {type(e).__name__}: {e}", flush=True)


if __name__ == "__main__":
    main()

"""Check the pipeline against hand labels of a clip and print per-rep diagnostics.

    python -m pullup_rom.label_check data/labels/<clip>.json

The labels file (under data/, never in this package) holds the person's visual reads: per rep an approximate
top time and a verdict (FULL / NOT FULL / UNKNOWN). Times are never hard-coded here: each label is snapped to
the rep the signal detects whose peak is nearest, within LABEL_SNAP_WINDOW_S. Agreement on a handful of reps is
a sanity check, not an accuracy figure.
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
from .rep_report import analyze
from .reps import Rep

HEAD_GROUPS: dict[str, tuple[int, ...]] = {
    "nose": (config.NOSE,),
    "mouth": (config.MOUTH_LEFT, config.MOUTH_RIGHT),
    "eyes": (config.LEFT_EYE, config.RIGHT_EYE),
    "ears": (config.LEFT_EAR, config.RIGHT_EAR),
}
TOP_HALF_WINDOW_FRAMES = 2  # head landmarks are read as the median over the peak frame +/- this many frames


def load_labels(path: str | Path) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    for key in ("keypoints", "reps"):
        if key not in data:
            raise ValueError(f"labels file {path} has no '{key}'")
    for r in data["reps"]:
        if r.get("label") not in ("FULL", "NOT FULL", "UNKNOWN"):
            raise ValueError(f"rep {r.get('index')}: label must be FULL, NOT FULL or UNKNOWN")
        if "approx_top_s" not in r:
            raise ValueError(f"rep {r.get('index')}: approx_top_s is required")
    return data


def snap_labels(labels: Sequence[dict], reps: Sequence[Rep], window_s: float) -> list[tuple[dict, Optional[Rep]]]:
    """Pair each labelled rep with the nearest unused detected rep peak within `window_s` seconds (else None)."""
    used: set[int] = set()
    out: list[tuple[dict, Optional[Rep]]] = []
    for lab in labels:
        best, best_d = None, window_s
        for r in reps:
            d = abs(r.peak_s - lab["approx_top_s"])
            if r.index not in used and d <= best_d:
                best, best_d = r, d
        if best is not None:
            used.add(best.index)
        out.append((lab, best))
    return out


def verdict_text(full: Optional[bool]) -> str:
    return {True: "FULL", False: "NOT FULL", None: "UNKNOWN"}[full]


def head_minus_bar_by_group(pose: PoseResult, signals: Signals, frame: int) -> dict[str, tuple[Optional[float], float]]:
    """Per head-landmark group: (height over the bar in body scales, median confidence), around `frame`.

    The bar is the pipeline's own bar line (the wrist line plus the configured BAR_OFFSET_BODY_SCALES, i.e. the bar centre). A group needs every landmark in it
    to be confident; the value is the median over the peak frame +/- TOP_HALF_WINDOW_FRAMES.
    """
    lo, hi = max(0, frame - TOP_HALF_WINDOW_FRAMES), min(pose.num_frames, frame + TOP_HALF_WINDOW_FRAMES + 1)
    kp = pose.keypoints[lo:hi]
    scale, bar = signals.body_scale_px[frame], signals.bar_height_px[frame]
    out: dict[str, tuple[Optional[float], float]] = {}
    for name, ids in HEAD_GROUPS.items():
        conf = kp[:, list(ids), 2]
        ok = (conf >= config.MIN_LANDMARK_CONFIDENCE).all(axis=1) & np.isfinite(kp[:, list(ids), 1]).all(axis=1)
        if not ok.any() or not np.isfinite(scale) or not np.isfinite(bar):
            out[name] = (None, float(np.median(conf)) if conf.size else float("nan"))
            continue
        heights = pose.frame_size[1] - kp[ok][:, list(ids), 1].mean(axis=1)
        out[name] = (float((np.median(heights) - bar) / scale), float(np.median(conf[ok])))
    return out


def _f(x: Optional[float], spec: str = "6.2f") -> str:
    return " " * (int(spec.split(".")[0]) - 3) + "n/a" if x is None or not np.isfinite(x) else format(x, spec)


def build_report(labels: dict, pose: PoseResult, source: str = config.ELBOW_ANGLE_SOURCE) -> list[str]:
    fps = pose.fps
    s_main, a_main = analyze(pose, source)
    _, a_world = analyze(pose, "world")
    _, a_image = analyze(pose, "image")
    pairs = snap_labels(labels["reps"], a_main.reps, config.LABEL_SNAP_WINDOW_S)
    by_idx_world = {r.index: r for r in a_world.reps}
    by_idx_image = {r.index: r for r in a_image.reps}
    lines = [
        f"clip {labels.get('clip', '?')}: {pose.num_frames} frames at {fps:.1f} fps; labels by {labels.get('labelled_by', '?')}, "
        f"visual times good to about {labels.get('time_accuracy_s', '?')} s; snapped to the nearest detected peak within {config.LABEL_SNAP_WINDOW_S} s",
        f"pipeline setting: {source} elbow angle, dead-hang threshold {config.DEAD_HANG_ELBOW_MIN_DEG[source]} (3D estimate; see config.py)",
        "",
        "1. Per rep (angles in degrees, windows in seconds)",
        "  label   top(label) -> detected rep (peak s)  pipeline    match | 3D bottom: start  end  LOWER | 2D bottom: start  end  LOWER | window: start  end",
    ]
    complete_world, complete_image = [], []
    for lab, rep in pairs:
        if rep is None:
            lines.append(f"  {lab['label']:<8}{lab['approx_top_s']:>5.1f}s    -> NO detected rep within the window")
            continue
        w, i = by_idx_world[rep.index], by_idx_image[rep.index]
        match = "agree" if verdict_text(rep.full) == lab["label"] else "DIFFERS"
        lines.append(
            f"  {lab['label']:<8}{lab['approx_top_s']:>5.1f}s    -> rep {rep.index} ({rep.peak_s:4.1f}s, d={rep.peak_s - lab['approx_top_s']:+.1f})  "
            f"{verdict_text(rep.full):<9}{match:>8} | {_f(w.elbow_start_deg, '4.0f')} {_f(w.elbow_end_deg, '4.0f')} {_f(w.bottom_elbow_deg, '5.0f')}"
            f"      | {_f(i.elbow_start_deg, '4.0f')} {_f(i.elbow_end_deg, '4.0f')} {_f(i.bottom_elbow_deg, '5.0f')}"
            f"      | {w.start_window_frames / fps:5.2f} {w.end_window_frames / fps:5.2f}"
        )
        if not w.clipped:
            complete_world.append(w)
            complete_image.append(i)
    for name, reps_ in (("3D", complete_world), ("2D", complete_image)):
        if len(reps_) >= 2:
            low = min(reps_, key=lambda r: r.bottom_elbow_deg)
            values = ", ".join(f"rep {r.index}: {r.bottom_elbow_deg:.1f}" for r in reps_)
            first = reps_[0]
            tie = [r.index for r in reps_ if abs(r.bottom_elbow_deg - low.bottom_elbow_deg) < 0.5]
            lines.append(
                f"  lowest {name} bottom among complete reps: rep {low.index} ({low.bottom_elbow_deg:.1f}); rep {first.index} "
                f"(first, starts from the first grab) is {'THE LOWEST' if low.index == first.index else 'not the lowest'}"
                f"{'' if len(tie) <= 1 else f' (within 0.5 deg of reps {tie})'}  [{values}]"
            )
    lines += ["", "2. Head over the bar at the top, per head landmark (body scales; the bar is the wrist line plus the configured offset, i.e. the bar centre)",
              "  rep   nose           mouth          eyes           ears        (value / confidence)"]
    per_group: dict[str, list[float]] = {k: [] for k in HEAD_GROUPS}
    for lab, rep in pairs:
        if rep is None:
            continue
        vals = head_minus_bar_by_group(pose, s_main, rep.peak_frame)
        lines.append("  " + f"{rep.index:<4}" + "".join(f"  {_f(vals[g][0], '6.2f')} / {vals[g][1]:.2f}" for g in HEAD_GROUPS))
        if lab["label"] == "FULL":
            for g in HEAD_GROUPS:
                if vals[g][0] is not None:
                    per_group[g].append(vals[g][0])
    lines.append("  over the labelled-FULL reps:")
    for g, v in per_group.items():
        if v:
            lines.append(f"    {g:<6} mean {np.mean(v):5.2f}  min {min(v):5.2f}  max {max(v):5.2f}  spread {max(v) - min(v):4.2f}  (n={len(v)}; above the wrist line in {sum(x >= 0 for x in v)} of {len(v)})")
    return lines


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Check the pipeline against hand labels and print per-rep diagnostics.")
    ap.add_argument("labels", help="labels JSON under data/labels/")
    ap.add_argument("--source", default=config.ELBOW_ANGLE_SOURCE, choices=["world", "image"])
    args = ap.parse_args(argv)
    labels = load_labels(args.labels)
    pose = load_pose(labels["keypoints"])
    print("\n".join(build_report(labels, pose, args.source)))


if __name__ == "__main__":
    main()

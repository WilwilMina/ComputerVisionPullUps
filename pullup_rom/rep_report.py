"""Print each detected rep with its bottom elbow angle, for inspection. Reads a keypoints .npz.

    python -m pullup_rom.rep_report data/cache/<clip>.npz [--source world|image]

Dead-hang thresholds are unset until calibrated, so a rep that clears the bar shows UNKNOWN, not FULL.
Everything here is provisional; nothing printed is an accuracy figure.
"""
from __future__ import annotations

import argparse
from typing import Optional, Sequence

import numpy as np

from . import config
from .metrics import MetricsParams, Signals, compute_signals
from .pose import PoseResult, load_pose
from .reps import Rep, RepAnalysis, RepParams, detect_reps


def analyze(pose: PoseResult, source: str = config.ELBOW_ANGLE_SOURCE) -> tuple[Signals, RepAnalysis]:
    s = compute_signals(pose, MetricsParams(elbow_angle_source=source))
    return s, detect_reps(s.head_minus_bar, s.elbow_deg, s.fps, RepParams.for_source(source), bar_carried=s.bar_carried)


def _label(rep: Rep) -> str:
    return {True: "FULL", False: "NOT FULL", None: "UNKNOWN"}[rep.full]


def _deg(x: float) -> str:
    return "  n/a" if not np.isfinite(x) else f"{x:5.0f}"


_FROM_CONFIG = object()


def format_reps(analysis: RepAnalysis, source: str, dead_hang_deg=_FROM_CONFIG, origin: str = "") -> list[str]:
    """Rep table lines. `dead_hang_deg` is the threshold actually used (None = unset); default reads config."""
    thr = config.DEAD_HANG_ELBOW_MIN_DEG[source] if dead_hang_deg is _FROM_CONFIG else dead_hang_deg
    lines = [
        f"reps: {len(analysis.reps)}  (elbow source: {source}; dead-hang threshold: "
        f"{'UNSET (uncalibrated)' if thr is None else f'{thr:g} deg' + (f' ({origin})' if origin else '')}; head-clear margin {config.HEAD_CLEAR_MARGIN_BODY_SCALES:g} body scales, provisional)"
    ]
    if analysis.reps:
        lines.append("  #  start   peak    end  pull time | peak h-b | elbow deg: start   end  BOTTOM | head elbow  label     | flags")
    for r in analysis.reps:
        elbow_ok = {True: "ok ", False: "no ", None: "?  "}[r.elbow_ok]
        flags = list(r.reasons) + (["bar_carried_at_peak"] if r.peak_bar_carried else [])
        pull = "   n/a" if r.pull_time_s is None else f"{r.pull_time_s:5.1f}s"
        lines.append(
            f" {r.index:2d} {r.start_s:6.1f}s {r.peak_s:5.1f}s {r.end_s:5.1f}s {pull} | {r.peak_head_minus_bar:8.2f} | "
            f"           {_deg(r.elbow_start_deg)} {_deg(r.elbow_end_deg)} {_deg(r.bottom_elbow_deg)} | "
            f"{'ok  ' if r.head_ok else 'no  '} {elbow_ok}  {_label(r):<9} | {', '.join(flags) if flags else '-'}"
        )
    for d in analysis.discarded:
        lines.append(f"  {d.label}: {d.start_s:.1f}s to {d.end_s:.1f}s (not counted as a rep)")
    return lines


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Print detected reps and their bottom elbow angles.")
    ap.add_argument("keypoints", help="keypoints .npz from pullup_rom.pose or the eval cache")
    ap.add_argument("--source", default=config.ELBOW_ANGLE_SOURCE, choices=["world", "image"])
    args = ap.parse_args(argv)
    _, analysis = analyze(load_pose(args.keypoints), args.source)
    print("\n".join(format_reps(analysis, args.source)))


if __name__ == "__main__":
    main()

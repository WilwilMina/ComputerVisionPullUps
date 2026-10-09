"""Command line: video in, annotated video and JSON summary out.

    python -m pullup_rom.cli input.mp4 -o out.mp4
    python -m pullup_rom.cli input.mp4 -o out.mp4 --keypoints cached.npz    # skip the pose model

All thresholds are provisional placeholders in config.py. The dead-hang elbow threshold is unset until calibrated,
so reps that clear the bar are reported UNKNOWN, never guessed. Nothing printed or written is an accuracy figure.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Optional, Sequence

from . import config
from .camera import CameraChecks, camera_checks, camera_warnings
from .metrics import MetricsParams, Signals, compute_signals
from .pose import MediaPipePoseBackend, PoseResult, load_pose, save_pose
from .rep_report import format_reps
from .render import RenderStats, check_video_matches, invalid_share, render_video, warning_text
from .reps import RepAnalysis, RepParams, detect_reps

NOTE = (
    "All thresholds are provisional placeholders. UNKNOWN means the dead-hang elbow threshold for this angle "
    "source is not calibrated. These are tracking and rep-detection outputs on one video, not accuracy figures."
)


def _num(x: float) -> Optional[float]:
    return None if x is None or (isinstance(x, float) and not math.isfinite(x)) else round(float(x), 4)


def build_summary(
    video: str,
    pose: PoseResult,
    signals: Signals,
    analysis: RepAnalysis,
    source: str,
    dead_hang_deg: Optional[float],
    dead_hang_origin: str,
    warnings: list[str],
    stats: Optional[RenderStats],
    camera: Optional[CameraChecks] = None,
) -> dict:
    n = pose.num_frames
    return {
        "input": Path(video).name,
        "note": NOTE,
        "fps": _num(pose.fps),
        "frames": n,
        "duration_s": _num(n / pose.fps),
        "elbow_angle_source": source,
        "dead_hang_threshold_deg": dead_hang_deg,
        "dead_hang_threshold_origin": dead_hang_origin,  # "unset" | "config" | "command line"
        "head_clear_margin_body_scales": config.HEAD_CLEAR_MARGIN_BODY_SCALES,
        "frames_without_person_share": _num(1.0 - float(pose.detected.mean())),
        "invalid_share": {
            "head": _num(float((~signals.head_valid).mean())),
            "elbow": _num(float((~signals.elbow_valid).mean())),
            "either": _num(invalid_share(signals)),
            "warn_at": config.WARN_INVALID_SHARE,
        },
        "warnings": warnings,
        "camera_checks": None if camera is None else {
            "shoulder_width_over_torso_length": _num(camera.shoulder_torso_ratio) if camera.shoulder_torso_ratio is not None else None,
            "wrist_height_gap_body_scales": _num(camera.wrist_gap_body_scales) if camera.wrist_gap_body_scales is not None else None,
        },
        "output_video": None if stats is None else {"frames": stats.frames, "width": stats.size[0], "height": stats.size[1]},
        "reps": [
            {
                "index": r.index,
                "start_s": _num(r.start_s),
                "peak_s": _num(r.peak_s),
                "end_s": _num(r.end_s),
                "pull_time_s": _num(r.pull_time_s),
                "ascent_s": _num(r.ascent_s),
                "peak_head_minus_bar_body_scales": _num(r.peak_head_minus_bar),
                "head_cleared_bar": r.head_ok,
                "arms_straightened": r.elbow_ok,  # true / false / null = UNKNOWN
                "bottom_elbow_deg": _num(r.bottom_elbow_deg),
                "elbow_start_deg": _num(r.elbow_start_deg),
                "elbow_end_deg": _num(r.elbow_end_deg),
                "full": r.full,  # true / false / null = UNKNOWN
                "bar_carried_at_peak": r.peak_bar_carried,
                "flags": list(r.reasons),
            }
            for r in analysis.reps
        ],
        "uncertain_attempts": [
            {"label": d.label, "start_s": _num(d.start_s), "end_s": _num(d.end_s)} for d in analysis.discarded
        ],
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Detect pull-up reps and render an annotated video plus a JSON summary.")
    ap.add_argument("video")
    ap.add_argument("-o", "--output", required=True, help="annotated .mp4 to write")
    ap.add_argument("--summary", help="JSON summary path (default: the output path with .json)")
    ap.add_argument("--keypoints", help="use these cached keypoints (.npz) instead of running the pose model")
    ap.add_argument("--cache", help="where to save extracted keypoints (default: next to the output)")
    ap.add_argument("--no-cache", action="store_true", help="do not save extracted keypoints")
    ap.add_argument("--source", default=config.ELBOW_ANGLE_SOURCE, choices=["world", "image"], help="elbow angle source")
    ap.add_argument("--variant", default=config.POSE_MODEL_VARIANT, choices=["lite", "full", "heavy"])
    ap.add_argument("--dead-hang-deg", type=float, help="dead-hang elbow angle for THIS source; use only a value you calibrated")
    args = ap.parse_args(argv)

    video, out = Path(args.video), Path(args.output)
    if not video.is_file():
        raise SystemExit(f"error: video not found: {video}")

    try:
        if args.keypoints:
            pose = load_pose(args.keypoints)
        else:
            pose = MediaPipePoseBackend(args.variant).extract(video)
            if not args.no_cache:
                cache = Path(args.cache) if args.cache else out.with_suffix(".keypoints.npz")
                save_pose(cache, pose)
                print(f"keypoints cached: {cache}")
        if args.keypoints:
            check_video_matches(video, pose)  # wrong cache file: refuse, never draw nonsense
        signals = compute_signals(pose, MetricsParams(elbow_angle_source=args.source))
    except (FileNotFoundError, ValueError, OSError) as e:
        raise SystemExit(f"error: {e}")

    if args.dead_hang_deg is not None:
        rep_params = RepParams.for_source(args.source, dead_hang_elbow_min_deg=args.dead_hang_deg)
        origin = "command line"
    else:
        rep_params = RepParams.for_source(args.source)
        origin = "unset" if rep_params.dead_hang_elbow_min_deg is None else "config"
    analysis = detect_reps(signals.head_minus_bar, signals.elbow_deg, signals.fps, rep_params, bar_carried=signals.bar_carried)

    warnings: list[str] = []
    w = warning_text(invalid_share(signals))
    if w:
        warnings.append(w)
    checks = camera_checks(pose)
    warnings.extend(camera_warnings(checks))
    try:
        stats = render_video(video, pose, signals, analysis, out, warnings, elbow_source=args.source)
    except (ValueError, OSError) as e:
        raise SystemExit(f"error: {e}")
    if stats.frames < pose.num_frames:
        warnings.append(f"WARNING: the video ended after {stats.frames} of {pose.num_frames} frames; the output is incomplete")

    summary_path = Path(args.summary) if args.summary else out.with_suffix(".json")
    summary = build_summary(str(video), pose, signals, analysis, args.source, rep_params.dead_hang_elbow_min_deg, origin, warnings, stats, checks)
    try:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    except OSError as e:
        raise SystemExit(f"error: cannot write the summary to {summary_path}: {e.strerror or e}")

    for line in format_reps(analysis, args.source, rep_params.dead_hang_elbow_min_deg, origin):
        print(line)
    for msg in warnings:
        print(msg)
    print(f"annotated video: {out}  ({stats.frames} frames, {stats.size[0]}x{stats.size[1]})")
    print(f"summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

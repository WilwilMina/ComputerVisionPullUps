"""Unsupported-style flag: large body swing at the bottom of the reps (kipping or swinging).

This tool judges range of motion on strict reps. A kipping or swinging set is a different style, not a range-of-motion
fault, and the arms and head readings on it are not trustworthy, so the CLI flags it as UNSUPPORTED STYLE next to the
camera warnings. It never changes a verdict or a measurement. The cutoff is provisional (config.py).

Measure: the angle of the torso (mid-hip to mid-shoulder, from the model's 3D world landmarks) away from vertical,
read at the bottom of each rep and taken as the median over reps. A strict rep hangs close to vertical at the bottom;
a kipping rep swings out. The 3D landmarks are the pose model's estimate and the lean toward or away from the camera
(the main component of a kip from the front or back) is the part the model estimates least well, so this is a coarse
advisory signal, not a measurement of the swing.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from . import config
from .pose import PoseResult
from .reps import Rep


@dataclass(frozen=True)
class StyleChecks:
    bottom_tilt_deg: Optional[float]  # median over reps of the torso tilt from vertical at the bottoms; None = not measurable
    reps_used: int  # reps with a usable bottom tilt


def torso_tilt_deg(pose: PoseResult) -> np.ndarray:
    """[F] angle of the world-space torso from vertical in degrees; NaN where the hips or shoulders are not confident."""
    if pose.world is None:
        return np.full(pose.num_frames, np.nan)
    w, kp, m = pose.world.astype(np.float64), pose.keypoints, config.MIN_LANDMARK_CONFIDENCE
    joints = (config.LEFT_SHOULDER, config.RIGHT_SHOULDER, config.LEFT_HIP, config.RIGHT_HIP)
    conf = np.all([kp[:, j, 2] >= m for j in joints], axis=0)
    v = (w[:, config.LEFT_SHOULDER] + w[:, config.RIGHT_SHOULDER]) / 2 - (w[:, config.LEFT_HIP] + w[:, config.RIGHT_HIP]) / 2
    n = np.linalg.norm(v, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        tilt = np.degrees(np.arccos(np.clip(-v[:, 1] / n, -1.0, 1.0)))  # world y points down: shoulders above hips = -y
    return np.where(conf & np.isfinite(tilt) & (n > 0), tilt, np.nan)


def style_checks(pose: PoseResult, reps: Sequence[Rep]) -> StyleChecks:
    tilt = torso_tilt_deg(pose)
    h = config.SWING_BOTTOM_HALF_WINDOW_FRAMES
    per_rep = []
    for r in reps:
        vals = []
        for f in (r.start_frame, r.end_frame if not r.clipped else None):
            if f is None:
                continue
            seg = tilt[max(0, f - h) : f + h + 1]
            if np.isfinite(seg).any():
                vals.append(float(np.nanmedian(seg)))
        if vals:
            per_rep.append(max(vals))  # the more swung-out of the two bottoms
    if not per_rep:
        return StyleChecks(None, 0)
    return StyleChecks(float(np.median(per_rep)), len(per_rep))


def style_warnings(checks: StyleChecks) -> list[str]:
    if checks.bottom_tilt_deg is None or checks.reps_used < config.UNSUPPORTED_SWING_MIN_REPS:
        return []
    if checks.bottom_tilt_deg >= config.UNSUPPORTED_SWING_BOTTOM_TILT_DEG:
        return ["UNSUPPORTED STYLE: large body swing at the bottom of the reps (kipping or swinging); this tool judges strict reps, so the verdicts below are not reliable"]
    return []

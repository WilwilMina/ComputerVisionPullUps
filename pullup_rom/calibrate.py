"""Per-person dead-hang threshold from a reference clip of the person's own known-good reps.

The pose model's 3D elbow angle is an estimate that differs by person and camera view (the same straight arms read
about 130 on one person and about 160 on another), so one global number does not transfer. A person can instead film a
short clip of reps they know are full range of motion (a real dead hang each time, chin over the bar); the threshold for
THEIR later clips is then derived from it, by the same one-sided rule the global 144 used: the strictest whole degree
that keeps every reference rep FULL. No tolerance or margin is invented. Consequences, stated plainly:
  - it says only "do not require a straighter arm than you showed in your own good reps"; it is a floor from positives;
  - the reference reps pass by construction, so running the reference clip itself proves nothing;
  - it needs the same camera view and setup as the clips it will judge, and honest reference reps: a lazy reference
    gives a lax threshold. Only bad reps filmed by the same person can say how far below the floor is "not full".
This touches the arms check only; the head check and every other setting are unchanged.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from . import config
from .metrics import MetricsParams, compute_signals
from .pose import PoseResult
from .reps import RepParams, detect_reps


@dataclass(frozen=True)
class Reference:
    threshold_deg: float  # whole degrees; the strictest value that keeps every used reference rep FULL
    bottoms_deg: tuple[float, ...]  # the bottom elbow angle of each rep used
    reps_found: int  # complete reps detected in the clip
    reps_used: int  # reps used: complete, head cleared the bar, bottom angle known
    source: str  # "world" | "image"
    tolerance_deg: float = 0.0  # degrees allowed below the lowest reference bottom (already subtracted from threshold_deg)

    @property
    def origin(self) -> str:
        tol = f", {self.tolerance_deg:g} deg tolerance" if self.tolerance_deg else ""
        return f"reference clip: lowest bottom {min(self.bottoms_deg):.1f} deg over {self.reps_used} reps{tol}"


def reference_threshold(bottoms_deg: Sequence[float]) -> float:
    """Largest whole degree at or below every reference bottom (same rule as the global calibration)."""
    if not bottoms_deg:
        raise ValueError("no reference bottoms")
    return float(math.floor(min(bottoms_deg)))


def calibrate_from_reference(
    pose: PoseResult, source: str = config.ELBOW_ANGLE_SOURCE, min_reps: int = config.REFERENCE_MIN_REPS, tolerance_deg: float = config.REFERENCE_TOLERANCE_DEG
) -> Reference:
    """Threshold from a clip of known-good reps. Raises ValueError (with the reason) if the clip cannot support one."""
    signals = compute_signals(pose, MetricsParams(elbow_angle_source=source))
    analysis = detect_reps(
        signals.head_minus_bar, signals.elbow_deg, signals.fps, RepParams.for_source(source, dead_hang_elbow_min_deg=None), bar_carried=signals.bar_carried
    )
    complete = [r for r in analysis.reps if not r.clipped]
    used = [r for r in complete if r.head_ok and math.isfinite(r.bottom_elbow_deg)]
    if len(used) < min_reps:
        raise ValueError(
            f"the reference clip gives {len(used)} usable rep(s) of {len(analysis.reps)} detected "
            f"({len(complete) - len(used)} complete but head short of the bar or no bottom angle); need at least {min_reps} known-good reps"
        )
    bottoms = tuple(float(r.bottom_elbow_deg) for r in used)
    if not (math.isfinite(tolerance_deg) and tolerance_deg >= 0):
        raise ValueError(f"tolerance must be a finite number of degrees, 0 or more, got {tolerance_deg}")
    threshold = reference_threshold(bottoms) - float(tolerance_deg)
    if threshold <= 0:
        raise ValueError(f"a tolerance of {tolerance_deg:g} deg would put the threshold at {threshold:g} deg, so every rep would pass; use a smaller tolerance")
    return Reference(threshold, bottoms, len(complete), len(used), source, float(tolerance_deg))

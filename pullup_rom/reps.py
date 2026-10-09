"""Rep state machine. Pure: arrays in, records out. No I/O, no OpenCV, no model imports.

Input is the slice-2 time series: `head_minus_bar` (unitless, >0 = head above the bar, NaN where unusable)
and `elbow_deg` (NaN where unusable). The head signal finds the reps; the elbow angle is read ONLY at the
bottom of each rep (the dead-hang regions before and after it) and never at the top.

A rep is FULL only if the head clears the bar AND the bottom elbow angle reaches the dead-hang threshold.
That threshold is a per-angle-source calibrated parameter with no default value: while it is None a rep that
clears the bar has full=None (unknown) rather than a guessed label.

A rep runs from the last frame at the bottom before the pull (start), through the peak, to the first frame the
head is back at the low level it started from (end). Attempts the logic cannot trust are never dropped
silently: they are returned in `RepAnalysis.discarded` with a time range.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from . import config

REASON_HEAD_BELOW_BAR = "head_below_bar"
REASON_ELBOW_NOT_EXTENDED = "elbow_not_extended"
REASON_ELBOW_UNCALIBRATED = "elbow_threshold_uncalibrated"
REASON_ELBOW_MISSING = "elbow_angle_missing"
REASON_CLIP_ENDED_MID_REP = "clip_ended_mid_rep"

DISCARD_TRACKING_GAP = "gap_longer_than_max_during_rep"
DISCARD_TOO_SHORT = "ascent_shorter_than_min_duration"


@dataclass(frozen=True)
class RepParams:
    """Defaults come from config.py (all provisional there)."""

    head_clear_margin: float = config.HEAD_CLEAR_MARGIN_BODY_SCALES  # body scales
    dead_hang_elbow_min_deg: Optional[float] = None  # no default: calibrated per angle source, see for_source()
    start_rise: float = config.REP_START_RISE_BODY_SCALES  # body scales
    end_drop: float = config.REP_END_DROP_BODY_SCALES  # body scales
    debounce_s: float = config.REP_DEBOUNCE_S
    bottom_band: float = config.REP_BOTTOM_BAND_BODY_SCALES  # body scales
    min_ascent_s: float = config.REP_MIN_ASCENT_S
    max_gap_s: float = config.REP_MAX_GAP_S

    def __post_init__(self) -> None:
        for name in ("start_rise", "end_drop", "bottom_band", "debounce_s", "min_ascent_s", "max_gap_s"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0")
        if self.start_rise == 0 or self.end_drop == 0:
            raise ValueError("start_rise and end_drop must be > 0 or noise will start and end reps")

    @classmethod
    def for_source(cls, source: str, **overrides) -> "RepParams":
        """Params with the dead-hang threshold configured for this elbow-angle source ('world' or 'image')."""
        if source not in config.DEAD_HANG_ELBOW_MIN_DEG:
            raise ValueError(f"unknown elbow angle source {source!r}")
        overrides.setdefault("dead_hang_elbow_min_deg", config.DEAD_HANG_ELBOW_MIN_DEG[source])
        return cls(**overrides)


@dataclass(frozen=True)
class Rep:
    index: int  # 1-based
    start_frame: int  # last frame at the bottom before the pull
    peak_frame: int
    end_frame: int  # first frame back at the low level the rep started from (last finite frame if the clip ended mid-rep)
    start_s: float
    peak_s: float
    end_s: float
    peak_head_minus_bar: float  # body scales
    elbow_start_deg: float  # median elbow angle over the bottom before the pull; NaN if none
    elbow_end_deg: float  # same after lowering; NaN if none or the clip ended mid-rep
    bottom_elbow_deg: float  # the LESS extended (lower) of the two bottoms; NaN if neither exists
    head_ok: bool
    elbow_ok: Optional[bool]  # None = unknown (threshold unset or no elbow data)
    full: Optional[bool]  # False if the head failed; None if the head passed but the elbow is unknown
    peak_bar_carried: bool  # the bar value at the peak came from neighbouring frames
    clipped: bool = False  # the clip ended before the head came back down
    reasons: tuple[str, ...] = field(default_factory=tuple)

    @property
    def ascent_s(self) -> float:
        """Bottom to top."""
        return self.peak_s - self.start_s

    @property
    def pull_time_s(self) -> Optional[float]:
        """Bottom to top and back to the bottom; None when the clip ended before the head came back down."""
        return None if self.clipped else self.end_s - self.start_s


@dataclass(frozen=True)
class DiscardedAttempt:
    """A rep-like movement the logic did not count. Reported, never dropped silently."""

    start_frame: int
    end_frame: int
    reason: str
    start_s: float
    end_s: float

    @property
    def label(self) -> str:
        if self.reason == DISCARD_TRACKING_GAP:
            return "uncertain (tracking gap)"
        if self.reason == DISCARD_TOO_SHORT:
            return "ignored (too short to be a rep)"
        return self.reason


@dataclass(frozen=True)
class RepAnalysis:
    reps: list[Rep]
    discarded: list[DiscardedAttempt]


def _region_from(head: np.ndarray, seed: int, lo: int, hi: int, limit: float) -> tuple[int, int]:
    """(first, last) of the contiguous finite frames around `seed`, within [lo, hi], with head <= limit."""
    a = seed
    while a - 1 >= lo and np.isfinite(head[a - 1]) and head[a - 1] <= limit:
        a -= 1
    d = seed
    while d + 1 <= hi and np.isfinite(head[d + 1]) and head[d + 1] <= limit:
        d += 1
    return a, d


def _median_elbow(elbow: np.ndarray, first: int, last: int) -> float:
    """Median elbow angle over a bottom region.

    A median, not a maximum: with noisy angles a maximum drifts upward and can lift a bent bottom over the
    dead-hang threshold. Most of the region is the hang plateau, so the median sits on the plateau.
    """
    vals = elbow[first : last + 1]
    vals = vals[np.isfinite(vals)]
    return float(np.median(vals)) if vals.size else float("nan")


def detect_reps(
    head_minus_bar: np.ndarray,
    elbow_deg: np.ndarray,
    fps: float,
    params: Optional[RepParams] = None,
    bar_carried: Optional[np.ndarray] = None,
) -> RepAnalysis:
    p = params or RepParams()
    head = np.asarray(head_minus_bar, dtype=np.float64)
    elbow = np.asarray(elbow_deg, dtype=np.float64)
    n = len(head)
    if elbow.shape != head.shape:
        raise ValueError("head_minus_bar and elbow_deg must have the same length")
    if bar_carried is not None and len(bar_carried) != n:
        raise ValueError("bar_carried must have the same length as head_minus_bar")
    if not fps > 0:
        raise ValueError("fps must be positive")

    debounce = max(1, int(round(p.debounce_s * fps)))
    max_gap = int(round(p.max_gap_s * fps))

    raw: list[dict] = []  # {start, low, peak, peak_v, valley, clipped}
    discarded: list[DiscardedAttempt] = []
    state = "hang"
    valley_i = valley_v = None
    cur: Optional[dict] = None  # the ascent in progress
    pending: Optional[dict] = None  # a finished ascent waiting for its lowest point after lowering
    up_count = down_count = 0
    gap = 0
    last_finite = -1

    def close_pending(valley: int) -> None:
        nonlocal pending
        if pending is not None:
            pending["valley"] = valley
            raw.append(pending)
            pending = None

    def departure_of(start_valley: int) -> int:
        return _region_from(head, start_valley, 0, n - 1, head[start_valley] + p.bottom_band)[1]

    for i in range(n):
        h = head[i]
        if not np.isfinite(h):
            gap += 1
            continue
        if gap > max_gap and last_finite >= 0:
            # A long hole in the data: never link across it. Report an unfinished ascent as uncertain, close a finished one.
            if state == "up":
                dep = departure_of(cur["start"])
                discarded.append(DiscardedAttempt(dep, i, DISCARD_TRACKING_GAP, dep / fps, i / fps))
                close_pending(cur["start"])
                cur = None
            else:
                close_pending(valley_i)
            state, valley_i, valley_v = "hang", i, h
            up_count = down_count = 0
        gap = 0
        last_finite = i

        if valley_i is None:
            valley_i, valley_v = i, h
        if state == "hang":
            if h < valley_v:
                valley_i, valley_v = i, h
            up_count = up_count + 1 if h - valley_v >= p.start_rise else 0
            if up_count >= debounce:
                close_pending(valley_i)  # the valley between two reps ends one and starts the next
                cur = {"start": valley_i, "low": valley_v, "peak": i, "peak_v": h, "clipped": False}
                state, up_count, down_count = "up", 0, 0
        else:
            if h > cur["peak_v"]:
                cur["peak"], cur["peak_v"] = i, h
            down_count = down_count + 1 if cur["peak_v"] - h >= p.end_drop else 0
            if down_count >= debounce:
                pending = cur
                cur = None
                state, valley_i, valley_v, up_count, down_count = "hang", i, h, 0, 0

    if state == "up" and cur is not None:  # clip ended before the head came back down
        cur["clipped"] = True
        cur["valley"] = last_finite
        raw.append(cur)
    else:
        close_pending(valley_i)

    def start_region(seq: list[dict], k: int) -> tuple[int, int]:
        lo = seq[k - 1]["peak"] if k > 0 else 0
        return _region_from(head, seq[k]["start"], lo, seq[k]["peak"], head[seq[k]["start"]] + p.bottom_band)

    def end_region(seq: list[dict], k: int) -> tuple[int, int]:
        """Arrival = first frame after the peak with the head back at the low level the rep started from.

        If it never gets back within `bottom_band` of that level, fall back to the region around its lowest point.
        """
        r = seq[k]
        hi = seq[k + 1]["peak"] if k + 1 < len(seq) else n - 1
        limit = r["low"] + p.bottom_band
        for j in range(r["peak"] + 1, r["valley"] + 1):
            if np.isfinite(head[j]) and head[j] <= limit:
                return _region_from(head, j, r["peak"], hi, limit)
        return _region_from(head, r["valley"], r["peak"], hi, head[r["valley"]] + p.bottom_band)

    # A rep's ascent runs from the last frame at the bottom to the peak. Anything shorter than
    # min_ascent_s is a tracking glitch, not a rep.
    min_ascent = int(round(p.min_ascent_s * fps))
    kept = []
    for k, r in enumerate(raw):
        departure = start_region(raw, k)[1]
        if r["peak"] - departure < min_ascent:
            discarded.append(DiscardedAttempt(departure, r["valley"], DISCARD_TOO_SHORT, departure / fps, r["valley"] / fps))
        else:
            kept.append(r)
    raw = kept
    discarded.sort(key=lambda d: d.start_frame)

    reps: list[Rep] = []
    for k, r in enumerate(raw):
        s_arr, s_dep = start_region(raw, k)
        elbow_start = _median_elbow(elbow, s_arr, s_dep)
        if r["clipped"]:
            e_arr, elbow_end = r["valley"], float("nan")
        else:
            e_arr, e_dep = end_region(raw, k)
            elbow_end = _median_elbow(elbow, e_arr, e_dep)
        bottoms = [v for v in (elbow_start, elbow_end) if np.isfinite(v)]
        bottom = min(bottoms) if bottoms else float("nan")

        reasons: list[str] = []
        head_ok = bool(r["peak_v"] >= p.head_clear_margin)
        if not head_ok:
            reasons.append(REASON_HEAD_BELOW_BAR)
        if r["clipped"]:
            reasons.append(REASON_CLIP_ENDED_MID_REP)
        if p.dead_hang_elbow_min_deg is None:
            elbow_ok = None
            reasons.append(REASON_ELBOW_UNCALIBRATED)
        elif r["clipped"]:
            elbow_ok = None  # the clip ended before the arms came back down: the return to dead hang was never seen
        elif not np.isfinite(bottom):
            elbow_ok = None
            reasons.append(REASON_ELBOW_MISSING)
        else:
            elbow_ok = bool(bottom >= p.dead_hang_elbow_min_deg)
            if not elbow_ok:
                reasons.append(REASON_ELBOW_NOT_EXTENDED)
        full = False if not head_ok else elbow_ok

        reps.append(
            Rep(
                index=k + 1,
                start_frame=s_dep,
                peak_frame=r["peak"],
                end_frame=e_arr,
                start_s=s_dep / fps,
                peak_s=r["peak"] / fps,
                end_s=e_arr / fps,
                peak_head_minus_bar=float(r["peak_v"]),
                elbow_start_deg=elbow_start,
                elbow_end_deg=elbow_end,
                bottom_elbow_deg=bottom,
                head_ok=head_ok,
                elbow_ok=elbow_ok,
                full=full,
                peak_bar_carried=bool(bar_carried[r["peak"]]) if bar_carried is not None else False,
                clipped=bool(r["clipped"]),
                reasons=tuple(reasons),
            )
        )
    return RepAnalysis(reps=reps, discarded=discarded)

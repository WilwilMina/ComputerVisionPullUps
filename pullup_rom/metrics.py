"""Keypoints -> smoothed, unitless signals for the rep state machine.

The only place image y (down) is flipped to height (up). Elbow angles are
computed from joint vectors, which a flip does not change, so they use the
raw coordinates.

Missing data is never turned into zeros: low-confidence or absent joints are
NaN, short gaps are interpolated, long gaps stay NaN and `Signals.head_valid` /
`Signals.elbow_valid` are False there. The bar line uses only wrist positions the
model was confident about; frames where it is carried from neighbours are flagged.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.signal import savgol_filter

from . import config
from .pose import PoseResult


# Bit flags for Signals.elbow_arms_used: which arm(s) contributed to elbow_deg in a frame.
ARM_LEFT = 1
ARM_RIGHT = 2  # both arms = ARM_LEFT | ARM_RIGHT; 0 = no arm had a usable angle


@dataclass(frozen=True)
class MetricsParams:
    """Defaults come from config.py (all provisional there); override per call or per person."""

    min_landmark_confidence: float = config.MIN_LANDMARK_CONFIDENCE
    max_gap_s: float = config.MAX_GAP_S
    smooth_window_s: float = config.SMOOTH_WINDOW_S
    smooth_polyorder: int = config.SMOOTH_POLYORDER
    head_landmarks: tuple[int, ...] = config.HEAD_LANDMARKS
    body_scale_mode: str = config.BODY_SCALE_MODE
    body_scale_window_s: float = config.BODY_SCALE_WINDOW_S
    bar_window_s: float = config.BAR_WINDOW_S
    bar_min_confident_share: float = config.BAR_MIN_CONFIDENT_SHARE
    bar_offset_body_scales: float = config.BAR_OFFSET_BODY_SCALES
    elbow_angle_source: str = config.ELBOW_ANGLE_SOURCE
    elbow_arm_combine: str = config.ELBOW_ARM_COMBINE


@dataclass
class Signals:
    fps: float
    t_s: np.ndarray  # [F] seconds
    head_height_px: np.ndarray  # [F] height-up pixels, smoothed (debug / overlay)
    wrist_height_px: np.ndarray  # [F] mean height-up pixels of the wrists the model was confident about; NaN otherwise (never interpolated)
    bar_height_px: np.ndarray  # [F] height-up pixels: rolling median over confident gripping frames, smoothed
    bar_support: np.ndarray  # [F] share of the bar window holding confident gripping frames (0..1)
    bar_carried: np.ndarray  # [F] bool: a bar value exists but this frame has no confident gripping wrist of its own (taken from neighbours)
    body_scale_px: np.ndarray  # [F] pixels per body-scale unit
    head_minus_bar: np.ndarray  # [F] unitless (body scales); >0 = head above the bar; NaN where not head_valid
    elbow_deg: np.ndarray  # [F] selected source, NaN where not elbow_valid
    elbow_image_deg: np.ndarray  # [F] 2D-pixel angle, smoothed (debug)
    elbow_world_deg: np.ndarray  # [F] 3D world angle, smoothed; all NaN if the backend has no world
    elbow_arms_used: np.ndarray  # [F] uint8 bit flags ARM_LEFT/ARM_RIGHT for the selected source, so a one-arm frame is visible
    gripping: np.ndarray  # [F] bool: a confident wrist is above the shoulders (hanging on the bar)
    in_hang_span: np.ndarray  # [F] bool: between the first and last confident gripping frame
    head_valid: np.ndarray  # [F] bool: head_minus_bar usable. Does NOT need wrists in this frame (see bar_carried)
    elbow_valid: np.ndarray  # [F] bool: elbow_deg usable (own validity: needs the arm joints, not the bar)


# --- small pure helpers (each unit-tested) ---

def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """(start, end_exclusive) of each contiguous True run."""
    m = np.concatenate(([False], mask, [False])).astype(np.int8)
    d = np.diff(m)
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def _frames(seconds: float, fps: float) -> int:
    return int(round(seconds * fps))


def _odd_frames(seconds: float, fps: float) -> int:
    n = max(1, _frames(seconds, fps))
    return n if n % 2 == 1 else n + 1


def interpolate_short_gaps(x: np.ndarray, max_gap_frames: int) -> np.ndarray:
    """Linearly fill NaN runs of at most `max_gap_frames` that have valid neighbours both sides."""
    out = np.asarray(x, dtype=np.float64).copy()
    out[~np.isfinite(out)] = np.nan
    for s, e in _runs(np.isnan(out)):
        if s == 0 or e == len(out) or (e - s) > max_gap_frames:
            continue
        out[s:e] = np.interp(np.arange(s, e), [s - 1, e], [out[s - 1], out[e]])
    return out


def smooth_valid_runs(x: np.ndarray, window_frames: int, polyorder: int) -> np.ndarray:
    """Savitzky-Golay on each NaN-free run; NaN stays NaN. Short runs use a shrunk odd window."""
    out = np.asarray(x, dtype=np.float64).copy()
    for s, e in _runs(np.isfinite(out)):
        n = e - s
        w = min(window_frames, n if n % 2 == 1 else n - 1)
        if w > polyorder:
            out[s:e] = savgol_filter(out[s:e], w, polyorder, mode="interp")
    return out


def rolling_median_masked(x: np.ndarray, window_frames: int, min_count: int = 1) -> np.ndarray:
    """Centred rolling median ignoring NaN; NaN where the window holds fewer than `min_count` samples.

    Windows are truncated at the series ends, so the first/last window/2 frames
    are biased toward the interior (under drift, by up to slope*window/4).
    """
    half = window_frames // 2
    padded = np.pad(np.asarray(x, dtype=np.float64), half, constant_values=np.nan)
    win = np.sort(sliding_window_view(padded, 2 * half + 1), axis=1)  # NaN sorts last
    count = np.isfinite(win).sum(axis=1)
    rows = np.arange(len(win))
    lo = win[rows, np.maximum(count - 1, 0) // 2]
    hi = win[rows, count // 2]
    return np.where(count >= max(min_count, 1), (lo + hi) / 2, np.nan)  # np.nanmedian is far slower on [F, w]


def rolling_share(mask: np.ndarray, window_frames: int) -> np.ndarray:
    """Share (0..1) of True frames in a centred window of `window_frames` (odd). Out-of-range counts as False."""
    kernel = np.ones(window_frames)
    return np.convolve(np.asarray(mask, dtype=np.float64), kernel, mode="same") / window_frames


def elbow_angle_deg(shoulder: np.ndarray, elbow: np.ndarray, wrist: np.ndarray) -> np.ndarray:
    """Interior angle at the elbow in degrees; 180 = arm straight. Inputs [F, D], D = 2 or 3."""
    v1, v2 = shoulder - elbow, wrist - elbow
    denom = np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        cos = np.einsum("ij,ij->i", v1, v2) / denom
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def _combine_arms(left: np.ndarray, right: np.ndarray, mode: str) -> np.ndarray:
    if mode not in ("mean", "min"):
        raise ValueError(f"elbow_arm_combine must be 'mean' or 'min', got {mode!r}")
    both = np.stack([left, right])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # both arms missing -> NaN
        return np.nanmean(both, axis=0) if mode == "mean" else np.nanmin(both, axis=0)


# --- main entry point ---

def compute_signals(pose: PoseResult, params: Optional[MetricsParams] = None) -> Signals:
    p = params or MetricsParams()
    if p.elbow_angle_source not in ("world", "image"):
        raise ValueError(f"elbow_angle_source must be 'world' or 'image', got {p.elbow_angle_source!r}")
    if p.body_scale_mode not in ("shoulder_width", "torso"):
        raise ValueError(f"body_scale_mode must be 'shoulder_width' or 'torso', got {p.body_scale_mode!r}")
    if len(p.head_landmarks) == 0:
        raise ValueError("head_landmarks must name at least one landmark")
    if p.elbow_angle_source == "world" and pose.world is None:
        raise ValueError("elbow_angle_source='world' but this backend produced no world landmarks; use 'image'")

    fps = pose.fps
    n_frames = pose.num_frames
    frame_h = pose.frame_size[1]
    gap = _frames(p.max_gap_s, fps)
    sm_win = _odd_frames(p.smooth_window_s, fps)

    def smooth(x: np.ndarray) -> np.ndarray:
        return smooth_valid_runs(x, sm_win, p.smooth_polyorder)

    def clean(arr: np.ndarray, ok: np.ndarray) -> np.ndarray:
        """[F, D] -> low-confidence/absent frames NaN, short gaps interpolated."""
        a = np.where(ok[:, None], arr, np.nan)
        return np.stack([interpolate_short_gaps(a[:, d], gap) for d in range(a.shape[1])], axis=1)

    kp = pose.keypoints
    ok_cache: dict[int, np.ndarray] = {}

    def ok_mask(j: int) -> np.ndarray:
        if j not in ok_cache:
            ok_cache[j] = (kp[:, j, 2] >= p.min_landmark_confidence) & np.isfinite(kp[:, j, :2]).all(axis=1)
        return ok_cache[j]

    def xy(j: int) -> np.ndarray:
        return clean(kp[:, j, :2].astype(np.float64), ok_mask(j))

    def xyz_world(j: int) -> np.ndarray:
        return clean(pose.world[:, j, :].astype(np.float64), ok_mask(j))

    def up(y_down: np.ndarray) -> np.ndarray:
        return frame_h - y_down  # the one and only y flip

    # Joints in image pixels (y down), cleaned.
    l_sh, r_sh = xy(config.LEFT_SHOULDER), xy(config.RIGHT_SHOULDER)
    l_el, r_el = xy(config.LEFT_ELBOW), xy(config.RIGHT_ELBOW)
    l_wr, r_wr = xy(config.LEFT_WRIST), xy(config.RIGHT_WRIST)
    l_hp, r_hp = xy(config.LEFT_HIP), xy(config.RIGHT_HIP)

    # Heights (up). Head = mean of the chosen landmarks, NaN if any is missing.
    head_h = up(np.mean([xy(j)[:, 1] for j in p.head_landmarks], axis=0))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        shoulder_h = np.nanmean([up(l_sh[:, 1]), up(r_sh[:, 1])], axis=0)
        # Wrists for the BAR use only positions the model was confident about: never interpolated, never guessed.
        wrist_conf_h = np.nanmean(
            [np.where(ok_mask(j), up(kp[:, j, 1].astype(np.float64)), np.nan) for j in (config.LEFT_WRIST, config.RIGHT_WRIST)],
            axis=0,
        )

    # Body scale: pixels per unit, rolling median so posture changes don't move it.
    if p.body_scale_mode == "shoulder_width":
        scale_raw = np.linalg.norm(l_sh - r_sh, axis=1)
    else:
        scale_raw = np.linalg.norm((l_sh + r_sh) / 2 - (l_hp + r_hp) / 2, axis=1)
    scale_raw = np.where(scale_raw > 0, scale_raw, np.nan)  # coincident joints are a tracking failure, not scale 0
    body_scale = rolling_median_masked(scale_raw, _odd_frames(p.body_scale_window_s, fps))

    # Bar line: wrists stay on the bar, so the rolling median of confident gripping-wrist height follows it,
    # including a slowly moving camera. A frame whose own wrists were lost still gets a bar value if enough
    # confident frames sit in its window; those frames are flagged in bar_carried.
    with np.errstate(invalid="ignore"):
        gripping = np.isfinite(wrist_conf_h) & np.isfinite(shoulder_h) & (wrist_conf_h > shoulder_h)
    bar_win = _odd_frames(p.bar_window_s, fps)
    bar_support = rolling_share(gripping, bar_win)
    min_count = int(np.ceil(p.bar_min_confident_share * bar_win))
    bar_raw = rolling_median_masked(np.where(gripping, wrist_conf_h, np.nan), bar_win, min_count=min_count)
    bar_raw = bar_raw + p.bar_offset_body_scales * body_scale
    bar_carried = np.isfinite(bar_raw) & ~gripping

    # Only frames between the first and last confident grip count as hanging on the bar.
    in_hang_span = np.zeros(n_frames, dtype=bool)
    grip_idx = np.flatnonzero(gripping)
    if grip_idx.size:
        in_hang_span[grip_idx[0] : grip_idx[-1] + 1] = True

    head_s, bar_s = smooth(head_h), smooth(bar_raw)
    with np.errstate(invalid="ignore", divide="ignore"):
        head_minus_bar = (head_s - bar_s) / body_scale

    # Elbow angles from both sources, per arm then combined.
    img_l = elbow_angle_deg(l_sh, l_el, l_wr)
    img_r = elbow_angle_deg(r_sh, r_el, r_wr)
    elbow_image = smooth(_combine_arms(img_l, img_r, p.elbow_arm_combine))
    if pose.world is not None:
        w_l = elbow_angle_deg(xyz_world(config.LEFT_SHOULDER), xyz_world(config.LEFT_ELBOW), xyz_world(config.LEFT_WRIST))
        w_r = elbow_angle_deg(xyz_world(config.RIGHT_SHOULDER), xyz_world(config.RIGHT_ELBOW), xyz_world(config.RIGHT_WRIST))
        elbow_world = smooth(_combine_arms(w_l, w_r, p.elbow_arm_combine))
    else:
        w_l = w_r = np.full(n_frames, np.nan)
        elbow_world = np.full(n_frames, np.nan)
    use_world = p.elbow_angle_source == "world"
    elbow = elbow_world if use_world else elbow_image
    sel_l, sel_r = (w_l, w_r) if use_world else (img_l, img_r)
    arms_used = (np.isfinite(sel_l) * ARM_LEFT + np.isfinite(sel_r) * ARM_RIGHT).astype(np.uint8)

    head_valid = np.isfinite(head_minus_bar) & in_hang_span
    elbow_valid = np.isfinite(elbow) & in_hang_span
    return Signals(
        fps=fps,
        t_s=np.arange(n_frames) / fps,
        head_height_px=head_s,
        wrist_height_px=wrist_conf_h,
        bar_height_px=bar_s,
        bar_support=bar_support,
        bar_carried=bar_carried,
        body_scale_px=body_scale,
        head_minus_bar=np.where(head_valid, head_minus_bar, np.nan),
        elbow_deg=np.where(elbow_valid, elbow, np.nan),
        elbow_image_deg=elbow_image,
        elbow_world_deg=elbow_world,
        elbow_arms_used=arms_used,
        gripping=gripping,
        in_hang_span=in_hang_span,
        head_valid=head_valid,
        elbow_valid=elbow_valid,
    )

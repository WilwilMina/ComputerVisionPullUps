"""Synthetic skeletons for tests. No video, model or network.

Geometry (all lengths in "pixels at scale 1", multiplied by `scale` for the image):
wrists on the bar at height `bar_up`; shoulders directly under them; the elbow sits
at the apex of an isosceles triangle so the elbow angle is exactly `elbow_deg`.
The elbow bends sideways (`bend_axis="x"`, visible to a front camera) or toward the
camera (`bend_axis="z"`, which foreshortens in 2D but not in the world coords).
"""
from __future__ import annotations

import numpy as np

from pullup_rom import config
from pullup_rom.pose import PoseResult, empty_frame

SHOULDER_WIDTH = 100.0
ARM = 110.0  # upper arm = forearm length
TORSO = 150.0
WORLD_UNITS_PER_PX = 0.005  # arbitrary metric scale; world angles do not depend on it


def make_pose(
    head_up,
    bar_up,
    elbow_deg,
    fps: float = 30.0,
    frame_size=(1280, 720),
    scale: float = 1.0,
    bend_axis: str = "x",
    grip=None,
    noise_px: float = 0.0,
    seed: int = 0,
    with_world: bool = True,
) -> PoseResult:
    """head_up, bar_up (px at scale 1, height up) and elbow_deg are scalars or [F] arrays."""
    head_up, bar_up, elbow_deg = (np.atleast_1d(np.asarray(a, dtype=np.float64)) for a in (head_up, bar_up, elbow_deg))
    n = max(len(head_up), len(bar_up), len(elbow_deg))
    head_up, bar_up, elbow_deg = (np.broadcast_to(a, (n,)) for a in (head_up, bar_up, elbow_deg))
    grip = np.ones(n, dtype=bool) if grip is None else np.asarray(grip, dtype=bool)

    W, L, T = SHOULDER_WIDTH, ARM, TORSO
    half = np.radians(elbow_deg) / 2
    d = 2 * L * np.sin(half)  # shoulder-to-wrist distance
    h = L * np.cos(half)  # elbow offset from that line
    shoulder_up = bar_up - d
    wrist_up = np.where(grip, bar_up, shoulder_up - 0.6 * T)  # hands down when not gripping
    elbow_up = (shoulder_up + wrist_up) / 2

    cx = frame_size[0] / 2 / scale  # centre in scale-1 units
    kp3 = np.full((n, config.NUM_LANDMARKS, 3), np.nan)  # x, up, z in scale-1 units
    for side, sgn in (("LEFT", -1.0), ("RIGHT", 1.0)):
        x = cx + sgn * W / 2
        ex = x + (sgn * h if bend_axis == "x" else np.zeros(n))
        ez = h if bend_axis == "z" else np.zeros(n)
        kp3[:, getattr(config, f"{side}_SHOULDER")] = np.stack([np.full(n, x), shoulder_up, np.zeros(n)], axis=1)
        kp3[:, getattr(config, f"{side}_ELBOW")] = np.stack([ex, elbow_up, ez], axis=1)
        kp3[:, getattr(config, f"{side}_WRIST")] = np.stack([np.full(n, x), wrist_up, np.zeros(n)], axis=1)
        kp3[:, getattr(config, f"{side}_HIP")] = np.stack([np.full(n, x), shoulder_up - T, np.zeros(n)], axis=1)
        kp3[:, getattr(config, f"{side}_EAR")] = np.stack([np.full(n, cx + sgn * 8.0), head_up, np.zeros(n)], axis=1)
    kp3[:, config.NOSE] = np.stack([np.full(n, cx), head_up, np.zeros(n)], axis=1)

    present = np.isfinite(kp3[:, :, 0])
    img = np.zeros((n, config.NUM_LANDMARKS, 3), dtype=np.float32)
    img[:, :, 0] = kp3[:, :, 0] * scale
    img[:, :, 1] = frame_size[1] - kp3[:, :, 1] * scale  # image y grows downward
    img[:, :, 2] = present.astype(np.float32)
    if noise_px:
        rng = np.random.default_rng(seed)
        img[:, :, :2] += rng.normal(0.0, noise_px, img[:, :, :2].shape).astype(np.float32)

    world = None
    if with_world:
        world = np.stack([kp3[:, :, 0], -kp3[:, :, 1], kp3[:, :, 2]], axis=2).astype(np.float32) * WORLD_UNITS_PER_PX

    # Joints we did not place behave like undetected ones: NaN position, zero confidence.
    absent = ~present
    img[absent, :2] = np.nan
    return PoseResult(
        keypoints=img,
        fps=fps,
        frame_size=frame_size,
        timestamps_ms=np.arange(n) * 1000.0 / fps,
        world=world,
    )


def drop_frames(pose: PoseResult, frames, low_conf: float | None = None) -> PoseResult:
    """Mark frames as no-person (NaN, conf 0) or, with `low_conf`, keep positions at that confidence."""
    frames = np.asarray(list(frames))
    if low_conf is None:
        pose.keypoints[frames] = empty_frame()
        if pose.world is not None:
            pose.world[frames] = np.nan
    else:
        pose.keypoints[frames, :, 2] = low_conf
    return pose


def make_rep_series(
    fps: float = 30.0,
    peaks=(0.4,),
    valley_elbows=(165.0, 165.0),
    bottom: float = -1.0,
    top_elbow_deg: float = 50.0,
    lead_s: float = 1.0,
    up_s: float = 1.0,
    hold_s: float = 0.3,
    down_s: float = 1.2,
    pause_s: float = 1.0,
    tail_s: float = 1.0,
    noise_head: float = 0.0,
    noise_elbow: float = 0.0,
    seed: int = 0,
):
    """Head-minus-bar (body scales) and elbow angle (deg) curves for rep logic. No pose, no video.

    `peaks[k]` is the head height at the top of rep k. `valley_elbows[k]` is the elbow angle on the hang plateau
    at valley k (valley 0 is before rep 0, valley k+1 follows rep k), so len(valley_elbows) == len(peaks) + 1 and
    a bent-elbow valley affects both reps that touch it, as it would in real footage.
    """
    peaks = list(peaks)
    assert len(valley_elbows) == len(peaks) + 1
    head, elbow = [], []

    def frames(seconds):
        return max(1, int(round(seconds * fps)))

    def hang(seconds, angle):
        k = frames(seconds)
        head.extend([bottom] * k)
        elbow.extend([angle] * k)

    def ramp(seconds, h0, h1, a0, a1):
        k = frames(seconds)
        t = (np.arange(1, k + 1) / k)
        s = 0.5 * (1 - np.cos(np.pi * t))  # smooth start and stop
        head.extend(list(h0 + (h1 - h0) * s))
        elbow.extend(list(a0 + (a1 - a0) * s))

    hang(lead_s, valley_elbows[0])
    for k, peak in enumerate(peaks):
        ramp(up_s, bottom, peak, valley_elbows[k], top_elbow_deg)
        k_hold = frames(hold_s)
        head.extend([peak] * k_hold)
        elbow.extend([top_elbow_deg] * k_hold)
        ramp(down_s, peak, bottom, top_elbow_deg, valley_elbows[k + 1])
        hang(pause_s if k + 1 < len(peaks) else tail_s, valley_elbows[k + 1])

    head, elbow = np.asarray(head), np.asarray(elbow)
    rng = np.random.default_rng(seed)
    if noise_head:
        head = head + rng.normal(0.0, noise_head, head.shape)
    if noise_elbow:
        elbow = elbow + rng.normal(0.0, noise_elbow, elbow.shape)
    return head, elbow

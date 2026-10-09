"""Unsupported-style flag on synthetic poses (hips swung toward the camera at the rep bottoms) and, when the dev clip
cache exists locally, on the real kipping clip. The real-clip test is skipped on a machine without data/."""
from pathlib import Path

import numpy as np
import pytest
from synthetic import SHOULDER_WIDTH, TORSO, WORLD_UNITS_PER_PX, make_pose, make_rep_series

from pullup_rom import config
from pullup_rom.metrics import compute_signals
from pullup_rom.reps import RepParams, detect_reps
from pullup_rom.style import StyleChecks, style_checks, style_warnings, torso_tilt_deg

FPS = 30.0


def _scene(n_reps=3, tilt_at_bottom_deg=0.0, tilt_at_top_deg=0.0):
    head, elbow = make_rep_series(peaks=[0.4] * n_reps, valley_elbows=[165] * (n_reps + 1), lead_s=0.5, up_s=0.8, hold_s=0.2, down_s=0.8, pause_s=0.5, tail_s=0.5)
    pose = make_pose(500.0 + head * SHOULDER_WIDTH, 500.0, elbow, fps=FPS, frame_size=(320, 360), scale=0.4)
    s = compute_signals(pose)
    analysis = detect_reps(s.head_minus_bar, s.elbow_deg, FPS, RepParams(dead_hang_elbow_min_deg=150.0))
    # Swing the hips out along the camera axis (world z): tilt = angle of the torso from vertical.
    near_bottom = np.zeros(pose.num_frames, dtype=bool)
    near_top = np.zeros(pose.num_frames, dtype=bool)
    for r in analysis.reps:
        for f in (r.start_frame, r.end_frame):
            near_bottom[max(0, f - 6) : f + 7] = True
        near_top[r.peak_frame - 6 : r.peak_frame + 7] = True
    tilt = np.where(near_bottom, tilt_at_bottom_deg, np.where(near_top, tilt_at_top_deg, 0.0))
    dz = TORSO * np.tan(np.radians(tilt)) * WORLD_UNITS_PER_PX
    for j in (config.LEFT_HIP, config.RIGHT_HIP):
        pose.world[:, j, 2] += dz.astype(np.float32)
    return pose, analysis


def test_a_vertical_torso_has_no_tilt_and_no_flag():
    pose, analysis = _scene()
    assert np.nanmax(torso_tilt_deg(pose)) < 1.0
    checks = style_checks(pose, analysis.reps)
    assert checks.reps_used == 3 and checks.bottom_tilt_deg < 1.0 and style_warnings(checks) == []


def test_a_large_swing_at_the_bottoms_raises_the_unsupported_style_flag():
    pose, analysis = _scene(tilt_at_bottom_deg=config.UNSUPPORTED_SWING_BOTTOM_TILT_DEG + 10)
    checks = style_checks(pose, analysis.reps)
    assert abs(checks.bottom_tilt_deg - (config.UNSUPPORTED_SWING_BOTTOM_TILT_DEG + 10)) < 2.0
    assert style_warnings(checks) and style_warnings(checks)[0].startswith("UNSUPPORTED STYLE")


def test_a_lean_only_at_the_top_is_a_normal_pull_not_a_swing():
    pose, analysis = _scene(tilt_at_top_deg=40.0)  # chest leaning back at the top is normal
    assert style_warnings(style_checks(pose, analysis.reps)) == []


def test_a_small_swing_below_the_cutoff_is_not_flagged():
    pose, analysis = _scene(tilt_at_bottom_deg=config.UNSUPPORTED_SWING_BOTTOM_TILT_DEG - 8)
    assert style_warnings(style_checks(pose, analysis.reps)) == []


def test_too_few_reps_or_no_hip_data_never_flags():
    pose, analysis = _scene(n_reps=1, tilt_at_bottom_deg=40.0)
    assert style_warnings(style_checks(pose, analysis.reps)) == []  # one rep is not enough to call a style
    pose, analysis = _scene(tilt_at_bottom_deg=40.0)
    pose.keypoints[:, config.LEFT_HIP, 2] = 0.0  # hips not confident: nothing measurable, so no flag and no guess
    assert style_checks(pose, analysis.reps) == StyleChecks(None, 0)
    pose.world = None
    assert np.isnan(torso_tilt_deg(pose)).all()


@pytest.mark.slow
def test_real_clips_if_present_the_kipping_clip_is_flagged_and_labelled_strict_clips_are_not():
    from pullup_rom.pose import load_pose
    from pullup_rom.rep_report import analyze

    cache = Path("data/cache")
    kip = cache / "pexels_14284164_pexels_14284164.npz"
    if not kip.exists():
        pytest.skip("dev clip caches are not on this machine")
    pose = load_pose(kip)
    _, analysis = analyze(pose, "world")
    assert style_warnings(style_checks(pose, analysis.reps)), "the kipping clip should be flagged"
    for name in ("pexels_10336041_pexels_10336041", "pexels_6389831", "pexels_15859716_pexels_15859716", "pexels_7187450_pexels_7187450"):
        p = cache / f"{name}.npz"
        if p.exists():
            pose = load_pose(p)
            _, analysis = analyze(pose, "world")
            assert style_warnings(style_checks(pose, analysis.reps)) == [], f"{name} is labelled as not kipping"

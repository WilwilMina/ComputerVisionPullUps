import numpy as np
import pytest
from synthetic import SHOULDER_WIDTH, drop_frames, make_pose

from pullup_rom import config
from pullup_rom.metrics import (
    ARM_LEFT,
    ARM_RIGHT,
    MetricsParams,
    compute_signals,
    elbow_angle_deg,
    interpolate_short_gaps,
    rolling_median_masked,
    smooth_valid_runs,
)

FPS = 30.0
N = 300  # 10 s
T = np.arange(N) / FPS
W = SHOULDER_WIDTH
BAR = 500.0


# ---------- helpers ----------

def test_interpolate_fills_short_gaps_only():
    x = np.arange(20, dtype=float)
    x[3:5] = np.nan  # 2 frames
    x[10:16] = np.nan  # 6 frames
    out = interpolate_short_gaps(x, max_gap_frames=3)
    np.testing.assert_allclose(out[3:5], [3, 4])
    assert np.isnan(out[10:16]).all()


def test_interpolate_never_extrapolates_at_the_ends():
    x = np.array([np.nan, np.nan, 1.0, 2.0, np.nan])
    assert np.isnan(interpolate_short_gaps(x, 10)[[0, 1, 4]]).all()


def test_smoothing_keeps_nan_and_does_not_bleed_across_gaps():
    x = np.sin(np.linspace(0, 6, 60))
    x[30:35] = np.nan
    out = smooth_valid_runs(x, window_frames=9, polyorder=2)
    assert np.isnan(out[30:35]).all()
    assert np.isfinite(out[:30]).all() and np.isfinite(out[35:]).all()


def test_rolling_median_ignores_nan_and_is_robust_to_outliers():
    x = np.full(50, 10.0)
    x[20] = 1000.0  # outlier
    x[30:40] = np.nan
    out = rolling_median_masked(x, window_frames=9)
    assert out[20] == 10.0
    assert out[35] == 10.0 or np.isnan(out[35])
    assert np.isnan(rolling_median_masked(np.full(20, np.nan), 5)).all()


@pytest.mark.parametrize("deg", [30.0, 90.0, 135.0, 180.0])
def test_elbow_angle_known_2d_and_3d(deg):
    a = np.radians(deg)
    sh = np.array([[1.0, 0.0]])
    el = np.array([[0.0, 0.0]])
    wr = np.array([[np.cos(a), np.sin(a)]])
    np.testing.assert_allclose(elbow_angle_deg(sh, el, wr), [deg], atol=1e-6)
    pad = lambda v: np.concatenate([v, np.zeros((1, 1))], axis=1)
    np.testing.assert_allclose(elbow_angle_deg(pad(sh), pad(el), pad(wr)), [deg], atol=1e-6)


def test_elbow_angle_degenerate_is_nan_not_zero():
    z = np.zeros((1, 2))
    assert np.isnan(elbow_angle_deg(z, z, z)).all()


# ---------- height-up flip and unitless scale ----------

def test_head_above_bar_is_positive_and_unitless():
    pose = make_pose(head_up=np.full(N, BAR + 50.0), bar_up=BAR, elbow_deg=170.0)
    s = compute_signals(pose, MetricsParams(elbow_angle_source="image"))
    # 50 px above the bar / 100 px shoulder width = 0.5 body scales, single flip of image y
    np.testing.assert_allclose(s.head_minus_bar[s.head_valid], 0.5, atol=1e-3)
    below = compute_signals(make_pose(head_up=np.full(N, BAR - 50.0), bar_up=BAR, elbow_deg=170.0), MetricsParams(elbow_angle_source="image"))
    np.testing.assert_allclose(below.head_minus_bar[below.head_valid], -0.5, atol=1e-3)


def test_signal_is_the_same_at_double_image_scale():
    head = BAR + 80.0 * np.sin(2 * np.pi * T / 3.0)
    base = compute_signals(make_pose(head, BAR, 170.0, scale=1.0))
    big = compute_signals(make_pose(head, BAR, 170.0, scale=2.0))
    np.testing.assert_allclose(base.head_minus_bar, big.head_minus_bar, atol=1e-3, equal_nan=True)
    assert abs(big.body_scale_px[N // 2] / base.body_scale_px[N // 2] - 2.0) < 1e-3


def test_body_scale_values_for_both_modes():
    for mode, px in (("shoulder_width", 100.0), ("torso", 150.0)):
        pose = make_pose(np.full(N, BAR + 50.0), BAR, 170.0)
        s = compute_signals(pose, MetricsParams(body_scale_mode=mode))
        np.testing.assert_allclose(s.body_scale_px, px, rtol=1e-3)


# ---------- bar line: static camera and drifting camera ----------

def _bar_error_in_body_scales(bar_up, params=None, noise_px=0.0, interior_s=1.0):
    head = np.full(N, 560.0)  # head motion must not leak into the bar estimate
    pose = make_pose(head, bar_up, 170.0, noise_px=noise_px)
    s = compute_signals(pose, params)
    err = (s.bar_height_px - bar_up) / W
    sl = slice(int(interior_s * FPS), N - int(interior_s * FPS))
    return err, sl


def test_bar_static_camera_default_window():
    err, sl = _bar_error_in_body_scales(np.full(N, BAR), noise_px=1.5)
    assert np.abs(err[sl]).max() < 0.05


def test_bar_linear_drift_default_window_interior():
    drift = BAR + 20.0 * T  # 20 px/s = 0.2 body scales per second
    err, sl = _bar_error_in_body_scales(drift, noise_px=1.5)
    assert np.abs(err[sl]).max() < 0.05


def test_bar_sinusoidal_drift_default_window():
    sway = BAR + 30.0 * np.sin(2 * np.pi * T / 8.0)
    err, sl = _bar_error_in_body_scales(sway, noise_px=1.5)
    assert np.abs(err[sl]).max() < 0.05


def test_bar_edges_under_drift_are_biased_by_at_most_slope_times_quarter_window():
    slope = 20.0
    err, _ = _bar_error_in_body_scales(BAR + slope * T)
    bound = slope * (config.BAR_WINDOW_S / 4) / W + 0.02
    assert np.abs(err).max() < bound


def test_bar_window_param_is_wired_longer_window_tracks_drift_worse():
    sway = BAR + 30.0 * np.sin(2 * np.pi * T / 8.0)
    short, sl = _bar_error_in_body_scales(sway, MetricsParams(bar_window_s=1.0))
    long_, _ = _bar_error_in_body_scales(sway, MetricsParams(bar_window_s=6.0))
    assert np.abs(long_[sl]).max() > 3 * np.abs(short[sl]).max()


def test_bar_ignores_frames_where_hands_are_down_and_marks_them_invalid():
    grip = np.ones(N, dtype=bool)
    grip[:60] = False  # first 2 s hands down
    pose = make_pose(np.full(N, 560.0), BAR, 170.0, grip=grip)
    s = compute_signals(pose)
    assert not s.head_valid[:50].any()
    assert s.head_valid[90:].all()
    assert np.abs(s.bar_height_px[90:] - BAR).max() / W < 0.02
    assert np.isnan(s.head_minus_bar[:50]).all()  # NaN, not zeros


# ---------- missing data ----------

def test_short_gap_interpolated_long_gap_invalid_never_zero():
    head = BAR + 60.0 * np.sin(2 * np.pi * T / 3.0)
    pose = make_pose(head, BAR, 170.0)
    drop_frames(pose, range(100, 105))  # 5 frames = 0.167 s, under the default max gap
    drop_frames(pose, range(150, 190))  # 40 frames, over it
    s = compute_signals(pose)
    assert s.head_valid[100:105].all()
    assert not s.head_valid[155:185].any()
    assert np.isnan(s.head_minus_bar[155:185]).all() and np.isnan(s.elbow_deg[155:185]).all()
    expected = (head - BAR) / W
    np.testing.assert_allclose(s.head_minus_bar[100:105], expected[100:105], atol=0.05)


def test_low_confidence_frames_are_treated_as_missing():
    pose = make_pose(np.full(N, BAR + 50.0), BAR, 170.0)
    drop_frames(pose, range(100, 140), low_conf=0.1)
    s = compute_signals(pose)
    assert not s.head_valid[105:135].any()
    assert s.head_valid[:90].all() and s.head_valid[150:].all()


def test_all_frames_missing_gives_all_invalid_without_crashing():
    pose = make_pose(np.full(N, BAR + 50.0), BAR, 170.0)
    drop_frames(pose, range(N))
    s = compute_signals(pose)
    assert not s.head_valid.any()
    assert np.isnan(s.head_minus_bar).all()


# ---------- elbow angle source ----------

@pytest.mark.parametrize("deg", [90.0, 120.0, 170.0])
def test_elbow_angle_image_and_world_agree_for_a_front_view_bend(deg):
    s = compute_signals(make_pose(np.full(N, BAR + 40.0), BAR, deg))
    np.testing.assert_allclose(s.elbow_image_deg[10:-10], deg, atol=0.5)
    np.testing.assert_allclose(s.elbow_world_deg[10:-10], deg, atol=0.5)


def test_world_angle_is_right_where_the_2d_angle_is_foreshortened():
    pose = make_pose(np.full(N, BAR + 40.0), BAR, 100.0, bend_axis="z")
    s = compute_signals(pose)
    np.testing.assert_allclose(s.elbow_world_deg[10:-10], 100.0, atol=0.5)
    assert s.elbow_image_deg[10:-10].min() > 170.0  # bend toward the camera looks straight in 2D


def test_elbow_source_selection_and_missing_world():
    pose = make_pose(np.full(N, BAR + 40.0), BAR, 100.0, bend_axis="z")
    world = compute_signals(pose, MetricsParams(elbow_angle_source="world"))
    image = compute_signals(pose, MetricsParams(elbow_angle_source="image"))
    assert abs(np.nanmedian(world.elbow_deg) - 100.0) < 1.0
    assert np.nanmedian(image.elbow_deg) > 170.0
    no_world = make_pose(np.full(N, BAR + 40.0), BAR, 100.0, with_world=False)
    with pytest.raises(ValueError, match="world"):
        compute_signals(no_world, MetricsParams(elbow_angle_source="world"))
    s = compute_signals(no_world, MetricsParams(elbow_angle_source="image"))
    assert np.isnan(s.elbow_world_deg).all() and s.head_valid.any()


def test_arm_combine_min_picks_the_more_bent_arm():
    pose = make_pose(np.full(N, BAR + 40.0), BAR, 170.0)
    # Bend only the left elbow in 2D by pulling it in toward the shoulder-wrist line.
    left_elbow = pose.keypoints[:, config.LEFT_ELBOW]
    left_elbow[:, 0] += 60.0
    mean = compute_signals(pose, MetricsParams(elbow_angle_source="image", elbow_arm_combine="mean"))
    low = compute_signals(pose, MetricsParams(elbow_angle_source="image", elbow_arm_combine="min"))
    assert np.nanmedian(low.elbow_deg) < np.nanmedian(mean.elbow_deg) < 170.0


# ---------- smoothing and contract ----------

def test_smoothing_reduces_jitter_and_keeps_a_clean_peak():
    head = BAR + 80.0 * np.sin(2 * np.pi * T / 3.0)
    clean_s = compute_signals(make_pose(head, BAR, 170.0))
    assert abs(np.nanmax(clean_s.head_minus_bar) - 0.8) < 0.02  # peak kept within 2% of body scale
    noisy = make_pose(head, BAR, 170.0, noise_px=3.0, seed=1)
    s = compute_signals(noisy)
    truth = (head - BAR) / W
    raw_noise = np.std(noisy.keypoints[:, config.NOSE, 1] - (720 - head)) / W
    resid = np.nanstd((s.head_minus_bar - truth)[s.head_valid])
    assert resid < raw_noise


def test_head_landmark_param_is_respected():
    pose = make_pose(np.full(N, BAR + 50.0), BAR, 170.0)
    ears = compute_signals(pose, MetricsParams(head_landmarks=(config.LEFT_EAR, config.RIGHT_EAR)))
    np.testing.assert_allclose(ears.head_minus_bar[ears.head_valid], 0.5, atol=1e-3)
    pose.keypoints[:, config.LEFT_EAR, 1] -= 30.0  # left ear 30 px higher in the image
    shifted = compute_signals(pose, MetricsParams(head_landmarks=(config.LEFT_EAR, config.RIGHT_EAR)))
    np.testing.assert_allclose(shifted.head_minus_bar[shifted.head_valid], 0.5 + 0.15, atol=1e-3)


def test_output_contract_shapes_and_validity():
    s = compute_signals(make_pose(np.full(N, BAR + 50.0), BAR, 170.0))
    for name in ("t_s", "head_height_px", "bar_height_px", "body_scale_px", "head_minus_bar", "elbow_deg",
                 "elbow_image_deg", "elbow_world_deg", "gripping", "head_valid", "elbow_valid", "bar_carried",
                 "bar_support", "in_hang_span"):
        assert getattr(s, name).shape == (N,), name
    assert s.head_valid.dtype == bool and s.gripping.dtype == bool and s.bar_carried.dtype == bool
    assert np.isnan(s.head_minus_bar[~s.head_valid]).all() and np.isnan(s.elbow_deg[~s.elbow_valid]).all()


def test_bad_params_raise():
    pose = make_pose(np.full(N, BAR + 50.0), BAR, 170.0)
    for bad in (MetricsParams(elbow_angle_source="x"), MetricsParams(body_scale_mode="x")):
        with pytest.raises(ValueError):
            compute_signals(pose, bad)
    with pytest.raises(ValueError):
        compute_signals(pose, MetricsParams(elbow_arm_combine="x"))


def test_empty_head_landmarks_is_a_clear_error():
    with pytest.raises(ValueError, match="head_landmarks"):
        compute_signals(make_pose(np.full(N, BAR + 50.0), BAR, 170.0), MetricsParams(head_landmarks=()))


def test_coincident_shoulders_give_nan_scale_not_zero():
    pose = make_pose(np.full(N, BAR + 50.0), BAR, 170.0)
    pose.keypoints[:, config.RIGHT_SHOULDER, :2] = pose.keypoints[:, config.LEFT_SHOULDER, :2]
    s = compute_signals(pose)
    assert np.isnan(s.body_scale_px).all()
    assert not s.head_valid.any() and np.isnan(s.head_minus_bar).all()


@pytest.mark.parametrize("frames", [1, 5])
def test_very_short_clips_do_not_crash(frames):
    s = compute_signals(make_pose(np.full(frames, BAR + 50.0), BAR, 170.0))
    assert s.head_valid.shape == (frames,)
    assert np.isfinite(s.head_minus_bar[s.head_valid]).all()


def test_elbow_arms_used_makes_the_one_arm_fallback_visible():
    pose = make_pose(np.full(N, BAR + 50.0), BAR, 170.0)
    both = compute_signals(pose)
    assert (both.elbow_arms_used[both.elbow_valid] == ARM_LEFT | ARM_RIGHT).all()

    for src in ("world", "image"):
        p = MetricsParams(elbow_angle_source=src)
        left_only = make_pose(np.full(N, BAR + 50.0), BAR, 170.0)
        drop_joints = (config.RIGHT_ELBOW, config.RIGHT_WRIST)
        for j in drop_joints:
            left_only.keypoints[100:200, j, 2] = 0.0  # low confidence
        s = compute_signals(left_only, p)
        assert (s.elbow_arms_used[120:180] == ARM_LEFT).all(), src
        assert s.elbow_valid[120:180].all()  # still valid, but no longer silent about it
        assert (s.elbow_arms_used[:90] == ARM_LEFT | ARM_RIGHT).all(), src

    neither = make_pose(np.full(N, BAR + 50.0), BAR, 170.0)
    for j in (config.LEFT_WRIST, config.RIGHT_WRIST):
        neither.keypoints[50:80, j, 2] = 0.0
    s = compute_signals(neither)
    assert (s.elbow_arms_used[60:70] == 0).all()


# ---------- bar estimate guardrails and separate validity ----------

def _lose_wrists(pose, frames, conf=0.0, y_shift_px=0.0):
    """Wrists unconfident (and optionally wildly misplaced) in `frames`, everything else untouched."""
    for j in (config.LEFT_WRIST, config.RIGHT_WRIST):
        pose.keypoints[frames, j, 2] = conf
        pose.keypoints[frames, j, 1] += y_shift_px
    return pose


def test_head_stays_valid_without_wrists_but_the_bar_is_flagged_as_carried():
    head = BAR + 60.0 * np.sin(2 * np.pi * T / 3.0)
    pose = _lose_wrists(make_pose(head, BAR, 170.0), slice(100, 140))  # 1.3 s without wrists, head still seen
    s = compute_signals(pose)
    assert s.head_valid[100:140].all()
    assert s.bar_carried[100:140].all() and not s.bar_carried[:90].any() and not s.bar_carried[150:].any()
    assert (np.abs(s.bar_height_px[100:140] - BAR) / W < 0.05).all()
    np.testing.assert_allclose(s.head_minus_bar[100:140], ((head - BAR) / W)[100:140], atol=0.06)


def test_elbow_keeps_its_own_validity_when_the_wrists_are_lost():
    pose = _lose_wrists(make_pose(np.full(N, BAR + 50.0), BAR, 170.0), slice(100, 140))
    s = compute_signals(pose)
    assert s.head_valid[100:140].all()
    assert not s.elbow_valid[105:135].any() and np.isnan(s.elbow_deg[105:135]).all()
    assert s.elbow_valid[:90].all() and s.elbow_valid[150:].all()


def test_the_bar_never_uses_positions_the_model_was_not_confident_about():
    head = np.full(N, BAR + 50.0)
    honest = compute_signals(_lose_wrists(make_pose(head, BAR, 170.0), slice(100, 140)))
    # Same frames, but the unconfident wrists are wildly misplaced: the bar must not move at all.
    wrong = compute_signals(_lose_wrists(make_pose(head, BAR, 170.0), slice(100, 140), y_shift_px=300.0))
    np.testing.assert_allclose(wrong.bar_height_px, honest.bar_height_px, atol=1e-6)
    # And a short gap is not bridged by interpolating the wrists for the bar.
    short = _lose_wrists(make_pose(head, BAR + 0 * T, 170.0), slice(100, 103), y_shift_px=-300.0)
    assert np.abs(compute_signals(short).bar_height_px - BAR).max() / W < 0.02


def test_bar_needs_a_minimum_share_of_confident_frames_in_its_window():
    pose = _lose_wrists(make_pose(np.full(N, BAR + 50.0), BAR, 170.0), slice(60, 240))  # 6 s: far longer than the window
    s = compute_signals(pose)
    mid = slice(120, 180)
    assert np.isnan(s.bar_height_px[mid]).all() and not s.head_valid[mid].any()
    assert not s.bar_carried[mid].any()  # no bar value at all, so nothing is "carried"
    assert s.head_valid[:55].all() and s.head_valid[250:].all()
    strict = compute_signals(_lose_wrists(make_pose(np.full(N, BAR + 50.0), BAR, 170.0), slice(100, 140)),
                             MetricsParams(bar_min_confident_share=0.95))
    assert not strict.head_valid[120:140].any()  # the share threshold is wired through


def test_only_frames_between_the_first_and_last_grip_count_as_hanging():
    grip = np.ones(N, dtype=bool)
    grip[:60] = False
    grip[240:] = False
    s = compute_signals(make_pose(np.full(N, 560.0), BAR, 170.0, grip=grip))
    assert not s.head_valid[:55].any() and not s.head_valid[245:].any()
    assert s.head_valid[70:230].all() and not s.in_hang_span[:55].any()

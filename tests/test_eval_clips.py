import dataclasses

import numpy as np
from synthetic import SHOULDER_WIDTH, make_pose

from pullup_rom import config
from pullup_rom.eval_clips import bar_stability, elbow_range, head_spread, report_line
from pullup_rom.metrics import MetricsParams
from pullup_rom.metrics import compute_signals as _compute_signals

N = 300
T = np.arange(N) / 30.0
W = SHOULDER_WIDTH


def compute_signals(pose, params=None):
    """These tests probe the bar-line mechanics with the synthetic bar placed AT the wrists, so the measured
    wrist-to-bar offset is switched off here. The offset itself is tested in test_bar_offset.py."""
    return _compute_signals(pose, dataclasses.replace(params or MetricsParams(), bar_offset_body_scales=0.0))


def _signals(bar_up, head_up=None, **kw):
    head_up = np.full(N, 560.0) if head_up is None else head_up
    pose = make_pose(head_up, bar_up, 170.0, **kw)
    return pose, compute_signals(pose)


def test_bar_stability_static_camera_has_no_drift_or_scatter():
    _, s = _signals(np.full(N, 500.0))
    drift, scatter = bar_stability(s)
    assert drift < 0.02 and scatter < 0.02


def test_bar_stability_reports_drift_in_body_scales():
    _, s = _signals(500.0 + 20.0 * T)  # 20 px/s; the first second is skipped, so 9 s = 180 px = 1.8 body scales
    drift, _ = bar_stability(s)
    assert abs(drift - 1.8) < 0.3  # edge bias of the rolling median trims the ends a little


def test_bar_stability_scatter_grows_when_wrists_wander_off_the_bar():
    _, steady = _signals(np.full(N, 500.0))
    pose = make_pose(np.full(N, 560.0), 500.0, 170.0)
    pose.keypoints[:, config.LEFT_WRIST, 1] += 25.0 * np.sin(2 * np.pi * T / 1.5)  # one wrist bobbing
    wandering = compute_signals(pose)
    assert bar_stability(wandering)[1] > bar_stability(steady)[1] + 0.05


def test_bar_stability_none_when_nothing_is_valid():
    pose = make_pose(np.full(N, 560.0), 500.0, 170.0)
    pose.keypoints[:, :, 2] = 0.0
    assert bar_stability(compute_signals(pose)) == (None, None)


def test_elbow_range_uses_valid_frames_only():
    angle = np.array([50.0, 100.0, 150.0, 170.0])
    valid = np.array([False, True, True, False])
    assert elbow_range(angle, valid) == (100.0, 150.0)
    assert elbow_range(angle, np.zeros(4, dtype=bool)) is None


def test_bar_drift_ignores_the_first_second_while_reaching_up():
    reach = np.where(T < 0.8, 400.0 + 125.0 * T, 500.0)  # bar (wrists) climbs 100 px in the first 0.8 s, then flat
    _, s = _signals(reach)
    drift, _ = bar_stability(s)
    assert drift < 0.1  # without the skip this would be about 1 body scale


def test_head_spread_is_zero_when_landmarks_agree_and_grows_when_one_is_off():
    pose, s = _signals(np.full(N, 560.0))
    med, p95, share = head_spread(pose, s)
    assert med < 0.02 and p95 < 0.02 and share == 1.0
    pose.keypoints[:, config.LEFT_EAR, 1] -= 30.0  # one ear 30 px higher = 0.3 body scales
    med, _, _ = head_spread(pose, s)
    assert abs(med - 0.3) < 0.03


def test_head_spread_needs_at_least_two_confident_landmarks():
    pose, s = _signals(np.full(N, 560.0))
    for j in (config.LEFT_EAR, config.RIGHT_EAR):
        pose.keypoints[:, j, 2] = 0.0  # only the nose is left (eyes are absent in the synthetic skeleton)
    assert head_spread(pose, s) is None


def test_report_line_is_one_line_and_labels_the_columns():
    pose, s = _signals(np.full(N, 500.0))
    line = report_line("clip", pose, s)
    assert "\n" not in line
    for label in ("person", "invalid head", "bar carried", "head spread", "bar drift", "scatter", "elbow deg 3D", "one-arm"):
        assert label in line

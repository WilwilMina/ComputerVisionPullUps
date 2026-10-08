import numpy as np
from synthetic import make_pose

from pullup_rom import config
from pullup_rom.camera import CameraChecks, camera_checks, camera_warnings

N = 60


def front_view():
    return make_pose(np.full(N, 560.0), 500.0, 165.0)


def test_a_front_or_back_view_with_a_level_bar_raises_no_warning():
    checks = camera_checks(front_view())
    assert 0.6 < checks.shoulder_torso_ratio < 0.7  # shoulder width 100 / torso 150 in the synthetic body
    assert checks.wrist_gap_body_scales < 0.01
    assert camera_warnings(checks) == []


def test_a_side_view_is_flagged_when_the_shoulders_nearly_overlap():
    pose = front_view()
    pose.keypoints[:, config.RIGHT_SHOULDER, 0] = pose.keypoints[:, config.LEFT_SHOULDER, 0] + 20.0  # 20 px apart, not 100
    warnings = camera_warnings(camera_checks(pose))
    assert any("looks like a side view" in w for w in warnings)


def test_a_tilted_bar_is_flagged_when_the_wrists_differ_in_height():
    pose = front_view()
    pose.keypoints[:, config.LEFT_WRIST, 1] -= 80.0  # one wrist 80 px higher in the image (0.8 shoulder widths)
    warnings = camera_warnings(camera_checks(pose))
    assert any("different heights" in w for w in warnings)
    assert not any("looks like a side view" in w for w in warnings)  # the shoulders are still wide apart


def test_cutoffs_are_the_configured_ones():
    just_under = CameraChecks(config.WARN_SIDE_VIEW_RATIO - 0.01, 0.0)
    just_over = CameraChecks(config.WARN_SIDE_VIEW_RATIO + 0.01, 0.0)
    assert camera_warnings(just_under) and not camera_warnings(just_over)
    assert camera_warnings(CameraChecks(0.6, config.WARN_BAR_TILT_WRIST_GAP + 0.01))
    assert not camera_warnings(CameraChecks(0.6, config.WARN_BAR_TILT_WRIST_GAP - 0.01))


def test_nothing_confident_means_no_checks_and_no_warnings():
    pose = front_view()
    pose.keypoints[:, :, 2] = 0.0
    checks = camera_checks(pose)
    assert checks == CameraChecks(None, None) and camera_warnings(checks) == []


def test_wrists_below_the_shoulders_are_not_treated_as_hanging():
    pose = make_pose(np.full(N, 560.0), 500.0, 165.0, grip=np.zeros(N, dtype=bool))  # hands down
    pose.keypoints[:, config.LEFT_WRIST, 1] -= 80.0
    assert camera_checks(pose).wrist_gap_body_scales is None


def test_one_frame_with_overlapping_shoulders_does_not_switch_off_the_tilt_check():
    pose = front_view()
    pose.keypoints[:, config.LEFT_WRIST, 1] -= 80.0
    pose.keypoints[0, config.RIGHT_SHOULDER, :2] = pose.keypoints[0, config.LEFT_SHOULDER, :2]  # one degenerate frame
    assert camera_checks(pose).wrist_gap_body_scales > config.WARN_BAR_TILT_WRIST_GAP


def test_an_infinite_coordinate_is_treated_as_missing_without_a_warning():
    import warnings

    pose = front_view()
    pose.keypoints[5, config.LEFT_WRIST, 1] = np.inf
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        camera_checks(pose)

"""Several people in frame: follow the one hanging on the bar, count people, warn. Synthetic skeletons only."""
from types import SimpleNamespace

import numpy as np

from pullup_rom import config
from pullup_rom.camera import CameraChecks, camera_warnings
from pullup_rom.pose import PoseResult, count_people, follow_agreement, is_hanging, is_person, load_pose, save_pose, select_person


def skeleton(cx, cy=400.0, width=100.0, hanging=True, conf=1.0):
    """A [J,3] pixel skeleton: shoulders `width` apart centred on cx, wrists above (hanging) or below the shoulders."""
    kp = np.full((config.NUM_LANDMARKS, 3), np.nan, dtype=np.float32)
    kp[:, 2] = 0.0
    for side, sgn in (("LEFT", -1.0), ("RIGHT", 1.0)):
        x = cx + sgn * width / 2
        kp[getattr(config, f"{side}_SHOULDER")] = (x, cy, conf)
        kp[getattr(config, f"{side}_HIP")] = (x, cy + 1.5 * width, conf)
        kp[getattr(config, f"{side}_WRIST")] = (x, cy - width if hanging else cy + width, conf)
    return kp


def test_a_person_needs_confident_shoulders_and_hanging_needs_wrists_above_them():
    assert is_person(skeleton(100)) and not is_person(skeleton(100, conf=0.1))
    assert is_hanging(skeleton(100)) and not is_hanging(skeleton(100, hanging=False))
    lifter = skeleton(100)
    lifter[[config.LEFT_WRIST, config.RIGHT_WRIST], 2] = 0.2  # wrists the model is unsure about still count by position
    assert is_hanging(lifter)
    lifter[config.LEFT_WRIST, :2] = np.nan
    assert not is_hanging(lifter)  # but a missing wrist cannot be judged
    assert count_people([skeleton(100), skeleton(500), skeleton(900, conf=0.1)]) == 2  # a stray low-confidence pose is not a person


def test_the_person_on_the_bar_is_followed_not_the_bystander_even_if_the_bystander_is_larger():
    lifter, bystander = skeleton(300, width=80, hanging=True), skeleton(700, width=140, hanging=False)
    assert select_person([bystander, lifter], None) == 1
    assert select_person([lifter, bystander], None) == 0


def test_with_history_the_nearest_person_is_followed_among_those_hanging():
    left, right = skeleton(200), skeleton(800)
    assert select_person([left, right], previous=skeleton(780)) == 1
    assert select_person([left, right], previous=skeleton(230)) == 0


def test_with_nobody_hanging_the_largest_person_is_the_fallback_and_no_detection_gives_none():
    small, big = skeleton(200, width=60, hanging=False), skeleton(800, width=120, hanging=False)
    assert select_person([small, big], None) == 1
    assert select_person([], None) is None


def test_follow_agreement_says_whether_the_followed_person_is_the_one_on_the_bar():
    lifter, other = skeleton(300), skeleton(700, hanging=False)
    cands = [other, lifter]
    chosen = select_person(cands, None)
    assert chosen == 1
    assert follow_agreement(cands, chosen, followed=skeleton(310)) == 1  # the measurement pass followed the lifter
    assert follow_agreement(cands, chosen, followed=skeleton(690)) == 0  # ... followed the bystander
    assert follow_agreement([lifter], 0, followed=skeleton(300)) == -1  # one person: nothing to judge
    assert follow_agreement([skeleton(300, hanging=False), skeleton(700, hanging=False)], 0, skeleton(300)) == -1  # nobody on the bar
    assert follow_agreement(cands, chosen, followed=np.full((config.NUM_LANDMARKS, 3), np.nan, dtype=np.float32)) == -1  # nobody followed


def test_people_count_round_trips_and_is_checked(tmp_path):
    n = 4
    kp = np.zeros((n, config.NUM_LANDMARKS, 3), dtype=np.float32)
    pose = PoseResult(kp, 30.0, (320, 240), np.arange(n) * 33.3, people_count=np.array([1, 2, 2, 0], dtype=np.int16), follow_check=np.array([-1, 1, 0, -1], dtype=np.int8))
    save_pose(tmp_path / "k.npz", pose)
    back = load_pose(tmp_path / "k.npz")
    assert back.people_count.tolist() == [1, 2, 2, 0] and back.follow_check.tolist() == [-1, 1, 0, -1]
    save_pose(tmp_path / "old.npz", PoseResult(kp, 30.0, (320, 240), np.arange(n) * 33.3))
    old = load_pose(tmp_path / "old.npz")
    assert old.people_count is None and old.follow_check is None  # old caches stay readable
    try:
        PoseResult(kp, 30.0, (320, 240), np.arange(n) * 33.3, people_count=np.zeros(3, dtype=np.int16))
    except ValueError:
        return
    raise AssertionError("a wrong-length people_count must be rejected")


def test_the_multi_person_warning_appears_at_the_configured_share_and_never_for_unknown():
    share = config.WARN_MULTI_PERSON_SHARE
    assert not camera_warnings(CameraChecks(0.55, 0.05, None))
    assert not camera_warnings(CameraChecks(0.55, 0.05, share - 0.01))
    msg = camera_warnings(CameraChecks(0.55, 0.05, share))
    assert len(msg) == 1 and "more than one person" in msg[0] and "not necessarily the one hanging on the bar" in msg[0]


def test_the_wrong_person_warning_needs_judged_frames_and_the_configured_share():
    share = config.WARN_WRONG_PERSON_SHARE
    assert not camera_warnings(CameraChecks(0.55, 0.05, None, None))
    assert not camera_warnings(CameraChecks(0.55, 0.05, 0.1, share - 0.01))
    msg = camera_warnings(CameraChecks(0.55, 0.05, 0.1, share))
    assert len(msg) == 1 and "NOT the one hanging on the bar" in msg[0]


def test_camera_checks_reads_the_counts_and_the_follow_check_from_the_pose():
    from pullup_rom.camera import camera_checks

    n = 10
    kp = np.stack([skeleton(300)] * n)
    pose = PoseResult(kp, 30.0, (640, 480), np.arange(n) * 33.3, people_count=np.array([1] * 4 + [2] * 6, dtype=np.int16),
                      follow_check=np.array([-1] * 4 + [1, 1, 0, 0, 0, 0], dtype=np.int8))
    c = camera_checks(pose)
    assert abs(c.multi_person_share - 0.6) < 1e-9 and abs(c.wrong_person_share - 4 / 6) < 1e-9
    plain = camera_checks(PoseResult(kp, 30.0, (640, 480), np.arange(n) * 33.3))
    assert plain.multi_person_share is None and plain.wrong_person_share is None  # an old cache: unknown, never a guess

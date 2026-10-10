"""Per-person reference calibration on synthetic series: the rule, the guards, and the CLI. Fixture numbers are arbitrary."""
import json

import numpy as np
import pytest
from synthetic import SHOULDER_WIDTH, make_pose, make_rep_series

from pullup_rom import cli, config
from pullup_rom.calibrate import calibrate_from_reference, reference_threshold
from pullup_rom.metrics import compute_signals
from pullup_rom.pose import save_pose
from pullup_rom.reps import RepParams, detect_reps

FPS, BAR, SIZE = 30.0, 500.0, (320, 360)


def _pose(valleys, peaks=None):
    peaks = peaks or [0.45] * (len(valleys) - 1)
    head, elbow = make_rep_series(peaks=list(peaks), valley_elbows=list(valleys), lead_s=0.3, up_s=0.5, hold_s=0.1, down_s=0.5, pause_s=0.3, tail_s=0.3)
    return make_pose(BAR + head * SHOULDER_WIDTH, BAR, elbow, fps=FPS, frame_size=SIZE, scale=0.4)


def test_the_threshold_is_the_strictest_whole_degree_that_keeps_every_reference_rep_full():
    assert reference_threshold([152.3, 150.9, 153.0]) == 150.0
    assert reference_threshold([144.86]) == 144.0  # how the global 144 was derived from 144.86
    assert reference_threshold([151.0]) == 151.0
    with pytest.raises(ValueError):
        reference_threshold([])


def test_every_reference_rep_is_full_under_its_own_threshold_and_it_is_not_higher_than_the_lowest_bottom():
    pose = _pose([160, 155, 158, 160])
    ref = calibrate_from_reference(pose, "world")
    assert ref.reps_used == 3 and ref.threshold_deg <= min(ref.bottoms_deg) and min(ref.bottoms_deg) - ref.threshold_deg < 1.0
    s = compute_signals(pose)
    reps = detect_reps(s.head_minus_bar, s.elbow_deg, FPS, RepParams.for_source("world", dead_hang_elbow_min_deg=ref.threshold_deg)).reps
    assert [r.full for r in reps] == [True, True, True]


def test_a_tolerance_lowers_the_threshold_by_exactly_that_many_degrees_and_must_not_be_negative():
    pose = _pose([160, 155, 158, 160])
    strict, loose = calibrate_from_reference(pose, "world"), calibrate_from_reference(pose, "world", tolerance_deg=4.0)
    assert strict.threshold_deg - loose.threshold_deg == 4.0 and "4 deg tolerance" in loose.origin and "tolerance" not in strict.origin
    for bad in (-1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="finite number"):
            calibrate_from_reference(pose, "world", tolerance_deg=bad)
    with pytest.raises(ValueError, match="every rep would pass"):
        calibrate_from_reference(pose, "world", tolerance_deg=500.0)


def test_two_people_get_different_thresholds_and_the_same_arms_pass_for_one_and_fail_for_the_other():
    loose = calibrate_from_reference(_pose([125, 125, 125, 125]), "world")  # a person whose good reps read low
    tight = calibrate_from_reference(_pose([160, 160, 160, 160]), "world")  # one whose good reps read high
    assert tight.threshold_deg - loose.threshold_deg >= 30
    probe = 140.0
    assert probe >= loose.threshold_deg and probe < tight.threshold_deg  # 140 deg is a full rep for one person, not the other


def test_a_reference_needs_enough_complete_reps_with_the_head_over_the_bar():
    with pytest.raises(ValueError, match="need at least 3"):
        calibrate_from_reference(_pose([160, 160, 160]), "world")  # two reps
    with pytest.raises(ValueError, match="head short of the bar"):
        calibrate_from_reference(_pose([160] * 5, peaks=[0.45, -0.3, -0.3, -0.3]), "world")  # only one rep clears the bar
    ok = calibrate_from_reference(_pose([160] * 5, peaks=[0.45, 0.45, 0.45, -0.3]), "world")
    assert ok.reps_used == 3 and ok.reps_found == 4  # the head-short rep is left out of the reference


def test_a_cut_off_last_rep_is_not_used():
    from pullup_rom.pose import PoseResult

    pose = _pose([160] * 6)  # five reps
    s = compute_signals(pose)
    reps = detect_reps(s.head_minus_bar, s.elbow_deg, FPS, RepParams.for_source("world", dead_hang_elbow_min_deg=None)).reps
    assert len(reps) == 5
    stop = reps[-1].peak_frame + 4  # the clip ends just after the last rep's top
    cut = PoseResult(pose.keypoints[:stop], pose.fps, pose.frame_size, pose.timestamps_ms[:stop], world=pose.world[:stop])
    ref = calibrate_from_reference(cut, "world")
    assert ref.reps_used == 4 and ref.reps_found == 4  # the clipped fifth rep is neither found as complete nor used


def _write_video(path, frames):
    import cv2

    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, SIZE)
    for i in range(frames):
        w.write(np.full((SIZE[1], SIZE[0], 3), 90 + (i % 5), dtype=np.uint8))
    w.release()


def _cli_files(tmp_path, pose):
    video, kp = tmp_path / "in.mp4", tmp_path / "kp.npz"
    _write_video(video, pose.num_frames)
    save_pose(kp, pose)
    return video, kp


@pytest.mark.slow
def test_cli_uses_the_reference_threshold_and_records_where_it_came_from(tmp_path, capsys):
    pose = _pose([140, 140, 140, 140])  # this person's good reps read 140
    video, kp = _cli_files(tmp_path, pose)
    ref, out = tmp_path / "ref.npz", tmp_path / "out.mp4"
    save_pose(ref, _pose([140, 140, 140, 140]))
    assert cli.main([str(video), "-o", str(out), "--keypoints", str(kp), "--reference", str(ref)]) == 0
    printed = capsys.readouterr().out
    assert "reference clip" in printed and "reps: 3" in printed
    summary = json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))
    assert summary["dead_hang_threshold_origin"].startswith("reference clip") and summary["dead_hang_threshold_deg"] <= 140.0
    assert [r["full"] for r in summary["reps"]] == [True, True, True]  # 140-deg arms are full for this person; the global 144 would fail them
    default = tmp_path / "default.mp4"
    cli.main([str(video), "-o", str(default), "--keypoints", str(kp)])
    assert [r["full"] for r in json.loads(default.with_suffix(".json").read_text(encoding="utf-8"))["reps"]] == [False, False, False]
    assert config.DEAD_HANG_ELBOW_MIN_DEG["world"] == 144.0  # the global threshold is untouched


def test_cli_refuses_a_conflicting_flag_a_short_reference_and_a_missing_one(tmp_path):
    video, kp = _cli_files(tmp_path, _pose([140, 140, 140, 140]))
    ref, out = tmp_path / "ref.npz", tmp_path / "out.mp4"
    save_pose(ref, _pose([140, 140, 140, 140]))
    with pytest.raises(SystemExit, match="either --reference or --dead-hang-deg"):
        cli.main([str(video), "-o", str(out), "--keypoints", str(kp), "--reference", str(ref), "--dead-hang-deg", "150"])
    short = tmp_path / "short.npz"
    save_pose(short, _pose([160, 160, 160]))
    with pytest.raises(SystemExit, match="reference clip: .*need at least"):
        cli.main([str(video), "-o", str(out), "--keypoints", str(kp), "--reference", str(short)])
    with pytest.raises(SystemExit, match="reference clip"):
        cli.main([str(video), "-o", str(out), "--keypoints", str(kp), "--reference", str(tmp_path / "missing.mp4")])
    assert not out.exists()  # nothing is rendered when the reference is refused


def test_cli_refuses_a_nonsense_tolerance_or_threshold_before_rendering(tmp_path):
    video, kp = _cli_files(tmp_path, _pose([140, 140, 140, 140]))
    ref, out = tmp_path / "ref.npz", tmp_path / "out.mp4"
    save_pose(ref, _pose([140, 140, 140, 140]))
    base = [str(video), "-o", str(out), "--keypoints", str(kp)]
    for tol in ("inf", "500", "-3", "nan"):
        with pytest.raises(SystemExit, match="error"):
            cli.main(base + ["--reference", str(ref), "--reference-tolerance-deg", tol])
    for deg in ("inf", "nan", "0", "-5"):
        with pytest.raises(SystemExit, match="--dead-hang-deg must be"):
            cli.main(base + ["--dead-hang-deg", deg])
    with pytest.raises(SystemExit, match="only applies together with --reference"):
        cli.main(base + ["--reference-tolerance-deg", "3"])
    assert not out.exists()

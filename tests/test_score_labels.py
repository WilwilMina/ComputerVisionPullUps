import json

from synthetic import SHOULDER_WIDTH, make_pose, make_rep_series

from pullup_rom.metrics import compute_signals
from pullup_rom.pose import save_pose
from pullup_rom.reps import RepParams, detect_reps
from pullup_rom.score_labels import build_report, expected_verdict, score_clip, split_of, tally

FPS, BAR = 30.0, 500.0


def _pose(valleys):
    head, elbow = make_rep_series(peaks=[0.4, 0.4], valley_elbows=list(valleys), lead_s=0.3, up_s=0.5, hold_s=0.1, down_s=0.5, pause_s=0.3, tail_s=0.3)
    return make_pose(BAR + head * SHOULDER_WIDTH, BAR, elbow, fps=FPS, frame_size=(320, 360), scale=0.4)


def _reps(pose):
    s = compute_signals(pose)
    return detect_reps(s.head_minus_bar, s.elbow_deg, FPS, RepParams(dead_hang_elbow_min_deg=144.0)).reps


def _data(reps, labels, excluded=False, split="held_out"):
    return {
        "assignment": {"calibration": [], "held_out": []} | {split: ["c1"]},
        "clips": [{"clip": "c1", "people": 1, "excluded_from_scoring": excluded, "exclusion_reason": "kipping",
                   "reps": [{"index": i + 1, "top_s": r.peak_s, "label": lab} for i, (r, lab) in enumerate(zip(reps, labels))]}],
    }


def test_expected_verdicts():
    assert expected_verdict("FULL") == "FULL"
    assert all(expected_verdict(x) == "NOT FULL" for x in ("SHORT_BOTTOM", "NOT_HIGH", "SHORT_BOTH"))
    assert expected_verdict("UNCLEAR") is None


def test_unclear_and_excluded_reps_are_never_scored_and_the_rest_are_counted():
    reps = _reps(_pose([165, 165, 165]))
    data = _data(reps, ["FULL", "UNCLEAR"])
    rows = score_clip(data, data["clips"][0], reps)
    assert [r.scored for r in rows] == [True, False] and rows[0].outcome == "agree" and rows[1].outcome == "not scored (UNCLEAR)"
    ex = _data(reps, ["FULL", "FULL"], excluded=True)
    assert not any(r.scored for r in score_clip(ex, ex["clips"][0], reps)) and tally(score_clip(ex, ex["clips"][0], reps)) == {}


def test_a_wrong_verdict_is_reported_as_differs_and_a_missing_rep_as_missing():
    reps = _reps(_pose([165, 120, 165]))  # rep 1 ends in a bent valley, so both reps touching it are NOT FULL
    data = _data(reps, ["FULL", "FULL"])
    rows = score_clip(data, data["clips"][0], reps)
    assert [r.outcome for r in rows] == ["differs", "differs"]
    far = _data(reps, ["FULL", "FULL"])
    far["clips"][0]["reps"][1]["top_s"] += 50
    assert score_clip(far, far["clips"][0], reps)[1].outcome == "no detected rep"


def test_split_must_be_assigned():
    data = _data([], [])
    assert split_of(data, "c1") == "held_out"
    try:
        split_of(data, "other")
    except ValueError:
        return
    raise AssertionError("an unassigned clip must raise")


def test_report_prints_counts_never_percentages_and_says_when_there_are_no_negatives(tmp_path):
    pose = _pose([165, 165, 165])
    save_pose(tmp_path / "c1.keypoints.npz", pose)
    reps = _reps(pose)
    text = "\n".join(build_report(_data(reps, ["FULL", "FULL"]), tmp_path))
    assert "NO ACCURACY FIGURE" in text and "%" not in text.replace("0 scored negative", "")
    assert "2 x FULL" in text


def test_a_malformed_labels_file_or_missing_keypoints_gives_a_clear_error(tmp_path):
    import pytest

    from pullup_rom.score_labels import load_label_file

    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"assignment": {}, "clips": [{"clip": "c1", "reps": [{"index": 1, "top_s": 1.0}]}]}))
    with pytest.raises(ValueError, match="no 'label'"):
        load_label_file(bad)
    data = _data([], [])
    data["status"] = None
    with pytest.raises(ValueError, match="no keypoints file"):
        build_report(data, tmp_path)  # a null status is fine; the missing file is the error

import json

import numpy as np
import pytest
from synthetic import SHOULDER_WIDTH, make_pose, make_rep_series

from pullup_rom import config
from pullup_rom.label_check import build_report, head_minus_bar_by_group, load_labels, snap_labels, verdict_text
from pullup_rom.metrics import compute_signals
from pullup_rom.reps import RepParams, detect_reps

FPS = 30.0
BAR = 500.0
W = SHOULDER_WIDTH


def _scene(peaks=(0.4, 0.4, 0.4), valleys=(165, 165, 165, 165)):
    head_mb, elbow = make_rep_series(
        peaks=list(peaks), valley_elbows=list(valleys), lead_s=0.3, up_s=0.5, hold_s=0.1, down_s=0.5, pause_s=0.3, tail_s=0.3
    )
    pose = make_pose(BAR + head_mb * W, BAR, elbow, fps=FPS, frame_size=(320, 360), scale=0.4)
    signals = compute_signals(pose)
    return pose, signals, detect_reps(signals.head_minus_bar, signals.elbow_deg, FPS, RepParams(dead_hang_elbow_min_deg=144.0))


def test_labels_file_is_validated(tmp_path):
    good = {"keypoints": "k.npz", "reps": [{"index": 1, "label": "FULL", "approx_top_s": 3.8}]}
    p = tmp_path / "ok.json"
    p.write_text(json.dumps(good))
    assert load_labels(p)["reps"][0]["label"] == "FULL"
    for bad in (
        {"reps": good["reps"]},
        {"keypoints": "k.npz", "reps": [{"index": 1, "label": "MAYBE", "approx_top_s": 1.0}]},
        {"keypoints": "k.npz", "reps": [{"index": 1, "label": "FULL"}]},
    ):
        p.write_text(json.dumps(bad))
        with pytest.raises(ValueError):
            load_labels(p)


def test_labels_snap_to_the_nearest_detected_peak_and_each_rep_is_used_once():
    _, _, analysis = _scene()
    reps = analysis.reps
    assert len(reps) == 3
    labels = [{"index": i + 1, "label": "FULL", "approx_top_s": r.peak_s + 0.3} for i, r in enumerate(reps)]  # visual times 0.3 s off
    assert [rep.index if rep else None for _, rep in snap_labels(labels, reps, 1.0)] == [1, 2, 3]
    far = [{"index": 1, "label": "FULL", "approx_top_s": reps[0].peak_s + 5.0}]
    assert snap_labels(far, reps, 1.0)[0][1] is None  # outside the window: no match, not a forced one
    twice = [{"index": 1, "label": "FULL", "approx_top_s": reps[0].peak_s}, {"index": 2, "label": "FULL", "approx_top_s": reps[0].peak_s + 0.1}]
    snapped = snap_labels(twice, reps, 1.0)
    assert snapped[0][1].index == 1 and snapped[1][1] is None  # the second label cannot claim the same rep


def test_head_landmark_heights_over_the_bar_are_read_per_group():
    pose, signals, analysis = _scene()
    frame = analysis.reps[0].peak_frame
    kp = pose.keypoints
    nose_y = kp[:, config.NOSE, 1].copy()
    for j, dy in ((config.MOUTH_LEFT, 0.4 * 20), (config.MOUTH_RIGHT, 0.4 * 20), (config.LEFT_EYE, -0.4 * 5), (config.RIGHT_EYE, -0.4 * 5)):
        kp[:, j, 0], kp[:, j, 1], kp[:, j, 2] = kp[:, config.NOSE, 0], nose_y + dy, 1.0  # mouth 20 body-px below the nose, eyes 5 above
    got = head_minus_bar_by_group(pose, signals, frame)
    nose = got["nose"][0]
    assert abs(got["mouth"][0] - (nose - 20.0 / W)) < 0.02
    assert abs(got["eyes"][0] - (nose + 5.0 / W)) < 0.02
    assert abs(got["ears"][0] - nose) < 0.02
    kp[:, config.MOUTH_LEFT, 2] = 0.0  # one unconfident mouth corner: the whole group is unavailable, not guessed
    assert head_minus_bar_by_group(pose, signals, frame)["mouth"][0] is None


def test_report_compares_labels_with_the_pipeline_and_flags_the_lowest_bottom():
    pose, signals, analysis = _scene()
    labels = {
        "clip": "synthetic", "keypoints": "unused.npz", "labelled_by": "test", "time_accuracy_s": 0.3,
        "reps": [{"index": i + 1, "label": "FULL", "approx_top_s": r.peak_s} for i, r in enumerate(analysis.reps)],
    }
    text = "\n".join(build_report(labels, pose, "world"))
    assert text.count("agree") >= 3 and "DIFFERS" not in text
    assert "lowest 3D bottom among complete reps" in text and "nose" in text and "mouth" in text
    labels["reps"][1]["label"] = "NOT FULL"
    assert "DIFFERS" in "\n".join(build_report(labels, pose, "world"))


def test_verdict_text():
    assert [verdict_text(x) for x in (True, False, None)] == ["FULL", "NOT FULL", "UNKNOWN"]

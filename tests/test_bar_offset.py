import json

import numpy as np
from synthetic import make_pose

from pullup_rom.bar_offset import offsets_in_body_scales, pick_frames, summarize, tick_step_px, write_crops
from pullup_rom.metrics import compute_signals

N = 90


def _scene():
    pose = make_pose(np.full(N, 560.0), 500.0, 165.0, fps=30.0, frame_size=(320, 360), scale=0.4)
    return pose, compute_signals(pose)


def test_pick_frames_spreads_evenly_and_only_uses_frames_with_confident_wrists():
    pose, signals = _scene()
    frames = pick_frames(pose, signals, 6)
    assert len(frames) == 6 and frames == sorted(frames) and frames[0] < 10 and frames[-1] > N - 12
    pose.keypoints[:45, 15, 2] = 0.0  # left wrist unconfident for the first half
    signals = compute_signals(pose)
    assert all(f >= 40 for f in pick_frames(pose, signals, 6))
    assert pick_frames(pose, signals, 1000) == sorted(set(pick_frames(pose, signals, 1000)))  # asking for more than exist is fine


def test_offsets_are_in_body_scales_and_unreadable_frames_are_skipped():
    manifest = [
        {"file": "a.png", "time_s": 1.0, "body_scale_px": 200.0},
        {"file": "b.png", "time_s": 2.0, "body_scale_px": 100.0},
        {"file": "c.png", "time_s": 3.0, "body_scale_px": 100.0},
    ]
    readings = {"a.png": {"center": 50, "top": 60, "bottom": 40}, "b.png": {"center": 30, "top": None, "bottom": None}, "c.png": None}
    rows = offsets_in_body_scales(manifest, readings)
    assert [r["file"] for r in rows] == ["a.png", "b.png"]
    assert abs(rows[0]["center_scales"] - 0.25) < 1e-9 and abs(rows[0]["thickness_scales"] - 0.1) < 1e-9
    assert abs(rows[1]["center_scales"] - 0.30) < 1e-9 and rows[1]["thickness_scales"] is None
    s = summarize(rows)
    assert s["n"] == 2 and abs(s["mean"] - 0.275) < 1e-9 and s["min"] == 0.25 and s["max"] == 0.30 and abs(s["std"] - np.std([0.25, 0.30], ddof=1)) < 1e-9
    assert summarize([]) is None


def test_tick_step_gets_coarser_as_the_body_gets_bigger():
    assert tick_step_px(40) < tick_step_px(100) < tick_step_px(300)


def test_crops_are_written_with_a_manifest(tmp_path):
    import cv2

    pose, signals = _scene()
    video = tmp_path / "v.mp4"
    w = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (320, 360))
    for _ in range(N):
        w.write(np.full((360, 320, 3), 90, dtype=np.uint8))
    w.release()
    manifest = write_crops(video, pose, signals, tmp_path / "crops", 4)
    assert len(manifest) == 4
    saved = json.loads((tmp_path / "crops" / "manifest.json").read_text())
    assert [m["file"] for m in saved] == [m["file"] for m in manifest]
    for m in manifest:
        img = cv2.imread(str(tmp_path / "crops" / m["file"]))
        assert img is not None and img.shape[0] > 100
        assert m["body_scale_px"] > 0 and m["ruler_step_px"] == tick_step_px(m["body_scale_px"])


def test_manifest_rows_without_a_usable_body_scale_are_skipped():
    manifest = [
        {"file": "a.png", "time_s": 1.0, "body_scale_px": 0.0},
        {"file": "b.png", "time_s": 2.0, "body_scale_px": float("nan")},
        {"file": "c.png", "time_s": 3.0},
        {"file": "d.png", "time_s": 4.0, "body_scale_px": 100.0},
    ]
    readings = {k: {"center": 20, "top": None, "bottom": None} for k in "abcd" for k in [k + ".png"]}
    rows = offsets_in_body_scales(manifest, readings)
    assert [r["file"] for r in rows] == ["d.png"]
    assert summarize(rows)["mean"] == 0.2


def test_summarize_single_reading_has_zero_std_and_thickness_is_optional():
    s = summarize([{"center_scales": 0.2, "thickness_scales": None}])
    assert s["n"] == 1 and s["std"] == 0.0 and s["thickness_mean"] is None


def test_wrists_outside_the_image_write_no_crop_instead_of_crashing(tmp_path):
    import cv2

    pose, signals = _scene()
    pose.keypoints[:, 15:17, 1] = -2000.0  # both wrists far above the image
    video = tmp_path / "v.mp4"
    w = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (320, 360))
    for _ in range(N):
        w.write(np.full((360, 320, 3), 90, dtype=np.uint8))
    w.release()
    assert write_crops(video, pose, compute_signals(pose), tmp_path / "crops", 4) == []


# ---------- the measured offset is APPLIED (geometry, not tuned to labels) ----------

def test_the_configured_offset_is_inside_the_range_that_was_measured():
    from pullup_rom import config

    assert 0.191 <= config.BAR_OFFSET_BODY_SCALES <= 0.266  # min and max of the 22 hand readings on two clips


def test_the_bar_line_sits_the_configured_offset_above_the_wrist_line():
    import dataclasses

    from pullup_rom import config
    from pullup_rom.metrics import MetricsParams

    pose, _ = _scene()
    on = compute_signals(pose)  # default: the configured offset
    off = compute_signals(pose, dataclasses.replace(MetricsParams(), bar_offset_body_scales=0.0))
    scale = on.body_scale_px[N // 2]
    rise_px = on.bar_height_px[N // 2] - off.bar_height_px[N // 2]  # height is up
    assert abs(rise_px / scale - config.BAR_OFFSET_BODY_SCALES) < 1e-6


def test_a_head_that_only_reaches_the_wrist_line_no_longer_counts_as_over_the_bar():
    from synthetic import make_rep_series

    from pullup_rom import config
    from pullup_rom.reps import RepParams, detect_reps

    head, elbow = make_rep_series(peaks=[0.1], valley_elbows=[165, 165])  # peak 0.1 body scales above the WRIST line
    params = RepParams(dead_hang_elbow_min_deg=150.0)
    before = detect_reps(head, elbow, 30.0, params).reps[0]
    after = detect_reps(head - config.BAR_OFFSET_BODY_SCALES, elbow, 30.0, params).reps[0]  # same head, bar moved up to the real bar
    assert before.head_ok and not after.head_ok and after.full is False

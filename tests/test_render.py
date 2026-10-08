"""Renderer and CLI on synthetic data: no real video, no model, no network."""
import json

import numpy as np
import pytest
from synthetic import SHOULDER_WIDTH, make_pose, make_rep_series

from pullup_rom import cli, config
from pullup_rom.metrics import compute_signals
from pullup_rom.pose import save_pose
from pullup_rom.render import (
    AMBER,
    CARRIED,
    check_video_matches,
    compute_layout,
    counter_texts,
    invalid_share,
    rep_row,
    render_video,
    table_entries,
    verdict_color,
    warning_text,
)
from pullup_rom.reps import RepParams, detect_reps

FPS = 30.0
BAR = 500.0
W = SHOULDER_WIDTH
SIZE = (320, 360)
SCALE = 0.4  # synthetic body scaled to fit the small frame


def _scene(peaks=(0.4, -0.2), threshold=150.0):
    head_mb, elbow = make_rep_series(
        peaks=list(peaks), valley_elbows=[165] * (len(peaks) + 1),
        lead_s=0.3, up_s=0.4, hold_s=0.1, down_s=0.4, pause_s=0.3, tail_s=0.3,
    )
    pose = make_pose(BAR + head_mb * W, BAR, elbow, fps=FPS, frame_size=SIZE, scale=SCALE)
    signals = compute_signals(pose)
    analysis = detect_reps(signals.head_minus_bar, signals.elbow_deg, FPS, RepParams(dead_hang_elbow_min_deg=threshold))
    return pose, signals, analysis


def _poor_scene(frames=24):
    """A short clip where the model was never confident: nothing is usable, so the warning must fire."""
    pose = make_pose(np.full(frames, BAR + 50.0), BAR, 165.0, fps=FPS, frame_size=SIZE, scale=SCALE)
    pose.keypoints[:, :, 2] = 0.0
    signals = compute_signals(pose)
    return pose, signals, detect_reps(signals.head_minus_bar, signals.elbow_deg, FPS, RepParams())


def _write_video(path, frames, size=SIZE, fps=FPS):
    import cv2

    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    assert w.isOpened()
    for i in range(frames):
        w.write(np.full((size[1], size[0], 3), 90 + (i % 5), dtype=np.uint8))
    w.release()


@pytest.fixture(scope="module")
def scene():
    return _scene()


# ---------- layout, text, warning (pure) ----------

@pytest.mark.parametrize("size", [(1080, 1920), (1280, 720), (320, 360), (2560, 1440), (321, 241)])
def test_layout_is_even_never_enlarges_and_fits_the_panel(size):
    lay = compute_layout(size)
    assert lay.scale <= 1.0 and lay.vid_h <= config.RENDER_VIDEO_MAX_HEIGHT_PX + 1
    assert lay.canvas_w % 2 == 0 and lay.canvas_h % 2 == 0
    assert lay.canvas_w == lay.vid_w + lay.panel_w + (lay.canvas_w - lay.vid_w - lay.panel_w)  # even padding only
    assert lay.canvas_h >= lay.vid_h + lay.graph_h


def test_rep_rows_say_yes_no_and_unknown_and_never_guess(scene):
    _, _, full_case = _scene(peaks=(0.4, -0.2), threshold=150.0)
    full, short = [rep_row(r) for r in full_case.reps]
    assert (full.head_cleared, full.arms_straight, full.verdict) == ("yes", "yes", "FULL")
    assert (short.head_cleared, short.verdict) == ("no", "NOT FULL")
    assert full.bottom_elbow.endswith(" deg") and full.pull_time.endswith(" s")

    _, _, unset = _scene(peaks=(0.4, -0.2), threshold=None)
    first, second = [rep_row(r) for r in unset.reps]
    assert (first.head_cleared, first.arms_straight, first.verdict) == ("yes", "UNKNOWN", "UNKNOWN")
    assert second.verdict == "NOT FULL"  # a failed head check is a definite no, even with no elbow threshold

    _, _, bent = _scene(peaks=(0.4,), threshold=170.0)
    assert (rep_row(bent.reps[0]).arms_straight, rep_row(bent.reps[0]).verdict) == ("no", "NOT FULL")


def test_verdict_colours_are_distinct():
    assert len({verdict_color("FULL"), verdict_color("NOT FULL"), verdict_color("UNKNOWN")}) == 3


def test_counter_only_counts_reps_that_have_finished(scene):
    _, _, analysis = scene
    first, second = analysis.reps
    assert counter_texts(analysis, 0)[0] == "REPS: 0"
    mid_first = (first.start_frame + first.end_frame) // 2
    assert counter_texts(analysis, mid_first)[0] == "REPS: 0" and counter_texts(analysis, mid_first)[2] == 1
    headline, tally, live = counter_texts(analysis, first.end_frame)
    assert headline == "REPS: 1" and "FULL 1" in tally and live is None
    headline, tally, _ = counter_texts(analysis, second.end_frame)
    assert headline == "REPS: 2" and "FULL 1" in tally and "NOT FULL 1" in tally


def test_table_grows_in_time_order_and_includes_uncertain_attempts(scene):
    _, _, analysis = scene
    assert table_entries(analysis, 0, FPS) == []
    after_all = table_entries(analysis, 10_000, FPS)
    assert [obj.index for kind, obj in after_all if kind == "rep"] == [1, 2]

    head, elbow = make_rep_series(peaks=[0.4], valley_elbows=[165, 165])
    head = head.copy()
    peak = int(np.argmax(head))
    head[peak - 15 : peak + 15] = np.nan
    gapped = detect_reps(head, elbow, FPS, RepParams(dead_hang_elbow_min_deg=150.0))
    kinds = [kind for kind, _ in table_entries(gapped, 10_000, FPS)]
    assert kinds == ["discard"]
    assert table_entries(gapped, 0, FPS) == []  # shown only once it is over


def test_warning_text_appears_at_the_configured_share_and_not_below(scene):
    _, signals, _ = scene
    assert warning_text(invalid_share(signals)) is None or invalid_share(signals) >= config.WARN_INVALID_SHARE
    assert warning_text(config.WARN_INVALID_SHARE - 0.01) is None
    text = warning_text(config.WARN_INVALID_SHARE)
    assert text and "WARNING" in text and "%" in text


# ---------- the video ----------

def test_render_writes_every_frame_at_the_expected_size_and_warns_when_tracking_is_poor(tmp_path):
    import cv2

    pose, signals, analysis = _poor_scene()
    assert invalid_share(signals) >= config.WARN_INVALID_SHARE and analysis.reps == []
    video, out = tmp_path / "in.mp4", tmp_path / "sub" / "out.mp4"
    _write_video(video, pose.num_frames)
    stats = render_video(video, pose, signals, analysis, out)
    lay = compute_layout(SIZE)
    assert stats.frames == pose.num_frames and stats.size == (lay.canvas_w, lay.canvas_h)

    cap = cv2.VideoCapture(str(out))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == pose.num_frames
    assert (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))) == stats.size
    assert abs(cap.get(cv2.CAP_PROP_FPS) - FPS) < 0.5
    ok, frame = cap.read()
    cap.release()
    banner = frame[3:10, 3:60].reshape(-1, 3).mean(axis=0)  # BGR; the warning banner is red
    assert ok and banner[2] > 120 and banner[0] < 90


def test_render_rejects_a_missing_video(tmp_path, scene):
    pose, signals, analysis = scene
    with pytest.raises(ValueError, match="cannot read video"):
        render_video(tmp_path / "nope.mp4", pose, signals, analysis, tmp_path / "out.mp4")


# ---------- the command line ----------

def test_cli_writes_video_and_summary_and_prints_reps(tmp_path, capsys, scene):
    import cv2

    pose, _, _ = scene
    video, npz, out = tmp_path / "in.mp4", tmp_path / "kp.npz", tmp_path / "out.mp4"
    _write_video(video, pose.num_frames)
    save_pose(npz, pose)
    code = cli.main([str(video), "-o", str(out), "--keypoints", str(npz), "--dead-hang-deg", "150"])
    assert code == 0
    printed = capsys.readouterr().out
    assert "reps: 2" in printed and "annotated video:" in printed and "150 deg" in printed

    summary = json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))
    assert summary["frames"] == pose.num_frames and summary["output_video"]["frames"] == pose.num_frames
    assert summary["dead_hang_threshold_origin"] == "command line" and summary["dead_hang_threshold_deg"] == 150.0
    first, second = summary["reps"]
    assert (first["head_cleared_bar"], first["arms_straightened"], first["full"]) == (True, True, True)
    assert (second["head_cleared_bar"], second["full"]) == (False, False)
    assert "accuracy" not in json.dumps(summary).lower().replace("not accuracy figures", "")
    lay = compute_layout(SIZE)
    cap = cv2.VideoCapture(str(out))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == pose.num_frames
    assert (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))) == (lay.canvas_w, lay.canvas_h)
    cap.set(cv2.CAP_PROP_POS_FRAMES, pose.num_frames - 1)
    ok, last = cap.read()
    cap.release()
    assert ok and last[: lay.vid_h, lay.vid_w :].std() > 5  # the counter and table were drawn, not left blank


def test_summary_marks_unknown_as_null_never_as_a_guess(scene):
    pose, signals, _ = scene
    analysis = detect_reps(signals.head_minus_bar, signals.elbow_deg, FPS, RepParams())  # threshold unset
    summary = cli.build_summary("clip.mp4", pose, signals, analysis, "world", None, "unset", [], None)
    first = summary["reps"][0]
    assert first["head_cleared_bar"] is True and first["arms_straightened"] is None and first["full"] is None
    assert summary["dead_hang_threshold_deg"] is None and summary["dead_hang_threshold_origin"] == "unset"
    json.dumps(summary, allow_nan=False)  # no NaN leaks into the JSON


@pytest.mark.slow
def test_cli_prints_a_warning_and_records_it_when_tracking_is_poor(tmp_path, capsys):
    pose, _, _ = _poor_scene()
    video, npz, out = tmp_path / "in.mp4", tmp_path / "kp.npz", tmp_path / "out.mp4"
    _write_video(video, pose.num_frames)
    save_pose(npz, pose)
    assert cli.main([str(video), "-o", str(out), "--keypoints", str(npz)]) == 0
    assert "WARNING" in capsys.readouterr().out
    assert json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))["warnings"]


def test_cli_reports_a_missing_video_and_unusable_keypoints_cleanly(tmp_path):
    with pytest.raises(SystemExit, match="video not found"):
        cli.main([str(tmp_path / "nope.mp4"), "-o", str(tmp_path / "out.mp4")])
    video = tmp_path / "in.mp4"
    _write_video(video, 5)
    with pytest.raises(SystemExit, match="error"):
        cli.main([str(video), "-o", str(tmp_path / "out.mp4"), "--keypoints", str(tmp_path / "missing.npz")])


# ---------- regressions from the QA review ----------

def test_graph_labels_do_not_collide_with_the_graph_title():
    top = config.RENDER_GRAPH_MARGINS_PX[2]
    title_baseline, text_height = 14, 12
    label_baseline = top - 6  # rep numbers and the '?' marker
    assert label_baseline - text_height > title_baseline  # the label's top edge is below the title's baseline


def test_carried_bar_colour_is_not_confusable_with_the_unknown_amber():
    distance = sum((a - b) ** 2 for a, b in zip(CARRIED, AMBER)) ** 0.5
    assert distance > 150


def test_layout_geometry_comes_from_config_and_short_sources_keep_a_usable_panel():
    tiny = compute_layout((65, 33))
    assert tiny.body_h >= config.RENDER_MIN_PANEL_HEIGHT_PX and tiny.canvas_h >= tiny.body_h + tiny.graph_h
    assert len(config.RENDER_TABLE_COLUMNS_X) == 6 and config.RENDER_TABLE_COLUMNS_X[-1] < config.RENDER_PANEL_WIDTH_PX


def test_a_tiny_source_renders_without_crashing(tmp_path):
    import cv2

    pose = make_pose(np.full(6, BAR + 50.0), BAR, 165.0, fps=FPS, frame_size=(64, 32), scale=0.1)
    signals = compute_signals(pose)
    analysis = detect_reps(signals.head_minus_bar, signals.elbow_deg, FPS, RepParams())
    video, out = tmp_path / "in.mp4", tmp_path / "out.mp4"
    _write_video(video, 6, size=(64, 32))
    stats = render_video(video, pose, signals, analysis, out)
    assert stats.frames == 6 and stats.size == (compute_layout((64, 32)).canvas_w, compute_layout((64, 32)).canvas_h)


def test_keypoints_that_do_not_match_the_video_are_refused(tmp_path, scene):
    pose, _, _ = scene
    video = tmp_path / "in.mp4"
    _write_video(video, pose.num_frames)
    check_video_matches(video, pose)  # the matching case passes

    import dataclasses

    wrong_size = dataclasses.replace(pose, frame_size=(640, 480))
    with pytest.raises(ValueError, match="wrong cache file"):
        check_video_matches(video, wrong_size)

    fewer = dataclasses.replace(
        pose, keypoints=pose.keypoints[:20], timestamps_ms=pose.timestamps_ms[:20], world=pose.world[:20]
    )
    with pytest.raises(ValueError, match="wrong cache file"):
        check_video_matches(video, fewer)

    one_off = dataclasses.replace(
        pose, keypoints=pose.keypoints[:-1], timestamps_ms=pose.timestamps_ms[:-1], world=pose.world[:-1]
    )
    check_video_matches(video, one_off)  # within the tolerance containers need


def test_cli_refuses_a_mismatched_cache_instead_of_stretching_the_video(tmp_path, scene):
    import dataclasses

    pose, _, _ = scene
    video, npz = tmp_path / "in.mp4", tmp_path / "kp.npz"
    _write_video(video, pose.num_frames)
    other = dataclasses.replace(pose, frame_size=(640, 480))
    save_pose(npz, other)
    with pytest.raises(SystemExit, match="wrong cache file"):
        cli.main([str(video), "-o", str(tmp_path / "out.mp4"), "--keypoints", str(npz)])
    assert not (tmp_path / "out.mp4").exists()


@pytest.mark.slow
def test_cli_turns_unwritable_paths_into_clean_errors(tmp_path):
    pose, _, _ = _poor_scene()
    video, npz = tmp_path / "in.mp4", tmp_path / "kp.npz"
    _write_video(video, pose.num_frames)
    save_pose(npz, pose)
    blocker = tmp_path / "a_file.txt"
    blocker.write_text("x")
    with pytest.raises(SystemExit, match="error: cannot create the output folder"):
        cli.main([str(video), "-o", str(blocker / "sub" / "out.mp4"), "--keypoints", str(npz)])
    folder = tmp_path / "is_a_folder"
    folder.mkdir()
    with pytest.raises(SystemExit, match="error: cannot write the summary"):
        cli.main([str(video), "-o", str(tmp_path / "out.mp4"), "--keypoints", str(npz), "--summary", str(folder)])


@pytest.mark.slow
def test_cli_image_source_runs_and_is_recorded(tmp_path):
    pose, _, _ = _poor_scene()
    video, npz, out = tmp_path / "in.mp4", tmp_path / "kp.npz", tmp_path / "out.mp4"
    _write_video(video, pose.num_frames)
    save_pose(npz, pose)
    assert cli.main([str(video), "-o", str(out), "--keypoints", str(npz), "--source", "image"]) == 0
    assert json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))["elbow_angle_source"] == "image"

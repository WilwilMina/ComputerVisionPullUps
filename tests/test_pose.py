from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest

from pullup_rom import config
from pullup_rom.pose import (
    PoseBackend,
    PoseResult,
    empty_frame,
    landmarks_to_array,
    load_pose,
    model_path,
    progress_line,
    save_pose,
)


def _result(frames=4, world=True):
    kp = np.random.default_rng(0).random((frames, config.NUM_LANDMARKS, 3)).astype(np.float32)
    return PoseResult(
        keypoints=kp,
        fps=30.0,
        frame_size=(640, 480),
        timestamps_ms=np.arange(frames) * (1000 / 30.0),
        world=kp.copy() if world else None,
    )


def test_landmarks_to_array_scales_to_pixels_and_maps_visibility():
    lms = [SimpleNamespace(x=0.5, y=0.25, visibility=0.9), SimpleNamespace(x=1.0, y=1.0, visibility=0.1)]
    out = landmarks_to_array(lms, width=640, height=480)
    assert out.dtype == np.float32
    np.testing.assert_allclose(out[0], [320, 120, 0.9], rtol=1e-6)
    np.testing.assert_allclose(out[1], [640, 480, 0.1], rtol=1e-6)


def test_no_detection_frame_is_nan_with_zero_confidence_not_zeros():
    f = empty_frame()
    assert f.shape == (config.NUM_LANDMARKS, 3)
    assert np.isnan(f[:, :2]).all()
    assert (f[:, 2] == 0).all()


def test_detected_mask_flags_empty_frames():
    r = _result(frames=3)
    r.keypoints[1] = empty_frame()
    assert r.detected.tolist() == [True, False, True]


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(keypoints=np.zeros((4, 33), dtype=np.float32)),  # wrong ndim
        dict(keypoints=np.zeros((4, 33, 3), dtype=np.float64)),  # wrong dtype
        dict(timestamps_ms=np.zeros(3)),  # wrong length
        dict(world=np.zeros((4, 33, 2), dtype=np.float32)),  # wrong world shape
        dict(fps=0.0),
    ],
)
def test_pose_result_rejects_bad_inputs(kwargs):
    base = dict(
        keypoints=np.zeros((4, 33, 3), dtype=np.float32),
        fps=30.0,
        frame_size=(640, 480),
        timestamps_ms=np.zeros(4),
    )
    base.update(kwargs)
    with pytest.raises(ValueError):
        PoseResult(**base)


@pytest.mark.parametrize("with_world", [True, False])
def test_npz_round_trip(tmp_path, with_world):
    r = _result(world=with_world)
    r.keypoints[2] = empty_frame()  # NaNs must survive
    path = tmp_path / "sub" / "kp.npz"
    save_pose(path, r)
    back = load_pose(path)
    np.testing.assert_array_equal(back.keypoints, r.keypoints)
    np.testing.assert_array_equal(back.timestamps_ms, r.timestamps_ms)
    assert back.fps == r.fps and back.frame_size == r.frame_size
    assert (back.world is None) == (not with_world)


def test_fake_backend_satisfies_protocol():
    class FakeBackend:
        def extract(self, video_path):
            return _result()

    backend: PoseBackend = FakeBackend()
    assert backend.extract("anything.mp4").num_frames == 4


@pytest.mark.model
def test_real_backend_blank_video_has_no_person(tmp_path):
    cv2 = pytest.importorskip("cv2")
    from pullup_rom.pose import MediaPipePoseBackend

    if not model_path(config.POSE_MODEL_VARIANT).exists():
        pytest.skip("model file not downloaded")
    video = tmp_path / "blank.mp4"
    w = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 30, (320, 240))
    for _ in range(30):
        w.write(np.zeros((240, 320, 3), dtype=np.uint8))
    w.release()

    r = MediaPipePoseBackend().extract(video)
    assert r.num_frames == 30
    assert not r.detected.any()
    assert np.isnan(r.keypoints[:, :, :2]).all()


def test_progress_line_shows_frames_and_percent_and_survives_an_unknown_total():
    assert progress_line(300, 900) == "pose model: 300/900 frames (33%)"
    assert progress_line(120, 0) == "pose model: 120 frames"
    assert progress_line(120, -1) == "pose model: 120 frames"


def test_progress_percent_never_passes_100_if_the_container_under_reports_its_length():
    assert progress_line(150, 100) == "pose model: 150/100 frames (100%)"

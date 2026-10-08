"""Video -> keypoints. Backend interface plus the MediaPipe Pose Landmarker backend.

Keypoints are [frames, joints, (x_px, y_px, confidence)] in image pixels
(y grows downward; flipping to height-up happens once, in metrics.py).
Frames with no detected person are NaN with confidence 0, never zeros.
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol, Sequence

import numpy as np

from . import config


@dataclass
class PoseResult:
    keypoints: np.ndarray  # float32 [F, J, 3] = (x_px, y_px, conf)
    fps: float
    frame_size: tuple[int, int]  # (width, height) in pixels
    timestamps_ms: np.ndarray  # float64 [F]
    world: Optional[np.ndarray] = None  # float32 [F, J, 3], metric 3D, if the backend has it

    def __post_init__(self) -> None:
        kp = self.keypoints
        if kp.ndim != 3 or kp.shape[2] != 3:
            raise ValueError(f"keypoints must be [frames, joints, 3], got {kp.shape}")
        if kp.dtype != np.float32:
            raise ValueError(f"keypoints must be float32, got {kp.dtype}")
        if self.timestamps_ms.shape != (kp.shape[0],):
            raise ValueError("timestamps_ms must have one entry per frame")
        if self.world is not None and self.world.shape != kp.shape:
            raise ValueError(f"world must match keypoints shape {kp.shape}, got {self.world.shape}")
        if not self.fps > 0:
            raise ValueError("fps must be positive")

    @property
    def num_frames(self) -> int:
        return self.keypoints.shape[0]

    @property
    def detected(self) -> np.ndarray:
        """bool [F]: frame has a person (any finite keypoint)."""
        return np.isfinite(self.keypoints[:, :, 0]).any(axis=1)


class PoseBackend(Protocol):
    def extract(self, video_path: str | Path) -> PoseResult: ...


def landmarks_to_array(landmarks: Sequence, width: int, height: int) -> np.ndarray:
    """Normalised landmarks (.x, .y, .visibility) -> float32 [J, 3] in pixels."""
    out = np.empty((len(landmarks), 3), dtype=np.float32)
    for j, lm in enumerate(landmarks):
        out[j] = (lm.x * width, lm.y * height, lm.visibility)
    return out


def world_to_array(landmarks: Sequence) -> np.ndarray:
    out = np.empty((len(landmarks), 3), dtype=np.float32)
    for j, lm in enumerate(landmarks):
        out[j] = (lm.x, lm.y, lm.z)
    return out


def empty_frame(num_joints: int = config.NUM_LANDMARKS) -> np.ndarray:
    """A no-person frame: NaN position, zero confidence."""
    out = np.full((num_joints, 3), np.nan, dtype=np.float32)
    out[:, 2] = 0.0
    return out


def model_path(variant: str) -> Path:
    return Path(config.POSE_MODEL_DIR) / f"pose_landmarker_{variant}.task"


def ensure_model(variant: str = config.POSE_MODEL_VARIANT) -> Path:
    """Return the local model file, downloading it on first use."""
    path = model_path(variant)
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    url = config.POSE_MODEL_URL_TEMPLATE.format(variant=variant)
    print(f"Downloading {url}")
    tmp = path.with_suffix(".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(path)
    return path


def progress_line(done: int, total: int) -> str:
    if total > 0:
        return f"pose model: {done}/{total} frames ({min(100, 100 * done // total)}%)"
    return f"pose model: {done} frames"


class MediaPipePoseBackend:
    def __init__(self, variant: str = config.POSE_MODEL_VARIANT, progress: bool = True):
        self.variant = variant
        self.progress = progress

    def extract(self, video_path: str | Path) -> PoseResult:
        import cv2
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        model = ensure_model(self.variant)
        if not Path(video_path).is_file():
            raise FileNotFoundError(f"Video not found: {video_path}")
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"OpenCV could not decode video (empty or corrupt?): {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0:
            cap.release()
            raise ValueError(f"Video reports invalid fps ({fps}): {video_path}")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        options = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model)),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=config.MIN_POSE_DETECTION_CONFIDENCE,
            min_pose_presence_confidence=config.MIN_POSE_PRESENCE_CONFIDENCE,
            min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE,
        )
        frames, worlds, stamps = [], [], []
        with vision.PoseLandmarker.create_from_options(options) as landmarker:
            i = 0
            while True:
                ok, bgr = cap.read()
                if not ok:
                    break
                # Synthetic, strictly increasing timestamps; container PTS can be unreliable.
                ts = int(round(i * 1000.0 / fps))
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                res = landmarker.detect_for_video(image, ts)
                if res.pose_landmarks:
                    frames.append(landmarks_to_array(res.pose_landmarks[0], width, height))
                    worlds.append(world_to_array(res.pose_world_landmarks[0]))
                else:
                    frames.append(empty_frame())
                    worlds.append(np.full((config.NUM_LANDMARKS, 3), np.nan, dtype=np.float32))
                stamps.append(float(ts))
                i += 1
                if self.progress and i % config.POSE_PROGRESS_EVERY_FRAMES == 0:
                    print(progress_line(i, total), file=sys.stderr, flush=True)
        cap.release()
        if not frames:
            raise ValueError(f"No frames decoded from {video_path}")
        return PoseResult(
            keypoints=np.stack(frames),
            fps=float(fps),
            frame_size=(width, height),
            timestamps_ms=np.asarray(stamps, dtype=np.float64),
            world=np.stack(worlds),
        )


def save_pose(path: str | Path, result: PoseResult) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    arrays = dict(
        keypoints=result.keypoints,
        fps=np.float64(result.fps),
        frame_size=np.asarray(result.frame_size, dtype=np.int64),
        timestamps_ms=result.timestamps_ms,
    )
    if result.world is not None:
        arrays["world"] = result.world
    np.savez_compressed(path, **arrays)


def load_pose(path: str | Path) -> PoseResult:
    with np.load(path) as z:
        return PoseResult(
            keypoints=z["keypoints"],
            fps=float(z["fps"]),
            frame_size=(int(z["frame_size"][0]), int(z["frame_size"][1])),
            timestamps_ms=z["timestamps_ms"],
            world=z["world"] if "world" in z.files else None,
        )


_REPORT_JOINTS = {
    "nose": [config.NOSE],
    "shoulders": [config.LEFT_SHOULDER, config.RIGHT_SHOULDER],
    "elbows": [config.LEFT_ELBOW, config.RIGHT_ELBOW],
    "wrists": [config.LEFT_WRIST, config.RIGHT_WRIST],
}


def summarize(result: PoseResult) -> str:
    det = result.detected
    lines = [
        f"frames: {result.num_frames}  fps: {result.fps:.2f}  size: {result.frame_size[0]}x{result.frame_size[1]}",
        f"frames with a person: {int(det.sum())}/{result.num_frames} ({100 * det.mean():.1f}%)",
    ]
    for name, idx in _REPORT_JOINTS.items():
        conf = result.keypoints[det][:, idx, 2]
        med = float(np.median(conf)) if conf.size else float("nan")
        lines.append(f"median confidence, {name}: {med:.3f}")
    return "\n".join(lines)


def write_preview(video_path: str | Path, result: PoseResult, out_path: str | Path) -> None:
    """Bare dots-and-lines debug video. Throwaway; the real overlay is render.py."""
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    w, h = result.frame_size
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), result.fps, (w, h))
    for i in range(result.num_frames):
        ok, frame = cap.read()
        if not ok:
            break
        kp = result.keypoints[i]
        for a, b in config.PREVIEW_BONES:
            if np.isfinite(kp[[a, b], :2]).all():
                cv2.line(frame, tuple(int(v) for v in kp[a, :2]), tuple(int(v) for v in kp[b, :2]), (0, 255, 0), 2)
        for j in set(k for bone in config.PREVIEW_BONES for k in bone):
            if np.isfinite(kp[j, :2]).all():
                cv2.circle(frame, (int(kp[j, 0]), int(kp[j, 1])), 4, (0, 0, 255), -1)
        writer.write(frame)
    cap.release()
    writer.release()


def main(argv: Optional[Sequence[str]] = None) -> None:
    p = argparse.ArgumentParser(description="Extract pose keypoints from a video.")
    p.add_argument("video")
    p.add_argument("-o", "--output", required=True, help="keypoints .npz path")
    p.add_argument("--preview", help="optional debug preview .mp4 path")
    p.add_argument("--variant", default=config.POSE_MODEL_VARIANT, choices=["lite", "full", "heavy"])
    args = p.parse_args(argv)

    try:
        result = MediaPipePoseBackend(args.variant).extract(args.video)
    except (FileNotFoundError, ValueError) as e:
        raise SystemExit(f"error: {e}")
    save_pose(args.output, result)
    print(summarize(result))
    if args.preview:
        write_preview(args.video, result, args.preview)
        print(f"preview: {args.preview}")


if __name__ == "__main__":
    main()

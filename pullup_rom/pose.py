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
    people_count: Optional[np.ndarray] = None  # int16 [F], people seen per frame (the tool follows one); None for old caches / backends that do not count
    follow_check: Optional[np.ndarray] = None  # int8 [F]: 1 the followed person is the one hanging on the bar, 0 not, -1 not applicable (one person / nobody hanging)

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
        if self.people_count is not None and self.people_count.shape != (kp.shape[0],):
            raise ValueError("people_count must have one entry per frame")
        if self.follow_check is not None and self.follow_check.shape != (kp.shape[0],):
            raise ValueError("follow_check must have one entry per frame")
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


def _confident(kp: np.ndarray, joints: Sequence[int]) -> bool:
    return bool((kp[list(joints), 2] >= config.MIN_LANDMARK_CONFIDENCE).all() and np.isfinite(kp[list(joints), :2]).all())


def is_person(kp: np.ndarray) -> bool:
    """A real detection: both shoulders confident (a stray low-confidence pose does not count as a person)."""
    return _confident(kp, (config.LEFT_SHOULDER, config.RIGHT_SHOULDER))


def is_hanging(kp: np.ndarray) -> bool:
    """Both wrists above the shoulders: hanging on the bar (image y grows downward).

    Wrist POSITIONS only, not their confidence: a lifter's wrists are often low-confidence at the bar (hands wrapped over it,
    seen from behind), and requiring confident wrists would hide exactly the person we want to find. A person whose arms
    are merely raised also qualifies; that is accepted.
    """
    if not is_person(kp):
        return False
    w = kp[[config.LEFT_WRIST, config.RIGHT_WRIST], 1]
    sh = kp[[config.LEFT_SHOULDER, config.RIGHT_SHOULDER], 1]
    return bool(np.isfinite(w).all() and (w < sh).all())


def _centre(kp: np.ndarray) -> np.ndarray:
    pts = kp[[config.LEFT_SHOULDER, config.RIGHT_SHOULDER, config.LEFT_HIP, config.RIGHT_HIP], :2]
    pts = pts[np.isfinite(pts).all(axis=1)]
    return pts.mean(axis=0) if len(pts) else np.array([np.nan, np.nan])


def _shoulder_width(kp: np.ndarray) -> float:
    w = float(np.linalg.norm(kp[config.LEFT_SHOULDER, :2] - kp[config.RIGHT_SHOULDER, :2]))
    return w if np.isfinite(w) else 0.0


def count_people(candidates: Sequence[np.ndarray]) -> int:
    return sum(is_person(c) for c in candidates)


def select_person(candidates: Sequence[np.ndarray], previous: Optional[np.ndarray]) -> Optional[int]:
    """Index of the person to follow among this frame's detections, or None if there are none.

    People hanging on the bar are preferred (a bystander or spotter is not hanging). Among the preferred, follow the
    one nearest to where the followed person was in the previous frame; with no history, the largest (nearest the camera).
    """
    if not candidates:
        return None
    pool = [i for i, c in enumerate(candidates) if is_hanging(c)] or [i for i, c in enumerate(candidates) if is_person(c)] or list(range(len(candidates)))
    if previous is not None and np.isfinite(_centre(previous)).all():
        prev = _centre(previous)
        dist = {i: float(np.linalg.norm(_centre(candidates[i]) - prev)) for i in pool}
        dist = {i: d for i, d in dist.items() if np.isfinite(d)}
        if dist:
            return min(dist, key=dist.get)
    return max(pool, key=lambda i: _shoulder_width(candidates[i]))


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


def follow_agreement(candidates: Sequence[np.ndarray], chosen: Optional[int], followed: np.ndarray) -> int:
    """Is the person the measurement pass followed the one hanging on the bar? 1 yes, 0 no, -1 not applicable.

    Only judged on frames with two or more people and a person hanging. The followed person matches when their body
    centre is nearer to the hanging person than to any other detected person (no distance cutoff).
    """
    if chosen is None or not is_hanging(candidates[chosen]) or count_people(candidates) < 2:
        return -1
    centre = _centre(followed)
    if not np.isfinite(centre).all():
        return -1
    people = [i for i, c in enumerate(candidates) if is_person(c)]
    nearest = min(people, key=lambda i: float(np.linalg.norm(_centre(candidates[i]) - centre)))
    return 1 if nearest == chosen else 0


class MediaPipePoseBackend:
    def __init__(self, variant: str = config.POSE_MODEL_VARIANT, progress: bool = True, check_people: bool = True):
        self.variant = variant
        self.progress = progress
        self.check_people = check_people

    def _frames(self, video_path: str | Path, num_poses: int, label: str):
        """Yield (frame index, timestamp ms, mediapipe result, width, height, fps) for every frame, running the model with num_poses."""
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
            num_poses=num_poses,
            min_pose_detection_confidence=config.MIN_POSE_DETECTION_CONFIDENCE,
            min_pose_presence_confidence=config.MIN_POSE_PRESENCE_CONFIDENCE,
            min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE,
        )
        try:
            with vision.PoseLandmarker.create_from_options(options) as landmarker:
                i = 0
                while True:
                    ok, bgr = cap.read()
                    if not ok:
                        break
                    # Synthetic, strictly increasing timestamps; container PTS can be unreliable.
                    ts = int(round(i * 1000.0 / fps))
                    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                    yield i, ts, landmarker.detect_for_video(image, ts), width, height, float(fps)
                    i += 1
                    if self.progress and i % config.POSE_PROGRESS_EVERY_FRAMES == 0:
                        print(progress_line(i, total).replace("pose model", label), file=sys.stderr, flush=True)
        finally:
            cap.release()

    def extract(self, video_path: str | Path) -> PoseResult:
        """Measurement pass: ONE person per frame (num_poses=1; the calibration and every measurement were made this way),
        then, unless disabled, a people-check pass that counts the people and checks the followed one is on the bar."""
        frames, worlds, stamps = [], [], []
        width = height = 0
        fps = 0.0
        for i, ts, res, width, height, fps in self._frames(video_path, config.POSE_NUM_POSES, "pose model"):
            if res.pose_landmarks:
                frames.append(landmarks_to_array(res.pose_landmarks[0], width, height))
                worlds.append(world_to_array(res.pose_world_landmarks[0]))
            else:
                frames.append(empty_frame())
                worlds.append(np.full((config.NUM_LANDMARKS, 3), np.nan, dtype=np.float32))
            stamps.append(float(ts))
        if not frames:
            raise ValueError(f"No frames decoded from {video_path}")
        result = PoseResult(
            keypoints=np.stack(frames),
            fps=fps,
            frame_size=(width, height),
            timestamps_ms=np.asarray(stamps, dtype=np.float64),
            world=np.stack(worlds),
        )
        if self.check_people:
            result.people_count, result.follow_check = self.check_people_pass(video_path, result)
        return result

    def check_people_pass(self, video_path: str | Path, result: PoseResult) -> tuple[np.ndarray, np.ndarray]:
        """Second pass with several poses allowed: (people seen per frame, follow check per frame). Leaves the keypoints alone."""
        counts = np.zeros(result.num_frames, dtype=np.int16)
        follow = np.full(result.num_frames, -1, dtype=np.int8)
        previous = None
        for i, _, res, width, height, _ in self._frames(video_path, config.POSE_COUNT_NUM_POSES, "people check"):
            if i >= result.num_frames:
                break
            cands = [landmarks_to_array(p, width, height) for p in res.pose_landmarks]
            counts[i] = count_people(cands)
            chosen = select_person(cands, previous)
            if chosen is not None:
                previous = cands[chosen]
            follow[i] = follow_agreement(cands, chosen, result.keypoints[i])
        return counts, follow


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
    if result.people_count is not None:
        arrays["people_count"] = result.people_count
    if result.follow_check is not None:
        arrays["follow_check"] = result.follow_check
    np.savez_compressed(path, **arrays)


def load_pose(path: str | Path) -> PoseResult:
    with np.load(path) as z:
        return PoseResult(
            keypoints=z["keypoints"],
            fps=float(z["fps"]),
            frame_size=(int(z["frame_size"][0]), int(z["frame_size"][1])),
            timestamps_ms=z["timestamps_ms"],
            world=z["world"] if "world" in z.files else None,
            people_count=z["people_count"] if "people_count" in z.files else None,
            follow_check=z["follow_check"] if "follow_check" in z.files else None,
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

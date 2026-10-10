"""Camera-setup checks from the keypoints: warn about a side view or a tilted bar.

The head and elbow measurements assume a front or back view of the athlete and a level bar. These checks only
raise warnings; they never change a measurement. Cutoffs are provisional (config.py) and were read off 9 dev clips.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from . import config
from .pose import PoseResult


@dataclass(frozen=True)
class CameraChecks:
    shoulder_torso_ratio: Optional[float]  # median shoulder width / torso length; small = seen from the side
    wrist_gap_body_scales: Optional[float]  # median left-right wrist height difference while hanging; large = tilted bar or side view
    multi_person_share: Optional[float] = None  # share of frames with two or more people; None = this keypoints file does not count people
    wrong_person_share: Optional[float] = None  # of the frames judged (several people, one hanging), the share where the followed person is NOT the one on the bar


def camera_checks(pose: PoseResult) -> CameraChecks:
    kp, m = pose.keypoints, config.MIN_LANDMARK_CONFIDENCE
    height = pose.frame_size[1]
    multi = None if pose.people_count is None else float((pose.people_count >= 2).mean())
    wrong = None
    if pose.follow_check is not None and (pose.follow_check >= 0).any():
        wrong = float((pose.follow_check[pose.follow_check >= 0] == 0).mean())

    def ok(j: int) -> np.ndarray:
        return (kp[:, j, 2] >= m) & np.isfinite(kp[:, j, :2]).all(axis=1)

    body = ok(config.LEFT_SHOULDER) & ok(config.RIGHT_SHOULDER) & ok(config.LEFT_HIP) & ok(config.RIGHT_HIP)
    if not body.any():
        return CameraChecks(None, None, multi, wrong)
    shoulder_w = np.linalg.norm(kp[:, config.LEFT_SHOULDER, :2] - kp[:, config.RIGHT_SHOULDER, :2], axis=1)
    mid_s = (kp[:, config.LEFT_SHOULDER, :2] + kp[:, config.RIGHT_SHOULDER, :2]) / 2
    mid_h = (kp[:, config.LEFT_HIP, :2] + kp[:, config.RIGHT_HIP, :2]) / 2
    torso = np.linalg.norm(mid_s - mid_h, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.where(body & (torso > 0), shoulder_w / torso, np.nan)
    ratio_med = float(np.nanmedian(ratio)) if np.isfinite(ratio).any() else None

    wrists = ok(config.LEFT_WRIST) & ok(config.RIGHT_WRIST) & body
    wrist_h = (2 * height - kp[:, config.LEFT_WRIST, 1] - kp[:, config.RIGHT_WRIST, 1]) / 2
    shoulder_h = height - mid_s[:, 1]
    hanging = wrists & (wrist_h > shoulder_h)  # wrists above the shoulders: on the bar
    gap_med = None
    usable = hanging & (shoulder_w > 0)  # a frame with the shoulders on top of each other cannot give a ratio
    if usable.any():
        gap = np.abs(kp[:, config.LEFT_WRIST, 1] - kp[:, config.RIGHT_WRIST, 1]) / np.where(shoulder_w > 0, shoulder_w, 1.0)
        gap_med = float(np.median(gap[usable]))
    return CameraChecks(ratio_med, gap_med, multi, wrong)


def camera_warnings(checks: CameraChecks) -> list[str]:
    out: list[str] = []
    if checks.shoulder_torso_ratio is not None and checks.shoulder_torso_ratio < config.WARN_SIDE_VIEW_RATIO:
        out.append("WARNING: this looks like a side view - film from the front or back; head and elbow readings are unreliable")
    if checks.wrist_gap_body_scales is not None and checks.wrist_gap_body_scales > config.WARN_BAR_TILT_WRIST_GAP:
        out.append("WARNING: the wrists are at very different heights (tilted bar or side view) - the bar line may be wrong")
    if checks.multi_person_share is not None and checks.multi_person_share >= config.WARN_MULTI_PERSON_SHARE:
        out.append(
            f"WARNING: {100 * checks.multi_person_share:.0f}% of frames show more than one person - the tool tracks one person (the pose model's own choice, not necessarily the one hanging on the bar); "
            "check the video that it is the right person"
        )
    if checks.wrong_person_share is not None and checks.wrong_person_share >= config.WARN_WRONG_PERSON_SHARE:
        out.append(
            f"WARNING: in {100 * checks.wrong_person_share:.0f}% of the frames with several people the tracked person is NOT the one hanging on the bar - "
            "the readings may be for the wrong person; film with only the lifter in frame"
        )
    return out

"""All tunable constants. Units are in the name or a comment.

Rep-logic thresholds (head-over-bar margin, dead-hang elbow angle, smoothing
window, debounce) are NOT set here yet. They are derived from labelled
calibration footage (see docs/PROGRESS.md) and added with provenance.
"""

# --- Pose model (MediaPipe Pose Landmarker, Tasks API) ---
POSE_MODEL_VARIANT = "heavy"  # "lite" | "full" | "heavy"; offline processing, so favour accuracy
POSE_MODEL_URL_TEMPLATE = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_{variant}/float16/latest/pose_landmarker_{variant}.task"
)
POSE_MODEL_DIR = "models"  # relative to the working directory; *.task is gitignored

# MediaPipe library defaults (0..1). Not tuned; to calibrate on real footage.
MIN_POSE_DETECTION_CONFIDENCE = 0.5
MIN_POSE_PRESENCE_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5

# --- BlazePose 33-landmark indices (verified against mediapipe PoseLandmark enum) ---
NUM_LANDMARKS = 33
NOSE = 0
LEFT_EYE, RIGHT_EYE = 2, 5
LEFT_EAR, RIGHT_EAR = 7, 8
MOUTH_LEFT, MOUTH_RIGHT = 9, 10
LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_ELBOW, RIGHT_ELBOW = 13, 14
LEFT_WRIST, RIGHT_WRIST = 15, 16
LEFT_HIP, RIGHT_HIP = 23, 24

# --- Metrics (slice 2) ---
# EVERYTHING IN THIS BLOCK IS PROVISIONAL: a placeholder so the code runs, not a
# value derived from footage. Each gets replaced, with provenance, once labelled
# calibration clips exist (see docs/PROGRESS.md).
MIN_LANDMARK_CONFIDENCE = 0.5  # PROVISIONAL, 0..1 visibility; below this a joint counts as missing
MAX_GAP_S = 0.2  # PROVISIONAL, seconds; shorter gaps are interpolated, longer ones stay missing
SMOOTH_WINDOW_S = 0.3  # PROVISIONAL, seconds; Savitzky-Golay window (rounded to odd frames)
SMOOTH_POLYORDER = 2  # PROVISIONAL, Savitzky-Golay polynomial order
HEAD_LANDMARKS = (NOSE,)  # PROVISIONAL; averaged if several. Alternatives: ears, mouth corners
BODY_SCALE_MODE = "shoulder_width"  # PROVISIONAL: "shoulder_width" | "torso" (mid-shoulder to mid-hip)
BODY_SCALE_WINDOW_S = 2.0  # PROVISIONAL, seconds; rolling median so pose changes don't move the scale
BAR_WINDOW_S = 2.0  # PROVISIONAL, seconds; centred rolling-median window for the bar line
BAR_MIN_CONFIDENT_SHARE = 0.25  # PROVISIONAL, 0..1; min share of the bar window with confident gripping wrists, else no bar value
BAR_OFFSET_BODY_SCALES = 0.0  # PROVISIONAL; wrist-joint-to-bar-top offset, in body scales, to measure
ELBOW_ANGLE_SOURCE = "world"  # PROVISIONAL: "world" (3D, view-independent) | "image" (2D pixels)
ELBOW_ARM_COMBINE = "mean"  # PROVISIONAL: "mean" | "min" of the two arms

# --- Rep logic (slice 3) ---
# ALL PROVISIONAL placeholders, none derived from footage yet (see docs/PROGRESS.md).
HEAD_CLEAR_MARGIN_BODY_SCALES = 0.0  # PROVISIONAL; head_minus_bar must reach this at the top. 0 = nose at the wrist line; BlazePose has no chin
# Dead-hang elbow angle (deg) a rep's bottom must reach, PER ANGLE SOURCE. Deliberately UNSET: the 3D and 2D angles
# read very differently (3D never exceeded 171 deg on the dev clips), so each needs its own value calibrated on
# labelled footage. While None, a rep that clears the bar is reported as UNCALIBRATED, not FULL/NOT FULL.
DEAD_HANG_ELBOW_MIN_DEG: dict[str, "float | None"] = {"world": None, "image": None}
REP_START_RISE_BODY_SCALES = 0.25  # PROVISIONAL; head must rise this far above its running bottom to start a rep
REP_END_DROP_BODY_SCALES = 0.25  # PROVISIONAL; head must fall this far below the rep's peak to end the ascent
REP_DEBOUNCE_S = 0.1  # PROVISIONAL, seconds; a rise/fall must hold this long before the state changes
REP_BOTTOM_BAND_BODY_SCALES = 0.15  # PROVISIONAL; frames within this of a valley's lowest head height form its "bottom"
REP_MIN_ASCENT_S = 0.3  # PROVISIONAL, seconds; start-to-peak shorter than this is a tracking glitch, not a rep (discarded, listed)
REP_MAX_GAP_S = 0.5  # PROVISIONAL, seconds; a missing-data gap longer than this mid-rep discards that attempt

# --- Rendering and CLI (slice 4) ---
RENDER_VIDEO_MAX_HEIGHT_PX = 720  # the source video is drawn at most this tall (never enlarged)
RENDER_PANEL_WIDTH_PX = 520  # side panel with the rep counter and table
RENDER_GRAPH_HEIGHT_PX = 170  # head-height graph strip under the video
RENDER_MIN_PANEL_HEIGHT_PX = 320  # the side panel is at least this tall, so a very short source does not squash it
RENDER_GRAPH_MARGINS_PX = (52, 12, 38, 22)  # graph plot area margins: left, right, top, bottom
RENDER_TABLE_TOP_PX = 118  # y of the table header in the side panel
RENDER_TABLE_ROW_H_PX = 26
RENDER_TABLE_COLUMNS_X = (12, 36, 136, 246, 346, 424)  # x of: rep #, head over bar, arms straight, bottom elbow, pull time, verdict
RENDER_BANNER_H_PX = 26  # warning banner height on the video
RENDER_FRAME_MATCH_TOLERANCE = 2  # frames of slack between a video's reported length and its keypoints (containers disagree by one)
WARN_INVALID_SHARE = 0.30  # PROVISIONAL, 0..1; when this share of frames has no usable head or elbow signal, the video and CLI warn

# --- Clip evaluation (diagnostic report only; never feeds rep logic) ---
HEAD_AGREEMENT_LANDMARKS = (NOSE, LEFT_EYE, RIGHT_EYE, LEFT_EAR, RIGHT_EAR)  # heights compared for the head-spread diagnostic
EVAL_DRIFT_SKIP_S = 1.0  # seconds ignored at the start of a clip for the bar-drift number (the person is still reaching up)
EVAL_VIDEO_SUFFIXES = (".mp4", ".webm", ".mov", ".mkv", ".m4v")
EVAL_CACHE_DIR = "data/cache"  # keypoint .npz per clip so re-runs skip the pose model
EVAL_PLOT_DIR = "data/plots"  # per-clip debug plot + contact sheet written by eval_clips

# Pairs of landmark indices for the throwaway debug preview skeleton.
PREVIEW_BONES = (
    (LEFT_SHOULDER, RIGHT_SHOULDER),
    (LEFT_SHOULDER, LEFT_ELBOW), (LEFT_ELBOW, LEFT_WRIST),
    (RIGHT_SHOULDER, RIGHT_ELBOW), (RIGHT_ELBOW, RIGHT_WRIST),
    (LEFT_SHOULDER, LEFT_HIP), (RIGHT_SHOULDER, RIGHT_HIP), (LEFT_HIP, RIGHT_HIP),
    (MOUTH_LEFT, MOUTH_RIGHT), (LEFT_EAR, NOSE), (NOSE, RIGHT_EAR),
)

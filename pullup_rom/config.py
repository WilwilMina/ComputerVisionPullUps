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
# Wrist-landmark-to-bar-CENTRE offset, in body scales (shoulder widths), bar above the wrist line. A GEOMETRY measurement, not tuned to
# any label: the hands wrap over the bar, so the bar sits above the wrist landmark. Measured 2026-10-09 by reading the bar against a pixel
# ruler on 22 frames of two clips, two people (pexels_6389831 mean 0.237, pexels_10336041 mean 0.218; pooled mean 0.228, sd 0.022, range
# 0.191-0.266; readings good to about 2-5 px; see docs/PROGRESS.md). Rounded to 0.23. It is the bar's centre, not its top edge (the top
# edge is about 0.06 higher); it differs a little per person and grip, so it is one estimate from two clips, not a constant of nature.
BAR_OFFSET_BODY_SCALES = 0.23
ELBOW_ANGLE_SOURCE = "world"  # PROVISIONAL: "world" (the model's estimated 3D angle) | "image" (2D pixels, depends on the camera view)
ELBOW_ARM_COMBINE = "mean"  # PROVISIONAL: "mean" | "min" of the two arms

# --- Rep logic (slice 3) ---
# ALL PROVISIONAL placeholders, none derived from footage yet (see docs/PROGRESS.md).
HEAD_CLEAR_MARGIN_BODY_SCALES = 0.0  # PROVISIONAL; head_minus_bar must reach this at the top. 0 = nose at the wrist line; BlazePose has no chin
# Dead-hang elbow angle (deg) a rep's bottom must reach, PER ANGLE SOURCE (a rep passes at or above it).
# "world" (3D): 144.0. This is a MODEL-ESTIMATED 3D elbow angle, not a true elbow angle: it reads ~145-155 on arms
#   that look nearly straight. Calibrated on 2026-10-09 from the user's labels: every rep in pexels_10336041 and
#   pexels_6389831 is a full-ROM rep. The 13 complete reps there have bottom angles 144.86 to 154.60 deg (3D);
#   144.0 is the lowest whole degree at or below all of them, i.e. the strictest value that keeps every
#   labelled rep FULL. ONE-SIDED: there were no bent-elbow examples, so it says nothing about where bent starts,
#   it only cannot be set higher. Two clips, two people. Re-derive with labelled bad reps and a held-out clip.
# "image" (2D): left unset. The same reps read 129.9 to 161.3 deg in 2D (the two clips differ by about 25 deg
#   because the 2D angle depends on the camera view), so no 2D value can be set from this data.
# A rep that clears the bar while its source has no threshold is reported UNKNOWN, never guessed.
DEAD_HANG_ELBOW_MIN_DEG: dict[str, "float | None"] = {"world": 144.0, "image": None}
REP_START_RISE_BODY_SCALES = 0.25  # PROVISIONAL; head must rise this far above its running bottom to start a rep
REP_END_DROP_BODY_SCALES = 0.25  # PROVISIONAL; head must fall this far below the rep's peak to end the ascent
REP_DEBOUNCE_S = 0.1  # PROVISIONAL, seconds; a rise/fall must hold this long before the state changes
REP_BOTTOM_BAND_BODY_SCALES = 0.15  # PROVISIONAL; frames within this of a valley's lowest head height form its "bottom"
REP_MIN_ASCENT_S = 0.3  # PROVISIONAL, seconds; start-to-peak shorter than this is a tracking glitch, not a rep (discarded, listed). Not a form judgement
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
RENDER_BANNER_LINE_H_PX = 20  # height of one line of the warning banner on the video
RENDER_BANNER_MAX_LINES = 8  # the banner wraps long warnings to the video width, up to this many lines
RENDER_ELBOW_READOUT_MARGIN_PX = 8  # the live elbow readout sits this far from the video's bottom-left corner
RENDER_ELBOW_READOUT_SIZE = 0.5  # its text size (OpenCV font scale)
RENDER_FRAME_MATCH_TOLERANCE = 2  # frames of slack between a video's reported length and its keypoints (containers disagree by one)
WARN_INVALID_SHARE = 0.30  # PROVISIONAL, 0..1; when this share of frames has no usable head or elbow signal, the video and CLI warn

# --- Camera-setup warnings (provisional; read off the 9 dev clips, one positive example each) ---
WARN_SIDE_VIEW_RATIO = 0.40  # PROVISIONAL; shoulder width / torso length below this = side view. Dev clips: front/back 0.46-0.62, side view 0.33
WARN_BAR_TILT_WRIST_GAP = 0.50  # PROVISIONAL; median left-right wrist height gap (body scales) above this = tilted bar or side view. Dev clips 0.04-0.22, tilted/side 0.83
POSE_PROGRESS_EVERY_FRAMES = 100  # the pose model prints a progress line this often

# --- Clip evaluation (diagnostic report only; never feeds rep logic) ---
LABEL_SNAP_WINDOW_S = 1.0  # a hand-labelled top time is matched to the nearest detected rep peak within this many seconds (visual times are good to ~0.3 s; the peak is not exactly 'chin at the bar')
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

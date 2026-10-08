# Architecture rules

- Pipeline: video -> `PoseBackend` (keypoints per frame) -> `metrics` (smoothed signals) -> `reps` (state machine) -> `render`. Each stage takes and returns plain data (NumPy arrays or dataclasses) so it can be tested alone.
- `reps.py` is pure: no OpenCV, no file I/O, no model imports. It consumes time series (head height, elbow angle, fps) and returns a list of rep records.
- `PoseBackend.extract(video_path)` returns keypoints shaped `[frames, joints, (x, y, confidence)]` plus fps. MediaPipe is the default. ViTPose or YOLO-pose must be swappable without touching other modules.
- Image y grows downward. Convert to "height up" exactly once, in `metrics.py`.
- Bar line is estimated from wrist position while hanging (the wrists stay on the bar). Only ask the user to annotate it if auto-estimation proves unreliable on real footage.
- Smooth signals before applying thresholds (Savitzky-Golay or moving average, window set in `config.py`). Debounce state changes with a minimum number of frames so one rep is never counted twice.
- Frames with no person or low confidence are interpolated or skipped explicitly, never silently treated as zeros.
- Every threshold is a named constant in `config.py` with its unit in the name or a comment. Per-person calibration overrides go through config, not code edits.

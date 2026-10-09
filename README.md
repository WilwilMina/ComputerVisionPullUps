# Pull-up ROM Detector

Takes a video of someone doing pull-ups, counts the reps, and checks each one for full range of motion (ROM):
did the head clear the bar, and did the arms get back to a dead hang. It writes an annotated video (skeleton,
rep counter, head-height graph, per-rep table) and a JSON summary.

**Status: work in progress.** Every threshold is a provisional placeholder in `pullup_rom/config.py`. The one
calibration so far is the dead-hang elbow threshold (144) for the model-estimated 3D elbow angle, set from two
stock clips whose reps the project owner labelled as full range of motion. That angle is the pose model's
estimate, not a true joint angle: it reads about 145-155 on arms that look nearly straight. It is one-sided: with no bent-elbow examples it is the
strictest value that keeps those reps FULL, and it says nothing about where a bent elbow starts. The 2D angle has
no threshold (it depends on the camera view), so with `--source image` the arms check shows **UNKNOWN**. This
repository makes **no accuracy claims**: nothing has been validated on held-out, labelled reps yet. There are still **no validated bad-rep examples**: bent-elbow-bottom and chin-short-of-bar detection are untested on real footage, and on two other labelled clips the 3D arms check disagreed with most labelled-FULL reps, so the threshold does not yet transfer between people and camera views.

## How a rep is judged
- The pose model (MediaPipe Pose Landmarker, behind a swappable `PoseBackend` interface) finds body points per frame.
- The bar line is estimated from the wrists while hanging; head height is measured against it in body-scale
  units, so it does not depend on image size.
- A rep runs from the bottom of a hang, up to the peak, and back to the bottom.
- **FULL** needs both: the head cleared the bar, and the bottom elbow angle reached the dead-hang threshold.
  A failed head check is NOT FULL. If the angle source has no calibrated threshold (currently the 2D angle), or the clip ends
  before the arms come back down, a rep that clears the bar is UNKNOWN.
  The elbow angle at the top of a rep is never used.
- **Range of motion only, not speed or tempo.** Pull time is shown in the table for information. It never changes a
  verdict or a colour. The only timing rule is a tracking filter: a movement whose rise lasts under
  `REP_MIN_ASCENT_S` is discarded as a glitch (and listed), not judged as a fast rep.
- Reps the tool cannot trust (tracking gaps) are listed as "uncertain", never silently dropped.

## Install
Python 3.13.

```
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The pose model file (about 30 MB, `models/pose_landmarker_heavy.task`) is downloaded from Google's public model
storage the first time the pose model runs.

## Run
```
.venv\Scripts\python.exe -m pullup_rom.cli your_video.mp4 -o out\result.mp4
```
It writes `out\result.mp4`, `out\result.json` and a keypoints cache `out\result.keypoints.npz`, and prints the rep
table. The pose model runs on the CPU and takes a while on long clips; it prints a progress line every 100
frames. Re-run in seconds with `--keypoints out\result.keypoints.npz`. Other options: `--source world|image`
(3D or 2D elbow angle) and `--dead-hang-deg X` (only for a value you calibrated for that source).

If a video will not open, re-export it as a standard H.264 `.mp4`.

## Reading the video
Left: the person, a green skeleton and a dashed bar line, and a live elbow readout at the bottom-left (labelled a model estimate; hidden on frames where the angle is not valid). Top right: the rep counter. Side table: rep number,
head over the bar (yes/no), arms straight (yes/no/UNKNOWN), bottom elbow angle, pull time, verdict. Bottom: head
height against the bar, with rep bands (green FULL, red NOT FULL, amber UNKNOWN). A red banner appears when many
frames could not be tracked, or when the camera looks like a side view or the bar looks tilted.

## Filming tips
Fixed camera on a tripod, front or back view (not the side), whole body and the bar in frame with the bar level,
even light, nothing in front of the bar, and the person large enough in frame.

## Limitations
- Calibrated on two clips only (see Status) and not validated: no accuracy figures exist. The 3D elbow angle varies
  with the person and the camera view, so one threshold from two clips may not transfer.
- Needs a front or back view. Side views and tilted bars are flagged but not supported.
- One person is tracked. The 3D elbow angle never reaches a perfectly straight 180 degrees on the footage tried
  so far, so the dead-hang threshold has to be found from data, separately for the 3D and 2D angle.
- BlazePose has no chin point; the nose stands in for the head.

## Development
```
.venv\Scripts\python.exe -m pytest -q            # fast suite, synthetic data only
.venv\Scripts\python.exe -m pytest -q -m slow    # throwaway-git-repo and end-to-end render tests
```
`pullup_rom/` holds the pipeline (`pose` -> `metrics` -> `reps` -> `render`, plus `cli`). `reps.py` is pure.
Project rules and workflow are in `CLAUDE.md` and `.claude/rules/`; progress notes are in `docs/PROGRESS.md`.
Videos, model files and data are never committed; a commit and push security scan
(`tools/scan_sensitive.py`, enabled with `git config core.hooksPath .githooks`) enforces that.
Development test clips are for local testing only and are not part of this repository.

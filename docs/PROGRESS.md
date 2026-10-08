# Progress

Source of truth for project state across sessions. Conversation resume is unreliable, so update this at the end of each session.

## Status
Slices 1 (pose), 2 (metrics) and 3 (rep state machine) are built and tested on synthetic data. Slices 1-2 have been run on 9 stock dev clips of different people (tracking diagnostics only, no ground truth). Slice 3 has run on those clips only to print reps and bottom elbow angles for inspection. Waiting for the user's check before slice 4 (renderer + CLI). No footage of the user yet, so every setting is a placeholder and no rep is labelled FULL or NOT FULL on elbow grounds (dead-hang threshold unset).

## Done
- Workspace set up per `docs/reference/` guide: `CLAUDE.md`, 2 rules, `qa` and `researcher` agents, one test hook, `.gitignore`, `requirements.txt`.
- **Slice 1** `pullup_rom/pose.py`: `PoseResult` (pixel keypoints `[F,33,(x,y,conf)]` + MediaPipe `world` 3D landmarks + fps), `PoseBackend` Protocol, `MediaPipePoseBackend` (heavy model, VIDEO mode), `.npz` cache, CLI with `--preview`. Frames with no person are NaN/conf 0, never zeros.
- **Slice 2** `pullup_rom/metrics.py` -> `Signals` (see "Slice 2 validity change" below for the current rules):
  - Low-confidence joints become NaN; gaps up to `MAX_GAP_S` are interpolated for the head, shoulders, elbows and body scale (NOT for the wrists that build the bar). Savitzky-Golay smoothing per contiguous valid run.
  - The one y flip to height-up happens in `compute_signals` (`up()`); angles do not need it.
  - `head_minus_bar` is divided by a body scale (shoulder width or torso length, rolling median) so it is unitless.
  - Elbow angle from both 3D world landmarks and 2D pixels; `ELBOW_ANGLE_SOURCE` picks which one the rep logic sees. `Signals.elbow_arms_used` flags which arm(s) fed each frame.
  - `pullup_rom/debug_plot.py`: head/bar/elbow plot. Grey = not valid, orange = bar carried from neighbouring frames.
- **Slice 2 validity change (this session).** `head_minus_bar` no longer needs visible wrists in the same frame:
  - The bar line is a rolling median (`BAR_WINDOW_S`) of wrist height over frames where a wrist was CONFIDENT and above the shoulders. Wrist positions are never interpolated or guessed for the bar.
  - A bar value needs at least `BAR_MIN_CONFIDENT_SHARE` of its window to be confident gripping frames, otherwise there is no bar value.
  - `Signals.bar_carried` marks frames that have a bar value although their own wrists were not confident; `bar_support` is the share per frame. `reps.Rep.peak_bar_carried` passes this to slice 3.
  - Only frames between the first and last confident grip count (`in_hang_span`). `head_valid` and `elbow_valid` replace the old single `valid`; the elbow keeps its own validity.
  - With the defaults the longest wrist loss the bar can bridge is about 1.5 s (window 2 s, share 0.25).
- **Slice 3** `pullup_rom/reps.py` (pure: numpy and config only): `detect_reps(head_minus_bar, elbow_deg, fps, RepParams, bar_carried)` -> `RepAnalysis(reps, discarded)`.
  - Hysteresis state machine on head height with debounce; a rep runs from the head's lowest point before the pull, through the peak, to its lowest point after lowering. The valley between two reps ends one and starts the next.
  - The elbow angle is read ONLY at the bottom: the MEDIAN angle over the bottom region (contiguous frames within `REP_BOTTOM_BAND_BODY_SCALES` of the valley's lowest head height) before and after the rep; `bottom_elbow_deg` is the lower (less extended) of the two. A median, not a maximum, because QA showed a maximum lets angle noise lift a bent bottom over the threshold (190 of 200 noisy reps); tested with 40 noise seeds both ways. The elbow at the top is never used (tested by overwriting it).
  - A rep's `start_frame` is the last frame at the bottom before the pull; its `end_frame` is the FIRST frame the head is back within `REP_BOTTOM_BAND_BODY_SCALES` of the low level the rep started from (not the lowest point of a long hang, which drifted late on 14293451). If it never returns that far it falls back to the region around its lowest point. `Rep.pull_time_s` = start to end (bottom to top to bottom; None if the clip ended mid-rep), `Rep.ascent_s` = bottom to top. Ascents shorter than `REP_MIN_ASCENT_S` (start to peak) are discarded as glitches (QA showed a 0.1 s blip counted as a rep).
  - FULL needs head clear AND bottom elbow at or above the dead-hang threshold. The threshold is `config.DEAD_HANG_ELBOW_MIN_DEG`, a per-source dict that is UNSET (`None`). While unset, a rep that clears the bar has `full=None` (shown UNKNOWN); a rep whose head fails is `full=False`. No "near 180 deg" value exists anywhere.
  - A missing-data gap longer than `REP_MAX_GAP_S` during a rep is kept conservative (user decision, no threshold changed to rescue the stock clips; revisit on the user's own footage), but never dropped silently: it is returned in `RepAnalysis.discarded` and printed as `uncertain (tracking gap)` with its time range. Glitches are listed as `ignored (too short to be a rep)`. A clip that ends mid-rep is flagged.
  - `pullup_rom/rep_report.py` prints each rep with start/end/bottom elbow angles: `python -m pullup_rom.rep_report data/cache/<clip>.npz [--source world|image]`; `eval_clips --reps` prints it per clip.
- **Evaluation** `python -m pullup_rom.eval_clips data/external [--cached-only] [--reps] [--no-plots]` (not in pytest; needs the model and videos; caches keypoints in `data/cache/`, writes `data/plots/`). Per clip: person %, invalid head / elbow / either %, bar carried %, median confidence for head/shoulders/elbows/wrists, head-landmark spread (nose, eyes, ears: how much they disagree on height, in body scales; agreement is not correctness, since from behind the face points are guessed), bar drift (first `EVAL_DRIFT_SKIP_S` seconds excluded) and scatter, elbow angle ranges for both sources, one-arm share. The old `head@top` column was removed: nose confidence stays high even from behind.
- Tests: 154 in the default suite (about 2.5 s); `tests/synthetic.py` builds 2D/3D-consistent skeletons and rep series. The 6 required cases in `.claude/rules/testing.md` are in `tests/test_reps.py` (case 5 runs 25 noise seeds). `.claude/rules/testing.md` budget relaxed by the user to about 3 s.
- Agents (three, the budget; user decision: add no more): `qa` (runs the tests and tries to break the code on synthetic inputs and edge cases; checks the rules), `researcher` (external lookups), and `clip-evaluator` (`.claude/agents/clip-evaluator.md`). Jobs `clip-evaluator` does that `qa` does not: re-run the evaluation over the 9 real dev clips from their keypoint caches, LOOK at the per-clip plots and contact sheets (camera motion, occlusion, cropping, lighting, tilted bar), compare before/after against the last report, and say whether printed rep counts look plausible against the plot. `qa` has no real-footage job; it judges code, not tracking. A new agent file is only picked up after a Claude Code restart; until then spawn a general-purpose agent told to follow that file.

## Dev clip set (development testing only: never in the demo, README or posts)
9 clips in `data/external/` (8 Pexels, 1 Wikimedia Commons), each in its own directory; provenance, license and SHA-256 in `data/external/SOURCES.md` (allowed hosts there include `videos.pexels.com`). All HTTPS, under 100 MB, confirmed as real video before the pose model ran. `data/` and `samples/` are gitignored.

### Invalid frames before and after the validity change (percent of frames)
"Before" = old combined `valid` (head, elbow and wrists in the same frame). "After" = new head_valid, elbow_valid and either-invalid. Same clips, same cached keypoints, all other settings unchanged.

| clip | before | after: head | after: elbow | after: either |
|---|---|---|---|---|
| 10336041 (indoor gym, back) | 19.7 | 6.5 | 7.2 | 7.2 |
| 14293451 (outdoor, small subject) | 0.0 | 0.0 | 0.0 | 0.0 |
| 26586395 (outdoor posts, back) | 43.4 | 12.7 | 6.6 | 16.8 |
| 35585615 (handheld close-up) | 72.4 | 44.0 | 72.4 | 73.3 |
| 38226256 (dark, backlit, hat) | 85.3 | 94.8 | 84.8 | 100.0 |
| 4964660 (street bar, back) | 17.9 | 0.0 | 2.5 | 2.5 |
| 7672115 (dark gym, moving camera) | 53.7 | 36.2 | 50.9 | 52.0 |
| 8233829 (outdoor, side view, tilted bar) | 37.6 | 23.5 | 29.2 | 33.7 |
| wikimedia demo (back view, title card) | 0.6 | 0.6 | 0.6 | 0.6 |

- The three badly tracked clips were not rescued on purpose. `38226256` got worse (head 85.3 -> 94.8) because its wrists are almost never confident, so the share rule refuses to build a bar. Keep them as examples of input the tool should eventually warn about.
- `BAR_MIN_CONFIDENT_SHARE` is a placeholder and matters. Head-invalid % at shares 0.10 / 0.25 / 0.50 / 0.75: 26586395: 0.0 / 12.7 / 47.2 / 77.5; 4964660: 0.0 / 0.0 / 7.0 / 56.1; 38226256: 35.1 / 94.8 / 100 / 100; 35585615: 34.2 / 44.0 / 64.8 / 83.0. A low share would also rescue the bad clips, which is the wrong reason to pick it.

### Tracking findings (not accuracy)
- Tracked well (fixed camera, back view, whole body and bar in frame, decent light): `4964660`, `26586395`, `10336041`, and the Wikimedia demo (title card at the start has no person). `14293451` tracks but the person is small: elbow confidence median 0.69, 38% of elbow-valid frames single-arm.
- Tracked badly: `38226256` (close, low back view, backlight, lamp glare, dark sleeves, hat; wrist/elbow confidence about 0.3), `35585615` (handheld close-up, wrists/arms out of frame, person leaves frame), `7672115` (panning camera, foreground pole, low light, person partly out of frame), and `8233829` (side view, tilted bar, person cropped; bar drift 2.60 and scatter 0.69 body scales: the mean of two wrists at different heights is not the bar line when the bar is slanted or the view is from the side).
- Elbow source: at the hang, 3D reads about 141-165 deg and 2D about 150-178 deg; 3D never exceeded 171 deg on any clip. At the top of a rep 2D collapses to about 0-40 deg while 3D stays higher. Thresholds must be calibrated per source.
- Recommended camera setup (from these clips): fixed tripod, back or front view (not side), whole body and bar in frame, bar level, person not tiny, even light, nothing in front of the bar, no backlight.

### Reps printed on the dev clips (final code, 3D source, dead-hang threshold unset so no rep is FULL; not accuracy)
| clip | reps printed | bottom elbow 3D (deg, median) | notes |
|---|---|---|---|
| 10336041 | 10 | 145-155; #10: 95 | #1 is a small bump below the bar; #10 is a long top-hold (peak +0.29), not a clean rep |
| 14293451 | 3 | 149-157 | rep 3's end time is 19.1 s although the head is back down by about 11 s (see limitations) |
| 26586395 | 3 (+1 discarded) | 136-141 | plot shows 4 peaks; the fourth was discarded because the head signal had a gap over 0.5 s at the top |
| 4964660 | 3 | 157-158 | matches the 3 clean peaks |
| wikimedia | 2 | 138-155 | matches the 2 peaks |
| 35585615 | 4 | 140-167, one n/a | poor tracking; weakly supported |
| 7672115 | 5 (+1 discarded) | 87-154, some n/a | poor tracking; sits on fragmented signal |
| 38226256 | 0 | none | no usable bar line, as intended |
| 8233829 | 0 (2 discarded) | none | side view, slanted bar |
The highest bottom elbow angle printed on any clip is 167 deg (3D); none reaches 171, let alone 180. Every rep that clears the bar shows UNKNOWN until the per-source dead-hang threshold is calibrated.

## PROVISIONAL (placeholders, not calibrated; all in `config.py`, marked)
Metrics: `MIN_LANDMARK_CONFIDENCE`, `MAX_GAP_S`, `SMOOTH_WINDOW_S`, `SMOOTH_POLYORDER`, `HEAD_LANDMARKS` (nose), `BODY_SCALE_MODE`, `BODY_SCALE_WINDOW_S`, `BAR_WINDOW_S`, `BAR_MIN_CONFIDENT_SHARE` (0.25), `BAR_OFFSET_BODY_SCALES` (0.0), `ELBOW_ANGLE_SOURCE` (world), `ELBOW_ARM_COMBINE` (mean).
Reps: `HEAD_CLEAR_MARGIN_BODY_SCALES` (0.0), `REP_START_RISE_BODY_SCALES`, `REP_END_DROP_BODY_SCALES`, `REP_DEBOUNCE_S`, `REP_BOTTOM_BAND_BODY_SCALES`, `REP_MIN_ASCENT_S`, `REP_MAX_GAP_S`. `DEAD_HANG_ELBOW_MIN_DEG` is deliberately unset for both sources.
None of these may appear in the README as chosen values.

## Known limitations
- Rolling-median edges: the first/last half window of a clip is biased toward the interior under camera drift (at most slope x window / 4).
- The wrist joint is not the top of the bar. `BAR_OFFSET_BODY_SCALES` exists for this and must be measured. A mean of two wrists is also wrong for a tilted bar or a side view (8233829).
- Head landmark: BlazePose has no chin; the nose is a placeholder, and from behind the face points are guessed. Head-spread agreement does not prove they are right.
- If one forearm is missing the elbow angle uses the other arm (flagged in `elbow_arms_used`, nothing yet acts on it). If a whole body side is missing, body scale is NaN. If any `HEAD_LANDMARKS` entry is missing the head signal is NaN.
- Rep detection is untested on real footage beyond the printed tables; the start-rise, end-drop and bottom-band values are placeholders and were not chosen from measured hang jitter.
- Rep end time is now the first return to the starting low level. If the head never returns within `REP_BOTTOM_BAND_BODY_SCALES` of it (a rep that finishes high), the end falls back to the region around its lowest point.
- A head-signal gap longer than `REP_MAX_GAP_S` mid-rep still makes that attempt uncertain: the bar is NaN because too few wrists were confident. This costs a real rep on 26586395 and 7672115 (now listed as `uncertain (tracking gap)` with a time range, not hidden). Revisit on the user's own footage.
- A rep's bottom is the less extended of the valleys before and after it, so one bent valley fails both neighbouring reps (by design, as a rep must start from and return to dead hang).

## Next
1. User checks slice 3 (rep tables per clip, bottom elbow angles).
2. Slice 4: overlay renderer + CLI (commit slices 1-3 locally first, see Git below).
3. User records tripod footage including deliberately bad reps (bent-elbow bottom, chin short of bar) plus a dead-hang segment, with a per-rep label CSV. Then calibrate every PROVISIONAL value, the per-source dead-hang threshold included, with provenance, and hold out a clip for validation.

## Constraints and decisions
- Sandbox proxy blocks model downloads (`storage.googleapis.com` returns 403). Pose extraction must run on the user's machine.
- MediaPipe 1.1.0 is installed here; the Tasks API (`PoseLandmarker`) is unchanged. Model file: `models/pose_landmarker_heavy.task` (gitignored).
- Python: `.venv` built with 3.13. The test hook prefers `.venv`.
- Downloads: nothing is fetched onto the user's PC without their say-so for that specific file. The dev-clip rules (sources, HEAD-check, 100 MB cap, own directory, verify before use, never delete) are recorded in `data/external/SOURCES.md` and apply to any new clip.
- BrowserOS neo: the researcher agent can now see it (verified with a diagnostic run), after the tools entry was corrected.
- Git: local repo on `main`, remote `origin` = https://github.com/WilwilMina/ComputerVisionPullUps.git (set, never pushed by Claude). Commit identity is repo-local: the user's name and their GitHub no-reply address. Rule (user, in CLAUDE.md): after each slice passes tests and `qa`, commit locally; run `git status` first and confirm nothing from `data/`, no videos, model files or `.env` is staged and the layout matches `.claude/rules/architecture.md`. Never push; the user pushes. History so far: scaffold commit, then one commit for slices 1-3, then one per slice. `docs/reference/*.pdf` is gitignored. Security scan: `tools/scan_sensitive.py` runs from `.githooks/pre-commit` (staged changes) and `.githooks/pre-push` (every commit being pushed). It blocks videos, model files, `.npz` caches, PDFs, archives, keys and `.env`; anything under `data/`, `samples/`, `out/`, `models/`; files over 1 MB; private keys and common API-token shapes; secret-looking assignments; personal absolute paths and email addresses; and commits whose author or committer email is not a GitHub no-reply address. Findings are redacted (emails show only the domain). QA review fixed: a quadratic regex on long lines (now linear; a 5 MB line scans in under a second), `sk-proj-` keys, JSON/dotenv-style secrets, case-variant file names, merge-commit content. Known limits: encoded or split secrets, secrets in commit messages, unquoted letters-only values; a few false positives (e.g. numpy `A@B.shape` reads as an email; use `scan:allow`). Its 3 throwaway-repo tests are marked `slow` (`pytest -m slow`) to keep the default suite near 2 s. Enable it after cloning with `git config core.hooksPath .githooks`; never bypass with `--no-verify`. Also worth switching on in the GitHub repo settings: secret scanning and push protection.

# What is left

Short checklist of everything between today and "finished". Detail lives in `docs/PROGRESS.md` (state and measurements) and `docs/FAULTS.md` (research). Last updated 2026-10-09. `[x]` done, `[ ]` open.

## Blocked on the owner's clips (these decide whether anything can be called validated)
- [ ] **Own clips, filmed in one session on a tripod** (front or back; bar, arms and head in frame at the top of every rep; 3-5 s dead hang at the start of each clip):
  - [ ] a full 3 reps
  - [ ] bent-elbow bottoms (come down only until the arms are clearly still bent), ideally **two clips** so one can be held out; include some mild ones
  - [ ] chin-short reps (stop a few inches below the bar), ideally two clips; include a "just below the bar" one
  - [ ] another full 3 reps
  - [ ] if possible, a second person doing full reps and some bent reps
- [ ] **Rep-by-rep notes** for each clip (what each rep really was), in the owner's words.
- [ ] **Labels for clips already added:** the friend's `IMG_0970` (6 reps, pipeline says all FULL, unverified) and the owner's `IMG_6975` (3 reps detected, bar leaves the frame at the top, 92% of frames unusable).

## Once the clips arrive (I do these, in this order)
- [ ] Commit the owner's labels first, assign calibration vs held-out by clip (before any verdict), then run the pipeline.
- [ ] Report counts per class on held-out clips only. An accuracy figure only if the held-out set has enough scored reps, real negatives and more than one person; otherwise counts only.
- [ ] On calibration clips, show how bottom elbow angle and head-minus-bar separate by label; propose thresholds with the evidence and wait for approval.
- [ ] Decide the per-person tolerance (`REFERENCE_TOLERANCE_DEG`, currently 0) from the owner's own good and bent reps.
- [ ] Re-check the two provisional things that rest on little data: the unsupported-style cutoff (one positive clip, one likely false flag) and the 0.23 bar offset (two clips, two people).
- [ ] Update README and PROGRESS with only what was measured; if bad reps still cannot be separated, say so.

## Decisions waiting on the owner
- [ ] Switch the head landmark from the nose to the two mouth corners? (Proposed: mouth is the nearer proxy to the chin; tracking confidence cannot separate them; it moves `head_minus_bar` down about 0.07-0.08.)
- [ ] Anything on the 144 deg global threshold? (Not changed. It read 130-136 on one person and 154-161 on another, so one absolute number does not transfer; lowering it would only fit one clip.)

## Built (for reference)
- [x] Pose extraction, signals, rep state machine, overlay video, CLI, JSON summary; security scan and git hooks; 300+ tests.
- [x] Pull time is information only, never part of a verdict.
- [x] Measured wrist-to-bar offset applied (0.23 body scales).
- [x] Live elbow readout on the video (model estimate, hidden when invalid).
- [x] Labels file, calibration/held-out split, `score_labels` (counts only, never a percentage).
- [x] Synthetic rule sweeps on real signals (margins are small: roughly 2.5-11 deg of bend).
- [x] Unsupported-style (kipping) advisory flag.
- [x] `docs/FAULTS.md` research (standards, fault table, datasets, person-to-person variation).
- [x] Several-person handling: a people-check pass counts people and checks the tracked person is the one on the bar, and the CLI warns. It does not switch person (asking the model for several people changed the single person's 3D elbow by about 10 deg).
- [x] Optional per-person reference calibration (`--reference`), built; not validated. A strict floor from 3 reference reps falsely rejected a later good rep in 2 of 4 people tried.
- [x] Tracking gaps reviewed: 6 of 16 dev clips have uncertain attempts (7 in total); the conservative rule is kept by the owner's decision and the attempts are listed, never hidden.

## Known gaps that will stay (state them in the README)
- No validated bad-rep examples yet: bent-elbow and chin-short detection are untested on real footage.
- The 3D elbow angle is a model estimate that varies by person and view.
- The head check uses the nose (no chin point); from behind the face points are guessed.
- Neck craning, legs driving, shrugged hang and chin resting on the bar are not detected.
- A front or back view and a level bar are needed; side views and tilted bars are flagged, not supported.
- Several people: warned about, not separated.

## Later, after the clips (portfolio side)
- [ ] Pick a demo clip the owner is happy to show; render the annotated video once; cite only what the code measured. Stock and dev clips stay out of the demo, README and posts.
- [ ] README final pass: run steps, measured numbers (if any), the limitations list above, "no validated accuracy" unless the held-out check supports more.
- [ ] LinkedIn write-up drafted from measured results only.
- [ ] Final `qa` pass, full test run, security scan, update PROGRESS, push.

## Optional
- [ ] Download the two Wikimedia demos (done, in `data/external/`; not yet run through the pipeline) and run them.
- [ ] Pixabay `65659` (back view at a bar, unverified content; its file host `cdn.pixabay.com` is not on the approved list).
- [ ] ViTPose comparison (needs PyTorch and a weights download; ask first). Only worth it if MediaPipe tracking turns out to be the weak link.

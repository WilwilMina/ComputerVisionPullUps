# Testing rules

- Rep-logic tests use synthetic time series built in `tests/synthetic.py` (head-height and elbow-angle curves with a given fps). No video files, no model files, no network. The whole suite should run in about 3 seconds (most of that is the scipy import, so the budget is relaxed from 2 s).
- These cases must exist and pass before rep logic is called done:
  1. Full rep: starts at dead hang, head clears the bar, returns to hang -> counted FULL.
  2. Bent-elbow start: head clears the bar but the bottom elbow angle never reaches dead hang -> counted, NOT full.
  3. Not high enough: dead hang but head never clears the bar -> counted or flagged, NOT full.
  4. Multiple reps with a pause between them -> correct count and per-rep ROM labels.
  5. Noisy full rep (jitter near the bar crossing) -> counted exactly once.
  6. Missing or low-confidence frames mid-rep -> no crash, no phantom reps.
- Test each stage as it is built (metrics before reps, reps before render).
- When a bug appears on real footage, add a regression fixture first, then fix.
- Done means pytest is green AND the `qa` agent confirms. Do not rerun the `qa` agent if a deterministic local check already proves the result.

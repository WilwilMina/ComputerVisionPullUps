# Pull-up ROM Detector

Computer-vision demo: takes a pull-up video, detects each rep, and classifies it as full range of motion (ROM) or not. Outputs an annotated video (skeleton, rep counter, head-height graph, per-rep stats table) plus a JSON summary. Portfolio / LinkedIn piece.

## Critical guardrails
- A rep is FULL only if the head clears the bar line AND the elbow angle at the bottom reaches the dead-hang threshold (arms ~straight). All thresholds live in `pullup_rom/config.py` as named constants with units, never inline.
- Never invent numbers (angles, timings, accuracy) in the README or posts. Report only what the code measured on real footage.
- Pose runs behind a `PoseBackend` interface. The sandbox cannot download model files (proxy 403), so real pose extraction is run and checked on the user's machine. All rep logic is tested on synthetic keypoints.
- Do not commit videos, model files (`*.task`, `*.pt`, `*.onnx`), outputs, or `.env`.

## Stack
Python 3.13, OpenCV, NumPy, SciPy (smoothing), MediaPipe Pose Landmarker (Tasks API; legacy `mp.solutions` is gone in 0.10.32), matplotlib.

## Layout (planned, update as built)
- `pullup_rom/pose.py` video -> keypoints (backend interface + MediaPipe backend)
- `pullup_rom/metrics.py` smoothing, head height vs bar (unitless, body-scaled), elbow angle
- `pullup_rom/debug_plot.py` slice-2 signals plot for eyeballing (not the final overlay)
- `pullup_rom/reps.py` rep state machine (pure functions, no I/O)
- `pullup_rom/render.py` overlay, graph, stats table -> mp4
- `pullup_rom/cli.py`, `pullup_rom/config.py`, `tests/`
- `pullup_rom/rep_report.py`, `pullup_rom/eval_clips.py` inspection tools (I/O lives here, not in `reps.py`)
- `tools/scan_sensitive.py` + `.githooks/` commit and push security scan

## Commands
- Install: `pip install -r requirements.txt`
- Test: `python -m pytest -q` (fast suite). Also `python -m pytest -q -m slow` after touching `tools/` or `.githooks/` (builds throwaway git repos).
- Run (once built): `python -m pullup_rom.cli input.mp4 -o out.mp4`

## Workflow
- Use Plan Mode for anything beyond one module. Build one slice at a time: keypoints -> metrics -> rep logic -> overlay. Check each before moving on.
- A `.py` edit triggers the fast test hook automatically. Fix failures before continuing.
- Need current external info? Delegate to `researcher`. Before calling a feature done, run the `qa` agent. Both return short reports; the parent fixes and decides.
- Re-checking tracking on the dev clips in `data/external/` (reading plots and contact sheets) goes to the `clip-evaluator` agent, so images do not fill the parent's context. Delegate any other large, mechanical read-only task the same way.
- After each slice passes tests and `qa`, commit it locally with a clear message. Before committing, run `git status` and confirm nothing from `data/`, no videos, model files, or `.env` is staged, and that the directory layout still matches `.claude/rules/architecture.md`. The `.githooks/` pre-commit hook runs `tools/scan_sensitive.py` (secrets, media, models, data, personal paths and emails); run `python tools/scan_sensitive.py --staged` yourself too, and never use `--no-verify`. Never push; the user pushes (the pre-push hook scans what they send).
- Rules are in `.claude/rules/` (architecture, testing). Read them before changing module boundaries or tests.
- Update `docs/PROGRESS.md` at the end of every session. Files and git history are the source of truth, not conversation memory.

## Research
- External research (pose model options, library/API changes, published ROM or dead-hang definitions) goes through the `researcher` agent, not the parent. It uses BrowserOS neo when connected and returns a short synthesis with source links.
- The user connects BrowserOS neo on demand: `claude mcp add --transport http browseros <mcp_url> --scope user` (URL from `chrome://browseros/mcp`). If it isn't connected, the researcher says so first and falls back to public web search/fetch only.
- Research informs decisions. Never take thresholds or accuracy numbers from it without a cited source.

## Reference
`docs/reference/` holds the agentic workspace guide this setup follows (not auto-loaded).

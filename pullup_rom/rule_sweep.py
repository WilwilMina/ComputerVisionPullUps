"""SYNTHETIC rule-logic sweeps built from the real signals of labelled-FULL reps. NOT an accuracy measure.

For each labelled-FULL rep that the pipeline currently calls FULL, two perturbations of that rep's real signals show how
far the rep is from flipping the verdict:
  (a) head peak: the head's height over the bar is capped lower and lower inside the rep until the head check fails;
  (b) bottom extension: the elbow angle is reduced by a growing bend (degrees) around the rep until the arms check fails.
The answer is a number per rep (body scales of headroom, degrees of bend), reported per clip. Because the rule is a
plain comparison against a threshold, the flip points are expected to equal the rep's margin to that threshold: the
sweep confirms the rule logic end to end (rep detection, bottom windows, verdict), and shows how much margin each real
rep has. It says nothing about whether the labels, the threshold or the model are right.

    python -m pullup_rom.rule_sweep export OUT.json CACHE.npz:CLIP:INDICES [...]    # INDICES = all or 1,2,3
    python -m pullup_rom.rule_sweep report tests/fixtures/real_signals.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from . import config
from .reps import Rep, RepParams, detect_reps

HEAD_STEP_BODY_SCALES = 0.01  # sweep resolution for the head-peak perturbation
BEND_STEP_DEG = 0.5  # sweep resolution for the bottom-extension perturbation
MAX_HEAD_DROP_BODY_SCALES = 3.0
MAX_BEND_DEG = 90.0
BOTTOM_PAD_S = 2.0  # the bend is applied from this long before a rep's start to this long after its end

VERDICT_TEXT = {True: "FULL", False: "NOT FULL", None: "UNKNOWN"}


def load_fixture(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def export_clip(cache_path: str | Path, clip: str, full_rep_indices, source: str = config.ELBOW_ANGLE_SOURCE) -> dict:
    """Real head/elbow signals of one clip (numbers only, no pixels) plus which reps are labelled FULL."""
    from .metrics import MetricsParams, compute_signals
    from .pose import load_pose

    pose = load_pose(cache_path)
    s = compute_signals(pose, MetricsParams(elbow_angle_source=source))

    def rounded(a):
        return [None if not np.isfinite(v) else round(float(v), 4) for v in a]

    return {
        "clip": clip, "fps": s.fps, "source": source, "full_rep_indices": full_rep_indices,
        "head_minus_bar": rounded(s.head_minus_bar), "elbow_deg": rounded(s.elbow_deg), "bar_carried": [bool(b) for b in s.bar_carried],
    }


def arrays(entry: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    def f(a):
        return np.array([np.nan if v is None else v for v in a], dtype=np.float64)

    return f(entry["head_minus_bar"]), f(entry["elbow_deg"]), np.array(entry["bar_carried"], dtype=bool)


def _detect(head, elbow, carried, fps, params):
    return detect_reps(head, elbow, fps, params, bar_carried=carried)


def _rep_at(reps: Sequence[Rep], peak_frame: int) -> Optional[Rep]:
    """The detected rep whose span holds the original peak frame (capping the head flattens the peak, so it can move)."""
    holding = [r for r in reps if r.start_frame <= peak_frame <= r.end_frame]
    return min(holding, key=lambda r: abs(r.peak_frame - peak_frame)) if holding else None


def lower_head_peak(head: np.ndarray, rep: Rep, drop: float) -> np.ndarray:
    """Cap the head's height over the bar inside the rep at (its peak - drop): a rep that stops short."""
    out = head.copy()
    seg = slice(rep.start_frame, rep.end_frame + 1)
    out[seg] = np.minimum(out[seg], rep.peak_head_minus_bar - drop)
    return out


def bend_bottoms(elbow: np.ndarray, rep: Rep, bend_deg: float, fps: float) -> np.ndarray:
    """Reduce the elbow angle by bend_deg around the rep, so both of its bottoms are less extended."""
    out = elbow.copy()
    pad = int(round(BOTTOM_PAD_S * fps))
    out[max(0, rep.start_frame - pad) : rep.end_frame + pad + 1] -= bend_deg
    return out


def _first_flip(values, make_signals, rep, params, fps):
    """First sweep value at which the rep is no longer FULL (or is no longer found), with how it failed."""
    for v in values:
        head, elbow, carried = make_signals(float(v))
        again = _rep_at(_detect(head, elbow, carried, fps, params).reps, rep.peak_frame)
        if again is None or again.full is not True:
            return round(float(v), 3), "rep no longer found" if again is None else "NOT FULL"
    return None, "never within the sweep"


def flip_points(entry: dict, params: RepParams, head_step: float = HEAD_STEP_BODY_SCALES, bend_step: float = BEND_STEP_DEG) -> list[dict]:
    """Per labelled-FULL rep: current verdict, margins, and the first head drop / bend that makes it not FULL."""
    head, elbow, carried = arrays(entry)
    fps = entry["fps"]
    reps = _detect(head, elbow, carried, fps, params).reps
    if entry["full_rep_indices"] == "all":
        wanted = [r for r in reps if not r.clipped]
    else:
        wanted = [r for r in reps if r.index in entry["full_rep_indices"]]
    rows = []
    for rep in wanted:
        row = {
            "rep": rep.index, "peak_s": round(rep.peak_s, 2), "verdict": VERDICT_TEXT[rep.full],
            "peak_head_minus_bar": round(rep.peak_head_minus_bar, 3), "bottom_elbow_deg": round(rep.bottom_elbow_deg, 2),
            "head_drop_to_flip": None, "head_flip_outcome": None, "bend_to_flip_deg": None, "bend_flip_outcome": None,
        }
        if rep.full is True:
            drops = np.arange(head_step, MAX_HEAD_DROP_BODY_SCALES, head_step)
            row["head_drop_to_flip"], row["head_flip_outcome"] = _first_flip(drops, lambda d: (lower_head_peak(head, rep, d), elbow, carried), rep, params, fps)
            bends = np.arange(bend_step, MAX_BEND_DEG, bend_step)
            row["bend_to_flip_deg"], row["bend_flip_outcome"] = _first_flip(bends, lambda b: (head, bend_bottoms(elbow, rep, b, fps), carried), rep, params, fps)
        rows.append(row)
    return rows


def report(fixture: dict, params: Optional[RepParams] = None) -> list[str]:
    p = params or RepParams.for_source(fixture["clips"][0]["source"])
    lines = [
        "SYNTHETIC perturbations of real signals; this tests the rule logic only, it is NOT an accuracy figure.",
        f"rule: head must clear the bar line (margin {p.head_clear_margin}) AND the bottom elbow must reach {p.dead_hang_elbow_min_deg} deg; "
        f"sweep steps: head {HEAD_STEP_BODY_SCALES} body scales, bend {BEND_STEP_DEG} deg",
        "",
    ]
    for entry in fixture["clips"]:
        rows = flip_points(entry, p)
        lines.append(f"{entry['clip']} (one person per clip; {len(rows)} labelled-FULL complete reps)")
        lines.append("   rep  peak_s  verdict    head over bar | head drop to flip | bottom elbow | bend to flip")
        for r in rows:
            full = r["verdict"] == "FULL"
            hd = f"{r['head_drop_to_flip']:.2f}" if full else "already not FULL"
            bd = f"{r['bend_to_flip_deg']:.1f} deg" if full else "already not FULL"
            lines.append(f"   {r['rep']:3d} {r['peak_s']:6.1f}  {r['verdict']:<9} {r['peak_head_minus_bar']:13.2f} | {hd:>17} | {r['bottom_elbow_deg']:9.1f} deg | {bd:>12}")
        flips = [r for r in rows if r["verdict"] == "FULL"]
        if flips:
            hs = [r["head_drop_to_flip"] for r in flips]
            bs = [r["bend_to_flip_deg"] for r in flips]
            lines.append(
                f"   FULL reps: head drop to flip {min(hs):.2f}-{max(hs):.2f} (median {np.median(hs):.2f}) body scales; "
                f"bend to flip {min(bs):.1f}-{max(bs):.1f} (median {np.median(bs):.1f}) deg"
            )
        if len(rows) - len(flips):
            lines.append(f"   {len(rows) - len(flips)} of {len(rows)} labelled-FULL reps are ALREADY not FULL before any perturbation")
        lines.append("")
    return lines


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Synthetic rule-logic sweeps on real signals (not accuracy).")
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("out")
    e.add_argument("specs", nargs="+", help="CACHE.npz:CLIP:INDICES (INDICES = all or 1,2,3)")
    r = sub.add_parser("report")
    r.add_argument("fixture")
    args = ap.parse_args(argv)
    if args.cmd == "export":
        clips = []
        for spec in args.specs:
            path, clip, idx = spec.split(":")
            clips.append(export_clip(path, clip, "all" if idx == "all" else [int(i) for i in idx.split(",")]))
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        note = "Numeric signals only (no pixels) from stock clips, for the synthetic rule sweep. full_rep_indices are the labelled-FULL reps."
        Path(args.out).write_text(json.dumps({"note": note, "clips": clips}), encoding="utf-8")
        print(f"wrote {args.out}: {len(clips)} clips")
    else:
        print("\n".join(report(load_fixture(args.fixture))))


if __name__ == "__main__":
    main()

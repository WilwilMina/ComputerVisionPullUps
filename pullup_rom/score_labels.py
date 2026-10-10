"""Score the pipeline's verdicts against a committed labels file (labels/*.json), counts only.

    python -m pullup_rom.score_labels labels/bad_form_round1.json --keypoints-dir out/bad_form

Each labelled rep is snapped to the detected rep whose peak is nearest its labelled top time (LABEL_SNAP_WINDOW_S).
UNCLEAR reps and clips marked excluded_from_scoring are listed but never counted. Results are reported as counts per
label class, per set (calibration / held-out), with the number of clips. No percentage is ever printed: a figure
needs real labelled negatives in the held-out set, and this report says plainly when they are missing.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

from . import config
from .label_check import snap_labels, verdict_text
from .pose import load_pose
from .rep_report import analyze

NEGATIVE_LABELS = ("SHORT_BOTTOM", "NOT_HIGH", "SHORT_BOTH")  # the pipeline should say NOT FULL for these


@dataclass(frozen=True)
class ScoredRep:
    clip: str
    split: str  # "calibration" | "held_out"
    index: int
    label: str
    detected_rep: Optional[int]
    verdict: str  # FULL | NOT FULL | UNKNOWN | no detected rep
    scored: bool
    outcome: str  # agree | differs | pipeline UNKNOWN | no detected rep | not scored (...)


def expected_verdict(label: str) -> Optional[str]:
    if label == "FULL":
        return "FULL"
    return "NOT FULL" if label in NEGATIVE_LABELS else None


def load_label_file(path: str | Path) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    for key in ("assignment", "clips"):
        if key not in data:
            raise ValueError(f"labels file {path} has no '{key}'")
    for c in data["clips"]:
        for r in c.get("reps", []):
            for key in ("index", "label", "top_s"):
                if key not in r:
                    raise ValueError(f"clip {c.get('clip', '?')}: a rep entry has no '{key}'")
    return data


def split_of(data: dict, clip: str) -> str:
    for name in ("calibration", "held_out"):
        if clip in data["assignment"].get(name, []):
            return name
    raise ValueError(f"clip {clip} is in neither the calibration nor the held-out assignment")


def score_clip(data: dict, clip_entry: dict, reps: Sequence) -> list[ScoredRep]:
    clip, split = clip_entry["clip"], split_of(data, clip_entry["clip"])
    excluded = bool(clip_entry.get("excluded_from_scoring"))
    labs = [{**r, "approx_top_s": r["top_s"]} for r in clip_entry["reps"]]
    rows = []
    for lab, rep in snap_labels(labs, reps, config.LABEL_SNAP_WINDOW_S):
        verdict = "no detected rep" if rep is None else verdict_text(rep.full)
        if excluded:
            scored, outcome = False, f"not scored (clip excluded: {clip_entry.get('exclusion_reason', 'excluded')})"
        elif expected_verdict(lab["label"]) is None:
            scored, outcome = False, f"not scored ({lab['label']})"
        elif rep is None:
            scored, outcome = True, "no detected rep"
        elif rep.full is None:
            scored, outcome = True, "pipeline UNKNOWN"
        else:
            scored, outcome = True, "agree" if verdict == expected_verdict(lab["label"]) else "differs"
        rows.append(ScoredRep(clip, split, lab["index"], lab["label"], None if rep is None else rep.index, verdict, scored, outcome))
    return rows


def tally(rows: Sequence[ScoredRep]) -> dict[str, dict[str, Counter]]:
    """{split: {label: Counter(verdict)}} over scored reps only."""
    out: dict[str, dict[str, Counter]] = {}
    for r in rows:
        if r.scored:
            out.setdefault(r.split, {}).setdefault(r.label, Counter())[r.verdict] += 1
    return out


def build_report(data: dict, keypoints_dir: str | Path, source: str = config.ELBOW_ANGLE_SOURCE) -> list[str]:
    rows: list[ScoredRep] = []
    lines = [f"labels {data.get('round', '?')}: {str(data.get('status') or '').replace('%', ' percent')[:70]}", f"pipeline: {source} elbow angle, dead-hang threshold {config.DEAD_HANG_ELBOW_MIN_DEG[source]}", ""]
    for c in data["clips"]:
        path = Path(keypoints_dir) / f"{c['clip']}.keypoints.npz"
        if not path.exists():
            raise ValueError(f"no keypoints file for {c['clip']}: expected {path}")
        _, analysis = analyze(load_pose(path), source)
        clip_rows = score_clip(data, c, analysis.reps)
        rows += clip_rows
        flag = "  [EXCLUDED: " + c.get("exclusion_reason", "") + "]" if c.get("excluded_from_scoring") else ""
        lines.append(f"{c['clip']} ({split_of(data, c['clip'])}, {c.get('people', '?')} person(s) in frame; detected reps {len(analysis.reps)}, labelled {len(c['reps'])}){flag}")
        for r in clip_rows:
            lines.append(f"   label rep {r.index}: {r.label:<8} -> pipeline rep {r.detected_rep if r.detected_rep else '-'}: {r.verdict:<16} {r.outcome}")
    lines += ["", "Scored reps only (UNCLEAR and excluded clips left out); counts, never percentages:"]
    counts = tally(rows)
    for split in ("calibration", "held_out"):
        clips = sorted({r.clip for r in rows if r.scored and r.split == split})
        lines.append(f"  {split}: {sum(sum(c.values()) for c in counts.get(split, {}).values())} scored reps in {len(clips)} clip(s) {clips}")
        for label, ctr in sorted(counts.get(split, {}).items()):
            lines.append(f"    labelled {label:<12} pipeline says: " + ", ".join(f"{v} x {k}" for k, v in sorted(ctr.items())))
    held = counts.get("held_out", {})
    negatives = sum(sum(held.get(n, Counter()).values()) for n in NEGATIVE_LABELS)
    positives = sum(held.get("FULL", Counter()).values())
    if negatives == 0 or positives == 0:
        lines.append(f"  NO ACCURACY FIGURE: the held-out set has {positives} scored FULL and {negatives} scored negative reps. Both classes are needed before any figure means anything.")
    else:
        lines.append(f"  Held-out has {positives} scored FULL and {negatives} scored negative reps. Even so, treat any figure as indicative only unless clips and distinct people are many; counts above are the result.")
    return lines


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Score pipeline verdicts against a committed labels file (counts only).")
    ap.add_argument("labels")
    ap.add_argument("--keypoints-dir", required=True, help="folder with <clip>.keypoints.npz files (the cli writes these next to its output)")
    ap.add_argument("--source", default=config.ELBOW_ANGLE_SOURCE, choices=["world", "image"])
    args = ap.parse_args(argv)
    print("\n".join(build_report(load_label_file(args.labels), args.keypoints_dir, args.source)))


if __name__ == "__main__":
    main()

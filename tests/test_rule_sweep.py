"""SYNTHETIC perturbations of REAL signals (tests/fixtures/real_signals.json: numeric head/elbow curves of labelled-FULL
reps, no pixels). They test the rule logic end to end, not accuracy: a rep that is FULL must stop being FULL exactly
when its head peak falls to the bar line or its bottom elbow falls to the threshold."""
import numpy as np
import pytest

from pullup_rom import config
from pullup_rom.reps import RepParams
from pullup_rom.rule_sweep import (
    BEND_STEP_DEG,
    HEAD_STEP_BODY_SCALES,
    arrays,
    bend_bottoms,
    flip_points,
    load_fixture,
    lower_head_peak,
    report,
)

FIXTURE = "tests/fixtures/real_signals.json"
P = RepParams.for_source("world")


@pytest.fixture(scope="module")
def fixture():
    return load_fixture(FIXTURE)


@pytest.fixture(scope="module")
def rows_by_clip(fixture):
    return {e["clip"]: flip_points(e, P) for e in fixture["clips"]}  # fine steps over every clip: slow


COARSE_HEAD, COARSE_BEND = 0.05, 2.0


def test_synthetic_one_real_clip_with_coarse_steps_flips_at_the_margin_fast(fixture):
    entry = next(e for e in fixture["clips"] if e["clip"] == "pexels_6389831")
    rows = flip_points(entry, P, head_step=COARSE_HEAD, bend_step=COARSE_BEND)
    assert len(rows) == 3 and all(r["verdict"] == "FULL" for r in rows)
    for r in rows:
        assert r["peak_head_minus_bar"] - 1e-6 <= r["head_drop_to_flip"] <= r["peak_head_minus_bar"] + 2 * COARSE_HEAD
        margin = r["bottom_elbow_deg"] - P.dead_hang_elbow_min_deg
        assert margin - 1e-6 <= r["bend_to_flip_deg"] <= margin + 2 * COARSE_BEND


def test_synthetic_the_fixture_covers_several_clips_and_only_holds_numbers(fixture):
    assert len(fixture["clips"]) >= 3
    for e in fixture["clips"]:
        n = len(e["head_minus_bar"])
        assert len(e["elbow_deg"]) == len(e["bar_carried"]) == n and e["fps"] > 0
        assert all(v is None or isinstance(v, (int, float)) for v in e["head_minus_bar"] + e["elbow_deg"])


@pytest.mark.slow
def test_synthetic_head_peak_flips_the_verdict_exactly_where_the_head_reaches_the_bar_line(rows_by_clip):
    seen = 0
    for clip, rows in rows_by_clip.items():
        for r in rows:
            if r["verdict"] != "FULL":
                continue
            seen += 1
            headroom = r["peak_head_minus_bar"] - P.head_clear_margin
            assert r["head_flip_outcome"] == "NOT FULL", (clip, r)
            assert headroom - 1e-6 <= r["head_drop_to_flip"] <= headroom + 2 * HEAD_STEP_BODY_SCALES, (clip, r)
    assert seen >= 10


@pytest.mark.slow
def test_synthetic_bend_flips_the_verdict_exactly_where_the_bottom_elbow_reaches_the_threshold(rows_by_clip):
    for clip, rows in rows_by_clip.items():
        for r in rows:
            if r["verdict"] != "FULL":
                continue
            margin = r["bottom_elbow_deg"] - P.dead_hang_elbow_min_deg
            assert r["bend_flip_outcome"] == "NOT FULL", (clip, r)
            assert margin - 1e-6 <= r["bend_to_flip_deg"] <= margin + 2 * BEND_STEP_DEG, (clip, r)


@pytest.mark.slow
def test_synthetic_a_rep_already_below_the_threshold_is_not_made_full_by_the_sweep(rows_by_clip):
    already = [r for rows in rows_by_clip.values() for r in rows if r["verdict"] != "FULL"]
    assert already and all(r["head_drop_to_flip"] is None and r["bend_to_flip_deg"] is None for r in already)


@pytest.mark.slow
def test_synthetic_one_step_before_the_flip_the_rep_is_still_full(fixture):
    from pullup_rom.rule_sweep import _detect, _rep_at

    entry = fixture["clips"][1]
    head, elbow, carried = arrays(entry)
    rep = next(r for r in _detect(head, elbow, carried, entry["fps"], P).reps if r.full)
    row = next(r for r in flip_points(entry, P) if r["rep"] == rep.index)
    for perturbed in (
        (lower_head_peak(head, rep, row["head_drop_to_flip"] - 2 * HEAD_STEP_BODY_SCALES), elbow),
        (head, bend_bottoms(elbow, rep, row["bend_to_flip_deg"] - 2 * BEND_STEP_DEG, entry["fps"])),
    ):
        again = _rep_at(_detect(perturbed[0], perturbed[1], carried, entry["fps"], P).reps, rep.peak_frame)
        assert again is not None and again.full is True


@pytest.mark.slow
def test_synthetic_the_report_says_it_is_not_an_accuracy_figure(fixture):
    text = "\n".join(report(fixture))
    assert "NOT an accuracy figure" in text and "pexels_6389831" in text and "%" not in text
    assert config.DEAD_HANG_ELBOW_MIN_DEG["world"] == P.dead_hang_elbow_min_deg  # the sweep uses the configured rule

"""Rep logic on synthetic series only. The numbers in P are arbitrary FIXTURE values, not calibrated thresholds."""
import numpy as np
import pytest
from synthetic import make_rep_series

from pullup_rom import config
from pullup_rom.reps import (
    REASON_CLIP_ENDED_MID_REP,
    REASON_ELBOW_MISSING,
    REASON_ELBOW_NOT_EXTENDED,
    REASON_ELBOW_UNCALIBRATED,
    REASON_HEAD_BELOW_BAR,
    RepParams,
    detect_reps,
)

FPS = 30.0
P = RepParams(
    head_clear_margin=0.0,
    dead_hang_elbow_min_deg=150.0,
    start_rise=0.25,
    end_drop=0.25,
    debounce_s=0.1,
    bottom_band=0.15,
    max_gap_s=0.5,
)


def run(head, elbow, params=P, fps=FPS, **kw):
    return detect_reps(head, elbow, fps, params, **kw)


def labels(analysis):
    return [r.full for r in analysis.reps]


# ---------- the 6 required cases (.claude/rules/testing.md) ----------

def test_1_full_rep_is_counted_full():
    head, elbow = make_rep_series(peaks=[0.4], valley_elbows=[165, 165])
    a = run(head, elbow)
    assert len(a.reps) == 1 and not a.discarded
    r = a.reps[0]
    assert r.full is True and r.head_ok and r.elbow_ok is True and r.reasons == ()
    assert abs(r.peak_head_minus_bar - 0.4) < 0.01
    assert abs(r.bottom_elbow_deg - 165) < 0.5


def test_2_bent_elbow_bottom_is_counted_but_not_full():
    head, elbow = make_rep_series(peaks=[0.4], valley_elbows=[120, 120])
    a = run(head, elbow)
    assert len(a.reps) == 1
    r = a.reps[0]
    assert r.head_ok and r.elbow_ok is False and r.full is False
    assert REASON_ELBOW_NOT_EXTENDED in r.reasons
    assert abs(r.bottom_elbow_deg - 120) < 0.5


def test_3_head_never_clears_the_bar_is_counted_but_not_full():
    head, elbow = make_rep_series(peaks=[-0.2], valley_elbows=[165, 165])
    a = run(head, elbow)
    assert len(a.reps) == 1
    r = a.reps[0]
    assert r.head_ok is False and r.elbow_ok is True and r.full is False
    assert REASON_HEAD_BELOW_BAR in r.reasons


def test_4_multiple_reps_with_pauses_get_the_right_labels():
    head, elbow = make_rep_series(peaks=[0.4, 0.5, -0.2, 0.4], valley_elbows=[165] * 5, pause_s=1.5)
    a = run(head, elbow)
    assert [r.index for r in a.reps] == [1, 2, 3, 4]
    assert labels(a) == [True, True, False, True]
    starts = [r.start_frame for r in a.reps]
    ends = [r.end_frame for r in a.reps]
    assert starts == sorted(starts) and all(e > s for s, e in zip(starts, ends))
    assert all(ends[k] <= starts[k + 1] for k in range(len(a.reps) - 1))  # the pause between reps sits between end and next start
    assert all(starts[k + 1] - ends[k] >= 20 for k in range(len(a.reps) - 1))  # a 1.5 s pause is visible in the timing


def test_4b_a_bent_valley_fails_both_reps_that_touch_it():
    head, elbow = make_rep_series(peaks=[0.4, 0.4, 0.4], valley_elbows=[165, 165, 120, 165])
    a = run(head, elbow)
    assert labels(a) == [True, False, False]  # rep 2 returns to a bent hang, rep 3 starts from it


@pytest.mark.parametrize("seed", range(25))
def test_5_noisy_full_rep_is_counted_exactly_once(seed):
    head, elbow = make_rep_series(peaks=[0.4], valley_elbows=[165, 165], noise_head=0.03, noise_elbow=2.0, seed=seed)
    rng = np.random.default_rng(1000 + seed)
    near_bar = np.abs(head) < 0.15  # extra jitter right where the head crosses the bar line
    head = head.copy()
    head[near_bar] += rng.normal(0.0, 0.05, near_bar.sum())
    a = run(head, elbow)
    assert len(a.reps) == 1 and not a.discarded
    assert a.reps[0].full is True


def test_6a_short_gaps_mid_rep_do_not_change_the_result():
    head, elbow = make_rep_series(peaks=[0.4], valley_elbows=[165, 165])
    peak = int(np.argmax(head))
    head, elbow = head.copy(), elbow.copy()
    head[peak - 25 : peak - 20] = np.nan  # 5 frames mid-ascent
    head[peak - 2 : peak + 8] = np.nan  # 10 frames across the top
    elbow[peak - 25 : peak - 20] = np.nan
    a = run(head, elbow)
    assert len(a.reps) == 1 and a.reps[0].full is True and not a.discarded


def test_6b_a_long_gap_mid_rep_discards_the_attempt_without_phantom_reps():
    head, elbow = make_rep_series(peaks=[0.4], valley_elbows=[165, 165])
    peak = int(np.argmax(head))
    head, elbow = head.copy(), elbow.copy()
    head[peak - 15 : peak + 15] = np.nan  # 1 s, longer than max_gap_s
    a = run(head, elbow)
    assert a.reps == []
    assert len(a.discarded) == 1 and a.discarded[0].reason == "gap_longer_than_max_during_rep"


def test_6c_missing_data_in_the_hang_and_everywhere_never_crashes_or_invents_reps():
    head, elbow = make_rep_series(peaks=[0.4, 0.4], valley_elbows=[165, 165, 165], pause_s=2.0)
    h = head.copy()
    h[5:25] = np.nan  # long hole in the lead-in hang
    assert len(run(h, elbow).reps) == 2
    assert run(np.full(100, np.nan), np.full(100, np.nan)).reps == []
    assert run(np.array([]), np.array([])).reps == []
    assert run(np.full(100, -1.0), np.full(100, 165.0)).reps == []  # a flat hang is not a rep


def test_6d_elbow_missing_at_the_bottom_gives_unknown_not_a_guess():
    head, elbow = make_rep_series(peaks=[0.4], valley_elbows=[165, 165])
    elbow = elbow.copy()
    elbow[head < -0.8] = np.nan  # no elbow data on either hang plateau
    r = run(head, elbow).reps[0]
    assert r.elbow_ok is None and r.full is None and REASON_ELBOW_MISSING in r.reasons
    assert np.isnan(r.bottom_elbow_deg)


# ---------- the constraints set for slice 3 ----------

def test_the_elbow_angle_at_the_top_of_a_rep_is_ignored():
    head, elbow = make_rep_series(peaks=[0.4], valley_elbows=[165, 165])
    top = head > -0.2  # well above the bottom band
    base = run(head, elbow).reps[0]
    for junk in (0.0, 999.0, np.nan):
        e = elbow.copy()
        e[top] = junk
        r = run(head, e).reps[0]
        assert r.full == base.full and r.bottom_elbow_deg == base.bottom_elbow_deg


def test_dead_hang_threshold_is_per_source_never_a_near_180_constant_and_unset_means_unknown():
    world, image = config.DEAD_HANG_ELBOW_MIN_DEG["world"], config.DEAD_HANG_ELBOW_MIN_DEG["image"]
    assert world is not None and world < 170  # calibrated from labelled reps; 3D readings never reach 180
    assert image is None  # the 2D angle depends on the camera view, so it stays unset until it can be calibrated
    assert RepParams().dead_hang_elbow_min_deg is None  # no source, no threshold
    assert RepParams.for_source("world").dead_hang_elbow_min_deg == world
    assert RepParams.for_source("image").dead_hang_elbow_min_deg is None
    assert RepParams.for_source("image", dead_hang_elbow_min_deg=150.0).dead_hang_elbow_min_deg == 150.0
    with pytest.raises(ValueError):
        RepParams.for_source("lidar")
    head, elbow = make_rep_series(peaks=[0.4], valley_elbows=[165, 165])
    assert run(head, elbow, RepParams.for_source("image")).reps[0].full is None  # unset source: UNKNOWN, never a guess


def test_uncalibrated_threshold_reports_unknown_for_a_rep_that_clears_the_bar():
    p = RepParams(dead_hang_elbow_min_deg=None)
    head, elbow = make_rep_series(peaks=[0.4, -0.2], valley_elbows=[165, 165, 165])
    a = run(head, elbow, p)
    assert labels(a) == [None, False]  # unknown vs definitely-not-full
    assert REASON_ELBOW_UNCALIBRATED in a.reps[0].reasons
    assert all(np.isfinite(r.bottom_elbow_deg) for r in a.reps)  # still measured, so the user can inspect it


def test_each_rep_exposes_its_bottom_elbow_angle_for_inspection():
    head, elbow = make_rep_series(peaks=[0.4, 0.4], valley_elbows=[170, 155, 140])
    a = run(head, elbow)
    r1, r2 = a.reps
    assert abs(r1.elbow_start_deg - 170) < 0.5 and abs(r1.elbow_end_deg - 155) < 0.5
    assert abs(r1.bottom_elbow_deg - 155) < 0.5  # the less extended of the two bottoms
    assert abs(r2.elbow_start_deg - 155) < 0.5 and abs(r2.bottom_elbow_deg - 140) < 0.5


def test_peak_bar_carried_flag_is_passed_through():
    head, elbow = make_rep_series(peaks=[0.4], valley_elbows=[165, 165])
    carried = np.zeros(len(head), dtype=bool)
    peak = int(np.argmax(head))
    carried[peak - 3 : peak + 3] = True
    assert run(head, elbow, bar_carried=carried).reps[0].peak_bar_carried is True
    assert run(head, elbow, bar_carried=np.zeros(len(head), dtype=bool)).reps[0].peak_bar_carried is False


# ---------- robustness ----------

@pytest.mark.parametrize("fps", [24.0, 30.0, 60.0])
def test_result_does_not_depend_on_frame_rate(fps):
    head, elbow = make_rep_series(fps=fps, peaks=[0.4, 0.4, -0.2], valley_elbows=[165, 165, 120, 165])
    a = run(head, elbow, fps=fps)
    assert labels(a) == [True, False, False]


def test_a_clip_that_starts_mid_pull_or_ends_mid_rep_does_not_invent_or_hide_reps():
    head, elbow = make_rep_series(peaks=[0.4], valley_elbows=[165, 165])
    peak = int(np.argmax(head))
    started_high = run(head[peak:], elbow[peak:])  # begins at the top, only the lowering is seen
    assert started_high.reps == []
    cut = run(head[: peak + 5], elbow[: peak + 5])  # ends before the head comes back down
    assert len(cut.reps) == 1 and REASON_CLIP_ENDED_MID_REP in cut.reps[0].reasons
    assert np.isnan(cut.reps[0].elbow_end_deg) and np.isfinite(cut.reps[0].elbow_start_deg)


def test_bad_inputs_and_params_raise():
    with pytest.raises(ValueError):
        detect_reps(np.zeros(5), np.zeros(4), FPS)
    with pytest.raises(ValueError):
        detect_reps(np.zeros(5), np.zeros(5), 0.0)
    with pytest.raises(ValueError):
        RepParams(start_rise=0.0)
    with pytest.raises(ValueError):
        RepParams(end_drop=-1.0)
    with pytest.raises(ValueError):
        detect_reps(np.zeros(5), np.zeros(5), FPS, bar_carried=np.zeros(3, dtype=bool))


# ---------- hardening from the QA review ----------

@pytest.mark.parametrize("seed", range(20))
def test_noise_in_the_bottom_elbow_does_not_lift_a_bent_bottom_over_the_threshold(seed):
    head, elbow = make_rep_series(peaks=[0.4], valley_elbows=[145, 145], noise_elbow=4.0, seed=seed)
    assert run(head, elbow).reps[0].full is False  # true bottom 145 < 150; the median stays near 145


@pytest.mark.parametrize("seed", range(20))
def test_noise_in_the_bottom_elbow_does_not_sink_a_straight_bottom_either(seed):
    head, elbow = make_rep_series(peaks=[0.4], valley_elbows=[165, 165], noise_elbow=4.0, seed=seed)
    assert run(head, elbow).reps[0].full is True


@pytest.mark.parametrize("fps", [24.0, 30.0, 60.0, 120.0])
def test_a_short_glitch_is_discarded_not_counted_as_a_rep(fps):
    head, elbow = make_rep_series(fps=fps, peaks=[0.4], valley_elbows=[165, 165], lead_s=2.0, tail_s=2.0)
    head = head.copy()
    blip_at = int(0.5 * fps)  # in the lead-in hang, long before the real rep
    head[blip_at : blip_at + max(3, int(round(0.12 * fps)))] += 1.5  # about 0.1 s, 1.5 body scales high
    a = run(head, elbow, fps=fps)
    assert len(a.reps) == 1 and a.reps[0].full is True
    assert any(d.reason == "ascent_shorter_than_min_duration" for d in a.discarded)


def test_min_ascent_is_a_validated_param():
    with pytest.raises(ValueError):
        RepParams(min_ascent_s=-0.1)


def test_a_long_hang_after_the_last_rep_does_not_stretch_its_end_to_the_clip_end():
    head, elbow = make_rep_series(peaks=[0.4], valley_elbows=[165, 165], tail_s=10.0)
    head = head + np.linspace(0.0, -0.05, len(head))  # the hang drifts slowly lower, so the argmin is the last frame
    r = run(head, elbow).reps[0]
    assert r.end_s < 5.0  # arrival at the bottom, not the 14 s mark


def test_the_start_of_a_rep_is_when_the_pull_begins_not_the_start_of_a_long_hang():
    head, elbow = make_rep_series(peaks=[0.4], valley_elbows=[165, 165], lead_s=6.0)
    r = run(head, elbow).reps[0]
    assert 5.5 < r.start_s < 7.0 and r.peak_s > r.start_s


# ---------- decisions: uncertain attempts are listed, rep end = first return to the low level ----------

def test_a_tracking_gap_attempt_is_reported_as_uncertain_with_its_time_range():
    head, elbow = make_rep_series(peaks=[0.4], valley_elbows=[165, 165])
    peak = int(np.argmax(head))
    head = head.copy()
    head[peak - 15 : peak + 15] = np.nan
    a = run(head, elbow)
    assert a.reps == [] and len(a.discarded) == 1
    d = a.discarded[0]
    assert d.label == "uncertain (tracking gap)"
    assert 0.5 < d.start_s < d.end_s <= len(head) / FPS  # starts as the pull begins, ends where tracking resumes
    assert d.start_s < peak / FPS < d.end_s


def test_a_glitch_is_labelled_as_ignored_not_as_uncertain():
    head, elbow = make_rep_series(peaks=[0.4], valley_elbows=[165, 165], lead_s=2.0)
    head = head.copy()
    head[15:19] += 1.5
    assert any(d.label.startswith("ignored") for d in run(head, elbow).discarded)


def test_rep_end_is_the_first_return_to_the_starting_low_level_even_if_the_hang_keeps_sinking():
    head, elbow = make_rep_series(peaks=[0.4], valley_elbows=[165, 165], tail_s=8.0)
    tail = len(head) - int(8.0 * FPS)
    sink = np.zeros(len(head))
    sink[tail:] = -np.linspace(0.0, 0.3, len(head) - tail)  # keeps dropping well past the bottom band
    r = run(head + sink, elbow).reps[0]
    first_back_at_bottom = int(np.argmax((np.arange(len(head)) > np.argmax(head)) & (head <= -1.0 + 0.15)))
    assert abs(r.end_frame - first_back_at_bottom) <= 1


def test_pull_time_is_bottom_to_top_to_bottom_and_none_for_a_clipped_rep():
    head, elbow = make_rep_series(peaks=[0.4], valley_elbows=[165, 165], up_s=1.0, hold_s=0.3, down_s=1.2)
    r = run(head, elbow).reps[0]
    assert 2.0 < r.pull_time_s < 2.8 and 0.7 < r.ascent_s < 1.2
    cut = run(head[: int(np.argmax(head)) + 5], elbow[: int(np.argmax(head)) + 5]).reps[0]
    assert cut.pull_time_s is None and cut.ascent_s > 0


def test_a_rep_cut_off_by_the_end_of_the_clip_is_unknown_not_full():
    head, elbow = make_rep_series(peaks=[0.4], valley_elbows=[165, 165])
    peak = int(np.argmax(head))
    cut = run(head[: peak + 5], elbow[: peak + 5]).reps[0]  # threshold set, head cleared, start bottom fine
    assert cut.clipped and cut.head_ok and cut.elbow_ok is None and cut.full is None
    assert REASON_CLIP_ENDED_MID_REP in cut.reasons

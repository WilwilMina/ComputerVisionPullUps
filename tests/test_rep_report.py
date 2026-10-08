import numpy as np
from synthetic import make_rep_series

from pullup_rom.rep_report import format_reps
from pullup_rom.reps import RepParams, detect_reps


def test_rep_table_shows_each_bottom_elbow_angle_and_says_when_the_threshold_is_unset():
    head, elbow = make_rep_series(peaks=[0.4, -0.2], valley_elbows=[170, 155, 140])
    analysis = detect_reps(head, elbow, 30.0, RepParams())  # no dead-hang threshold
    text = "\n".join(format_reps(analysis, "world"))
    assert "reps: 2" in text and "UNSET (uncalibrated)" in text
    assert "BOTTOM" in text and "155" in text and "140" in text
    assert "UNKNOWN" in text and "NOT FULL" in text  # clears the bar but unknown elbow / head below the bar
    assert "FULL " not in text.replace("NOT FULL", "")  # nothing is called FULL while uncalibrated


def test_rep_table_with_a_threshold_labels_full_and_lists_discarded_attempts():
    head, elbow = make_rep_series(peaks=[0.4], valley_elbows=[165, 165])
    peak = int(np.argmax(head))
    head = head.copy()
    head[peak - 15 : peak + 15] = np.nan
    analysis = detect_reps(head, elbow, 30.0, RepParams(dead_hang_elbow_min_deg=150.0))
    text = " ".join(format_reps(analysis, "image"))
    assert "uncertain (tracking gap)" in text and " to " in text

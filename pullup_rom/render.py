"""Annotated overlay video: skeleton, bar line, rep counter, head-height graph, per-rep table.

The video is drawn on the left, a side panel (counter + table) on the right and a head-height graph underneath.
UNKNOWN is shown as unknown, never guessed: a rep is FULL only if the head cleared the bar AND the arms
straightened, and the arms column says UNKNOWN until the dead-hang threshold for the elbow source is calibrated.

Pure helpers (layout, table text, warning text) are separate from the drawing so they can be tested without video.
Layout geometry lives in config.py (RENDER_*); colours and font sizes are style and stay here.
Nothing here is an accuracy figure.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from . import config
from .metrics import Signals
from .pose import PoseResult
from .reps import DiscardedAttempt, Rep, RepAnalysis

# BGR colours
GREEN = (80, 200, 80)
RED = (70, 70, 230)
AMBER = (0, 170, 255)
GREY = (150, 150, 150)
WHITE = (240, 240, 240)
DARK = (28, 28, 28)
PANEL_BG = (38, 38, 38)
CARRIED = (255, 160, 60)  # blue: a bar value carried from neighbouring frames (not orange, which reads like AMBER)
PURPLE = (200, 120, 170)
BANNER_RED = (30, 30, 190)

_FONT = None  # set lazily so importing this module does not need OpenCV


@dataclass(frozen=True)
class Layout:
    scale: float  # source frame -> drawn video
    vid_w: int
    vid_h: int
    body_h: int  # height of the video row (at least the panel's minimum height)
    panel_w: int
    graph_h: int
    canvas_w: int
    canvas_h: int


@dataclass(frozen=True)
class RepRow:
    number: str
    head_cleared: str  # "yes" | "no"
    arms_straight: str  # "yes" | "no" | "UNKNOWN"
    bottom_elbow: str  # e.g. "151 deg" | "n/a"
    pull_time: str  # e.g. "2.4 s" | "n/a"
    verdict: str  # "FULL" | "NOT FULL" | "UNKNOWN"


@dataclass(frozen=True)
class RenderStats:
    frames: int
    size: tuple[int, int]  # (width, height) of the output video


def _even(x: int) -> int:
    return x + (x % 2)


def compute_layout(frame_size: tuple[int, int]) -> Layout:
    w, h = frame_size
    scale = min(1.0, config.RENDER_VIDEO_MAX_HEIGHT_PX / h)
    vid_w, vid_h = max(2, _even(int(round(w * scale)))), max(2, _even(int(round(h * scale))))
    body_h = max(vid_h, config.RENDER_MIN_PANEL_HEIGHT_PX)
    panel_w = config.RENDER_PANEL_WIDTH_PX
    graph_h = config.RENDER_GRAPH_HEIGHT_PX
    return Layout(scale, vid_w, vid_h, body_h, panel_w, graph_h, _even(vid_w + panel_w), _even(body_h + graph_h))


# ---------- text (pure) ----------

def _yes_no(flag: Optional[bool]) -> str:
    return "UNKNOWN" if flag is None else ("yes" if flag else "no")


def rep_row(rep: Rep) -> RepRow:
    return RepRow(
        number=str(rep.index),
        head_cleared="yes" if rep.head_ok else "no",
        arms_straight=_yes_no(rep.elbow_ok),
        bottom_elbow="n/a" if not np.isfinite(rep.bottom_elbow_deg) else f"{rep.bottom_elbow_deg:.0f} deg",
        pull_time="n/a" if rep.pull_time_s is None else f"{rep.pull_time_s:.1f} s",
        verdict={True: "FULL", False: "NOT FULL", None: "UNKNOWN"}[rep.full],
    )


def verdict_color(verdict: str) -> tuple[int, int, int]:
    return {"FULL": GREEN, "NOT FULL": RED}.get(verdict, AMBER)


def invalid_share(signals: Signals) -> float:
    """Share of frames where the head signal or the elbow angle could not be used."""
    return float((~(signals.head_valid & signals.elbow_valid)).mean()) if len(signals.head_valid) else 1.0


def warning_text(share: float) -> Optional[str]:
    if share >= config.WARN_INVALID_SHARE:
        return f"WARNING: {share:.0%} of frames lack a usable head or elbow signal - results unreliable"
    return None


def counter_texts(analysis: RepAnalysis, frame: int) -> tuple[str, str, Optional[int]]:
    """(headline, tally, in-progress rep number). Only reps that have finished are counted."""
    done = [r for r in analysis.reps if r.end_frame <= frame and not r.clipped]
    full = sum(r.full is True for r in done)
    not_full = sum(r.full is False for r in done)
    unknown = sum(r.full is None for r in done)
    live = next((r.index for r in analysis.reps if r.start_frame <= frame < r.end_frame or (r.clipped and r.start_frame <= frame)), None)
    return f"REPS: {len(done)}", f"FULL {full}   NOT FULL {not_full}   UNKNOWN {unknown}", live


def table_entries(analysis: RepAnalysis, frame: int, fps: float) -> list[tuple[str, object]]:
    """Rows to show at `frame`, in time order: finished reps and attempts the logic could not trust."""
    entries: list[tuple[float, str, object]] = []
    for r in analysis.reps:
        if r.end_frame <= frame or (r.clipped and r.peak_frame <= frame):
            entries.append((r.end_s, "rep", r))
    for d in analysis.discarded:
        if d.end_frame <= frame:
            entries.append((d.end_s, "discard", d))
    entries.sort(key=lambda e: e[0])
    return [(kind, obj) for _, kind, obj in entries]


# ---------- video and keypoint consistency ----------

def video_info(video_path: str | Path) -> tuple[int, int, int]:
    """(width, height, reported frame count) of a video; raises ValueError if it cannot be opened."""
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"cannot read video: {video_path}")
    info = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    cap.release()
    return info


def check_video_matches(video_path: str | Path, pose: PoseResult) -> None:
    """Refuse keypoints that were not made from this video (wrong size or length), instead of drawing nonsense."""
    w, h, frames = video_info(video_path)
    pw, ph = pose.frame_size
    if (w, h) != (pw, ph):
        raise ValueError(f"these keypoints are for a {pw}x{ph} video but this video is {w}x{h}; wrong cache file?")
    if frames > 0 and abs(frames - pose.num_frames) > config.RENDER_FRAME_MATCH_TOLERANCE:
        raise ValueError(f"these keypoints cover {pose.num_frames} frames but this video has {frames}; wrong cache file?")


# ---------- drawing ----------

def _font():
    global _FONT
    if _FONT is None:
        import cv2

        _FONT = cv2.FONT_HERSHEY_SIMPLEX
    return _FONT


def _text(img, s: str, org: tuple[int, int], size: float = 0.5, color=WHITE, thick: int = 1) -> None:
    import cv2

    cv2.putText(img, s, org, _font(), size, color, thick, cv2.LINE_AA)


def draw_skeleton(frame: np.ndarray, kp: np.ndarray, scale: float) -> None:
    import cv2

    thick = max(1, int(round(frame.shape[0] / 240)))
    for a, b in config.PREVIEW_BONES:
        if np.isfinite(kp[[a, b], :2]).all():
            pa = tuple(int(v * scale) for v in kp[a, :2])
            pb = tuple(int(v * scale) for v in kp[b, :2])
            cv2.line(frame, pa, pb, (0, 230, 0), thick, cv2.LINE_AA)
    for j in {k for bone in config.PREVIEW_BONES for k in bone}:
        if np.isfinite(kp[j, :2]).all():
            conf = kp[j, 2]
            color = (0, 0, 230) if conf >= config.MIN_LANDMARK_CONFIDENCE else (150, 150, 150)  # grey = low confidence
            cv2.circle(frame, (int(kp[j, 0] * scale), int(kp[j, 1] * scale)), thick + 2, color, -1, cv2.LINE_AA)


def draw_bar_line(frame: np.ndarray, signals: Signals, i: int, frame_h: int, scale: float) -> None:
    """Dashed line at the estimated bar height; blue when the value is carried from neighbouring frames."""
    import cv2

    bar = signals.bar_height_px[i]
    if not np.isfinite(bar):
        return
    y = int(round((frame_h - bar) * scale))
    if not 0 <= y < frame.shape[0]:
        return
    color = CARRIED if signals.bar_carried[i] else (255, 200, 0)
    for x in range(0, frame.shape[1], 14):
        cv2.line(frame, (x, y), (min(x + 7, frame.shape[1] - 1), y), color, 1, cv2.LINE_AA)
    _text(frame, "bar" + (" (carried)" if signals.bar_carried[i] else ""), (6, max(12, y - 5)), 0.4, color)


def _legend(img: np.ndarray, x: int, y: int, x_max: int) -> None:
    import cv2

    for label, color in (
        ("FULL", GREEN), ("NOT FULL", RED), ("UNKNOWN", AMBER), ("untracked", GREY), ("bar carried", CARRIED), ("? uncertain", PURPLE),
    ):
        (w, _), _ = cv2.getTextSize(label, _font(), 0.38, 1)
        if x + w > x_max:
            break
        _text(img, label, (x, y), 0.38, color)
        x += w + 14


def build_graph(signals: Signals, analysis: RepAnalysis, layout: Layout) -> tuple[np.ndarray, tuple]:
    """The static head-height graph (image, geometry): head minus bar over time, bar line at 0, rep bands, invalid spans."""
    import cv2

    w, h = layout.canvas_w, layout.graph_h
    img = np.full((h, w, 3), DARK, dtype=np.uint8)
    m_left, m_right, m_top, m_bottom = config.RENDER_GRAPH_MARGINS_PX
    left, right, top, bottom = m_left, w - m_right, m_top, h - m_bottom
    t = signals.t_s
    t0, t1 = float(t[0]), max(float(t[-1]), float(t[0]) + 1e-6)
    vals = signals.head_minus_bar[np.isfinite(signals.head_minus_bar)]
    lo = min(-0.5, float(vals.min()) - 0.1) if vals.size else -1.0
    hi = max(0.5, float(vals.max()) + 0.1) if vals.size else 1.0

    def X(sec):
        return int(left + (sec - t0) / (t1 - t0) * (right - left))

    def Y(v):
        return int(bottom - (v - lo) / (hi - lo) * (bottom - top))

    def shade(a, b, color, alpha):
        x0, x1 = max(left, X(a)), min(right, max(X(b), X(a) + 1))
        if x1 > x0:
            roi = img[top:bottom, x0:x1]
            img[top:bottom, x0:x1] = cv2.addWeighted(roi, 1 - alpha, np.full_like(roi, color), alpha, 0)

    def runs(mask):
        m = np.concatenate(([False], mask, [False])).astype(np.int8)
        d = np.diff(m)
        return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))

    for s, e in runs(~signals.head_valid):
        shade(t[s], t[min(e, len(t)) - 1], GREY, 0.30)
    for s, e in runs(signals.bar_carried & signals.head_valid):
        shade(t[s], t[min(e, len(t)) - 1], CARRIED, 0.30)
    label_y = top - 6  # rep numbers sit between the title (above) and the plot (below)
    for r in analysis.reps:
        color = verdict_color(rep_row(r).verdict)
        shade(r.start_s, r.end_s, color, 0.28)
        _text(img, str(r.index), (max(left, X(r.peak_s) - 4), label_y), 0.45, color, 1)
    for d in analysis.discarded:
        shade(d.start_s, d.end_s, PURPLE, 0.35)
        _text(img, "?", (max(left, X((d.start_s + d.end_s) / 2) - 4), label_y), 0.5, PURPLE, 1)

    cv2.line(img, (left, Y(0.0)), (right, Y(0.0)), (255, 200, 0), 1, cv2.LINE_AA)  # the bar line
    _text(img, "bar", (6, Y(0.0) + 4), 0.45, (255, 200, 0))
    cv2.rectangle(img, (left, top), (right, bottom), (80, 80, 80), 1)
    pts_run: list[tuple[int, int]] = []
    for sec, v in zip(t, signals.head_minus_bar):
        if np.isfinite(v):
            pts_run.append((X(sec), Y(v)))
        elif pts_run:
            cv2.polylines(img, [np.array(pts_run, dtype=np.int32)], False, WHITE, 1, cv2.LINE_AA)
            pts_run = []
    if pts_run:
        cv2.polylines(img, [np.array(pts_run, dtype=np.int32)], False, WHITE, 1, cv2.LINE_AA)
    _text(img, "head height vs bar (body scales; above the line = head over the bar)", (left, 14), 0.42, GREY)
    _text(img, "0s", (left, h - 6), 0.4, GREY)
    _text(img, f"{t1:.0f}s", (right - 28, h - 6), 0.4, GREY)
    _legend(img, left + 40, h - 6, right - 40)
    return img, (left, right, top, bottom, t0, t1, lo, hi)


def draw_panel(panel: np.ndarray, analysis: RepAnalysis, frame: int, fps: float) -> None:
    import cv2

    panel[:] = PANEL_BG
    headline, tally, live = counter_texts(analysis, frame)
    _text(panel, headline, (14, 40), 1.1, WHITE, 2)
    _text(panel, tally, (14, 66), 0.5, GREY)
    if live is not None:
        _text(panel, f"rep {live} in progress", (14, 88), 0.5, AMBER)
    xs = config.RENDER_TABLE_COLUMNS_X
    y0, row_h = config.RENDER_TABLE_TOP_PX, config.RENDER_TABLE_ROW_H_PX
    for name, x in zip(("#", "Head over bar", "Arms straight", "Bottom elbow", "Pull time", "Verdict"), xs):
        _text(panel, name, (x, y0), 0.38, GREY)
    cv2.line(panel, (10, y0 + 6), (panel.shape[1] - 10, y0 + 6), (90, 90, 90), 1)
    capacity = max(2, (panel.shape[0] - (y0 + 14) - 30) // row_h)
    entries = table_entries(analysis, frame, fps)
    hidden = max(0, len(entries) - capacity)
    shown = entries[hidden:]
    y = y0 + row_h
    if hidden:
        _text(panel, f"... {hidden + 1} earlier row(s) not shown", (xs[0], y), 0.4, GREY)
        y += row_h
        shown = shown[1:]
    for kind, obj in shown:
        if kind == "rep":
            row = rep_row(obj)  # type: ignore[arg-type]
            _text(panel, row.number, (xs[0], y), 0.5, WHITE)
            _text(panel, row.head_cleared, (xs[1], y), 0.5, GREEN if row.head_cleared == "yes" else RED)
            _text(panel, row.arms_straight, (xs[2], y), 0.5, {"yes": GREEN, "no": RED}.get(row.arms_straight, AMBER))
            _text(panel, row.bottom_elbow, (xs[3], y), 0.5, WHITE)
            _text(panel, row.pull_time, (xs[4], y), 0.5, WHITE)
            _text(panel, row.verdict, (xs[5], y), 0.45, verdict_color(row.verdict), 1)
        else:
            d: DiscardedAttempt = obj  # type: ignore[assignment]
            _text(panel, f"? {d.label}  {d.start_s:.1f}-{d.end_s:.1f} s", (xs[0], y), 0.42, PURPLE)
        y += row_h
    _text(panel, "all thresholds provisional; UNKNOWN = not calibrated", (xs[0], panel.shape[0] - 10), 0.36, GREY)


def wrap_text(text: str, max_width_px: int, scale: float = 0.42) -> list[str]:
    """Split `text` into lines no wider than `max_width_px` at the given font scale (a single over-long word keeps its own line)."""
    import cv2

    lines: list[str] = []
    current = ""
    for word in text.split():
        trial = (current + " " + word).strip()
        if not current or cv2.getTextSize(trial, _font(), scale, 1)[0][0] <= max_width_px:
            current = trial
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def banner_lines(warnings: list[str], width_px: int) -> list[str]:
    lines: list[str] = []
    for w in warnings:
        lines.extend(wrap_text(w, width_px - 16))
    if len(lines) > config.RENDER_BANNER_MAX_LINES:
        lines = lines[: config.RENDER_BANNER_MAX_LINES]
        lines[-1] = lines[-1] + " ..."
    return lines


def draw_banner(frame: np.ndarray, lines: list[str]) -> None:
    """A red banner across the top of the video, one text line per row, so nothing is cut off at the edge."""
    import cv2

    line_h = config.RENDER_BANNER_LINE_H_PX
    cv2.rectangle(frame, (0, 0), (frame.shape[1], len(lines) * line_h + 8), BANNER_RED, -1)
    for i, line in enumerate(lines):
        _text(frame, line, (8, 8 + (i + 1) * line_h - 6), 0.42, WHITE, 1)


def render_video(
    video_path: str | Path,
    pose: PoseResult,
    signals: Signals,
    analysis: RepAnalysis,
    out_path: str | Path,
    warnings: Optional[list[str]] = None,
) -> RenderStats:
    """Write the annotated video; one output frame per pose frame. Raises ValueError if it cannot be read or written."""
    import cv2

    layout = compute_layout(pose.frame_size)
    graph, geom = build_graph(signals, analysis, layout)
    left, right, top, bottom, t0, t1, lo, hi = geom
    if warnings is None:
        w = warning_text(invalid_share(signals))
        warnings = [w] if w else []

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"cannot read video: {video_path}")
    try:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        cap.release()
        raise ValueError(f"cannot create the output folder for {out_path}: {e.strerror or e}")
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), pose.fps, (layout.canvas_w, layout.canvas_h))
    if not writer.isOpened():
        cap.release()
        raise ValueError(f"cannot write video: {out_path}")

    frame_h = pose.frame_size[1]
    written = 0
    for i in range(pose.num_frames):
        ok, frame = cap.read()
        if not ok:
            break
        small = frame.copy() if frame.shape[:2] == (layout.vid_h, layout.vid_w) else cv2.resize(frame, (layout.vid_w, layout.vid_h))
        draw_bar_line(small, signals, i, frame_h, layout.scale)
        draw_skeleton(small, pose.keypoints[i], layout.scale)
        if warnings:
            draw_banner(small, banner_lines(warnings, layout.vid_w))

        canvas = np.zeros((layout.canvas_h, layout.canvas_w, 3), dtype=np.uint8)
        canvas[: layout.vid_h, : layout.vid_w] = small
        panel = np.empty((layout.body_h, layout.canvas_w - layout.vid_w, 3), dtype=np.uint8)
        draw_panel(panel, analysis, i, pose.fps)
        canvas[: layout.body_h, layout.vid_w :] = panel
        g = graph.copy()
        x = int(left + (signals.t_s[i] - t0) / (t1 - t0) * (right - left))
        cv2.line(g, (x, top), (x, bottom), (0, 200, 255), 1, cv2.LINE_AA)
        v = signals.head_minus_bar[i]
        if np.isfinite(v):
            cv2.circle(g, (x, int(bottom - (v - lo) / (hi - lo) * (bottom - top))), 4, (0, 200, 255), -1, cv2.LINE_AA)
        canvas[layout.body_h : layout.body_h + layout.graph_h, :] = g[: layout.canvas_h - layout.body_h, : layout.canvas_w]
        writer.write(canvas)
        written += 1
    cap.release()
    writer.release()
    return RenderStats(frames=written, size=(layout.canvas_w, layout.canvas_h))

"""Debug plot of slice-2 signals from a keypoints .npz. Not the final overlay (render.py)."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from .metrics import MetricsParams, Signals, compute_signals
from .pose import load_pose


def plot_signals(s: Signals, out_path: str | Path, title: str = "") -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    ax[0].plot(s.t_s, s.wrist_height_px, color="tab:red", lw=0.8, alpha=0.5, label="wrists (raw mean)")
    ax[0].plot(s.t_s, s.head_height_px, label="head height")
    ax[0].plot(s.t_s, s.bar_height_px, label="bar line (rolling median of gripping wrists)")
    ax[0].set_ylabel("height up (px)")
    ax[0].legend(loc="upper right")

    ax[1].plot(s.t_s, s.head_minus_bar, color="tab:green")
    ax[1].axhline(0.0, color="k", lw=0.8, ls="--")
    ax[1].set_ylabel("head - bar\n(body scales)")

    ax[2].plot(s.t_s, s.elbow_world_deg, label="3D world angle", alpha=0.9)
    ax[2].plot(s.t_s, s.elbow_image_deg, label="2D image angle", alpha=0.9)
    ax[2].set_ylabel("elbow angle (deg)")
    ax[2].set_xlabel("time (s)")
    ax[2].legend(loc="lower right")

    for i, a in enumerate(ax):
        invalid = ~s.elbow_valid if i == 2 else ~s.head_valid
        for start, end in _spans(invalid):  # grey: the rep logic must ignore these frames
            a.axvspan(s.t_s[start], s.t_s[min(end, len(s.t_s)) - 1], color="grey", alpha=0.2, lw=0)
        if i < 2:
            for start, end in _spans(s.bar_carried & s.head_valid):  # orange: bar value carried from neighbouring frames
                a.axvspan(s.t_s[start], s.t_s[min(end, len(s.t_s)) - 1], color="tab:orange", alpha=0.18, lw=0)
    ax[0].set_title(title or "Slice 2 signals (provisional settings; grey = not valid, orange = bar carried from neighbours)")
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def _spans(mask: np.ndarray) -> list[tuple[int, int]]:
    m = np.concatenate(([False], mask, [False])).astype(np.int8)
    d = np.diff(m)
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def main(argv: Optional[Sequence[str]] = None) -> None:
    p = argparse.ArgumentParser(description="Plot slice-2 signals from a keypoints .npz.")
    p.add_argument("keypoints")
    p.add_argument("-o", "--output", required=True)
    args = p.parse_args(argv)
    s = compute_signals(load_pose(args.keypoints), MetricsParams())
    plot_signals(s, args.output)
    print(f"head valid: {int(s.head_valid.sum())}/{len(s.head_valid)}  elbow valid: {int(s.elbow_valid.sum())}/{len(s.elbow_valid)}  plot: {args.output}")


if __name__ == "__main__":
    main()

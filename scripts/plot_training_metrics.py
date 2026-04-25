#!/usr/bin/env python3
"""
Plot reward proxy + entropy from runs/training_metrics.jsonl (written by JsonlTrainingMetricsCallback).

Usage (from agentguard-gym/):
  uv run python scripts/plot_training_metrics.py
  # optional:  uv run python scripts/plot_training_metrics.py --jsonl custom.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_JSONL = Path("runs/training_metrics.jsonl")
OUT_DIR = Path("results")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--jsonl", type=Path, default=DEFAULT_JSONL)
    p.add_argument("--nash-x", type=float, default=None, help="optional vertical line (e.g. curriculum Nash step)")
    args = p.parse_args()
    if not args.jsonl.is_file():
        print(f"Missing {args.jsonl} — run train_adversarial.py first to append metrics from JsonlTrainingMetricsCallback.")
        raise SystemExit(1)
    rows = [json.loads(line) for line in args.jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        print("JSONL is empty.")
        raise SystemExit(1)

    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        print("Install matplotlib: uv sync --extra dev  (or pip install matplotlib)")
        raise e

    steps = [int(r["step"]) for r in rows]
    win = [r.get("win_rate") for r in rows]
    fpr = [r.get("fp_rate") for r in rows]
    ent = [r.get("entropy") for r in rows]

    OUT_DIR.mkdir(exist_ok=True)
    out = OUT_DIR / "training_curve.png"

    fig, ax1 = plt.subplots(figsize=(9, 4))
    ax2 = None
    if any(x is not None for x in win):
        ax1.plot(steps, win, label="win_rate", color="#1D9E75", linewidth=1.5)
    if any(x is not None for x in fpr):
        ax1.plot(steps, fpr, label="fp_rate", color="#C73E1D", linewidth=1.0, alpha=0.8)
    ax1.set_xlabel("GRPO step")
    ax1.set_ylabel("rate")
    ax1.grid(True, alpha=0.3)

    if any(x is not None for x in ent):
        ax2 = ax1.twinx()
        ax2.plot(steps, ent, label="entropy", color="#185FA5", alpha=0.7, linestyle="--")
        ax2.set_ylabel("entropy")

    if args.nash_x is not None:
        ax1.axvline(x=args.nash_x, color="#BA7517", linestyle="--", label=f"Nash @ {args.nash_x}", alpha=0.8)

    lines, labels = ax1.get_legend_handles_labels()
    if ax2 is not None:
        l2, lab2 = ax2.get_legend_handles_labels()
        lines += l2
        labels += lab2
    if lines:
        ax1.legend(lines, labels, loc="best")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()

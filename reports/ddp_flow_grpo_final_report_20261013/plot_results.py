#!/usr/bin/env python3
"""Render the report figure from its checked-in aggregate CSV (requires matplotlib)."""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent


def main():
    with (ROOT / "final_metrics.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    colors = ["#4169a1", "#198273"]
    plt.rcParams.update({"font.size": 11, "svg.fonttype": "none", "svg.hashsalt": "ddp-flow-grpo-final-20261013"})
    fig, ax = plt.subplots(figsize=(8.8, 3.9), layout="constrained")
    for index, (row, color) in enumerate(zip(rows, colors)):
        mean, low, high = (float(row[key]) for key in ("delta_points", "ci95_low_points", "ci95_high_points"))
        y = 1 - index
        ax.errorbar(mean, y, xerr=[[mean - low], [high - mean]], fmt="o", color=color, capsize=6, markersize=8, linewidth=2)
        ax.text(1.035, y, f"{mean:+.4f}\n[{low:+.4f}, {high:+.4f}]", transform=ax.get_yaxis_transform(), va="center", ha="left", color=color, fontsize=10)
    ax.axvline(0, color="#666666", linestyle="--", linewidth=1)
    ax.set(yticks=[1, 0], yticklabels=["PDMS (v1.1)", "EPDMS (v2)"], ylim=(-0.6, 1.6), xlim=(-0.065, 0.41))
    ax.set_xticks([-0.05, 0, 0.1, 0.2, 0.3, 0.4])
    ax.set_xlabel("RL minus own SFT (score points; 100-point scale)")
    ax.set_title("Final DDP Flow-GRPO: small gains, conditional uncertainty", loc="left", pad=30, fontweight="bold", fontsize=13)
    ax.text(0, 1.07, "Update 12912 | 12,146 scenes / 136 logs | 5 inference seeds / 1 training seed", transform=ax.transAxes, color="#555555", fontsize=10)
    ax.grid(axis="x", color="#e4e7eb", linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.text(0.02, -0.035, "Intervals: paired whole-log bootstrap, 2,000 draws, seed 42; average inference seeds within scene first.", fontsize=9, color="#555555")
    svg_path = ROOT / "final_paired_gain.svg"
    fig.savefig(svg_path, bbox_inches="tight", metadata={"Date": None})
    svg_path.write_text("\n".join(line.rstrip() for line in svg_path.read_text().splitlines()) + "\n")
    fig.savefig(ROOT / "final_paired_gain.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()

"""Archive complete fixed-milestone BEV evaluations and plot learning curves."""
import argparse
import csv
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root, output = Path(args.artifacts), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for variant in ("tasks", "control"):
        for step, passes in ((456, 1), (911, 2), (1821, 4)):
            source = root / f"bev_{variant}_dev_tasks_step{step}"
            metrics = json.loads((source / "metrics.json").read_text())
            with (source / "metrics.csv").open() as handle:
                scenes = list(csv.DictReader(handle))
            if metrics["failed"] or metrics["scenes"] != 1696 or len(scenes) != 1696:
                raise ValueError(f"Incomplete evaluation: {source}")
            destination = output / variant / f"dev_step{step}"
            destination.mkdir(parents=True, exist_ok=True)
            for name in ("metrics.json", "metrics.csv", "manifest.json"):
                shutil.copyfile(source / name, destination / name)
            rows.append(dict(variant=variant, step=step, passes=passes, **metrics))
    with (output / "task_learning_curves.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    figure, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for ax, key, title in zip(
        axes,
        ("occupied_iou", "dense_motion_ADE", "pair_geometry_loss"),
        ("Current occupied IoU (higher better)", "GT-cell motion ADE, m (lower better)", "Pair geometry loss (lower better)"),
    ):
        for variant in ("tasks", "control"):
            selected = [r for r in rows if r["variant"] == variant]
            ax.plot([r["passes"] for r in selected], [r[key] for r in selected], "o-", label=variant)
        if key == "dense_motion_ADE":
            ax.axhline(rows[0]["stationary_dense_ADE"], linestyle="--", color="gray", label="stationary")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Training passes over 7284 scenes")
        ax.set_xticks([1, 2, 4])
        ax.grid(alpha=.25)
        ax.legend()
    figure.suptitle("Same 1696 development scenes; one training seed; task metrics, not planning scores", fontsize=10)
    figure.tight_layout()
    for extension in ("png", "pdf"):
        figure.savefig(output / f"task_learning_curves.{extension}", dpi=160)
    plt.close(figure)
    report = {
        "rows": rows,
        "selection": "All fixed 1/2/4-pass milestones retained; no checkpoint chosen by PDMS.",
        "scope": "Motion at GT current occupied cells, not end-to-end actor forecasting; current occupancy only.",
        "convergence": "Motion still improves while current IoU is nonmonotonic; stable convergence is not established.",
    }
    (output / "SUMMARY.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"evaluations": len(rows), "scene_failures": sum(r["failed"] for r in rows)}))


if __name__ == "__main__":
    main()

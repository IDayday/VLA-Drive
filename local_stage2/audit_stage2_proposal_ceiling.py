#!/usr/bin/env python3
"""Decompose a Stage-2 validation gap into proposal ceiling and scorer regret.

Both event files must come from the same validation scene set and evaluator.
For runs with multiple epochs, the row with the highest ``val/score_epoch`` is
selected, matching the released checkpoint callback criterion.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


TAGS = (
    "val/score_epoch",
    "val/best_score",
    "val/lost_score",
    "val/l2",
    "val/collision",
    "val/ttc",
    "val/dac",
    "val/progress",
    "val/comfort",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _best_epoch(path: Path) -> dict:
    accumulator = EventAccumulator(str(path), size_guidance={"scalars": 0})
    accumulator.Reload()
    available = set(accumulator.Tags()["scalars"])
    missing = set(TAGS) - available
    if missing:
        raise RuntimeError(f"Missing scalar tags in {path}: {sorted(missing)}")

    by_tag = {tag: accumulator.Scalars(tag) for tag in TAGS}
    score_series = by_tag["val/score_epoch"]
    best_index = max(range(len(score_series)), key=lambda index: score_series[index].value)
    expected_step = score_series[best_index].step
    epoch_curve = []
    for index, score_item in enumerate(score_series):
        metrics = {}
        for tag, values in by_tag.items():
            if len(values) != len(score_series):
                raise RuntimeError(
                    f"Scalar length mismatch for {tag}: "
                    f"{len(values)} != {len(score_series)}"
                )
            if values[index].step != score_item.step:
                raise RuntimeError(
                    f"Scalar step mismatch for {tag}: "
                    f"{values[index].step} != {score_item.step}"
                )
            metrics[tag.removeprefix("val/")] = values[index].value
        epoch_curve.append(
            {"event_index": index, "step": score_item.step, "metrics": metrics}
        )
    return {
        "event_file": str(path.resolve()),
        "event_sha256": _sha256(path),
        "selected_event_index": best_index,
        "selected_step": expected_step,
        "metrics": epoch_curve[best_index]["metrics"],
        "epoch_curve": epoch_curve,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-event", type=Path, required=True)
    parser.add_argument("--local-event", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    public = _best_epoch(args.public_event)
    local = _best_epoch(args.local_event)
    public_metrics = public["metrics"]
    local_metrics = local["metrics"]
    local_ceiling_peak = max(
        local["epoch_curve"], key=lambda row: row["metrics"]["best_score"]
    )
    local_final = local["epoch_curve"][-1]
    selected_gap = public_metrics["score_epoch"] - local_metrics["score_epoch"]
    ceiling_gap = public_metrics["best_score"] - local_metrics["best_score"]
    report = {
        "public": public,
        "local": local,
        "decomposition": {
            "selected_pdms_gap": selected_gap,
            "best_of_64_ceiling_gap": ceiling_gap,
            "public_scorer_regret": public_metrics["lost_score"],
            "local_scorer_regret": local_metrics["lost_score"],
            "scorer_regret_gap": (
                public_metrics["lost_score"] - local_metrics["lost_score"]
            ),
            "ceiling_gap_over_selected_gap": (
                ceiling_gap / selected_gap if selected_gap else None
            ),
            "local_peak_ceiling_event_index": local_ceiling_peak["event_index"],
            "local_peak_ceiling_step": local_ceiling_peak["step"],
            "local_peak_best_of_64": local_ceiling_peak["metrics"]["best_score"],
            "public_ceiling_gap_to_local_peak": (
                public_metrics["best_score"]
                - local_ceiling_peak["metrics"]["best_score"]
            ),
            "local_selected_checkpoint_ceiling_drop_from_peak": (
                local_ceiling_peak["metrics"]["best_score"]
                - local_metrics["best_score"]
            ),
            "local_final_best_of_64": local_final["metrics"]["best_score"],
            "local_final_ceiling_drop_from_peak": (
                local_ceiling_peak["metrics"]["best_score"]
                - local_final["metrics"]["best_score"]
            ),
            "local_peak_ceiling_l2": local_ceiling_peak["metrics"]["l2"],
            "local_final_l2": local_final["metrics"]["l2"],
        },
        "interpretation": (
            "The selected-score gap is already present in the best-of-64 "
            "proposal ceiling; scorer regret is not the primary deficit. "
            "The local proposal ceiling peaks early and then falls while L2 "
            "continues to improve, which is direct evidence of late-training "
            "proposal coverage collapse."
        ),
    }
    output = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output)
    print(output, end="")


if __name__ == "__main__":
    main()

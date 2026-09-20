"""Read-only summary of native Flow-GRPO logs; never changes training.

Fresh behavior metrics are counted only at inner_epoch=0. Quantiles already
aggregated per batch retain that designation; averaging them is not a global
sample quantile. Reward means across different tokens are not learning curves.
"""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np


FIELDS = [
    "update", "policy_version", "inner_epoch", "scene_count", "reward",
    "reward_zero_fraction", "all_zero_groups", "all_equal_groups",
    "advantage_nonzero_group_fraction", "advantage_absolute_mean",
    "candidate_pair_ade_scene_median_m", "candidate_pair_fde_scene_median_m",
    "candidate_xy_effective_rank", "pre_update_ratio_clip_fraction",
    "pre_update_ratio_min", "pre_update_ratio_max", "pre_update_ratio_mean",
    "post_update_probe_ratio_clip_fraction", "post_update_probe_ratio_min",
    "post_update_probe_ratio_max", "reference", "grpo", "sft", "loss",
    "pre_clip_grad_norm", "clip_scale", "reward_errors",
    "reward_component/ego_progress", "reward_component/no_at_fault_collisions",
    "reward_component/time_to_collision_within_bound",
]


def summary(rows):
    result = {}
    for name in rows[0]:
        values = np.array([r[name] for r in rows if r[name] is not None], dtype=float)
        if len(values):
            result[name] = dict(count=len(values), mean=float(values.mean()),
                                min=float(values.min()), max=float(values.max()),
                                median=float(np.median(values)))
    result["updates_with_gradient_clipping"] = sum(
        r["clip_scale"] is not None and r["clip_scale"] < 0.999999 for r in rows)
    return result


def analyze(source, output, through):
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    scenes = []
    progress_only = []
    digest = hashlib.sha256()
    with source.open("rb") as f:
        for raw in f:
            data = json.loads(raw)
            if data["update"] > through:
                break
            digest.update(raw)
            row = {k: data.get(k) for k in FIELDS}
            row["reference_coefficient"] = data["loss_coefficients"]["reference"]
            row["lr_used"] = data.get("lr_used", data["lr"])[0]
            for key in ["reward_group_std_quantiles", "reward_group_span_quantiles",
                        "pre_update_ratio_quantiles"]:
                for q in ["0.0", "0.01", "0.5", "0.99", "1.0"]:
                    row[f"{key}/batch_q{q}"] = data.get(key, {}).get(q)
            rows.append(row)
            if data["inner_epoch"] == 0:
                for rank in data["ranks"]:
                    # Each rank has exactly two scene microbatches. Their local
                    # q0/q1 are the two actual values, not inferred quantiles.
                    if rank["scene_count"] != 2:
                        raise ValueError("per-scene reconstruction requires two local scenes")
                    for q in ["0.0", "1.0"]:
                        scenes.append(dict(update=data["update"],
                            std=rank["reward_group_std_quantiles"][q],
                            span=rank["reward_group_span_quantiles"][q]))
                    other = ["no_at_fault_collisions", "drivable_area_compliance",
                             "driving_direction_compliance", "traffic_light_compliance",
                             "time_to_collision_within_bound", "lane_keeping", "history_comfort"]
                    all_one = all(rank[f"reward_component/{k}"] == 1 for k in other)
                    progress_only.append(dict(update=data["update"],
                        active_scenes=rank["advantage_nonzero_group_fraction"] * 2,
                        active_progress_only_lower_bound=(
                            rank["advantage_nonzero_group_fraction"] * 2 if all_one else 0)))
    if not rows or len({r["update"] for r in rows}) != len(rows):
        raise ValueError("empty log or duplicate updates")
    with gzip.open(output / "per_update.csv.gz", "wt") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    windows = [(1, 388), (389, 1000), (1001, 3000), (3001, 6000),
               (6001, 8700), (8701, through), (1, min(8700, through)), (1, through)]
    results = []
    for low, high in dict.fromkeys(windows):
        window = [r for r in rows if low <= r["update"] <= high]
        if not window:
            continue
        fresh = [r for r in window if r["inner_epoch"] == 0]
        reused = [r for r in window if r["inner_epoch"] == 1]
        selected_scenes = [s for s in scenes if low <= s["update"] <= high]
        stds = np.array([s["std"] for s in selected_scenes])
        spans = np.array([s["span"] for s in selected_scenes])
        selected_progress = [s for s in progress_only if low <= s["update"] <= high]
        active = sum(s["active_scenes"] for s in selected_progress)
        only_progress = sum(s["active_progress_only_lower_bound"] for s in selected_progress)
        distribution = dict(count=len(stds),
            std_quantiles=dict(zip([0, .25, .5, .75, .9, .99, 1],
                                  np.quantile(stds, [0, .25, .5, .75, .9, .99, 1]).tolist())),
            span_quantiles=dict(zip([0, .25, .5, .75, .9, .99, 1],
                                   np.quantile(spans, [0, .25, .5, .75, .9, .99, 1]).tolist())),
            zero_std_count=int((stds == 0).sum()),
            nonzero_std_below_1e_3_count=int(((stds > 0) & (stds < 1e-3)).sum()),
            nonzero_span_below_1e_2_count=int(((spans > 0) & (spans < 1e-2)).sum()),
            active_scenes=active, active_progress_only_lower_bound=only_progress,
            active_progress_only_fraction_lower_bound=only_progress / active if active else None,
            note="std and span sorted independently locally; no scene-wise association claimed")
        results.append(dict(first_update=window[0]["update"], last_update=window[-1]["update"],
                            per_scene_distribution=distribution,
                            all_updates=summary(window),
                            fresh_behavior=summary(fresh) if fresh else {},
                            second_inner=summary(reused) if reused else {}))
    report = dict(schema=1, source=str(source.resolve()), source_prefix_sha256=digest.hexdigest(),
                  last_update=rows[-1]["update"], count=len(rows),
                  caveats=["No double counting behavior batches across inner epochs",
                           "Batch quantile averages are not global sample quantiles",
                           "Different training tokens: reward trend is not a paired learning curve",
                           "Loss magnitudes cannot establish gradient competition"], windows=results)
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--through", required=True, type=int)
    args = parser.parse_args()
    result = analyze(args.source, args.output, args.through)
    print(json.dumps({k: v for k, v in result.items() if k != "windows"}))

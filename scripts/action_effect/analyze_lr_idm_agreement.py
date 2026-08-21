#!/usr/bin/env python3
"""Report stratified log-replay versus NAVSIM-v2 IDM pair agreement."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from research.action_effect.agreement import (  # noqa: E402
    binary_agreement_metrics,
    candidate_is_unsafe,
    replay_divergent_pair,
    scene_interaction_flags,
)
from research.action_effect.cache_io import write_json  # noqa: E402
from research.action_effect.probe_data import iter_jsonl  # noqa: E402


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise TypeError(f"configuration must be a mapping: {path}")
    return value


def _metric(value: float | int) -> str:
    if isinstance(value, int):
        return str(value)
    return "n/a" if not math.isfinite(float(value)) else f"{float(value):.4f}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPOSITORY_ROOT / "configs/action_effect/agreement.yaml",
    )
    parser.add_argument("--consequence-cache", type=Path, required=True)
    parser.add_argument("--pair-cache", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPOSITORY_ROOT / "reports/action_effect_world_model/lr_idm_agreement_artifacts",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=REPOSITORY_ROOT / "reports/action_effect_world_model/lr_idm_agreement.md",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = _load_yaml(args.config.resolve())
    rows = list(iter_jsonl(args.consequence_cache.resolve() / "consequences.jsonl"))
    candidate = {str(row["candidate_id"]): row for row in rows}
    flags = scene_interaction_flags(
        rows,
        low_ttc_seconds=float(config["low_ttc_seconds"]),
        dynamic_clearance_m=float(config["dynamic_clearance_m"]),
        score_tolerance=float(config["score_tolerance"]),
    )
    pair_rows = [
        row
        for row in iter_jsonl(args.pair_cache.resolve() / "pairs.jsonl")
        if row.get("reactive_order") is not None
        and row.get("log_replay_hard_relation") is not None
        and row.get("reactive_hard_relation") is not None
    ]
    with (args.pair_cache.resolve() / "thresholds.json").open("r", encoding="utf-8") as stream:
        pair_thresholds = json.load(stream)
    replay_divergent_threshold = float(pair_thresholds["divergent"])
    reactive_scenes = sorted(set(str(row["scene_id"]) for row in pair_rows))
    replay_positive = np.asarray(
        [row["log_replay_hard_relation"] == "different" for row in pair_rows], dtype=bool
    )
    reactive_positive = np.asarray(
        [row["reactive_hard_relation"] == "different" for row in pair_rows], dtype=bool
    )
    replay_order = np.asarray([row["log_replay_order"] for row in pair_rows], dtype=np.int8)
    reactive_order = np.asarray([row["reactive_order"] for row in pair_rows], dtype=np.int8)
    scene_ids = np.asarray([str(row["scene_id"]) for row in pair_rows], dtype=str)
    at_least_one_unsafe = np.asarray(
        [
            candidate_is_unsafe(candidate[str(row["candidate_i"])], "log_replay")
            or candidate_is_unsafe(candidate[str(row["candidate_j"])], "log_replay")
            for row in pair_rows
        ],
        dtype=bool,
    )
    masks = {
        "all_pairs": np.ones(len(pair_rows), dtype=bool),
        "divergent_pairs": np.asarray(
            [
                replay_divergent_pair(row, soft_threshold=replay_divergent_threshold)
                for row in pair_rows
            ],
            dtype=bool,
        ),
        "safety_boundary_pairs": np.asarray(
            [bool(row["safety_boundary"]) for row in pair_rows], dtype=bool
        ),
        "at_least_one_unsafe_pairs": at_least_one_unsafe,
        "dynamic_interaction_scenes": np.isin(scene_ids, list(flags["dynamic_interaction"])),
        "low_ttc_scenes": np.isin(scene_ids, list(flags["low_ttc"])),
        "idm_reacted_scenes": np.isin(scene_ids, list(flags["idm_reacted"])),
    }
    results: list[dict[str, Any]] = []
    for subset, mask in masks.items():
        metrics = binary_agreement_metrics(
            replay_positive[mask],
            reactive_positive[mask],
            replay_order[mask],
            reactive_order[mask],
        )
        results.append(
            {
                "subset": subset,
                "scene_count": int(len(set(scene_ids[mask].tolist()))),
                **metrics,
            }
        )
    critical_mask = (
        masks["safety_boundary_pairs"]
        | masks["at_least_one_unsafe_pairs"]
        | masks["low_ttc_scenes"]
        | masks["idm_reacted_scenes"]
    )
    # Confidence is relevant when either the hard pair relation or the
    # candidate ordering changes across traffic assumptions.
    disagreement = (replay_positive != reactive_positive) | (replay_order != reactive_order)
    critical_disagreement = int(np.sum(disagreement & critical_mask))
    critical_count = int(np.sum(critical_mask))
    relevance = config["confidence_relevance"]
    confidence_meaningful = bool(
        len(reactive_scenes) >= int(relevance["minimum_reactive_scenes"])
        and critical_disagreement >= int(relevance["minimum_disagreement_pairs"])
        and critical_disagreement / max(critical_count, 1)
        >= float(relevance["minimum_critical_disagreement_rate"])
    )
    decision = {
        "confidence_weighting_meaningful": confidence_meaningful,
        "reactive_scene_count": len(reactive_scenes),
        "critical_pair_count": critical_count,
        "critical_disagreement_count": critical_disagreement,
        "critical_disagreement_rate": critical_disagreement / max(critical_count, 1),
        "configured_requirements": relevance,
        "phase6_action": (
            "run confidence_aee as a required method"
            if confidence_meaningful
            else "retain confidence_aee configuration but do not treat it as a primary Gate-3 method"
        ),
    }
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "decision.json", decision)
    with (output_dir / "agreement.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(results[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(results)
    lines = [
        "# Stratified log-replay / IDM agreement",
        "",
        "The binary agreement target is whether each candidate pair has a hard-consequence "
        "difference under the corresponding traffic assumption. Ranking correlation uses the "
        "three-way score order (-1/tie/+1) and Kendall tau-b, so ties are retained.",
        "The divergent subset is reconstructed from replay-side hard/soft thresholds before "
        "identifiability confidence can relabel a conflicting pair as ambiguous.",
        "The confidence-support count uses the union of hard-relation and ranking disagreements "
        "inside the declared critical subsets.",
        "",
        "| Subset | Scenes | Pairs | Hard disagreements | Rank disagreements | Raw agreement | Positive agreement | Cohen κ | MCC | Kendall τ-b |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in results:
        formatted = {
            key: (str(value) if key == "subset" else _metric(value))
            for key, value in row.items()
        }
        lines.append(
            "| {subset} | {scene_count} | {pair_count} | {disagreement_count} | {ranking_disagreement_count} | {raw_agreement} | {positive_agreement} | {cohen_kappa} | {mcc} | {tie_aware_kendall} |".format(
                **formatted
            )
        )
    lines.extend(
        [
            "",
            "## Confidence-weighting decision",
            "",
            f"- Reactive scenes: {decision['reactive_scene_count']}.",
            f"- Critical disagreements: {decision['critical_disagreement_count']} / "
            f"{decision['critical_pair_count']} ({decision['critical_disagreement_rate']:.2%}).",
            f"- Confidence weighting meaningful for the primary Phase-6 matrix: "
            f"**{decision['confidence_weighting_meaningful']}**.",
            f"- Action: {decision['phase6_action']}.",
            "",
            "These labels remain `log_replay` and `reactive_model`; neither is described as a "
            "ground-truth counterfactual.",
            "",
        ]
    )
    args.report_path.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.report_path.resolve().write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

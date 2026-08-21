#!/usr/bin/env python3
"""Audit raw/effect structured channels for within-scene action dependence."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np
import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from research.action_effect.cache_io import read_manifest, write_json, write_jsonl  # noqa: E402
from research.action_effect.effect_tube import EFFECT_TUBE_CHANNELS  # noqa: E402
from research.action_effect.probe_data import load_probe_arrays, load_structured_targets  # noqa: E402
from research.action_effect.structured_audit import structured_channel_audit  # noqa: E402
from research.action_effect.structured_future import STRUCTURED_CHANNELS  # noqa: E402


def _root(variable: str) -> Path:
    value = os.environ.get(variable, "").strip()
    if not value:
        raise ValueError(f"source load_env.sh or set {variable}")
    return Path(value).resolve()


def _resolve(explicit: Path | None, variable: str, relative: str) -> Path:
    return explicit.resolve() if explicit is not None else _root(variable) / relative


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise TypeError(f"configuration must be a mapping: {path}")
    return value


def _load_prediction(path: Path | None, key: str) -> np.ndarray | None:
    if path is None:
        return None
    with np.load(path.resolve()) as payload:
        return np.asarray(payload[key], dtype=np.float32)


def _format(value: float) -> str:
    return "n/a" if not math.isfinite(value) else f"{value:.6f}"


def _table(title: str, rows: Sequence[dict[str, Any]]) -> list[str]:
    result = [
        f"## {title}",
        "",
        "| Channel | Within variance | Between variance | Action ratio | Target AG | Pred/target | Target shuffle | Prediction shuffle gap | Class |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        result.append(
            "| {channel} | {within} | {between} | {ratio} | {gap} | {sensitivity} | {target_shuffle} | {prediction_shuffle} | {classification} |".format(
                channel=row["channel"],
                within=_format(row["within_scene_candidate_variance"]),
                between=_format(row["between_scene_variance"]),
                ratio=_format(row["action_variance_ratio"]),
                gap=_format(row["target_action_gap"]),
                sensitivity=_format(row["predicted_target_sensitivity_ratio"]),
                target_shuffle=_format(row["target_candidate_shuffle_effect"]),
                prediction_shuffle=_format(row["prediction_candidate_shuffle_gap"]),
                classification=row["classification"],
            )
        )
    result.append("")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPOSITORY_ROOT / "configs/action_effect/effect_tube.yaml",
    )
    parser.add_argument("--candidate-cache", type=Path)
    parser.add_argument("--consequence-cache", type=Path)
    parser.add_argument("--pair-cache", type=Path)
    parser.add_argument("--scene-feature-cache", type=Path)
    parser.add_argument("--split-file", type=Path)
    parser.add_argument("--raw-cache", type=Path)
    parser.add_argument("--effect-cache", type=Path)
    parser.add_argument("--raw-prediction", type=Path)
    parser.add_argument("--effect-prediction", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPOSITORY_ROOT / "reports/action_effect_world_model/structured_target_artifacts",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=REPOSITORY_ROOT / "reports/action_effect_world_model/structured_target_audit.md",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = _load_yaml(args.config.resolve())
    candidate_cache = _resolve(args.candidate_cache, "ACTION_EFFECT_CACHE_ROOT", "candidates/pilot_tiny/expert")
    consequence_cache = _resolve(args.consequence_cache, "ACTION_EFFECT_CACHE_ROOT", "consequences/pilot_tiny/expert")
    pair_cache = _resolve(args.pair_cache, "ACTION_EFFECT_CACHE_ROOT", "pairs/pilot_tiny/expert")
    scene_feature_cache = _resolve(args.scene_feature_cache, "ACTION_EFFECT_CACHE_ROOT", "scene_features/pilot_tiny/qwen_dit_100k")
    split_file = (
        args.split_file.resolve()
        if args.split_file is not None
        else _root("ACTION_EFFECT_OUTPUT_ROOT") / "factual_only/pilot_tiny/split.json"
    )
    raw_cache = _resolve(args.raw_cache, "ACTION_EFFECT_CACHE_ROOT", "structured_future/pilot_tiny/expert_log_replay_32")
    effect_cache = (
        args.effect_cache.resolve()
        if args.effect_cache is not None
        else _root("ACTION_EFFECT_CACHE_ROOT") / "effect_tube/pilot_tiny/expert_log_replay_32"
    )
    for path, name in (
        (candidate_cache, "candidate"),
        (consequence_cache, "consequence"),
        (pair_cache, "pair"),
        (scene_feature_cache, "scene feature"),
        (raw_cache, "raw structured"),
    ):
        if read_manifest(path) is None:
            raise FileNotFoundError(f"published {name} cache is missing: {path}")
    with split_file.open("r", encoding="utf-8") as stream:
        split = json.load(stream)
    fit_scenes = [str(value) for value in split.get("fit", split.get("train", []))]
    heldout_scenes = [str(value) for value in split.get("heldout", split.get("test", []))]
    all_scenes = fit_scenes + heldout_scenes
    arrays, _, _, _, _, _ = load_probe_arrays(
        candidate_cache=candidate_cache,
        consequence_cache=consequence_cache,
        scene_feature_cache=scene_feature_cache,
        fit_scene_ids=fit_scenes,
        assumption="log_replay",
    )
    raw_target, raw_valid = load_structured_targets(raw_cache, arrays)
    raw_prediction = _load_prediction(args.raw_prediction, "structured_future_prediction")
    audit_cfg = config["audit"]
    raw_rows, raw_details = structured_channel_audit(
        arrays=arrays,
        target=raw_target,
        valid=raw_valid,
        channels=STRUCTURED_CHANNELS,
        pair_path=pair_cache / "pairs.jsonl",
        selected_scene_ids=all_scenes,
        minimum_action_variance_ratio=float(audit_cfg["minimum_action_variance_ratio"]),
        minimum_target_action_gap=float(audit_cfg["minimum_target_action_gap"]),
        raw_prediction=raw_prediction,
        binary_channels=(0, 1, 2, 3),
    )
    effect_rows: list[dict[str, Any]] = []
    effect_details: dict[str, np.ndarray] | None = None
    if read_manifest(effect_cache) is not None:
        effect_target, effect_valid = load_structured_targets(effect_cache, arrays)
        effect_prediction = _load_prediction(args.effect_prediction, "effect_tube_prediction")
        effect_rows, effect_details = structured_channel_audit(
            arrays=arrays,
            target=effect_target,
            valid=effect_valid,
            channels=EFFECT_TUBE_CHANNELS,
            pair_path=pair_cache / "pairs.jsonl",
            selected_scene_ids=all_scenes,
            minimum_action_variance_ratio=float(audit_cfg["minimum_action_variance_ratio"]),
            minimum_target_action_gap=float(audit_cfg["minimum_target_action_gap"]),
            raw_prediction=effect_prediction,
            binary_channels=(0, 7, 8),
        )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "raw_channel_audit.jsonl", raw_rows)
    write_json(
        output_dir / "raw_action_dependent_channels.json",
        [row["channel"] for row in raw_rows if row["classification"].startswith("action_effect")],
    )
    np.savez_compressed(output_dir / "raw_pair_distances.npz", **raw_details)
    if effect_rows and effect_details is not None:
        write_jsonl(output_dir / "effect_channel_audit.jsonl", effect_rows)
        write_json(
            output_dir / "effect_action_dependent_channels.json",
            [row["channel"] for row in effect_rows if row["classification"].startswith("action_effect")],
        )
        np.savez_compressed(output_dir / "effect_pair_distances.npz", **effect_details)
    csv_path = output_dir / "channel_audit.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        fieldnames = ["target_kind", *raw_rows[0].keys()]
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for kind, rows in (("raw_map", raw_rows), ("effect_tube", effect_rows)):
            for row in rows:
                writer.writerow({"target_kind": kind, **row})
    lines = [
        "# Structured target action-dependence audit",
        "",
        "The original raw map tube is retained as a diagnostic only. Main action-collapse and "
        "Phase-6 representation metrics must use channels classified as action-dependent below; "
        "action-invariant channels are excluded rather than allowed to dominate raster averages.",
        "",
        "Within-scene candidate variance and target Action Gap measure target action dependence. "
        "Between-scene variance quantifies the competing scene prior. If predictions are supplied, "
        "the sensitivity ratio and shuffle gap use the corresponding scene-action probe.",
        "",
        *_table("Retained Phase-5 raw map diagnostic", raw_rows),
    ]
    if effect_rows:
        lines.extend(_table("Trajectory-aligned effect tube", effect_rows))
    else:
        lines.extend(
            [
                "## Trajectory-aligned effect tube",
                "",
                "NOT RUN: the versioned effect-tube cache was not supplied.",
                "",
            ]
        )
    lines.extend(
        [
            "## Target contract",
            "",
            "The effect tube contains candidate-relative dynamic occupancy, map signed-distance "
            "fields, occupied-agent relative velocity, dynamic clearance/collision fields, and the "
            "ego swept footprint at 1/2/4 seconds on a 32×32 candidate-aligned grid. All logged "
            "future actors remain target-only under `log_replay`; true interactive response remains unknown.",
            "",
        ]
    )
    args.report_path.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.report_path.resolve().write_text("\n".join(lines), encoding="utf-8")
    print(
        json.dumps(
            {
                "raw_action_dependent": sum(row["classification"].startswith("action_effect") for row in raw_rows),
                "effect_action_dependent": sum(row["classification"].startswith("action_effect") for row in effect_rows),
                "report": str(args.report_path.resolve()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

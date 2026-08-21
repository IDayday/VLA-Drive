#!/usr/bin/env python3
"""Fit train-only robust scales and cache action-effect pair labels."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from research.action_effect.cache_io import (  # noqa: E402
    CacheManifest,
    cache_is_reusable,
    content_hash,
    file_sha256,
    finalize_manifest,
    read_manifest,
    write_json,
    write_jsonl,
)
from research.action_effect.metric_cache_io import iter_jsonl  # noqa: E402
from research.action_effect.pair_builder import (  # noqa: E402
    build_scene_pairs,
    fit_pair_thresholds,
    fit_robust_scales,
    geometric_distance,
    hard_vector,
    normalized_soft_vector,
    soft_distance,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPOSITORY_ROOT / "configs/action_effect/pairs.yaml")
    parser.add_argument("--candidate-cache", type=Path)
    parser.add_argument("--consequence-cache", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument(
        "--statistics-scene-file",
        type=Path,
        help="JSON split whose `train` scenes exclusively fit scales and pair thresholds.",
    )
    return parser.parse_args()


def _resolve(explicit: Path | None, suffix: str) -> Path:
    if explicit is not None:
        return explicit.resolve()
    root = os.environ.get("ACTION_EFFECT_CACHE_ROOT", "").strip()
    if not root:
        raise ValueError("pass paths explicitly or source load_env.sh to set ACTION_EFFECT_CACHE_ROOT")
    return (Path(root) / suffix).resolve()


def main() -> None:
    args = parse_args()
    with args.config.resolve().open("r", encoding="utf-8") as stream:
        config: dict[str, Any] = yaml.safe_load(stream)
    candidate_cache = _resolve(args.candidate_cache, "candidates/pilot_tiny/expert")
    consequence_cache = _resolve(args.consequence_cache, "consequences/pilot_tiny/expert")
    output_root = _resolve(args.cache_dir, "pairs/pilot_tiny/expert")
    candidate_manifest = read_manifest(candidate_cache)
    consequence_manifest = read_manifest(consequence_cache)
    if candidate_manifest is None or consequence_manifest is None:
        raise FileNotFoundError("candidate and consequence caches require manifests")
    code_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPOSITORY_ROOT, text=True).strip()
    code_sources = (Path(__file__), REPOSITORY_ROOT / "research/action_effect/pair_builder.py")
    code_revision = f"{code_commit}+tree.{content_hash({str(path.relative_to(REPOSITORY_ROOT)): file_sha256(path) for path in code_sources})[:12]}"
    statistics_scenes: set[str] | None = None
    statistics_excluded_perturbation: str | None = None
    statistics_identity = "all_source_scenes"
    if args.statistics_scene_file is not None:
        statistics_path = args.statistics_scene_file.resolve()
        with statistics_path.open("r", encoding="utf-8") as stream:
            split_payload = json.load(stream)
        if not isinstance(split_payload, dict) or not isinstance(split_payload.get("train"), list):
            raise TypeError("statistics scene file must contain a `train` list")
        statistics_scenes = {str(value) for value in split_payload["train"]}
        if not statistics_scenes:
            raise ValueError("statistics train scene list is empty")
        excluded = split_payload.get("held_out_perturbation_family")
        statistics_excluded_perturbation = str(excluded) if excluded is not None else None
        statistics_identity = content_hash(
            {
                "scenes": sorted(statistics_scenes),
                "excluded_perturbation": statistics_excluded_perturbation,
            }
        )
    manifest = CacheManifest(
        cache_kind="action_effect_equivalence",
        cache_version=str(config["cache_version"]),
        dataset_version=consequence_manifest.dataset_version,
        code_commit=code_revision,
        config_hash=content_hash(config),
        evaluator_hash=consequence_manifest.evaluator_hash,
        split=consequence_manifest.split,
        seed=consequence_manifest.seed,
        inputs={
            "candidate_manifest": candidate_manifest.compatibility_identity(),
            "consequence_manifest": consequence_manifest.compatibility_identity(),
            "statistics_scenes_sha256": statistics_identity,
        },
    )
    required = ("pairs.jsonl", "robust_scales.json", "thresholds.json", "summary.json")
    if cache_is_reusable(output_root, manifest, required):
        print(f"[action-effect] reusable pair cache: {output_root}")
        return

    consequence_rows = list(iter_jsonl(consequence_cache / "consequences.jsonl"))
    statistics_rows = (
        consequence_rows
        if statistics_scenes is None
        else [
            row
            for row in consequence_rows
            if str(row["scene_id"]) in statistics_scenes
            and (
                statistics_excluded_perturbation is None
                or str(row.get("perturbation_type")) != statistics_excluded_perturbation
            )
        ]
    )
    if statistics_scenes is not None:
        observed = {str(row["scene_id"]) for row in statistics_rows}
        missing = sorted(statistics_scenes - observed)
        if missing:
            raise KeyError(f"statistics scenes are absent from consequence cache: {missing[:3]}")
    fields = [str(field) for field in config["soft_consequence_fields"]]
    normalization = config["normalization"]
    scales = fit_robust_scales(
        statistics_rows,
        fields,
        minimum_coverage=float(normalization["minimum_coverage"]),
        minimum_scale=float(normalization["minimum_scale"]),
    )
    candidate_trajectories = np.load(candidate_cache / "candidates.npz")["trajectories"]
    trajectory_by_id: dict[str, np.ndarray] = {}
    for row in consequence_rows:
        trajectory_by_id[row["candidate_id"]] = candidate_trajectories[int(row["candidate_index"])]

    preliminary_distances: list[float] = []
    rows_by_scene: dict[str, list[dict[str, Any]]] = {}
    for row in consequence_rows:
        rows_by_scene.setdefault(row["scene_id"], []).append(row)
    for scene_id, rows in rows_by_scene.items():
        if statistics_scenes is not None and scene_id not in statistics_scenes:
            continue
        accepted = [row for row in rows if row.get("candidate_accepted") and row["log_replay"].get("available")]
        if statistics_excluded_perturbation is not None:
            accepted = [
                row
                for row in accepted
                if str(row.get("perturbation_type")) != statistics_excluded_perturbation
            ]
        vectors = {
            row["candidate_id"]: normalized_soft_vector(
                row,
                scales,
                clip=float(normalization["clip_normalized"]),
            )
            for row in accepted
        }
        for index, left in enumerate(accepted):
            for right in accepted[index + 1 :]:
                if np.array_equal(hard_vector(left), hard_vector(right)):
                    geometry = geometric_distance(
                        trajectory_by_id[left["candidate_id"]],
                        trajectory_by_id[right["candidate_id"]],
                        heading_weight_m_per_rad=float(config["distance"]["heading_weight_m_per_rad"]),
                        terminal_weight=float(config["distance"]["terminal_weight"]),
                    )
                    if geometry >= float(config["distance"]["minimum_nonduplicate_geometry_m"]):
                        preliminary_distances.append(
                            soft_distance(vectors[left["candidate_id"]], vectors[right["candidate_id"]])
                        )
    threshold_cfg = config["thresholds"]
    thresholds = fit_pair_thresholds(
        preliminary_distances,
        equivalent_quantile=float(threshold_cfg["equivalent_quantile"]),
        divergent_quantile=float(threshold_cfg["divergent_quantile"]),
        equivalent_floor=float(threshold_cfg["equivalent_floor"]),
        divergent_floor=float(threshold_cfg["divergent_floor"]),
    )

    pairs: list[dict[str, Any]] = []
    for scene_id in sorted(rows_by_scene):
        pairs.extend(build_scene_pairs(rows_by_scene[scene_id], trajectory_by_id, scales, thresholds, config))
    pair_counts = Counter(pair["pair_type"] for pair in pairs)
    confidence_counts = Counter(pair["pair_confidence"] for pair in pairs)
    scene_pair_types: dict[str, set[str]] = {}
    for pair in pairs:
        scene_pair_types.setdefault(pair["scene_id"], set()).add(pair["pair_type"])
    both = sum(
        {"effect_equivalent", "effect_divergent"}.issubset(pair_types)
        for pair_types in scene_pair_types.values()
    )
    safety_boundaries = sum(bool(pair["safety_boundary"]) for pair in pairs)
    output_root.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_root / "pairs.jsonl", pairs)
    write_json(output_root / "robust_scales.json", [asdict(scale) for scale in scales])
    write_json(output_root / "thresholds.json", asdict(thresholds))
    summary = {
        "scene_count": len(rows_by_scene),
        "pair_count": len(pairs),
        "pair_type_counts": dict(sorted(pair_counts.items())),
        "pair_confidence_counts": dict(sorted(confidence_counts.items())),
        "scenes_with_equivalent_and_divergent": both,
        "scenes_with_equivalent_and_divergent_rate": both / max(len(rows_by_scene), 1),
        "safety_boundary_pair_count": safety_boundaries,
        "safety_boundary_pair_rate": safety_boundaries / max(len(pairs), 1),
        "threshold_fit_pair_count": len(preliminary_distances),
        "statistics_split": "train" if statistics_scenes is not None else consequence_manifest.split,
        "statistics_scene_count": (
            len(statistics_scenes) if statistics_scenes is not None else len(rows_by_scene)
        ),
        "statistics_excluded_perturbation": statistics_excluded_perturbation,
    }
    write_json(output_root / "summary.json", summary)
    finalize_manifest(output_root, manifest)
    print(json.dumps({"cache_dir": str(output_root), **summary}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

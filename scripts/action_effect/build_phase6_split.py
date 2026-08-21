#!/usr/bin/env python3
"""Publish the exact scene-disjoint pilot_small split used by Phase 6."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

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
)
from research.action_effect.phase6_data import deterministic_three_way_split  # noqa: E402
from research.action_effect.probe_data import iter_jsonl  # noqa: E402


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise TypeError(f"configuration must be a mapping: {path}")
    return value


def _resolve(explicit: Path | None, suffix: str) -> Path:
    if explicit is not None:
        return explicit.resolve()
    root = os.environ.get("ACTION_EFFECT_CACHE_ROOT", "").strip()
    if not root:
        raise ValueError("pass paths explicitly or source load_env.sh")
    return (Path(root) / suffix).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPOSITORY_ROOT / "configs/action_effect/phase6_split.yaml",
    )
    parser.add_argument("--candidate-cache", type=Path)
    parser.add_argument("--consequence-cache", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = _load_yaml(args.config.resolve())
    candidate_cache = _resolve(args.candidate_cache, "candidates/pilot_small/expert")
    consequence_cache = _resolve(args.consequence_cache, "consequences/pilot_small/expert")
    output_dir = _resolve(args.cache_dir, "splits/pilot_small/phase6")
    candidate_manifest = read_manifest(candidate_cache)
    consequence_manifest = read_manifest(consequence_cache)
    if candidate_manifest is None or consequence_manifest is None:
        raise FileNotFoundError("candidate and consequence caches must be published")
    rows = list(iter_jsonl(consequence_cache / "consequences.jsonl"))
    eligible = sorted(
        {
            str(row["scene_id"])
            for row in rows
            if row.get("perturbation_type") == "anchor"
            and row.get("candidate_accepted")
            and row["log_replay"].get("available")
        }
    )
    counts = config["scene_counts"]
    split = deterministic_three_way_split(
        eligible,
        train_count=int(counts["train"]),
        validation_count=int(counts["validation"]),
        test_count=int(counts["test"]),
        seed=int(config["seed"]),
    )
    source = REPOSITORY_ROOT / "research/action_effect/phase6_data.py"
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPOSITORY_ROOT, text=True
    ).strip()
    manifest = CacheManifest(
        cache_kind="phase6_scene_split",
        cache_version=str(config["cache_version"]),
        dataset_version=consequence_manifest.dataset_version,
        code_commit=f"{commit}+tree.{file_sha256(source)[:12]}",
        config_hash=content_hash(config),
        evaluator_hash=consequence_manifest.evaluator_hash,
        split=consequence_manifest.split,
        seed=int(config["seed"]),
        inputs={
            "candidate_manifest": candidate_manifest.compatibility_identity(),
            "consequence_manifest": consequence_manifest.compatibility_identity(),
            "eligible_scenes_sha256": content_hash(eligible),
        },
    )
    if cache_is_reusable(output_dir, manifest, ("split.json", "summary.json")):
        print(f"[action-effect] reusable Phase-6 split: {output_dir}")
        return
    heldout_family = str(config["held_out_perturbation_family"])
    candidate_counts = {name: 0 for name in ("train", "validation", "test")}
    heldout_counts = {name: 0 for name in candidate_counts}
    split_lookup = {
        scene: name
        for name in candidate_counts
        for scene in split[name]
    }
    for row in rows:
        name = split_lookup.get(str(row["scene_id"]))
        if name is None or not row.get("candidate_accepted"):
            continue
        candidate_counts[name] += 1
        heldout_counts[name] += int(row.get("perturbation_type") == heldout_family)
    payload = {
        **split,
        "seed": int(config["seed"]),
        "held_out_perturbation_family": heldout_family,
        "statistics_split": "train",
    }
    summary = {
        "eligible_scene_count": len(eligible),
        "scene_counts": {name: len(split[name]) for name in ("train", "validation", "test")},
        "unused_scene_count": len(split["unused"]),
        "accepted_candidate_counts": candidate_counts,
        "held_out_family_candidate_counts": heldout_counts,
        "held_out_perturbation_family": heldout_family,
        "scene_disjoint": len(set(split["train"]) & set(split["validation"])) == 0
        and len(set(split["train"]) & set(split["test"])) == 0
        and len(set(split["validation"]) & set(split["test"])) == 0,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "split.json", payload)
    write_json(output_dir / "summary.json", summary)
    finalize_manifest(output_dir, manifest)
    print(json.dumps({"cache_dir": str(output_dir), **summary}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

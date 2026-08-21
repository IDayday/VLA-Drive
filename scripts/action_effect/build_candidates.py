#!/usr/bin/env python3
"""Build a deterministic, versioned cache of policy-local trajectories."""

from __future__ import annotations

import argparse
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
    write_json,
    write_jsonl,
    write_npz,
)
from research.action_effect.candidate_generator import (  # noqa: E402
    CandidateGeneratorConfig,
    PolicyLocalCandidateGenerator,
)
from research.action_effect.trajectory_io import (  # noqa: E402
    load_expert_anchor,
    load_policy_anchor,
)


def _load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise TypeError(f"configuration must be a mapping: {path}")
    return value


def _code_revision(paths: list[Path]) -> str:
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPOSITORY_ROOT, text=True
    ).strip()
    tree_hash = content_hash({str(path.relative_to(REPOSITORY_ROOT)): file_sha256(path) for path in paths})
    return f"{commit}+tree.{tree_hash[:12]}"


def _select_scene_ids(
    scene_ids: list[str], max_scenes: int | None, sampling: str, seed: int
) -> list[str]:
    if max_scenes is None or max_scenes >= len(scene_ids):
        return list(scene_ids)
    if max_scenes < 1:
        raise ValueError("max_scenes must be positive or null")
    if sampling == "seeded_random":
        rng = np.random.default_rng(seed)
        indices = rng.choice(len(scene_ids), size=max_scenes, replace=False)
        return [scene_ids[int(index)] for index in indices]
    if sampling == "first":
        return scene_ids[:max_scenes]
    raise ValueError(f"unsupported scene sampling mode: {sampling}")


def _resolve_default_cache(profile: str, anchor_type: str) -> Path:
    root = os.environ.get("ACTION_EFFECT_CACHE_ROOT", "").strip()
    if not root:
        raise ValueError("pass --cache-dir or source load_env.sh to set ACTION_EFFECT_CACHE_ROOT")
    return Path(root) / "candidates" / profile / anchor_type


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPOSITORY_ROOT / "configs/action_effect/pilot_tiny.yaml",
    )
    parser.add_argument("--datalist", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument("--anchor-type", choices=("expert", "policy"), default="expert")
    parser.add_argument("--policy-prediction-root", type=Path)
    parser.add_argument(
        "--policy-anchor-id",
        help="Logical checkpoint/config hash required for policy anchors; paths alone are not identities.",
    )
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--max-scenes", type=int)
    parser.add_argument("--seed", type=int, default=20260821)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = _load_config(config_path)
    experiment = config["experiment"]
    profile = str(experiment["name"])
    datalist = (args.datalist or Path(os.environ.get("NAVSIM_DATALIST_PATH", ""))).resolve()
    data_root = (args.data_root or Path(os.environ.get("DATA_ROOT", ""))).resolve()
    if not datalist.is_file():
        raise FileNotFoundError(f"datalist is missing: {datalist}")
    if not data_root.is_dir():
        raise FileNotFoundError(f"processed data root is missing: {data_root}")
    if args.anchor_type == "policy":
        if args.policy_prediction_root is None or args.policy_anchor_id is None:
            raise ValueError("policy anchors require --policy-prediction-root and --policy-anchor-id")
        policy_root = args.policy_prediction_root.resolve()
    else:
        policy_root = None

    cache_dir = (args.cache_dir or _resolve_default_cache(profile, args.anchor_type)).resolve()
    generator_config = CandidateGeneratorConfig.from_mapping(config["candidate_generation"])
    generator = PolicyLocalCandidateGenerator(generator_config)
    max_scenes = args.max_scenes if args.max_scenes is not None else experiment.get("max_scenes")
    with datalist.open("r", encoding="utf-8") as stream:
        all_scene_ids = json.load(stream)
    if not isinstance(all_scene_ids, list) or not all(isinstance(item, str) for item in all_scene_ids):
        raise TypeError("datalist must be a JSON list of scene token strings")
    selected = _select_scene_ids(
        all_scene_ids,
        None if max_scenes is None else int(max_scenes),
        str(experiment.get("scene_sampling", "seeded_random")),
        args.seed,
    )

    code_revision = _code_revision(
        [
            Path(__file__),
            REPOSITORY_ROOT / "research/action_effect/candidate_generator.py",
            REPOSITORY_ROOT / "research/action_effect/trajectory_io.py",
            REPOSITORY_ROOT / "research/action_effect/cache_io.py",
        ]
    )
    inputs = {
        "datalist_sha256": file_sha256(datalist),
        "selected_scenes_sha256": content_hash(selected),
        "anchor_type": args.anchor_type,
        "policy_anchor_id": args.policy_anchor_id or "not_applicable",
    }
    manifest = CacheManifest(
        cache_kind="policy_local_candidate",
        cache_version=str(experiment["cache_version"]),
        dataset_version=str(experiment["dataset_version"]),
        code_commit=code_revision,
        config_hash=content_hash(config),
        evaluator_hash="not_applicable",
        split=args.split,
        seed=args.seed,
        inputs=inputs,
    )
    required = ("candidates.npz", "metadata.jsonl", "scene_index.json", "summary.json")
    if cache_is_reusable(cache_dir, manifest, required):
        print(f"[action-effect] reusable candidate cache: {cache_dir}")
        return

    cache_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    trajectories: list[np.ndarray] = []
    scene_index: dict[str, dict[str, int]] = {}
    valid_count = 0
    route_valid_count = 0
    for scene_number, scene_id in enumerate(selected):
        if args.anchor_type == "expert":
            anchor = load_expert_anchor(data_root, args.split, scene_id)
        else:
            assert policy_root is not None
            anchor = load_policy_anchor(policy_root, args.split, scene_id)
        candidates = generator.generate(
            anchor,
            scene_id=scene_id,
            anchor_type=args.anchor_type,
            seed=args.seed,
        )
        start = len(rows)
        for candidate in candidates:
            trajectory_index = len(trajectories)
            trajectories.append(candidate.trajectory)
            validation = candidate.validation
            valid_count += int(validation.kinematic_valid)
            route_valid_count += int(validation.route_valid)
            rows.append(
                {
                    "scene_id": scene_id,
                    "split": args.split,
                    "anchor_type": args.anchor_type,
                    "candidate_id": candidate.candidate_id,
                    "trajectory": {"array": "trajectories", "index": trajectory_index},
                    "perturbation_type": candidate.perturbation_type,
                    "perturbation_parameters": dict(candidate.perturbation_parameters),
                    "kinematic_valid": validation.kinematic_valid,
                    "route_valid": validation.route_valid,
                    "validation_reasons": list(validation.reasons),
                    "validation_metrics": {
                        key: value
                        for key, value in asdict(validation).items()
                        if key not in {"kinematic_valid", "route_valid", "reasons"}
                    },
                    "generation_seed": args.seed,
                    "cache_version": manifest.cache_version,
                }
            )
        scene_index[scene_id] = {"start": start, "count": len(candidates), "scene_number": scene_number}

    trajectory_array = np.stack(trajectories).astype(np.float32)
    write_npz(cache_dir / "candidates.npz", trajectories=trajectory_array)
    write_jsonl(cache_dir / "metadata.jsonl", rows)
    write_json(cache_dir / "scene_index.json", scene_index)
    summary = {
        "profile": profile,
        "scene_count": len(selected),
        "candidate_count": len(rows),
        "candidates_per_scene": generator_config.candidate_count(),
        "kinematic_valid_count": valid_count,
        "kinematic_valid_rate": valid_count / max(len(rows), 1),
        "route_valid_count": route_valid_count,
        "route_valid_rate": route_valid_count / max(len(rows), 1),
        "trajectory_shape": list(trajectory_array.shape),
    }
    write_json(cache_dir / "summary.json", summary)
    finalize_manifest(cache_dir, manifest)
    print(json.dumps({"cache_dir": str(cache_dir), **summary}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

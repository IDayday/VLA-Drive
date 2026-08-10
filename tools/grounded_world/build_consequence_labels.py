#!/usr/bin/env python3
"""Build training-only physical consequence labels for GT perturbations.

A local provider ``module:function`` wraps NAVSIM metric-cache/static-map APIs.
Unavailable quantities must be returned with ``valid_mask=False``; this tool
never substitutes aggregate EPDMS or demonstrated future as counterfactual GT.
"""

from __future__ import annotations

import argparse
import importlib
import json
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from starVLA.dataloader.field2plan_cache import (
    atomic_write_json,
    atomic_write_npz,
    hash_tokens,
    sha256_file,
)
from starVLA.dataloader.grounded_world_cache import ConsequenceCacheReader
from starVLA.model.modules.field2plan.temporal_alignment import se2_poses_to_transforms
from starVLA.model.modules.grounded_world.perturbations import (
    build_consequence_perturbations,
)
from starVLA.model.modules.grounded_world.teacher_protocols import (
    PhysicalConsequenceOutput,
)


COMPONENTS = (
    "clearance",
    "ttc",
    "collision",
    "lane_distance",
    "progress",
    "comfort",
)


def _factory(spec: str):
    if ":" not in spec:
        raise ValueError("provider factory must use module:function")
    module_name, function_name = spec.split(":", 1)
    try:
        module = importlib.import_module(module_name)
    except (ImportError, ModuleNotFoundError) as error:
        raise RuntimeError(
            f"cannot import consequence provider {module_name!r}; "
            "provide its local dependencies explicitly"
        ) from error
    value = getattr(module, function_name, None)
    if not callable(value):
        raise RuntimeError(f"consequence provider factory is not callable: {spec}")
    return value


def _physical_gt(metadata: dict) -> np.ndarray:
    poses = torch.as_tensor(
        metadata["glo_status"]["global_poses"][:12], dtype=torch.float32
    )
    transforms = se2_poses_to_transforms(poses)
    current_from_ego = torch.linalg.inv(transforms[3]) @ transforms[4:12]
    heading = torch.atan2(current_from_ego[:, 1, 0], current_from_ego[:, 0, 0])
    return torch.stack(
        (current_from_ego[:, 0, 3], current_from_ego[:, 1, 3], heading), dim=-1
    ).numpy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider-factory", required=True)
    parser.add_argument("--metric-cache-root", type=Path, required=True)
    parser.add_argument("--datalist", type=Path, required=True)
    parser.add_argument("--meta-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        tokens = json.loads(args.datalist.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid datalist: {args.datalist}") from error
    tokens = tokens[: args.max_samples] if args.max_samples > 0 else tokens
    if not tokens or not all(isinstance(token, str) and token for token in tokens):
        raise ValueError("datalist must contain non-empty tokens")
    if args.validate_only:
        reader = ConsequenceCacheReader(args.output_dir, args.split)
        reader.validate_dataset_binding(tokens, args.datalist)
        for token in tokens:
            reader.load(token)
        print("[consequence-cache] validation OK")
        return
    if not args.metric_cache_root.is_dir():
        raise FileNotFoundError(f"NAVSIM metric cache not found: {args.metric_cache_root}")
    provider = _factory(args.provider_factory)(metric_cache_root=args.metric_cache_root)
    provider_metadata = dict(getattr(provider, "metadata", {}))
    if provider_metadata.get("returns_aggregate_epdms") is not False:
        raise ValueError("provider metadata must declare returns_aggregate_epdms=false")
    if provider_metadata.get("reactive_counterfactual") is not False:
        raise ValueError("provider must declare reactive_counterfactual=false")
    if provider_metadata.get("uses_logged_future_agents") is True and (
        provider_metadata.get("logged_future_agents_role")
        != "non_reactive_proxy_only"
    ):
        raise ValueError(
            "logged future agents may only be declared as non_reactive_proxy_only"
        )
    if tuple(provider_metadata.get("components", ())) != COMPONENTS:
        raise ValueError("provider physical component ordering is invalid")
    output_split = args.output_dir / args.split
    output_split.mkdir(parents=True, exist_ok=True)
    source_names = None
    candidate_count = None
    for token in tokens:
        output_path = output_split / f"{token}.npz"
        if output_path.is_file() and not args.overwrite:
            continue
        meta_path = args.meta_root / f"{token}.pkl"
        try:
            with meta_path.open("rb") as stream:
                metadata = pickle.load(stream)
        except (OSError, pickle.UnpicklingError, EOFError) as error:
            raise ValueError(f"corrupt NAVSIM metadata: {meta_path}") from error
        perturbations = build_consequence_perturbations(
            torch.from_numpy(_physical_gt(metadata))
        )
        trajectories = perturbations.physical[0].numpy().astype(np.float32)
        result = provider.label(token, metadata, trajectories)
        if not isinstance(result, PhysicalConsequenceOutput):
            raise TypeError("provider must return PhysicalConsequenceOutput")
        result.validate(trajectories.shape[0])
        source_names = perturbations.source_names
        candidate_count = trajectories.shape[0]
        atomic_write_npz(
            output_path,
            token=np.asarray(token),
            physical_trajectories=trajectories,
            values=result.values.astype(np.float32),
            valid_mask=result.valid_mask.astype(np.bool_),
        )
    if candidate_count is None:
        with np.load(output_split / f"{tokens[0]}.npz", allow_pickle=False) as payload:
            candidate_count = int(payload["values"].shape[0])
        source_names = tuple(f"candidate_{index}" for index in range(candidate_count))
    missing = [token for token in tokens if not (output_split / f"{token}.npz").is_file()]
    if missing:
        raise RuntimeError(f"consequence cache incomplete: {missing[:10]}")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    manifest = {
        "schema_version": 1,
        "cache_type": "grounded_world_consequence",
        "status": "complete",
        "producer": {
            "source": "navsim_physical_components",
            "contains_aggregate_epdms": False,
            "provider_factory": args.provider_factory,
            "provider_metadata": provider_metadata,
            "metric_cache_root": str(args.metric_cache_root.resolve()),
        },
        "generator": {
            "git_commit": commit,
            "tool": "tools/grounded_world/build_consequence_labels.py",
        },
        "splits": {
            args.split: {
                "entry_count": len(tokens),
                "tokens_sha256": hash_tokens(tokens),
                "datalist_sha256": sha256_file(args.datalist),
            }
        },
        "tensor_schema": {
            "components": list(COMPONENTS),
            "candidate_sources": list(source_names),
            "physical_trajectories": {
                "shape": [candidate_count, 8, 3], "dtype": "float32"
            },
            "values": {"shape": [candidate_count, 6], "dtype": "float32"},
            "valid_mask": {"shape": [candidate_count, 6], "dtype": "bool"},
        },
    }
    atomic_write_json(args.output_dir / "manifest.json", manifest)
    reader = ConsequenceCacheReader(args.output_dir, args.split)
    reader.validate_dataset_binding(tokens, args.datalist)
    print(f"[consequence-cache] complete entries={len(tokens)}")


if __name__ == "__main__":
    main()

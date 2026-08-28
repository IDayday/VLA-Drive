#!/usr/bin/env python3
"""Build an oracle feature store for frozen EpisodeDrive proposal candidates."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .build_oracle_store import ARRAY_SPECS, _build_scene, _open_arrays, _shape
from .common import (
    DEFAULT_CACHE_DIR,
    DEFAULT_REPORT_DIR,
    append_command,
    ensure_dir,
    require_gate,
    sha256_file,
    write_json,
    write_markdown,
)


def build(args: argparse.Namespace) -> dict[str, Any]:
    require_gate(args.output_dir, "target_v3")
    report_dir = ensure_dir(args.output_dir)
    cache_dir = ensure_dir(args.cache_dir)
    store_dir = ensure_dir(args.store_dir or cache_dir / "model_candidate_oracle_store")
    manifest_path = cache_dir / "model_candidates/combined_selected_manifest.parquet"
    metric_path = cache_dir / "model_candidates/combined_selected_metrics.parquet"
    if not manifest_path.is_file() or not metric_path.is_file():
        raise FileNotFoundError("Aggregate the EpisodeDrive model candidate bank first")
    manifest = pd.read_parquet(manifest_path)
    metrics = pd.read_parquet(metric_path)
    if manifest.scene_token.nunique() * args.num_candidates != len(manifest):
        raise RuntimeError("Model candidate manifest is not fixed-K")
    inventory_path = (
        report_dir / "all_scene_inventory.parquet"
        if (report_dir / "all_scene_inventory.parquet").is_file()
        else report_dir / "balanced_scene_manifest.parquet"
    )
    balanced = pd.read_parquet(inventory_path)[
        ["scene_token", "selection_index"]
    ]
    metadata = (
        manifest[["scene_token", "log_name", "fold"]]
        .drop_duplicates("scene_token")
        .merge(balanced, on="scene_token", how="left", validate="one_to_one")
        .sort_values(["fold", "log_name", "selection_index", "scene_token"])
        .reset_index(drop=True)
    )
    if metadata.selection_index.isna().any():
        raise RuntimeError("Model proposal scene is absent from the balanced all-log manifest")
    metadata["scene_index"] = np.arange(len(metadata), dtype=np.int64)
    scenes = len(metadata)
    family_names = ["model_proposal"]
    expected_config = {
        "schema_version": "1.0.0-model-proposal",
        "scene_count": scenes,
        "candidates_per_scene": args.num_candidates,
        "candidate_manifest_sha256": sha256_file(manifest_path),
        "metric_sha256": sha256_file(metric_path),
        "family_names": family_names,
        "array_specs": {
            name: {
                "dtype": np.dtype(spec[0]).name,
                "shape": list(_shape(spec, scenes, args.num_candidates)),
            }
            for name, spec in ARRAY_SPECS.items()
        },
    }
    config_path = store_dir / "store_config.json"
    create = args.force or not config_path.is_file()
    if config_path.is_file() and not args.force:
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        if existing != expected_config:
            raise RuntimeError("Existing model-candidate oracle store has another identity")
    arrays = _open_arrays(store_dir, scenes, args.num_candidates, create=create)
    if create:
        write_json(config_path, expected_config)
    metadata.to_parquet(
        store_dir / "scene_metadata.parquet",
        index=False,
        engine="pyarrow",
        compression="zstd",
    )

    failures: list[dict[str, str]] = []
    processed = 0
    started = time.time()
    family_to_id = {"model_proposal": 0}
    for position, row in enumerate(metadata.itertuples(index=False)):
        scene_index = int(row.scene_index)
        if arrays["completed"][scene_index] and not args.force:
            continue
        token = str(row.scene_token)
        safe_log = str(row.log_name).replace("/", "_")
        target_path = cache_dir / "model_candidates/targets_v3" / safe_log / f"{token}.npz"
        try:
            _build_scene(
                arrays,
                scene_index,
                target_path,
                manifest[manifest.scene_token == token].copy(),
                metrics[metrics.scene_token == token].copy(),
                family_to_id,
            )
            processed += 1
        except Exception as exc:
            failures.append(
                {
                    "scene_token": token,
                    "log_name": str(row.log_name),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
        if (position + 1) % 100 == 0:
            print(
                f"scenes={position + 1}/{scenes} complete={int(arrays['completed'].sum())} "
                f"failures={len(failures)}",
                flush=True,
            )
    for array in arrays.values():
        if hasattr(array, "flush"):
            array.flush()
    completed = int(arrays["completed"].sum())
    result = {
        "scene_count": scenes,
        "log_count": int(metadata.log_name.nunique()),
        "completed_scene_count": completed,
        "success_rate": completed / scenes if scenes else 0.0,
        "processed_this_run": processed,
        "failure_count": len(failures),
        "failure_examples": failures[:30],
        "store_dir": str(store_dir),
        "elapsed_seconds": time.time() - started,
        "ground_truth_inserted": False,
        "official_scores_are_targets_only": True,
    }
    write_json(report_dir / "model_candidate_oracle_store_summary.json", result)
    if result["success_rate"] < 0.98:
        raise RuntimeError(f"Model candidate oracle store coverage below 98%: {result}")
    write_markdown(
        report_dir / "MODEL_CANDIDATE_ORACLE_STORE.md",
        f"""# Model-candidate Oracle Store

- Frozen EpisodeDrive proposal scenes/logs: {completed:,}/{scenes:,} / {result['log_count']:,}
- Candidates per scene: {args.num_candidates}
- Ground-truth trajectory inserted: no
- Success rate: {result['success_rate']:.3%}
- Store: `{store_dir}` (large cache; not committed)

The store contains offline official outcomes only as learning/evaluation targets.
They are never accepted by the inference input contract.
""",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-candidates", type=int, default=16)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR / "all")
    parser.add_argument("--store-dir", type=Path)
    args = parser.parse_args()
    result = build(args)
    print(json.dumps(result, sort_keys=True))
    append_command(
        args.output_dir,
        "python -m tools.shared_future_candidate_consequence.build_model_oracle_store "
        + " ".join(sys.argv[1:]),
    )


if __name__ == "__main__":
    main()

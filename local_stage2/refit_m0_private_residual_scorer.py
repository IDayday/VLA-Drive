#!/usr/bin/env python3
"""Rebuild an M0 scorer on every training log after held-out selection.

This wrapper reconstructs the training command from the selected artifact so
architecture and objective switches cannot be changed accidentally.  The
trainer independently revalidates the same provenance before reading a batch.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import List, Mapping

import torch


_SCALAR_ARGUMENTS = (
    "seed",
    "batch_size",
    "eval_batch_size",
    "num_workers",
    "learning_rate",
    "weight_decay",
    "model_dim",
    "dynamic_queries",
    "private_layers",
    "trajectory_layers",
    "candidate_layers",
    "fine_layers",
    "private_fine_top_k",
    "residual_layers",
    "reference_hidden_dim",
    "reference_layers",
    "reference_gain_quantile_index",
    "reference_minimum_lcb_gain",
    "reference_maximum_safety_worse_probability",
    "reference_minimum_safe_improvement_probability",
    "residual_top_k",
    "score_mode",
    "max_residual",
    "dropout",
    "minimum_pair_delta",
    "factor_rank_minimum_delta",
    "target_temperature",
    "prediction_temperature",
    "top_set_tolerance",
    "pairwise_weight",
    "base_pairwise_weight",
    "listwise_weight",
    "top_set_weight",
    "expected_regret_weight",
    "top_regret_weight",
    "top_regret_minimum_delta",
    "factor_weight",
    "private_factor_weight",
    "factor_rank_weight",
    "relative_safety_weight",
    "residual_l2_weight",
    "reference_weight",
    "reference_quantile_weight",
    "reference_median_rank_weight",
    "reference_safety_weight",
    "reference_improvement_weight",
    "reference_false_switch_weight",
    "reference_missed_improvement_weight",
    "reference_safety_worse_positive_weight",
    "reference_safe_improvement_positive_weight",
    "reference_switch_margin_temperature",
    "reference_minimum_improvement_target",
    "reference_factor_epsilon",
    "shared_future_weight",
    "current_actor_weight",
    "semantic_bev_weight",
    "candidate_relative_weight",
    "safety_negative_weight",
    "factor_loss_scope",
    "bootstrap_replicates",
    "max_scenes_per_source",
    "scene_sampling_mode",
    "risk_scene_max_multiplier",
)
_BOOLEAN_ARGUMENTS = (
    "m0_context_fusion",
    "m0_candidate_fusion",
    "m0_candidate_only",
    "conservative_reference",
    "shared_future_relabeling",
    "shared_future_constant_velocity_residual",
    "trajectory_observation_attention",
    "current_actor_cv_relabeling",
    "semantic_bev_fusion",
)
_OPTIONAL_PATH_ARGUMENTS = (
    "private_observation_root",
    "current_actor_target_root",
    "semantic_bev_target_root",
    "shared_future_target_root",
)
_LEGACY_DEFAULTS = {
    "top_regret_weight": 0.0,
    "top_regret_minimum_delta": 0.01,
    "factor_loss_scope": "all",
    "reference_hidden_dim": 512,
    "reference_layers": 2,
    "reference_gain_quantile_index": 1,
    "reference_minimum_lcb_gain": 0.0,
    "reference_maximum_safety_worse_probability": 0.1,
    "reference_minimum_safe_improvement_probability": 0.7,
    "reference_weight": 0.0,
    "reference_quantile_weight": 1.0,
    "reference_median_rank_weight": 0.25,
    "reference_safety_weight": 1.0,
    "reference_improvement_weight": 0.5,
    "reference_false_switch_weight": 0.5,
    "reference_missed_improvement_weight": 0.0,
    "reference_safety_worse_positive_weight": 10.0,
    "reference_safe_improvement_positive_weight": 3.0,
    "reference_switch_margin_temperature": 0.05,
    "reference_minimum_improvement_target": 0.005,
    "reference_factor_epsilon": 1.0e-6,
    "current_actor_cv_relabeling": False,
    "semantic_bev_weight": 0.0,
    "semantic_bev_fusion": False,
    "scene_sampling_mode": "log_balanced",
    "risk_scene_max_multiplier": 4.0,
}


def _flag(name: str) -> str:
    return "--" + name.replace("_", "-")


def build_refit_command(
    selection_artifact: Path,
    output_dir: Path,
    *,
    device: str,
    python_bin: Path,
    trainer_path: Path,
) -> List[str]:
    selected = torch.load(selection_artifact, map_location="cpu", weights_only=False)
    if not isinstance(selected, Mapping):
        raise RuntimeError("selection artifact has the wrong schema")
    fold = selected.get("fold_manifest")
    if not isinstance(fold, Mapping) or not isinstance(fold.get("args"), Mapping):
        raise RuntimeError("selection artifact lacks serialized training arguments")
    saved = dict(fold["args"])
    source_rows = saved.get("source")
    if not isinstance(source_rows, list) or not source_rows:
        raise RuntimeError("selection artifact lacks replay sources")
    selected_epoch = int(selected.get("epoch", -1))
    if selected_epoch < 0:
        raise RuntimeError("selection artifact lacks a valid selected epoch")

    command = [str(python_bin), str(trainer_path)]
    for row in source_rows:
        if not isinstance(row, (list, tuple)) or len(row) != 3:
            raise RuntimeError("selection replay source has the wrong schema")
        command.extend(("--source", *(str(value) for value in row)))
    for name in _OPTIONAL_PATH_ARGUMENTS:
        value = saved.get(name)
        if value is not None:
            command.extend((_flag(name), str(value)))
    for required in ("split_manifest", "selection_source"):
        value = saved.get(required)
        if not value:
            raise RuntimeError(f"selection artifact lacks {required}")
        command.extend((_flag(required), str(value)))
    for name in _SCALAR_ARGUMENTS:
        value = saved.get(name, _LEGACY_DEFAULTS.get(name))
        if value is None:
            raise RuntimeError(f"selection artifact lacks locked argument {name}")
        command.extend((_flag(name), str(value)))
    for name in _BOOLEAN_ARGUMENTS:
        if bool(saved.get(name, False)):
            command.append(_flag(name))
    command.extend(
        (
            "--epochs",
            str(selected_epoch + 1),
            "--device",
            device,
            "--output-dir",
            str(output_dir),
            "--refit-all-logs",
            "--refit-selection-artifact",
            str(selection_artifact),
        )
    )
    return command


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection-artifact", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not args.selection_artifact.is_file():
        raise FileNotFoundError(args.selection_artifact)
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    trainer_path = Path(__file__).with_name("train_m0_private_residual_scorer.py")
    command = build_refit_command(
        args.selection_artifact,
        args.output_dir,
        device=args.device,
        python_bin=args.python,
        trainer_path=trainer_path,
    )
    print(shlex.join(command), flush=True)
    if args.dry_run:
        return
    environment = dict(os.environ)
    environment.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    subprocess.run(command, check=True, env=environment)
    command_record = {
        "selection_artifact": str(args.selection_artifact.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "command": command,
        "future_or_evaluator_input": False,
        "navtest_used_for_selection": False,
    }
    (args.output_dir / "REFIT_COMMAND.json").write_text(
        json.dumps(command_record, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()

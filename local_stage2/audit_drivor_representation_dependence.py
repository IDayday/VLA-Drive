#!/usr/bin/env python3
"""Measure how much the released DrivOR scorer depends on its own context.

The scorer weights and proposal bank are held fixed.  Current-observation
registers and current ego status are then zeroed or shuffled across physical
logs.  PDM labels are joined only after proposal selection, so this is an
offline representation-dependence audit rather than an inference feature.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

from local_stage2.train_drivor_initialized_ranker import _split_indices
from local_stage2.train_independent_scorer import (
    ReplaySource,
    _log_bootstrap_ci,
    _sha256,
    load_replay_sources,
)
from navsim.agents.EpisodeDrive.score_module.drivor_ranker import (
    DrivORInitializedProposalRanker,
    DrivORRankerConfig,
)
from navsim.agents.EpisodeDrive.score_module.independent_ranker import (
    pdms_factor_log_utility,
)


MODES = (
    "correct",
    "scene_cross_log_shuffle",
    "scene_zero",
    "status_cross_log_shuffle",
    "status_zero",
    "scene_and_status_cross_log_shuffle",
    "scene_and_status_zero",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--label-root", type=Path, required=True)
    parser.add_argument("--private-observation-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _cross_log_derangement(
    physical_logs: Sequence[str], seed: int
) -> np.ndarray:
    """Return a deterministic row permutation with no same-log donor."""

    logs = np.asarray([str(value) for value in physical_logs])
    names, counts = np.unique(logs, return_counts=True)
    if len(names) < 2 or int(counts.max()) * 2 > len(logs):
        raise RuntimeError("a single physical log prevents a row derangement")
    # Keep each log in one circular block, randomize block order deterministically,
    # and rotate by the largest block.  Both circular displacements are at
    # least that block's width, so no row can receive a donor from its own log.
    rng = np.random.default_rng(seed)
    block_names = names[rng.permutation(len(names))]
    ordered = np.concatenate(
        [np.flatnonzero(logs == name) for name in block_names]
    )
    donor_ordered = np.roll(ordered, -int(counts.max()))
    permutation = np.empty(len(logs), dtype=np.int64)
    permutation[ordered] = donor_ordered
    if not bool(np.all(logs != logs[permutation])):
        raise RuntimeError("constructed cross-log permutation is not a derangement")
    return permutation


def _pairwise_counts(
    utility: torch.Tensor,
    target: torch.Tensor,
    minimum_delta: float = 1.0e-6,
) -> tuple[int, int]:
    if utility.shape != target.shape or utility.ndim != 2:
        raise ValueError("utility and target must share shape [B,K]")
    count = 0
    correct = 0
    candidate_count = utility.shape[1]
    for left in range(candidate_count - 1):
        target_delta = target[:, left, None] - target[:, left + 1 :]
        prediction_delta = utility[:, left, None] - utility[:, left + 1 :]
        valid = target_delta.abs() > minimum_delta
        count += int(valid.sum())
        correct += int(
            ((target_delta * prediction_delta) > 0.0)[valid].sum()
        )
    return correct, count


def _mode_inputs(
    mode: str,
    scene: torch.Tensor,
    status: torch.Tensor,
    donor_scene: torch.Tensor,
    donor_status: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if mode == "correct":
        return scene, status
    if mode == "scene_cross_log_shuffle":
        return donor_scene, status
    if mode == "scene_zero":
        return torch.zeros_like(scene), status
    if mode == "status_cross_log_shuffle":
        return scene, donor_status
    if mode == "status_zero":
        return scene, torch.zeros_like(status)
    if mode == "scene_and_status_cross_log_shuffle":
        return donor_scene, donor_status
    if mode == "scene_and_status_zero":
        return torch.zeros_like(scene), torch.zeros_like(status)
    raise ValueError(f"unknown mode: {mode}")


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    for path in (
        args.feature_root,
        args.label_root,
        args.private_observation_root,
        args.split_manifest,
        args.checkpoint,
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    torch.use_deterministic_algorithms(True)

    data, lineage = load_replay_sources(
        [ReplaySource("public_base", args.feature_root, args.label_root)],
        private_observation_root=args.private_observation_root,
    )
    _train_indices, validation_indices, split_lineage = _split_indices(
        data, args.split_manifest
    )
    validation_logs = [data.physical_logs[index] for index in validation_indices]
    donor_local = _cross_log_derangement(validation_logs, args.seed)
    validation = torch.as_tensor(validation_indices, dtype=torch.long)
    donors = validation[torch.as_tensor(donor_local, dtype=torch.long)]
    donor_logs = [data.physical_logs[int(index)] for index in donors]
    if any(left == right for left, right in zip(validation_logs, donor_logs)):
        raise RuntimeError("cross-log donor permutation contains same-log rows")

    model = DrivORInitializedProposalRanker(DrivORRankerConfig())
    initialization = model.load_drivor_checkpoint(args.checkpoint)
    device = torch.device(args.device)
    model.to(device).eval()

    selected_values: Dict[str, List[np.ndarray]] = {mode: [] for mode in MODES}
    selected_indices: Dict[str, List[np.ndarray]] = {mode: [] for mode in MODES}
    pairwise_correct = {mode: 0 for mode in MODES}
    pairwise_count = {mode: 0 for mode in MODES}
    oracle_values: List[np.ndarray] = []
    base_values: List[np.ndarray] = []

    for start in range(0, len(validation), args.batch_size):
        source_rows = validation[start : start + args.batch_size]
        donor_rows = donors[start : start + args.batch_size]
        observation_rows = data.observation_row_indices[source_rows]
        donor_observation_rows = data.observation_row_indices[donor_rows]
        proposals = data.proposals[source_rows].to(device, non_blocking=True)
        scene = data.observation_tokens[observation_rows].to(
            device=device, dtype=torch.float32, non_blocking=True
        )
        status = data.ego_features[observation_rows].to(
            device=device, dtype=torch.float32, non_blocking=True
        )
        donor_scene = data.observation_tokens[donor_observation_rows].to(
            device=device, dtype=torch.float32, non_blocking=True
        )
        donor_status = data.ego_features[donor_observation_rows].to(
            device=device, dtype=torch.float32, non_blocking=True
        )
        target = data.target_factors[source_rows, :, -1].to(
            device=device, dtype=torch.float32, non_blocking=True
        )
        base = data.base_scores_for_evaluation[source_rows].argmax(dim=1)
        rows = torch.arange(len(source_rows))
        base_values.append(target.cpu()[rows, base].numpy())
        oracle_values.append(target.max(dim=1).values.cpu().numpy())

        for mode in MODES:
            mode_scene, mode_status = _mode_inputs(
                mode, scene, status, donor_scene, donor_status
            )
            output = model(
                mode_scene,
                mode_status,
                proposals,
                scene_valid_mask=torch.ones(
                    mode_scene.shape[:2], dtype=torch.bool, device=device
                ),
            )
            utility = pdms_factor_log_utility(output["factor_logits"])
            selected = utility.argmax(dim=1)
            values = target[torch.arange(len(target), device=device), selected]
            correct, count = _pairwise_counts(utility, target)
            pairwise_correct[mode] += correct
            pairwise_count[mode] += count
            selected_values[mode].append(values.cpu().numpy())
            selected_indices[mode].append(selected.cpu().numpy())

    values_by_mode = {
        mode: np.concatenate(parts).astype(np.float64)
        for mode, parts in selected_values.items()
    }
    indices_by_mode = {
        mode: np.concatenate(parts).astype(np.int16)
        for mode, parts in selected_indices.items()
    }
    correct_values = values_by_mode["correct"]
    correct_indices = indices_by_mode["correct"]
    metrics: Dict[str, Mapping[str, object]] = {}
    for mode in MODES:
        delta = values_by_mode[mode] - correct_values
        ci = _log_bootstrap_ci(
            delta,
            validation_logs,
            args.seed,
            replicates=args.bootstrap_replicates,
        )
        metrics[mode] = {
            "selected_pdms": float(values_by_mode[mode].mean()),
            "scorer_regret": float(
                np.concatenate(oracle_values).mean()
                - values_by_mode[mode].mean()
            ),
            "pairwise_accuracy": float(
                pairwise_correct[mode] / pairwise_count[mode]
            ),
            "non_tied_pair_count": pairwise_count[mode],
            "delta_vs_correct": float(delta.mean()),
            "delta_vs_correct_log_bootstrap_95ci": [float(ci[0]), float(ci[1])],
            "selection_switch_rate_vs_correct": float(
                np.mean(indices_by_mode[mode] != correct_indices)
            ),
        }

    payload = {
        "audit": "released_drivor_representation_dependence",
        "split": "held_out_physical_logs",
        "scene_count": len(validation),
        "physical_log_count": len(set(validation_logs)),
        "candidate_count": 64,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": _sha256(args.checkpoint),
        "initialization": initialization,
        "split_lineage": split_lineage,
        "source_lineage": lineage,
        "current_observation_only": True,
        "future_or_evaluator_input": False,
        "official_score_input_present_during_inference": False,
        "labels_joined_after_selection": True,
        "cross_log_shuffle_has_no_same_log_donor": True,
        "base_selected_pdms": float(np.concatenate(base_values).mean()),
        "best_of_64_pdms": float(np.concatenate(oracle_values).mean()),
        "metrics": metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

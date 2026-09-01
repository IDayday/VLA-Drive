#!/usr/bin/env python3
"""Audit whether an independent scorer actually uses its current observation.

The proposal bank and scorer weights stay fixed. Current visual tokens and ego
context are then zeroed or replaced by a deterministic donor from a different
physical log. Official PDM labels are used only after each mode has selected a
proposal; they are never model inputs.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

from local_stage2.audit_drivor_representation_dependence import (
    MODES,
    _cross_log_derangement,
    _mode_inputs,
    _pairwise_counts,
)
from local_stage2.train_independent_scorer import (
    ReplaySource,
    _log_bootstrap_ci,
    _sha256,
    load_replay_sources,
)
from navsim.agents.EpisodeDrive.score_module.independent_ranker import (
    IndependentProposalRanker,
    IndependentRankerConfig,
    pdms_factor_log_utility,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument(
        "--ensemble-artifact",
        type=Path,
        action="append",
        default=[],
        help="Optional additional same-mode artifacts for equal-score ensembling.",
    )
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--label-root", type=Path, required=True)
    parser.add_argument("--private-observation-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--score-mode",
        choices=("artifact", "direct", "coarse", "factor"),
        default="artifact",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _score_output(output: Mapping[str, torch.Tensor], mode: str) -> torch.Tensor:
    if mode == "direct":
        return output["utility"]
    if mode == "coarse":
        return output["coarse_utility"]
    if mode == "factor":
        return pdms_factor_log_utility(output["factor_logits"])
    raise ValueError(f"unsupported score mode: {mode}")


def _mode_mask(
    mode: str,
    source_mask: torch.Tensor,
    donor_mask: torch.Tensor,
) -> torch.Tensor:
    if mode in ("scene_cross_log_shuffle", "scene_and_status_cross_log_shuffle"):
        return donor_mask
    if mode in MODES:
        return source_mask
    raise ValueError(f"unknown mode: {mode}")


def _validation_indices(
    physical_logs: Sequence[str], split_manifest: Path
) -> Tuple[List[int], Dict[str, object]]:
    split = json.loads(split_manifest.read_text())
    allowed = {str(value) for value in split["validation_physical_logs"]}
    indices = [
        index for index, log_name in enumerate(physical_logs) if log_name in allowed
    ]
    actual = {str(physical_logs[index]) for index in indices}
    if not indices or actual != allowed:
        raise RuntimeError(
            "validation split/replay physical-log mismatch: "
            f"rows={len(indices)} missing={len(allowed.difference(actual))}"
        )
    return indices, {
        "path": str(split_manifest.resolve()),
        "sha256": _sha256(split_manifest),
        "physical_log_count": len(allowed),
    }


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    artifact_paths = [args.artifact, *args.ensemble_artifact]
    if len(set(artifact_paths)) != len(artifact_paths):
        raise RuntimeError("ensemble artifact paths must be unique")
    for path in (
        *artifact_paths,
        args.feature_root,
        args.label_root,
        args.private_observation_root,
        args.split_manifest,
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
    validation_indices, split_lineage = _validation_indices(
        data.physical_logs, args.split_manifest
    )
    validation_logs = [data.physical_logs[index] for index in validation_indices]
    donor_local = _cross_log_derangement(validation_logs, args.seed)
    validation = torch.as_tensor(validation_indices, dtype=torch.long)
    donors = validation[torch.as_tensor(donor_local, dtype=torch.long)]
    donor_logs = [data.physical_logs[int(index)] for index in donors]
    if any(left == right for left, right in zip(validation_logs, donor_logs)):
        raise RuntimeError("cross-log donor permutation contains same-log rows")

    artifacts = [
        torch.load(path, map_location="cpu", weights_only=False)
        for path in artifact_paths
    ]
    if any(
        artifact.get("architecture") != "IndependentProposalRanker"
        for artifact in artifacts
    ):
        raise RuntimeError("artifact is not an IndependentProposalRanker")
    artifact_modes = {
        str(artifact.get("selection_mode", "direct")) for artifact in artifacts
    }
    if len(artifact_modes) != 1:
        raise RuntimeError(
            f"ensemble artifacts have different score modes: {artifact_modes}"
        )
    artifact_mode = next(iter(artifact_modes))
    score_mode = artifact_mode if args.score_mode == "artifact" else args.score_mode
    configs = [
        IndependentRankerConfig(**artifact["model_config"])
        for artifact in artifacts
    ]
    if any(config != configs[0] for config in configs[1:]):
        raise RuntimeError("ensemble artifacts have different model configurations")
    device = torch.device(args.device)
    models = []
    for artifact, config in zip(artifacts, configs):
        model = IndependentProposalRanker(config)
        model.load_state_dict(artifact["state_dict"], strict=True)
        model.to(device).eval()
        models.append(model)

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
        scene_mask = data.observation_valid_masks[observation_rows].to(
            device=device, non_blocking=True
        )
        status = data.ego_features[observation_rows].to(
            device=device, dtype=torch.float32, non_blocking=True
        )
        donor_scene = data.observation_tokens[donor_observation_rows].to(
            device=device, dtype=torch.float32, non_blocking=True
        )
        donor_mask = data.observation_valid_masks[donor_observation_rows].to(
            device=device, non_blocking=True
        )
        donor_status = data.ego_features[donor_observation_rows].to(
            device=device, dtype=torch.float32, non_blocking=True
        )

        # Targets stay outside the model call and are joined only after each
        # mode has produced an immutable selected index.
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
            outputs = [
                model(
                    mode_scene,
                    mode_status,
                    proposals,
                    observation_valid_mask=_mode_mask(
                        mode, scene_mask, donor_mask
                    ),
                )
                for model in models
            ]
            utility = torch.stack(
                [_score_output(output, score_mode) for output in outputs]
            ).mean(dim=0)
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
    oracle = np.concatenate(oracle_values).astype(np.float64)
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
            "scorer_regret": float(oracle.mean() - values_by_mode[mode].mean()),
            "pairwise_accuracy": float(
                pairwise_correct[mode] / max(pairwise_count[mode], 1)
            ),
            "non_tied_pair_count": pairwise_count[mode],
            "delta_vs_correct": float(delta.mean()),
            "delta_vs_correct_log_bootstrap_95ci": [float(ci[0]), float(ci[1])],
            "selection_switch_rate_vs_correct": float(
                np.mean(indices_by_mode[mode] != correct_indices)
            ),
        }

    payload = {
        "audit": "independent_scorer_representation_dependence",
        "split": "held_out_physical_logs",
        "scene_count": len(validation),
        "physical_log_count": len(set(validation_logs)),
        "candidate_count": 64,
        "artifact": str(args.artifact.resolve()),
        "artifact_sha256": _sha256(args.artifact),
        "artifact_epoch": int(artifacts[0]["epoch"]),
        "artifact_selection_mode": artifact_mode,
        "ensemble_artifacts": [
            {
                "path": str(path.resolve()),
                "sha256": _sha256(path),
                "epoch": int(artifact["epoch"]),
            }
            for path, artifact in zip(artifact_paths, artifacts)
        ],
        "ensemble_size": len(artifacts),
        "ensemble_method": "equal_mean_score",
        "evaluated_score_mode": score_mode,
        "split_lineage": split_lineage,
        "source_lineage": lineage,
        "current_observation_only": True,
        "future_or_evaluator_input": False,
        "official_score_input_present_during_inference": False,
        "labels_joined_after_selection": True,
        "cross_log_shuffle_has_no_same_log_donor": True,
        "base_selected_pdms": float(np.concatenate(base_values).mean()),
        "best_of_64_pdms": float(oracle.mean()),
        "metrics": metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

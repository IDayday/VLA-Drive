#!/usr/bin/env python3
"""Evaluate a validation-promoted independent DrivOR ranker on Navtest."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch

from local_stage2.train_independent_scorer import (
    load_private_observation_table,
    physical_log_name,
)
from local_stage2.train_public_base_residual_scorer import _log_bootstrap_ci
from navsim.agents.EpisodeDrive.score_module.drivor_ranker import (
    DrivORInitializedProposalRanker,
    DrivORRankerConfig,
)
from navsim.agents.EpisodeDrive.score_module.independent_ranker import (
    pdms_factor_log_utility,
)


def _sha256(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    checkpoint = parser.add_mutually_exclusive_group(required=True)
    checkpoint.add_argument(
        "--artifact",
        type=Path,
        help="Validation-promoted DrivORInitializedProposalRanker artifact.",
    )
    checkpoint.add_argument(
        "--released-drivor-checkpoint",
        type=Path,
        help=(
            "Released DrivOR checkpoint used without any scorer re-fitting. "
            "This is a diagnostic proposal-bank transfer evaluation and is "
            "not subject to the learned-artifact promotion gate."
        ),
    )
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--private-observation-root", type=Path, required=True)
    parser.add_argument("--candidate-matrix", type=Path, required=True)
    parser.add_argument("--public-audit-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _require_fp32_register_cache(root: Path) -> None:
    manifests = sorted(root.glob("**/manifest.json"))
    if not manifests:
        raise RuntimeError("private register cache has no completed manifests")
    precisions = {
        str(json.loads(path.read_text()).get("precision")) for path in manifests
    }
    if precisions != {"fp32_compute_float32_cache"}:
        raise RuntimeError(
            "Navtest claim requires FP32 current-register cache, got "
            f"{sorted(precisions)}"
        )


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    checkpoint = args.artifact or args.released_drivor_checkpoint
    assert checkpoint is not None
    required = (
        checkpoint,
        args.feature_cache,
        args.private_observation_root,
        args.candidate_matrix,
        args.public_audit_dir / "summary.json",
        args.public_audit_dir / "per_scene_candidate_quality.csv",
    )
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    _require_fp32_register_cache(args.private_observation_root)

    released_transfer = args.released_drivor_checkpoint is not None
    artifact = None
    initialization_audit = None
    if released_transfer:
        score_mode = "factor"
        selection_source = "released_drivor_no_refit"
        validation = None
    else:
        assert args.artifact is not None
        artifact = torch.load(
            args.artifact, map_location="cpu", weights_only=False
        )
        if artifact.get("architecture") != "DrivORInitializedProposalRanker":
            raise RuntimeError("artifact architecture mismatch")
        score_mode = str(artifact.get("selection_mode"))
        if score_mode not in {"direct", "factor"}:
            raise RuntimeError(
                f"unsupported artifact selection mode: {score_mode}"
            )
        selection_source = str(
            artifact["training_manifest"]["checkpoint_selection_source"]
        )
        validation = artifact["validation_by_source"][selection_source]
        ci_key = (
            "factor_selected_delta_log_bootstrap_95ci"
            if score_mode == "factor"
            else "selected_delta_log_bootstrap_95ci"
        )
        if float(validation[ci_key][0]) <= 0.0:
            raise RuntimeError(
                "artifact did not pass the held-out-log positive-CI "
                "promotion gate"
            )
    with args.feature_cache.open("rb") as stream:
        feature_cache = pickle.load(stream)
    with np.load(args.candidate_matrix, allow_pickle=False) as matrix:
        tokens = [str(value) for value in matrix["tokens"]]
        log_names = [str(value) for value in matrix["log_names"]]
        candidate_scores = np.asarray(matrix["candidate_scores"], dtype=np.float32)
        candidate_factors = np.asarray(matrix["candidate_factors"], dtype=np.float32)
        factor_names = [str(value) for value in matrix["candidate_factor_names"]]
        base_matrix = np.asarray(matrix["predicted_scores"], dtype=np.float32)
    if len(tokens) != 12_146 or len(set(tokens)) != 12_146:
        raise RuntimeError("Navtest must contain exactly 12,146 unique scenes")
    if len(set(log_names)) != 136:
        raise RuntimeError("Navtest must contain exactly 136 segment logs")
    if candidate_scores.shape != (12_146, 64):
        raise RuntimeError("candidate score matrix shape mismatch")
    if candidate_factors.shape != (12_146, 64, 7):
        raise RuntimeError("candidate factor matrix shape mismatch")
    if base_matrix.shape != (12_146, 64):
        raise RuntimeError("Base score matrix shape mismatch")
    if factor_names[-1] != "score":
        raise RuntimeError("candidate factor matrix lacks aggregate score")
    if set(tokens) != set(feature_cache):
        raise RuntimeError("feature cache and candidate matrix token sets differ")

    private = load_private_observation_table(args.private_observation_root)
    if set(private.tokens) != set(tokens):
        raise RuntimeError("private register and candidate token sets differ")
    private_rows = {token: index for index, token in enumerate(private.tokens)}
    device = torch.device(args.device)
    if released_transfer:
        assert args.released_drivor_checkpoint is not None
        model = DrivORInitializedProposalRanker(DrivORRankerConfig())
        initialization_audit = model.load_drivor_checkpoint(
            args.released_drivor_checkpoint
        )
        cache_checkpoint_sha = str(
            private.lineage.get("checkpoint_sha256", "")
        )
        released_checkpoint_sha = _sha256(args.released_drivor_checkpoint)
        if cache_checkpoint_sha != released_checkpoint_sha:
            raise RuntimeError(
                "DrivOR register cache/checkpoint SHA mismatch: "
                f"cache={cache_checkpoint_sha}, "
                f"checkpoint={released_checkpoint_sha}"
            )
        model.to(device)
    else:
        assert artifact is not None
        model = DrivORInitializedProposalRanker(
            DrivORRankerConfig(**artifact["model_config"])
        ).to(device)
        model.load_state_dict(artifact["state_dict"], strict=True)
    model.eval()
    utility_parts: List[np.ndarray] = []
    index_parts: List[np.ndarray] = []
    for start in range(0, len(tokens), args.batch_size):
        batch_tokens = tokens[start : start + args.batch_size]
        rows = torch.as_tensor(
            [private_rows[token] for token in batch_tokens], dtype=torch.long
        )
        proposals = torch.as_tensor(
            np.stack([feature_cache[token]["proposals"] for token in batch_tokens]),
            dtype=torch.float32,
            device=device,
        )
        scene = private.observation_tokens[rows].to(device=device, dtype=torch.float32)
        mask = private.observation_valid_masks[rows].to(device=device)
        status = private.status_features[rows].to(device=device, dtype=torch.float32)
        with torch.inference_mode():
            output = model(scene, status, proposals, scene_valid_mask=mask)
            utility = (
                pdms_factor_log_utility(output["factor_logits"])
                if score_mode == "factor"
                else output["direct_utility"]
            )
            selected = utility.argmax(dim=1)
        utility_parts.append(utility.float().cpu().numpy())
        index_parts.append(selected.cpu().numpy())

    predicted_scores = np.concatenate(utility_parts).astype(np.float32)
    selected_indices = np.concatenate(index_parts).astype(np.int16)
    oracle_indices = candidate_scores.argmax(axis=1).astype(np.int16)
    rows = np.arange(len(tokens))
    selected_values = candidate_scores[rows, selected_indices]
    oracle_values = candidate_scores[rows, oracle_indices]
    base_indices = base_matrix.argmax(axis=1)
    base_values = candidate_scores[rows, base_indices]
    delta = selected_values - base_values
    physical_logs = [physical_log_name(value) for value in log_names]
    ci = _log_bootstrap_ci(
        delta, physical_logs, args.seed, replicates=args.bootstrap_replicates
    )

    public_frame = pd.read_csv(
        args.public_audit_dir / "per_scene_candidate_quality.csv"
    ).set_index("token")
    if set(public_frame.index.astype(str)) != set(tokens):
        raise RuntimeError("public audit CSV token set differs")
    frame = public_frame.loc[tokens].reset_index()
    frame["selected_index"] = selected_indices
    frame["oracle_index"] = oracle_indices
    frame["selected_pdms"] = selected_values
    frame["standard_selected_pdms"] = selected_values
    frame["selected_score_parity_abs"] = 0.0
    frame["scorer_regret"] = (
        frame["best_of_64_pdms"].to_numpy(dtype=np.float64)
        - frame["selected_pdms"].to_numpy(dtype=np.float64)
    )
    factor_index = {name: index for index, name in enumerate(factor_names)}
    for name in factor_names[:-1]:
        frame[f"selected_{name}"] = candidate_factors[
            rows, selected_indices, factor_index[name]
        ]

    args.output_dir.mkdir(parents=True, exist_ok=False)
    frame.to_csv(
        args.output_dir / "per_scene_candidate_quality.csv",
        index=False,
        float_format="%.10f",
    )
    np.savez_compressed(
        args.output_dir / "candidate_scores.npz",
        tokens=np.asarray(tokens),
        log_names=np.asarray(log_names),
        candidate_scores=candidate_scores,
        predicted_scores=predicted_scores,
        selected_indices=selected_indices,
        oracle_indices=oracle_indices,
        candidate_factors=candidate_factors,
        candidate_factor_names=np.asarray(factor_names),
    )
    proposal_predictions = {
        token: {
            "proposals": np.asarray(feature_cache[token]["proposals"], dtype=np.float32),
            "predicted_scores": predicted_scores[index],
        }
        for index, token in enumerate(tokens)
    }
    proposal_path = args.output_dir / "proposal_predictions.pkl"
    with proposal_path.open("wb") as stream:
        pickle.dump(proposal_predictions, stream, protocol=pickle.HIGHEST_PROTOCOL)

    public_summary = json.loads((args.public_audit_dir / "summary.json").read_text())
    metrics = dict(public_summary["metrics"])
    metrics.update(
        {
            "selected_pdms": float(selected_values.mean()),
            "standard_selected_pdms": float(selected_values.mean()),
            "selected_score_parity_abs": 0.0,
            "scorer_regret": float(frame["scorer_regret"].mean()),
        }
    )
    for name in factor_names[:-1]:
        metrics[f"selected_{name}"] = float(
            candidate_factors[rows, selected_indices, factor_index[name]].mean()
        )
    wins = int((delta > 1.0e-9).sum())
    losses = int((delta < -1.0e-9).sum())
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "split": "navtest",
        "scene_count": 12_146,
        "valid_scene_count": 12_146,
        "invalid_scene_count": 0,
        "log_count": 136,
        "candidate_count": 64,
        "candidate_factor_matrix_present": True,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": _sha256(checkpoint),
        "agent_target": (
            "DrivORInitializedProposalRanker(released_weights)"
            if released_transfer
            else "DrivORInitializedProposalRanker"
        ),
        "precision": 32,
        "score_mode": score_mode,
        "released_drivor_zero_refit_transfer": released_transfer,
        "released_checkpoint_initialization_audit": initialization_audit,
        "validation_selection_source": selection_source,
        "validation_locked_metrics": validation,
        "proposal_predictions_path": str(proposal_path.resolve()),
        "proposal_predictions_sha256": _sha256(proposal_path),
        "feature_cache_path": str(args.feature_cache.resolve()),
        "feature_cache_sha256": _sha256(args.feature_cache),
        "private_observation_lineage": private.lineage,
        "candidate_matrix_source": str(args.candidate_matrix.resolve()),
        "candidate_matrix_source_sha256": _sha256(args.candidate_matrix),
        "inference_inputs_only": True,
        "future_target_present_during_inference": False,
        "official_score_input_present_during_inference": False,
        "official_candidate_matrix_joined_after_selection": True,
        "base_score_used_by_independent_ranker": False,
        "max_selected_score_parity_abs": 0.0,
        "metrics": metrics,
        "comparison_to_public_base": {
            "public_selected_pdms": float(base_values.mean()),
            "selected_delta": float(delta.mean()),
            "selected_delta_log_bootstrap_95ci": [float(ci[0]), float(ci[1])],
            "wins": wins,
            "losses": losses,
            "ties": int(len(delta) - wins - losses),
            "switch_rate": float(np.mean(selected_indices != base_indices)),
            "physical_log_count": len(set(physical_logs)),
        },
        "offline_oracle_candidate_bank_upper_bound": float(oracle_values.mean()),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

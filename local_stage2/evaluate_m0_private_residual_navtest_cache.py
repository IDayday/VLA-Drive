#!/usr/bin/env python3
"""Evaluate an M0-private residual scorer on the locked full Navtest cache.

The model forward pass consumes only deployable M0 tensors: current visual
tokens, current context, proposals, released Base factor logits, and released
Base scorer values.  Official candidate factors are joined only after the
selected proposal index has been frozen.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

from local_stage2.train_independent_scorer import (
    load_private_observation_table,
    physical_log_name,
)
from local_stage2.train_public_base_residual_scorer import _log_bootstrap_ci
from local_stage2.m0_native_private_scorer_agent import (
    M0NativePrivateScorerAgent,
    _private_vision_config,
)
from navsim.agents.EpisodeDrive.score_module.m0_private_residual_ranker import (
    M0PrivateResidualConfig,
    M0PrivateResidualRanker,
)
from navsim.agents.EpisodeDrive.score_module.independent_ranker import (
    IndependentRankerConfig,
)


def _sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while block := file.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--private-observation-root", type=Path, required=True)
    parser.add_argument("--candidate-matrix", type=Path, required=True)
    parser.add_argument("--public-audit-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _required_feature(entry: Dict[str, object], key: str, shape: tuple[int, ...]) -> np.ndarray:
    if key not in entry:
        raise RuntimeError(f"locked M0 feature cache lacks {key}")
    value = np.asarray(entry[key], dtype=np.float32)
    if value.shape != shape:
        raise RuntimeError(f"unexpected {key} shape: {value.shape} != {shape}")
    if not np.isfinite(value).all():
        raise RuntimeError(f"non-finite {key} in locked M0 feature cache")
    return value


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    required_paths = (
        args.artifact,
        args.feature_cache,
        args.private_observation_root,
        args.candidate_matrix,
        args.public_audit_dir / "summary.json",
        args.public_audit_dir / "per_scene_candidate_quality.csv",
    )
    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(path)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    artifact = torch.load(args.artifact, map_location="cpu", weights_only=False)
    if artifact.get("artifact_type") != M0NativePrivateScorerAgent.ARTIFACT_TYPE:
        raise RuntimeError("deployment artifact type mismatch")
    if artifact.get("scorer_architecture") != "M0PrivateResidualRanker":
        raise RuntimeError("deployment artifact architecture mismatch")
    expected_schema = (
        "m0_current_f0_l0_r0_b0_images",
        "m0_current_ego_navigation_status",
        "m0_proposals",
        "m0_base_factor_logits",
        "m0_base_scores",
    )
    declared_schema = tuple(artifact.get("inference_input_schema", ()))
    if declared_schema != expected_schema:
        raise RuntimeError("deployment artifact inference schema mismatch")

    with args.feature_cache.open("rb") as file:
        feature_cache = pickle.load(file)
    private_table = load_private_observation_table(args.private_observation_root)
    matrix = np.load(args.candidate_matrix, allow_pickle=False)
    tokens = [str(value) for value in matrix["tokens"]]
    log_names = [str(value) for value in matrix["log_names"]]
    candidate_scores = np.asarray(matrix["candidate_scores"], dtype=np.float32)
    candidate_factors = np.asarray(matrix["candidate_factors"], dtype=np.float32)
    factor_names = [str(value) for value in matrix["candidate_factor_names"]]

    if len(tokens) != 12_146 or len(set(tokens)) != 12_146:
        raise RuntimeError("Navtest must contain exactly 12,146 unique scenes")
    if len(set(log_names)) != 136:
        raise RuntimeError("Navtest must contain exactly 136 segment logs")
    if candidate_scores.shape != (12_146, 64):
        raise RuntimeError(f"unexpected candidate score shape: {candidate_scores.shape}")
    if candidate_factors.shape != (12_146, 64, 7):
        raise RuntimeError(f"unexpected candidate factor shape: {candidate_factors.shape}")
    if factor_names[-1] != "score":
        raise RuntimeError("candidate factor matrix lacks aggregate score")
    if set(tokens) != set(feature_cache):
        raise RuntimeError("M0 feature cache and candidate matrix token sets differ")
    if set(tokens) != set(private_table.tokens):
        raise RuntimeError(
            "private-observation cache and candidate matrix token sets differ"
        )
    evaluation_vision_config = _private_vision_config(
        args.private_observation_root
    )
    declared_vision_config = artifact["private_vision_config"]
    for key in (
        "m0_checkpoint_sha256",
        "camera_names",
        "max_dynamic_tiles",
        "max_crops_per_camera",
        "pool_grid",
        "visual_token_count",
        "visual_width",
        "visual_model_wrapper_chain",
    ):
        if evaluation_vision_config[key] != declared_vision_config[key]:
            raise RuntimeError(f"Navtest private vision config differs for {key}")

    private_row_for_token = {
        token: index for index, token in enumerate(private_table.tokens)
    }
    base_matrix = np.asarray(matrix["predicted_scores"], dtype=np.float32)
    locked_base_matrix = np.stack(
        [
            _required_feature(feature_cache[token], "predicted_scores", (64,))
            for token in tokens
        ]
    )
    base_score_parity = float(np.max(np.abs(base_matrix - locked_base_matrix)))
    if base_score_parity > 1.0e-8:
        raise RuntimeError(
            "locked Base score cache differs from candidate matrix: "
            f"max_abs={base_score_parity}"
        )

    device = torch.device(args.device)
    model = M0PrivateResidualRanker(
        IndependentRankerConfig(**dict(artifact["private_config"])),
        M0PrivateResidualConfig(**dict(artifact["residual_config"])),
    ).to(device)
    model.load_state_dict(artifact["scorer_state_dict"], strict=True)
    model.eval()

    prediction_parts: List[np.ndarray] = []
    selected_parts: List[np.ndarray] = []
    for start in range(0, len(tokens), args.batch_size):
        batch_tokens = tokens[start : start + args.batch_size]
        private_rows = torch.tensor(
            [private_row_for_token[token] for token in batch_tokens],
            dtype=torch.long,
        )
        observation = private_table.observation_tokens[private_rows].to(
            device, non_blocking=True
        ).float()
        observation_mask = private_table.observation_valid_masks[private_rows].to(
            device, non_blocking=True
        )
        status = private_table.status_features[private_rows].to(
            device, non_blocking=True
        ).float()
        proposals = torch.as_tensor(
            np.stack(
                [
                    _required_feature(feature_cache[token], "proposals", (64, 8, 3))
                    for token in batch_tokens
                ]
            ),
            dtype=torch.float32,
            device=device,
        )
        base_scores = torch.as_tensor(
            np.stack(
                [
                    _required_feature(feature_cache[token], "predicted_scores", (64,))
                    for token in batch_tokens
                ]
            ),
            dtype=torch.float32,
            device=device,
        )
        base_factor_logits = torch.as_tensor(
            np.stack(
                [
                    _required_feature(
                        feature_cache[token], "base_factor_logits", (64, 6)
                    )
                    for token in batch_tokens
                ]
            ),
            dtype=torch.float32,
            device=device,
        )
        with torch.no_grad():
            output = model(
                observation,
                status,
                proposals,
                base_factor_logits,
                base_scores,
                observation_valid_mask=observation_mask,
            )
            prediction = output["selection_scores"]
            selected = prediction.argmax(dim=1)
        prediction_parts.append(prediction.float().cpu().numpy())
        selected_parts.append(selected.cpu().numpy())

    # The official candidate-factor matrix is first touched for labels here,
    # after every inference-time selected index has been frozen.
    predicted_scores = np.concatenate(prediction_parts).astype(np.float32)
    selected_indices = np.concatenate(selected_parts).astype(np.int16)
    oracle_indices = candidate_scores.argmax(axis=1).astype(np.int16)
    rows = np.arange(len(tokens))
    selected_values = candidate_scores[rows, selected_indices]
    oracle_values = candidate_scores[rows, oracle_indices]
    base_indices = locked_base_matrix.argmax(axis=1)
    base_values = candidate_scores[rows, base_indices]
    delta = selected_values - base_values
    physical_logs = [physical_log_name(value) for value in log_names]
    ci = _log_bootstrap_ci(
        delta,
        physical_logs,
        args.seed,
        replicates=args.bootstrap_replicates,
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
    with proposal_path.open("wb") as file:
        pickle.dump(proposal_predictions, file, protocol=pickle.HIGHEST_PROTOCOL)

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
        "checkpoint": str(args.artifact.resolve()),
        "checkpoint_sha256": _sha256(args.artifact),
        "checkpoint_class": (
            "local_stage2.m0_native_private_scorer_agent."
            "M0NativePrivateScorerAgent"
        ),
        "ranker_class": (
            "navsim.agents.EpisodeDrive.score_module.m0_private_residual_ranker."
            "M0PrivateResidualRanker"
        ),
        "source_ranker_artifact_sha256": artifact[
            "source_ranker_artifact_sha256"
        ],
        "checkpoint_epoch": int(artifact["source_ranker_epoch"]),
        "checkpoint_validation": artifact.get("source_ranker_validation"),
        "agent_target": (
            "local_stage2.m0_native_private_scorer_agent."
            "M0NativePrivateScorerAgent"
        ),
        "precision": 32,
        "proposal_predictions_path": str(proposal_path.resolve()),
        "proposal_predictions_sha256": _sha256(proposal_path),
        "feature_cache_path": str(args.feature_cache.resolve()),
        "feature_cache_sha256": _sha256(args.feature_cache),
        "private_observation_lineage": private_table.lineage,
        "candidate_matrix_source": str(args.candidate_matrix.resolve()),
        "candidate_matrix_source_sha256": _sha256(args.candidate_matrix),
        "base_score_cache_parity_max_abs": base_score_parity,
        "inference_inputs_only": True,
        "future_target_present_during_inference": False,
        "official_score_input_present_during_inference": False,
        "official_candidate_matrix_joined_after_selection": True,
        "m0_base_model_score_used_as_input": True,
        "m0_base_factor_logits_used_as_input": True,
        "external_model_representation_or_weight_used": False,
        "drivor_representation_or_weight_used": False,
        "score_mode": artifact["score_mode"],
        "residual_top_k": int(artifact["residual_config"]["top_k"]),
        "max_selected_score_parity_abs": 0.0,
        "metrics": metrics,
        "comparison_to_public_base": {
            "public_selected_pdms": float(base_values.mean()),
            "selected_delta": float(delta.mean()),
            "selected_delta_log_bootstrap_95ci": [float(ci[0]), float(ci[1])],
            "wins": wins,
            "losses": losses,
            "ties": int(len(delta) - wins - losses),
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

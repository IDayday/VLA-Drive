#!/usr/bin/env python3
"""Evaluate a validation-locked conservative-reference scorer on Navtest."""

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

from local_stage2.train_independent_scorer import physical_log_name
from local_stage2.train_independent_scorer import load_private_observation_table
from local_stage2.train_public_base_residual_scorer import _log_bootstrap_ci
from navsim.agents.EpisodeDrive.score_module.drivor_ranker import (
    DrivORRankerConfig,
    DrivORReferenceGateRanker,
)
from navsim.agents.EpisodeDrive.score_module.independent_ranker import (
    ConservativeReferenceConfig,
    IndependentConservativeReferenceRanker,
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
    parser.add_argument("--policy-evaluation", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument(
        "--private-observation-root",
        type=Path,
        default=None,
        help=(
            "Optional current-only chunk cache produced by "
            "export_private_visual_navtest.py.  When supplied, its visual "
            "tokens, mask, and current status replace the legacy cached "
            "scene/ego tensors; proposals and Base fallback indices still "
            "come from --feature-cache."
        ),
    )
    parser.add_argument("--candidate-matrix", type=Path, required=True)
    parser.add_argument("--public-audit-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    required = (
        args.artifact,
        args.policy_evaluation,
        args.feature_cache,
        args.candidate_matrix,
        args.public_audit_dir / "summary.json",
        args.public_audit_dir / "per_scene_candidate_quality.csv",
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    artifact = torch.load(args.artifact, map_location="cpu", weights_only=False)
    architecture = str(artifact.get("architecture"))
    supported_architectures = {
        "IndependentConservativeReferenceRanker",
        "DrivORReferenceGateRanker",
    }
    if architecture not in supported_architectures:
        raise RuntimeError(f"artifact architecture mismatch: {architecture}")
    policy_audit = json.loads(args.policy_evaluation.read_text())
    if "metrics" in policy_audit:
        policy_metrics = policy_audit["metrics"]
    elif "history" in policy_audit and "best_epoch" in policy_audit:
        records = [
            record
            for record in policy_audit["history"]
            if int(record["epoch"]) == int(policy_audit["best_epoch"])
        ]
        if len(records) != 1:
            raise RuntimeError("policy training summary has no unique best epoch")
        selected_record = records[0]
        selection_source = policy_audit.get("selection_source")
        if selection_source and "validation_by_source" in selected_record:
            policy_metrics = selected_record["validation_by_source"][
                selection_source
            ]
        else:
            policy_metrics = selected_record["validation"]
    else:
        raise RuntimeError("unrecognized policy-evaluation schema")
    policy = policy_metrics["best_policy"]
    if not bool(policy_metrics["any_positive_ci_policy"]):
        raise RuntimeError("validation policy did not pass positive cluster CI")
    if float(policy["delta_log_bootstrap_95ci"][0]) <= 0.0:
        raise RuntimeError("selected policy has non-positive validation CI")
    artifact_policy = artifact.get("selected_policy")
    if artifact_policy is not None and artifact_policy != policy:
        raise RuntimeError("artifact and validation-locked policies differ")

    with args.feature_cache.open("rb") as file:
        feature_cache = pickle.load(file)
    matrix = np.load(args.candidate_matrix, allow_pickle=False)
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
    if set(tokens) != set(feature_cache):
        raise RuntimeError("feature cache and candidate matrix token sets differ")
    if factor_names[-1] != "score":
        raise RuntimeError("candidate factor matrix lacks aggregate score")

    private_table = None
    private_row_for_token = None
    if args.private_observation_root is not None:
        if not args.private_observation_root.is_dir():
            raise FileNotFoundError(args.private_observation_root)
        private_table = load_private_observation_table(
            args.private_observation_root
        )
        if set(private_table.tokens) != set(tokens):
            raise RuntimeError(
                "private-observation cache and candidate matrix token sets differ"
            )
        private_row_for_token = {
            token: index for index, token in enumerate(private_table.tokens)
        }

    device = torch.device(args.device)
    if architecture == "IndependentConservativeReferenceRanker":
        model = IndependentConservativeReferenceRanker(
            IndependentRankerConfig(**artifact["ranker_config"]),
            ConservativeReferenceConfig(**artifact["reference_config"]),
        ).to(device)
        agent_target = "IndependentConservativeReferenceRanker"
    else:
        if private_table is None:
            raise RuntimeError(
                "DrivORReferenceGateRanker requires --private-observation-root"
            )
        model = DrivORReferenceGateRanker(
            DrivORRankerConfig(**artifact["ranker_config"]),
            ConservativeReferenceConfig(**artifact["reference_config"]),
            alternative_mode=str(artifact["alternative_mode"]),
            alternative_count=int(artifact.get("alternative_count", 1)),
        ).to(device)
        agent_target = "DrivORReferenceGateRanker"
    model.load_state_dict(artifact["state_dict"], strict=True)
    model.eval()
    selection_parts: List[np.ndarray] = []
    index_parts: List[np.ndarray] = []
    quantile_index = int(policy["gain_quantile_index"])
    for start in range(0, len(tokens), args.batch_size):
        batch_tokens = tokens[start : start + args.batch_size]
        proposals = torch.as_tensor(
            np.stack([feature_cache[token]["proposals"] for token in batch_tokens]),
            dtype=torch.float32,
            device=device,
        )
        observation_valid_mask = None
        if private_table is None:
            scene = torch.as_tensor(
                np.stack(
                    [
                        feature_cache[token]["scene_features"]
                        for token in batch_tokens
                    ]
                ),
                dtype=torch.float32,
                device=device,
            )
            ego = torch.as_tensor(
                np.stack(
                    [feature_cache[token]["ego_features"] for token in batch_tokens]
                ),
                dtype=torch.float32,
                device=device,
            )
        else:
            assert private_row_for_token is not None
            private_rows = torch.as_tensor(
                [private_row_for_token[token] for token in batch_tokens],
                dtype=torch.long,
            )
            scene = private_table.observation_tokens[private_rows].to(
                device=device, dtype=torch.float32
            )
            observation_valid_mask = private_table.observation_valid_masks[
                private_rows
            ].to(device=device)
            ego = private_table.status_features[private_rows].to(
                device=device, dtype=torch.float32
            )
        if ego.ndim == 3 and ego.shape[1] == 1:
            ego = ego[:, 0]
        # Scores are used outside model.forward only to identify the deployable
        # Base fallback.  Their numeric values never enter the learned scorer.
        batch_base = torch.as_tensor(
            np.stack(
                [feature_cache[token]["predicted_scores"] for token in batch_tokens]
            ),
            dtype=torch.float32,
            device=device,
        )
        references = batch_base.argmax(dim=1)
        with torch.inference_mode():
            if architecture == "IndependentConservativeReferenceRanker":
                output = model(
                    scene,
                    ego,
                    proposals,
                    references,
                    observation_valid_mask=observation_valid_mask,
                    minimum_lcb_gain=float(policy["minimum_lcb_gain"]),
                    maximum_safety_worse_probability=float(
                        policy["maximum_safety_worse_probability"]
                    ),
                    minimum_safe_improvement_probability=float(
                        policy["minimum_safe_improvement_probability"]
                    ),
                )
                allowed_candidate_mask = None
            else:
                output = model(
                    scene,
                    ego,
                    proposals,
                    references,
                    scene_valid_mask=observation_valid_mask,
                    minimum_lcb_gain=float(policy["minimum_lcb_gain"]),
                    maximum_safety_worse_probability=float(
                        policy["maximum_safety_worse_probability"]
                    ),
                    minimum_safe_improvement_probability=float(
                        policy["minimum_safe_improvement_probability"]
                    ),
                )
                allowed_candidate_mask = output["allowed_candidate_mask"]
            # forward defaults to q10; recompute only when validation locked a
            # different quantile.  This is still the exact same learned output.
            from navsim.agents.EpisodeDrive.score_module.independent_ranker import (
                conservative_reference_selection_scores,
            )

            scores = conservative_reference_selection_scores(
                output["gain_quantiles"],
                output["safety_worse_logits"],
                output["safe_improvement_logit"],
                references,
                gain_quantile_index=quantile_index,
                minimum_lcb_gain=float(policy["minimum_lcb_gain"]),
                maximum_safety_worse_probability=float(
                    policy["maximum_safety_worse_probability"]
                ),
                minimum_safe_improvement_probability=float(
                    policy["minimum_safe_improvement_probability"]
                ),
                allowed_candidate_mask=allowed_candidate_mask,
            )
            selected = scores.argmax(dim=1)
        selection_parts.append(scores.float().cpu().numpy())
        index_parts.append(selected.cpu().numpy())

    predicted_scores = np.concatenate(selection_parts).astype(np.float32)
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
            "proposals": np.asarray(
                feature_cache[token]["proposals"], dtype=np.float32
            ),
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
    wins = int((delta > 1e-9).sum())
    losses = int((delta < -1e-9).sum())
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
        "agent_target": agent_target,
        "precision": 32,
        "proposal_predictions_path": str(proposal_path.resolve()),
        "proposal_predictions_sha256": _sha256(proposal_path),
        "feature_cache_path": str(args.feature_cache.resolve()),
        "feature_cache_sha256": _sha256(args.feature_cache),
        "private_observation_lineage": (
            private_table.lineage if private_table is not None else None
        ),
        "candidate_matrix_source": str(args.candidate_matrix.resolve()),
        "candidate_matrix_source_sha256": _sha256(args.candidate_matrix),
        "policy_evaluation": str(args.policy_evaluation.resolve()),
        "policy_evaluation_sha256": _sha256(args.policy_evaluation),
        "validation_locked_policy": policy,
        "inference_inputs_only": True,
        "future_target_present_during_inference": False,
        "official_score_input_present_during_inference": False,
        "official_candidate_matrix_joined_after_selection": True,
        "base_numeric_score_used_by_reference_ranker": False,
        "base_selection_index_used_as_reference": True,
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
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

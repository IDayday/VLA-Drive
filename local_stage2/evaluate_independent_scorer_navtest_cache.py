#!/usr/bin/env python3
"""Evaluate a standalone current-observation scorer on locked M0 Navtest proposals."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from local_stage2.train_independent_scorer import (
    load_private_observation_table,
    physical_log_name,
)
from navsim.agents.EpisodeDrive.score_module.independent_ranker import (
    IndependentProposalRanker,
    IndependentRankerConfig,
    pdms_factor_log_utility,
)


def _sha256(path: Path, block_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def _log_bootstrap_ci(
    values: np.ndarray,
    log_names: Sequence[str],
    seed: int,
    replicates: int,
) -> Tuple[float, float]:
    grouped: Dict[str, List[float]] = {}
    for value, log_name in zip(values, log_names):
        grouped.setdefault(str(log_name), []).append(float(value))
    ordered = sorted(grouped)
    sums = np.asarray([np.sum(grouped[name]) for name in ordered])
    counts = np.asarray([len(grouped[name]) for name in ordered])
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(ordered), size=(replicates, len(ordered)))
    estimates = sums[sampled].sum(1) / counts[sampled].sum(1)
    return float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))


def _score_mode(output: Dict[str, torch.Tensor], mode: str) -> torch.Tensor:
    if mode == "direct":
        return output["utility"]
    if mode == "coarse":
        return output["coarse_utility"]
    if mode == "factor":
        return pdms_factor_log_utility(output["factor_logits"])
    raise RuntimeError(f"Unsupported independent score mode: {mode}")


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
    parser.add_argument("--private-observation-root", type=Path, required=True)
    parser.add_argument("--proposal-pickle", type=Path, required=True)
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
    for path in (
        args.artifact,
        *args.ensemble_artifact,
        args.private_observation_root,
        args.proposal_pickle,
        args.candidate_matrix,
        args.public_audit_dir / "summary.json",
        args.public_audit_dir / "per_scene_candidate_quality.csv",
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    artifact_paths = [args.artifact, *args.ensemble_artifact]
    if len(set(artifact_paths)) != len(artifact_paths):
        raise RuntimeError("ensemble artifact paths must be unique")
    artifacts = [
        torch.load(path, map_location="cpu", weights_only=False)
        for path in artifact_paths
    ]
    for artifact in artifacts:
        if artifact.get("architecture") != "IndependentProposalRanker":
            raise RuntimeError("artifact is not an IndependentProposalRanker")
    modes = {str(artifact.get("selection_mode", "direct")) for artifact in artifacts}
    if len(modes) != 1:
        raise RuntimeError(f"ensemble artifacts have different score modes: {modes}")
    mode = next(iter(modes))
    configs = [IndependentRankerConfig(**artifact["model_config"]) for artifact in artifacts]
    if any(config != configs[0] for config in configs[1:]):
        raise RuntimeError("ensemble artifacts have different model configurations")
    models = []
    for artifact, config in zip(artifacts, configs):
        model = IndependentProposalRanker(config).to(device)
        model.load_state_dict(artifact["state_dict"], strict=True)
        model.eval()
        models.append(model)

    private = load_private_observation_table(args.private_observation_root)
    private_index = {token: row for row, token in enumerate(private.tokens)}
    with args.proposal_pickle.open("rb") as stream:
        proposal_bank = pickle.load(stream)

    # Read only identity fields before inference.  Official candidate labels
    # are deliberately loaded only after every selected index is fixed.
    with np.load(args.candidate_matrix, allow_pickle=False) as archive:
        tokens = archive["tokens"].astype(str).tolist()
        log_names = archive["log_names"].astype(str).tolist()
        candidate_count = int(archive["candidate_scores"].shape[1])
    if len(tokens) != 12_146 or len(set(tokens)) != 12_146:
        raise RuntimeError("Navtest requires exactly 12,146 unique scenes")
    if len(set(log_names)) != 136 or candidate_count != 64:
        raise RuntimeError("Navtest identity/candidate cardinality mismatch")
    if set(tokens) != set(private_index) or set(tokens) != set(proposal_bank):
        raise RuntimeError("private observation, proposal, and matrix token sets differ")

    utility_parts: List[np.ndarray] = []
    factor_parts: List[np.ndarray] = []
    selected_parts: List[np.ndarray] = []
    for start in range(0, len(tokens), args.batch_size):
        batch_tokens = tokens[start : start + args.batch_size]
        rows = torch.as_tensor(
            [private_index[token] for token in batch_tokens], dtype=torch.long
        )
        observation = private.observation_tokens[rows].to(
            device=device, dtype=torch.float32
        )
        mask = private.observation_valid_masks[rows].to(device=device)
        status = private.status_features[rows].to(device=device, dtype=torch.float32)
        proposals = torch.as_tensor(
            np.stack(
                [proposal_bank[token]["proposals"] for token in batch_tokens]
            ),
            device=device,
            dtype=torch.float32,
        )
        with torch.inference_mode():
            outputs = [
                model(
                    observation,
                    status,
                    proposals,
                    observation_valid_mask=mask,
                )
                for model in models
            ]
            utility = torch.stack(
                [_score_mode(output, mode) for output in outputs]
            ).mean(dim=0)
            factor_logits = torch.stack(
                [output["factor_logits"] for output in outputs]
            ).mean(dim=0)
            selected = utility.argmax(dim=1)
        utility_parts.append(utility.float().cpu().numpy())
        factor_parts.append(factor_logits.float().cpu().numpy())
        selected_parts.append(selected.cpu().numpy())
    predicted_scores = np.concatenate(utility_parts).astype(np.float32)
    predicted_factor_logits = np.concatenate(factor_parts).astype(np.float32)
    selected_indices = np.concatenate(selected_parts).astype(np.int16)

    # Offline evaluation join starts here, after selection is immutable.
    with np.load(args.candidate_matrix, allow_pickle=False) as archive:
        candidate_scores = np.asarray(archive["candidate_scores"], dtype=np.float32)
        candidate_factors = np.asarray(archive["candidate_factors"], dtype=np.float32)
        factor_names = archive["candidate_factor_names"].astype(str).tolist()
        base_scores = np.asarray(archive["predicted_scores"], dtype=np.float32)
    if candidate_scores.shape != (12_146, 64):
        raise RuntimeError("candidate score matrix shape mismatch")
    if candidate_factors.shape != (12_146, 64, 7) or factor_names[-1] != "score":
        raise RuntimeError("candidate factor matrix schema mismatch")
    rows = np.arange(len(tokens))
    base_indices = base_scores.argmax(axis=1)
    oracle_indices = candidate_scores.argmax(axis=1).astype(np.int16)
    selected_values = candidate_scores[rows, selected_indices]
    base_values = candidate_scores[rows, base_indices]
    oracle_values = candidate_scores[rows, oracle_indices]
    delta = selected_values - base_values
    physical_logs = [physical_log_name(value) for value in log_names]
    ci = _log_bootstrap_ci(
        delta, physical_logs, args.seed, args.bootstrap_replicates
    )

    public_frame = pd.read_csv(
        args.public_audit_dir / "per_scene_candidate_quality.csv"
    ).set_index("token")
    if set(public_frame.index.astype(str)) != set(tokens):
        raise RuntimeError("public audit token set differs")
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
        predicted_factor_logits=predicted_factor_logits,
        selected_indices=selected_indices,
        oracle_indices=oracle_indices,
        candidate_factors=candidate_factors,
        candidate_factor_names=np.asarray(factor_names),
    )
    output_proposals = {
        token: {
            "proposals": np.asarray(
                proposal_bank[token]["proposals"], dtype=np.float32
            ),
            "predicted_scores": predicted_scores[index],
        }
        for index, token in enumerate(tokens)
    }
    proposal_path = args.output_dir / "proposal_predictions.pkl"
    with proposal_path.open("wb") as stream:
        pickle.dump(output_proposals, stream, protocol=pickle.HIGHEST_PROTOCOL)

    public_summary = json.loads(
        (args.public_audit_dir / "summary.json").read_text()
    )
    metrics = dict(public_summary["metrics"])
    metrics.update(
        selected_pdms=float(selected_values.mean()),
        standard_selected_pdms=float(selected_values.mean()),
        selected_score_parity_abs=0.0,
        scorer_regret=float(frame.scorer_regret.mean()),
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
        "checkpoint": str(args.artifact.resolve()),
        "checkpoint_sha256": _sha256(args.artifact),
        "ensemble_artifacts": [
            {
                "path": str(path.resolve()),
                "sha256": _sha256(path),
            }
            for path in artifact_paths
        ],
        "ensemble_size": len(artifact_paths),
        "ensemble_method": "equal_mean_score",
        "agent_target": (
            "IndependentProposalRanker"
            if len(artifact_paths) == 1
            else "IndependentProposalRankerEnsemble"
        ),
        "score_mode": mode,
        "precision": 32,
        "proposal_predictions_path": str(proposal_path.resolve()),
        "proposal_predictions_sha256": _sha256(proposal_path),
        "private_observation_lineage": private.lineage,
        "candidate_matrix_source": str(args.candidate_matrix.resolve()),
        "candidate_matrix_source_sha256": _sha256(args.candidate_matrix),
        "inference_inputs_only": True,
        "future_target_present_during_inference": False,
        "official_score_input_present_during_inference": False,
        "official_candidate_matrix_joined_after_selection": True,
        "base_numeric_score_used_by_model": False,
        "base_rank_used_by_model": False,
        "drivor_checkpoint_or_representation_used": False,
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

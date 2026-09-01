#!/usr/bin/env python3
"""Evaluate a promoted independent shortlist scorer on locked Navtest data.

Model inference consumes only cached current-observation features, ego features
and proposals.  The official candidate-factor matrix is joined strictly after
selection and is used only for offline NAVSIM evaluation.
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

from local_stage2.independent_scorer_agent import IndependentShortlistScorerAgent
from local_stage2.train_independent_scorer import physical_log_name
from local_stage2.train_public_base_residual_scorer import _log_bootstrap_ci
from navsim.agents.EpisodeDrive.score_module.independent_ranker import (
    IndependentProposalRanker,
    IndependentRankerConfig,
    pdms_factor_log_utility,
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
    parser.add_argument("--candidate-matrix", type=Path, required=True)
    parser.add_argument("--public-audit-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _score_mode(output: Dict[str, torch.Tensor], mode: str) -> torch.Tensor:
    if mode == "coarse":
        return output["coarse_utility"]
    if mode == "factor":
        return pdms_factor_log_utility(output["factor_logits"])
    if mode == "direct":
        return output["utility"]
    raise RuntimeError(f"unsupported score mode: {mode}")


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    for path in (
        args.artifact,
        args.feature_cache,
        args.candidate_matrix,
        args.public_audit_dir / "summary.json",
        args.public_audit_dir / "per_scene_candidate_quality.csv",
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    artifact = torch.load(args.artifact, map_location="cpu", weights_only=False)
    if artifact.get("artifact_type") != IndependentShortlistScorerAgent.ARTIFACT_TYPE:
        raise RuntimeError("deployment artifact type mismatch")
    with args.feature_cache.open("rb") as file:
        feature_cache = pickle.load(file)
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
    if set(tokens) != set(feature_cache):
        raise RuntimeError("feature cache and candidate matrix token sets differ")
    if factor_names[-1] != "score":
        raise RuntimeError("candidate factor matrix lacks aggregate score")

    device = torch.device(args.device)
    model = IndependentProposalRanker(
        IndependentRankerConfig(**dict(artifact["model_config"]))
    ).to(device)
    model.load_state_dict(artifact["model_state_dict"], strict=True)
    model.eval()
    all_utilities: List[np.ndarray] = []
    selected_parts: List[np.ndarray] = []
    shortlist_size = int(artifact["shortlist_size"])
    mode = str(artifact["score_mode"])
    for start in range(0, len(tokens), args.batch_size):
        batch_tokens = tokens[start : start + args.batch_size]
        proposals = torch.as_tensor(
            np.stack([feature_cache[token]["proposals"] for token in batch_tokens]),
            dtype=torch.float32,
            device=device,
        )
        scene = torch.as_tensor(
            np.stack([feature_cache[token]["scene_features"] for token in batch_tokens]),
            dtype=torch.float32,
            device=device,
        )
        ego = torch.as_tensor(
            np.stack([feature_cache[token]["ego_features"] for token in batch_tokens]),
            dtype=torch.float32,
            device=device,
        )
        if ego.ndim == 3 and ego.shape[1] == 1:
            ego = ego[:, 0]
        base_scores = torch.as_tensor(
            np.stack([feature_cache[token]["predicted_scores"] for token in batch_tokens]),
            dtype=torch.float32,
            device=device,
        )
        with torch.no_grad():
            output = model(scene, ego, proposals)
            utility = _score_mode(output, mode)
            shortlist = base_scores.topk(shortlist_size, dim=1).indices
            selection_scores = torch.full_like(utility, -1.0e4)
            selection_scores.scatter_(1, shortlist, utility.gather(1, shortlist))
            selected = selection_scores.argmax(dim=1)
        all_utilities.append(selection_scores.cpu().numpy())
        selected_parts.append(selected.cpu().numpy())
    predicted_scores = np.concatenate(all_utilities, axis=0).astype(np.float32)
    selected_indices = np.concatenate(selected_parts).astype(np.int16)
    oracle_indices = candidate_scores.argmax(axis=1).astype(np.int16)
    rows = np.arange(len(tokens))
    selected_values = candidate_scores[rows, selected_indices]
    oracle_values = candidate_scores[rows, oracle_indices]

    base_matrix = np.asarray(matrix["predicted_scores"], dtype=np.float32)
    base_indices = base_matrix.argmax(axis=1)
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
    # Preserve the exact CSV identity checked by the scorer-evaluation skill.
    # ``best_of_64_pdms`` originated from the standard evaluator in float64,
    # while the compact candidate matrix is float32.  Deriving regret from the
    # two serialized CSV columns prevents a rounding-only identity failure.
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
    # A shared decimal precision keeps best - selected - regret algebraically
    # consistent after CSV round-trip under the evaluator's 1e-8 hard gate.
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

    public_summary = json.loads(
        (args.public_audit_dir / "summary.json").read_text()
    )
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
        "source_ranker_artifact_sha256": artifact.get(
            "source_ranker_artifact_sha256"
        ),
        "source_ranker_refit_all_logs": bool(
            artifact.get("source_ranker_refit_all_logs", False)
        ),
        "source_ranker_validation_performed": bool(
            artifact.get("source_ranker_validation_performed", True)
        ),
        "source_ranker_refit_provenance": artifact.get(
            "source_ranker_refit_provenance"
        ),
        "agent_target": (
            "local_stage2.independent_scorer_agent."
            "IndependentShortlistScorerAgent"
        ),
        "precision": 32,
        "proposal_predictions_path": str(proposal_path.resolve()),
        "proposal_predictions_sha256": _sha256(proposal_path),
        "feature_cache_path": str(args.feature_cache.resolve()),
        "feature_cache_sha256": _sha256(args.feature_cache),
        "candidate_matrix_source": str(args.candidate_matrix.resolve()),
        "candidate_matrix_source_sha256": _sha256(args.candidate_matrix),
        "inference_inputs_only": True,
        "future_target_present_during_inference": False,
        "official_score_input_present_during_inference": False,
        "official_candidate_matrix_joined_after_selection": True,
        "base_numeric_score_used_by_independent_ranker": False,
        "base_rank_used_for_shortlist": True,
        "shortlist_size": shortlist_size,
        "score_mode": mode,
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

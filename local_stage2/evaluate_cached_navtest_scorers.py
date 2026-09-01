"""Evaluate many residual scorers against one immutable FP32 Navtest cache.

The model-side pass consumes only current-observation features exported by the
released EpisodeDrive checkpoint.  PDM candidate matrices are joined strictly
after selection and are used only for offline evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

# Must be configured before torch creates a CuBLAS handle. The evaluator uses
# strict deterministic algorithms so repeated Navtest campaigns can be compared
# bit-for-bit; setting this after importing torch is too late.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

from local_stage2.public_base_residual_scorer import (
    PublicBaseResidualRanker,
    PublicBaseResidualScorerAgent,
    ResidualScorerConfig,
)
from local_stage2.temporal_consequence_scorer import (
    TemporalConsequenceConfig,
    TemporalConsequenceRanker,
    TemporalConsequenceScorerAgent,
)


EXPECTED_SCENES = 12_146
EXPECTED_LOGS = 136
EXPECTED_CANDIDATES = 64


def _sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while block := file.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _parse_artifacts(values: Sequence[str]) -> List[Tuple[str, Path]]:
    parsed: List[Tuple[str, Path]] = []
    names = set()
    for value in values:
        if "=" not in value:
            raise ValueError(f"Artifact must be NAME=PATH, got {value!r}")
        name, raw_path = value.split("=", 1)
        if not name or name in names:
            raise ValueError(f"Empty or duplicate artifact name: {name!r}")
        path = Path(raw_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        names.add(name)
        parsed.append((name, path))
    return parsed


def _load_artifact_manifest(path: Path) -> List[Tuple[str, Path]]:
    payload = json.loads(path.read_text())
    parsed = []
    for record in payload.get("artifacts", []):
        name = str(record["name"])
        artifact_path = Path(str(record["path"]))
        if not artifact_path.is_file():
            raise FileNotFoundError(artifact_path)
        parsed.append((name, artifact_path))
    return parsed


def _load_feature_cache(path: Path) -> Dict[str, Dict[str, np.ndarray]]:
    with path.open("rb") as file:
        cache = pickle.load(file)
    if not isinstance(cache, dict) or not cache:
        raise ValueError(f"Malformed feature cache: {path}")
    required = {
        "proposals",
        "predicted_scores",
        "base_factor_logits",
        "candidate_features",
    }
    for token, item in cache.items():
        missing = required - set(item)
        if missing:
            raise ValueError(f"Cache token {token} lacks {sorted(missing)}")
    return cache


def _load_candidate_matrix(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def _cluster_bootstrap(
    delta: np.ndarray,
    log_names: np.ndarray,
    *,
    iterations: int,
    seed: int,
) -> Dict[str, float]:
    unique_logs, inverse = np.unique(log_names.astype(str), return_inverse=True)
    counts = np.bincount(inverse).astype(np.float64)
    sums = np.bincount(inverse, weights=delta.astype(np.float64))
    generator = np.random.default_rng(seed)
    draws = np.empty(iterations, dtype=np.float64)
    for index in range(iterations):
        sampled = generator.integers(0, len(unique_logs), size=len(unique_logs))
        draws[index] = sums[sampled].sum() / counts[sampled].sum()
    low, high = np.quantile(draws, [0.025, 0.975])
    return {
        "mean": float(delta.mean()),
        "ci95_low": float(low),
        "ci95_high": float(high),
        "iterations": int(iterations),
        "seed": int(seed),
    }


def _batches(values: Sequence[str], size: int) -> Iterable[Sequence[str]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def _stack(cache, tokens: Sequence[str], key: str, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(
        np.stack([np.asarray(cache[token][key], dtype=np.float32) for token in tokens])
    ).to(device)


def _score_artifact(
    artifact_path: Path,
    cache: Dict[str, Dict[str, np.ndarray]],
    tokens: Sequence[str],
    *,
    device: torch.device,
    batch_size: int,
) -> Tuple[np.ndarray, np.ndarray, Mapping[str, object]]:
    payload = torch.load(artifact_path, map_location="cpu")
    artifact_type = payload.get("artifact_type")
    config_values = dict(payload["model_config"])
    if artifact_type == PublicBaseResidualScorerAgent.ARTIFACT_TYPE:
        if int(payload.get("artifact_version", 1)) < 4:
            config_values.setdefault("base_anchored_topk", False)
        model = PublicBaseResidualRanker(ResidualScorerConfig(**config_values))
        incompatible = model.load_state_dict(payload["model_state_dict"], strict=False)
        allowed_missing = {
            key
            for key in model.state_dict()
            if key.startswith(("safety_delta_head.", "relative_safety_head."))
        }
        if set(incompatible.missing_keys) - allowed_missing or incompatible.unexpected_keys:
            raise RuntimeError(
                f"Artifact state mismatch: missing={incompatible.missing_keys}, "
                f"unexpected={incompatible.unexpected_keys}"
            )
    elif artifact_type == TemporalConsequenceScorerAgent.ARTIFACT_TYPE:
        model = TemporalConsequenceRanker(
            TemporalConsequenceConfig(**config_values)
        )
        model.load_state_dict(payload["model_state_dict"], strict=True)
    else:
        raise ValueError(f"Unsupported scorer artifact type: {artifact_type!r}")
    model.to(device).eval()
    selected_parts: List[np.ndarray] = []
    score_parts: List[np.ndarray] = []
    with torch.inference_mode():
        for batch_tokens in _batches(tokens, batch_size):
            candidate_features = _stack(cache, batch_tokens, "candidate_features", device)
            proposals = _stack(cache, batch_tokens, "proposals", device)
            factor_logits = _stack(cache, batch_tokens, "base_factor_logits", device)
            base_scores = _stack(cache, batch_tokens, "predicted_scores", device)
            scene_features = (
                _stack(cache, batch_tokens, "scene_features", device)
                if "scene_features" in cache[batch_tokens[0]]
                else None
            )
            ego_features = (
                _stack(cache, batch_tokens, "ego_features", device)
                if "ego_features" in cache[batch_tokens[0]]
                else None
            )
            output = model(
                candidate_features,
                proposals,
                factor_logits,
                base_scores,
                scene_features,
                ego_features,
            )
            selection_scores = output["selection_scores"]
            selected_parts.append(selection_scores.argmax(dim=1).cpu().numpy())
            score_parts.append(selection_scores.float().cpu().numpy())
    return (
        np.concatenate(selected_parts).astype(np.int16),
        np.concatenate(score_parts).astype(np.float32),
        payload,
    )


def _evaluate(
    name: str,
    artifact_path: Path,
    selected: np.ndarray,
    model_scores: np.ndarray,
    matrix: Dict[str, np.ndarray],
    matrix_rows: np.ndarray,
    output_root: Path,
    feature_cache_path: Path,
    feature_manifest: Mapping[str, object],
    payload: Mapping[str, object],
    bootstrap_iterations: int,
    bootstrap_seed: int,
) -> Dict[str, object]:
    candidate_scores = matrix["candidate_scores"][matrix_rows].astype(np.float64)
    base_predictions = matrix["predicted_scores"][matrix_rows].astype(np.float64)
    log_names = matrix["log_names"][matrix_rows].astype(str)
    rows = np.arange(len(selected))
    base_selected = base_predictions.argmax(axis=1)
    selected_pdms = candidate_scores[rows, selected]
    base_pdms = candidate_scores[rows, base_selected]
    oracle_pdms = candidate_scores.max(axis=1)
    delta = selected_pdms - base_pdms

    method_dir = output_root / name
    method_dir.mkdir(parents=True, exist_ok=True)
    frame_values = {
        "token": matrix["tokens"][matrix_rows].astype(str),
        "log_name": log_names,
        "base_index": base_selected,
        "selected_index": selected,
        "oracle_index": candidate_scores.argmax(axis=1),
        "base_pdms": base_pdms,
        "selected_pdms": selected_pdms,
        "best_of_64_pdms": oracle_pdms,
        "pdms_delta": delta,
        "scorer_regret": oracle_pdms - selected_pdms,
    }
    metrics: Dict[str, float] = {
        "selected_pdms": float(selected_pdms.mean()),
        "public_base_selected_pdms": float(base_pdms.mean()),
        "best_of_64_pdms": float(oracle_pdms.mean()),
        "scorer_regret": float((oracle_pdms - selected_pdms).mean()),
        "public_base_scorer_regret": float((oracle_pdms - base_pdms).mean()),
        "pdms_delta": float(delta.mean()),
        "regret_reduction_fraction": float(
            1.0 - (oracle_pdms - selected_pdms).mean() / (oracle_pdms - base_pdms).mean()
        ),
        "switch_rate": float(np.mean(selected != base_selected)),
        "win_rate": float(np.mean(delta > 1e-8)),
        "loss_rate": float(np.mean(delta < -1e-8)),
        "tie_rate": float(np.mean(np.abs(delta) <= 1e-8)),
    }

    candidate_factors = matrix.get("candidate_factors")
    factor_names = matrix.get("candidate_factor_names")
    if candidate_factors is not None and factor_names is not None:
        factors = candidate_factors[matrix_rows].astype(np.float64)
        selected_factors = factors[rows, selected]
        base_factors = factors[rows, base_selected]
        for index, factor_name in enumerate(factor_names.astype(str)):
            frame_values[f"selected_{factor_name}"] = selected_factors[:, index]
            frame_values[f"base_{factor_name}"] = base_factors[:, index]
            metrics[f"selected_{factor_name}"] = float(selected_factors[:, index].mean())
            metrics[f"base_{factor_name}"] = float(base_factors[:, index].mean())

    pd.DataFrame(frame_values).to_csv(method_dir / "per_scene.csv", index=False)
    with (method_dir / "selections.npz").open("wb") as file:
        np.savez_compressed(
            file,
            tokens=np.asarray(frame_values["token"]),
            selected_indices=selected,
            model_scores=model_scores,
        )

    reference_parity = feature_manifest.get("reference_parity", {})
    acceptance = {
        "scene_count_12146": len(selected) == EXPECTED_SCENES,
        "log_count_136": len(set(log_names)) == EXPECTED_LOGS,
        "candidate_count_64": candidate_scores.shape[1] == EXPECTED_CANDIDATES,
        "zero_invalid": bool(np.isfinite(selected_pdms).all()),
        "public_reference_parity_1e_8": bool(reference_parity.get("passes_1e_8", False)),
        "candidate_proposal_lineage_match": bool(matrix.get("_lineage_ok", False)),
        "candidate_factor_matrix_present": candidate_factors is not None,
    }
    summary: Dict[str, object] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": name,
        "artifact_path": str(artifact_path.resolve()),
        "artifact_sha256": _sha256(artifact_path),
        "artifact_type": payload.get("artifact_type"),
        "artifact_version": payload.get("artifact_version"),
        "base_checkpoint_path": payload.get("base_checkpoint_path"),
        "base_checkpoint_sha256": payload.get("base_checkpoint_sha256"),
        "model_config": payload.get("model_config"),
        "feature_cache_path": str(feature_cache_path.resolve()),
        "feature_cache_sha256": feature_manifest.get("proposal_predictions_sha256"),
        "candidate_matrix_path": str(Path(matrix["_path"]).resolve()),
        "candidate_matrix_sha256": matrix["_sha256"],
        "candidate_matrix_summary_path": matrix.get("_summary_path"),
        "scene_count": len(selected),
        "log_count": len(set(log_names)),
        "candidate_count": int(candidate_scores.shape[1]),
        "invalid_scene_count": int((~np.isfinite(selected_pdms)).sum()),
        "inference_inputs_only": True,
        "official_pdm_used_only_after_selection": True,
        "metrics": metrics,
        "pdms_delta_log_cluster_bootstrap": _cluster_bootstrap(
            delta,
            log_names,
            iterations=bootstrap_iterations,
            seed=bootstrap_seed,
        ),
        "acceptance_gates": acceptance,
        "accepted": all(acceptance.values()),
    }
    _atomic_json(method_dir / "summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--feature-manifest", type=Path, required=True)
    parser.add_argument("--candidate-matrix", type=Path, required=True)
    parser.add_argument("--candidate-summary", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--artifact", action="append", default=[])
    parser.add_argument("--artifact-manifest", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--bootstrap-iterations", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260831)
    parser.add_argument("--allow-missing-factors", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifacts = _parse_artifacts(args.artifact)
    if args.artifact_manifest is not None:
        artifacts.extend(_load_artifact_manifest(args.artifact_manifest))
    names = [name for name, _path in artifacts]
    if not artifacts:
        raise ValueError("At least one --artifact or --artifact-manifest is required")
    if len(names) != len(set(names)):
        raise ValueError("Duplicate method names across artifact inputs")
    feature_manifest = json.loads(args.feature_manifest.read_text())
    if not feature_manifest.get("inference_inputs_only"):
        raise RuntimeError("Feature cache is not marked inference-input-only")
    parity = feature_manifest.get("reference_parity", {})
    if not parity.get("passes_1e_8"):
        raise RuntimeError(f"Feature cache failed public reference parity: {parity}")

    cache = _load_feature_cache(args.feature_cache)
    matrix = _load_candidate_matrix(args.candidate_matrix)
    matrix["_path"] = str(args.candidate_matrix)
    matrix["_sha256"] = _sha256(args.candidate_matrix)
    candidate_summary_path = args.candidate_summary or (
        args.candidate_matrix.parent / "summary.json"
    )
    candidate_summary = None
    if candidate_summary_path.is_file():
        candidate_summary = json.loads(candidate_summary_path.read_text())
        matrix["_summary_path"] = str(candidate_summary_path.resolve())
    reference_hash = feature_manifest.get("reference_predictions_sha256")
    matrix["_lineage_ok"] = bool(
        candidate_summary is not None
        and reference_hash
        and candidate_summary.get("proposal_predictions_sha256") == reference_hash
    )
    if not matrix["_lineage_ok"] and not args.allow_missing_factors:
        raise RuntimeError(
            "Candidate matrix does not trace to the feature cache's locked "
            "reference proposal SHA256"
        )
    if "candidate_factors" not in matrix and not args.allow_missing_factors:
        raise RuntimeError("Formal audit requires the all-candidate factor matrix")

    tokens = sorted(cache)
    matrix_index = {token: index for index, token in enumerate(matrix["tokens"].astype(str))}
    if set(tokens) != set(matrix_index):
        raise RuntimeError(
            f"Feature/matrix token mismatch: cache={len(tokens)}, matrix={len(matrix_index)}"
        )
    matrix_rows = np.asarray([matrix_index[token] for token in tokens])
    cache_base = np.stack(
        [np.asarray(cache[token]["predicted_scores"], dtype=np.float32) for token in tokens]
    )
    matrix_base = matrix["predicted_scores"][matrix_rows].astype(np.float32)
    base_max_abs = float(np.max(np.abs(cache_base.astype(np.float64) - matrix_base)))
    if base_max_abs > 1e-8:
        raise RuntimeError(f"Feature/matrix Base score mismatch: {base_max_abs}")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.use_deterministic_algorithms(True, warn_only=False)
    summaries = []
    for name, artifact_path in artifacts:
        selected, model_scores, payload = _score_artifact(
            artifact_path,
            cache,
            tokens,
            device=device,
            batch_size=args.batch_size,
        )
        summaries.append(
            _evaluate(
                name,
                artifact_path,
                selected,
                model_scores,
                matrix,
                matrix_rows,
                args.output_dir,
                args.feature_cache,
                feature_manifest,
                payload,
                args.bootstrap_iterations,
                args.bootstrap_seed,
            )
        )
    campaign = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "feature_cache": str(args.feature_cache.resolve()),
        "feature_cache_sha256": _sha256(args.feature_cache),
        "candidate_matrix": str(args.candidate_matrix.resolve()),
        "candidate_matrix_sha256": matrix["_sha256"],
        "candidate_proposal_lineage_match": matrix["_lineage_ok"],
        "base_score_max_abs": base_max_abs,
        "methods": summaries,
    }
    _atomic_json(args.output_dir / "campaign_summary.json", campaign)
    print(json.dumps(campaign, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

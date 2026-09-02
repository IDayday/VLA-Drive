#!/usr/bin/env python3
"""Sweep one fixed M0 conservative-reference artifact on its log-heldout fold.

The neural weights, stop epoch, candidate bank and validation logs are fixed
before this program runs.  Only deployment thresholds are varied.  The sweep
never reads Navtest and stores per-physical-log sufficient statistics so one
common policy can later be selected across all folds with a clustered
bootstrap.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch.utils.data import DataLoader

from local_stage2.train_independent_scorer import (
    ReplaySource,
    TARGET_TO_MODEL_FACTOR_ORDER,
    _atomic_json_dump,
    _sha256,
    load_replay_sources,
)
from local_stage2.train_m0_private_residual_scorer import (
    ResidualReplayDataset,
    load_replay_base_candidate_features,
    load_replay_base_factor_logits,
)
from navsim.agents.EpisodeDrive.score_module.independent_ranker import (
    FACTOR_KEYS,
    IndependentRankerConfig,
    conservative_reference_selection_scores,
)
from navsim.agents.EpisodeDrive.score_module.m0_private_residual_ranker import (
    M0PrivateResidualConfig,
    M0PrivateResidualRanker,
)


REFERENCE_POLICY_FIELDS = (
    "gain_quantile_index",
    "minimum_lcb_gain",
    "maximum_safety_worse_probability",
    "minimum_safe_improvement_probability",
)


def conservative_reference_policy_grid() -> List[Dict[str, object]]:
    """Return the Wave-12 deployment grid in deterministic order.

    q10 and q50 are both included because earlier fixed-split experiments
    showed that q10 was often too conservative while q50 switched too freely.
    q90 is deliberately excluded: it is not a conservative gain estimate.
    """

    rows: List[Dict[str, object]] = []
    for values in itertools.product(
        (0, 1),
        (-0.01, 0.0, 0.0025, 0.005, 0.01, 0.02),
        (0.02, 0.05, 0.10, 0.20),
        (0.50, 0.70, 0.80, 0.90),
    ):
        policy = dict(zip(REFERENCE_POLICY_FIELDS, values))
        policy["policy_id"] = len(rows)
        rows.append(policy)
    return rows


def _policy_without_id(policy: Mapping[str, object]) -> Dict[str, object]:
    return {key: policy[key] for key in REFERENCE_POLICY_FIELDS}


@torch.inference_mode()
def collect_reference_policy_tensors(
    model: M0PrivateResidualRanker,
    data,
    base_factor_logits: torch.Tensor,
    m0_candidate_features: Optional[torch.Tensor],
    indices: Sequence[int],
    device: torch.device,
    batch_size: int,
) -> Dict[str, torch.Tensor]:
    """Collect frozen predictions needed by every deployment policy."""

    if not model.residual_config.conservative_reference:
        raise ValueError("artifact does not contain a conservative-reference head")
    model.eval()
    names = (
        "gain_quantiles",
        "safety_worse_logits",
        "safe_improvement_logit",
        "shortlist_mask",
        "base_scores",
        "target_factors",
    )
    parts: Dict[str, List[torch.Tensor]] = {name: [] for name in names}
    loader = DataLoader(
        ResidualReplayDataset(
            data,
            base_factor_logits,
            indices,
            m0_candidate_features=m0_candidate_features,
            include_m0_context=model.residual_config.m0_context_fusion,
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    for batch in loader:
        (
            proposals,
            observation,
            observation_valid_mask,
            status,
            base_scores,
            factor_logits,
            target_factors,
            _source_indices,
        ) = batch[:8]
        cursor = 8
        candidate_features = None
        if model.residual_config.m0_candidate_fusion:
            candidate_features = batch[cursor].to(
                device, non_blocking=True
            ).float()
            cursor += 1
        m0_scene_features = None
        m0_ego_features = None
        if model.residual_config.m0_context_fusion:
            m0_scene_features = batch[cursor].to(
                device, non_blocking=True
            ).float()
            m0_ego_features = batch[cursor + 1].to(
                device, non_blocking=True
            ).float()
        output = model(
            observation.to(device, non_blocking=True).float(),
            status.to(device, non_blocking=True).float(),
            proposals.to(device, non_blocking=True),
            factor_logits.to(device, non_blocking=True),
            base_scores.to(device, non_blocking=True),
            observation_valid_mask=observation_valid_mask.to(
                device, non_blocking=True
            ),
            m0_scene_features=m0_scene_features,
            m0_ego_features=m0_ego_features,
            m0_candidate_features=candidate_features,
        )
        for name in names[:4]:
            parts[name].append(output[name].detach().float().cpu())
        parts["base_scores"].append(base_scores.float())
        parts["target_factors"].append(target_factors.float())
    return {name: torch.cat(values) for name, values in parts.items()}


def evaluate_reference_policy(
    tensors: Mapping[str, torch.Tensor],
    physical_logs: Sequence[str],
    policy: Mapping[str, object],
    *,
    device: torch.device,
) -> Dict[str, object]:
    """Evaluate one threshold tuple and retain clustered sufficient stats."""

    base_scores = tensors["base_scores"].to(device)
    target_factors = tensors["target_factors"].to(device)
    if len(physical_logs) != int(base_scores.shape[0]):
        raise ValueError("physical-log metadata does not align with tensors")
    base_indices = base_scores.argmax(dim=1)
    selection_scores = conservative_reference_selection_scores(
        tensors["gain_quantiles"].to(device),
        tensors["safety_worse_logits"].to(device),
        tensors["safe_improvement_logit"].to(device),
        base_indices,
        gain_quantile_index=int(policy["gain_quantile_index"]),
        minimum_lcb_gain=float(policy["minimum_lcb_gain"]),
        maximum_safety_worse_probability=float(
            policy["maximum_safety_worse_probability"]
        ),
        minimum_safe_improvement_probability=float(
            policy["minimum_safe_improvement_probability"]
        ),
        allowed_candidate_mask=tensors["shortlist_mask"].to(device).bool(),
    )
    selected = selection_scores.argmax(dim=1)
    rows = torch.arange(len(base_scores), device=device)
    target_scores = target_factors[..., -1]
    selected_scores = target_scores[rows, selected]
    base_values = target_scores[rows, base_indices]
    delta = selected_scores - base_values
    target_six = target_factors[..., list(TARGET_TO_MODEL_FACTOR_ORDER)]
    selected_six = target_six[rows, selected]
    base_six = target_six[rows, base_indices]
    factor_delta = selected_six - base_six

    delta_cpu = delta.float().cpu().numpy().astype(np.float64, copy=False)
    factor_delta_cpu = (
        factor_delta.float().cpu().numpy().astype(np.float64, copy=False)
    )
    grouped: Dict[str, Dict[str, object]] = {}
    for index, log_name in enumerate(physical_logs):
        name = str(log_name)
        row = grouped.setdefault(
            name,
            {
                "scene_count": 0,
                "delta_sum": 0.0,
                "factor_delta_sum": np.zeros(len(FACTOR_KEYS), dtype=np.float64),
            },
        )
        row["scene_count"] = int(row["scene_count"]) + 1
        row["delta_sum"] = float(row["delta_sum"]) + float(delta_cpu[index])
        row["factor_delta_sum"] += factor_delta_cpu[index]
    serialized_logs = {
        name: {
            "scene_count": int(row["scene_count"]),
            "delta_sum": float(row["delta_sum"]),
            "factor_delta_sum": [
                float(value) for value in row["factor_delta_sum"]
            ],
        }
        for name, row in sorted(grouped.items())
    }
    selected_factor_mean = selected_six.float().mean(dim=0).cpu()
    base_factor_mean = base_six.float().mean(dim=0).cpu()
    return {
        "policy_id": int(policy["policy_id"]),
        "policy": _policy_without_id(policy),
        "scene_count": int(len(base_scores)),
        "physical_log_count": len(grouped),
        "selected_pdms": float(selected_scores.mean()),
        "base_selected_pdms": float(base_values.mean()),
        "selected_delta": float(delta.mean()),
        "switch_rate": float((selected != base_indices).float().mean()),
        "wins": int((delta > 1.0e-9).sum()),
        "losses": int((delta < -1.0e-9).sum()),
        "ties": int((delta.abs() <= 1.0e-9).sum()),
        "selected_factors": {
            key: float(selected_factor_mean[index])
            for index, key in enumerate(FACTOR_KEYS)
        },
        "base_selected_factors": {
            key: float(base_factor_mean[index])
            for index, key in enumerate(FACTOR_KEYS)
        },
        "factor_delta": {
            key: float(selected_factor_mean[index] - base_factor_mean[index])
            for index, key in enumerate(FACTOR_KEYS)
        },
        "per_log_sufficient_statistics": serialized_logs,
    }


def _default_policy(config: M0PrivateResidualConfig) -> Dict[str, object]:
    return {
        "gain_quantile_index": config.gain_quantile_index,
        "minimum_lcb_gain": config.minimum_lcb_gain,
        "maximum_safety_worse_probability": (
            config.maximum_safety_worse_probability
        ),
        "minimum_safe_improvement_probability": (
            config.minimum_safe_improvement_probability
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument(
        "--source",
        action="append",
        nargs=3,
        metavar=("NAME", "FEATURE_ROOT", "LABEL_ROOT"),
        required=True,
    )
    parser.add_argument("--private-observation-root", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in (args.artifact, args.private_observation_root, args.split_manifest):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.manual_seed(20260902)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(20260902)
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    torch.use_deterministic_algorithms(True)

    artifact = torch.load(args.artifact, map_location="cpu", weights_only=False)
    if artifact.get("architecture") != "M0PrivateResidualRanker":
        raise RuntimeError("artifact architecture mismatch")
    if not bool(artifact["residual_config"].get("conservative_reference")):
        raise RuntimeError("artifact is not a conservative-reference model")
    sources = [
        ReplaySource(name, Path(feature_root), Path(label_root))
        for name, feature_root, label_root in args.source
    ]
    selection_source = str(artifact["checkpoint_selection_source"])
    data, source_lineage = load_replay_sources(
        sources,
        private_observation_root=args.private_observation_root,
        retain_m0_context=bool(
            artifact["residual_config"].get("m0_context_fusion", False)
        ),
    )
    factor_tokens, base_factor_logits = load_replay_base_factor_logits(sources)
    if factor_tokens != data.tokens:
        raise RuntimeError("Base factor logits do not match replay token order")
    m0_candidate_features = None
    if bool(artifact["residual_config"].get("m0_candidate_fusion", False)):
        candidate_tokens, m0_candidate_features = (
            load_replay_base_candidate_features(sources)
        )
        if candidate_tokens != data.tokens:
            raise RuntimeError("M0 candidate features do not match replay token order")
    split = json.loads(args.split_manifest.read_text())
    validation_logs = {str(value) for value in split["validation_physical_logs"]}
    validation_indices = [
        index
        for index, (log_name, source_name) in enumerate(
            zip(data.physical_logs, data.source_names)
        )
        if str(log_name) in validation_logs and str(source_name) == selection_source
    ]
    if not validation_indices:
        raise RuntimeError("selection source has no validation scenes")
    observed_logs = {str(data.physical_logs[index]) for index in validation_indices}
    if observed_logs != validation_logs:
        raise RuntimeError("validation log coverage differs from split manifest")

    private_config = IndependentRankerConfig(**dict(artifact["private_config"]))
    residual_config = M0PrivateResidualConfig(**dict(artifact["residual_config"]))
    device = torch.device(args.device)
    model = M0PrivateResidualRanker(private_config, residual_config).to(device)
    model.load_state_dict(artifact["state_dict"], strict=True)
    tensors = collect_reference_policy_tensors(
        model,
        data,
        base_factor_logits,
        m0_candidate_features,
        validation_indices,
        device,
        args.eval_batch_size,
    )
    physical_logs = [data.physical_logs[index] for index in validation_indices]
    policies = [
        evaluate_reference_policy(tensors, physical_logs, policy, device=device)
        for policy in conservative_reference_policy_grid()
    ]

    default = _default_policy(residual_config)
    matches = [row for row in policies if row["policy"] == default]
    if len(matches) != 1:
        raise RuntimeError("deployment grid does not contain the artifact policy")
    artifact_metrics = artifact.get("validation_by_source", {}).get(
        selection_source, artifact.get("validation")
    )
    if not isinstance(artifact_metrics, Mapping):
        raise RuntimeError("artifact lacks validation metrics")
    default_delta_error = abs(
        float(matches[0]["selected_delta"])
        - float(artifact_metrics["selected_delta"])
    )
    default_pdms_error = abs(
        float(matches[0]["selected_pdms"])
        - float(artifact_metrics["selected_pdms"])
    )
    if max(default_delta_error, default_pdms_error) > 1.0e-6:
        raise RuntimeError("frozen-policy replay does not reproduce artifact metrics")

    payload = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "artifact": str(args.artifact.resolve()),
        "artifact_sha256": _sha256(args.artifact),
        "artifact_epoch": int(artifact["epoch"]),
        "selection_source": selection_source,
        "fold_index": int(split.get("fold_index", -1)),
        "num_folds": int(split.get("num_folds", 1)),
        "split_manifest": str(args.split_manifest.resolve()),
        "split_manifest_sha256": _sha256(args.split_manifest),
        "validation_scene_count": len(validation_indices),
        "validation_physical_logs": sorted(observed_logs),
        "source_lineage": source_lineage,
        "policy_grid_size": len(policies),
        "policy_grid": policies,
        "artifact_policy": default,
        "artifact_policy_replay_pdms_error": default_pdms_error,
        "artifact_policy_replay_delta_error": default_delta_error,
        "navtest_used_for_policy_sweep": False,
        "future_or_evaluator_input_at_inference": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json_dump(payload, args.output)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "scene_count": len(validation_indices),
                "physical_log_count": len(observed_logs),
                "policy_grid_size": len(policies),
                "artifact_policy_replay_pdms_error": default_pdms_error,
                "artifact_policy_replay_delta_error": default_delta_error,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

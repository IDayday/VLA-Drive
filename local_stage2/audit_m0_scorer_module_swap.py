"""Attribute M0 scorer differences to context, attention, or factor heads.

The audit keeps one proposal bank fixed and evaluates a predeclared Cartesian
product of:

* cached current-observation ``scene_features`` / ``ego_features``;
* ``pos_embed`` + ``scorer_attention`` checkpoint weights; and
* the six EpisodeDrive factor heads.

PDM targets are never passed to a model component.  They are joined only after
each combination has selected an index, so this script is an offline attribution
audit rather than a deployable scorer or a Navtest hyper-parameter search.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np
import torch
from omegaconf import OmegaConf

from navsim.agents.EpisodeDrive.action_decoder import ActionDecoder


FACTOR_KEYS: Tuple[str, ...] = (
    "no_at_fault_collisions",
    "drivable_area_compliance",
    "driving_direction_compliance",
    "time_to_collision_within_bound",
    "ego_progress",
    "comfort",
)
TARGET_FACTOR_KEYS: Tuple[str, ...] = (
    "no_at_fault_collisions",
    "drivable_area_compliance",
    "ego_progress",
    "time_to_collision_within_bound",
    "comfort",
    "driving_direction_compliance",
    "score",
)
_SEGMENT_SUFFIX = re.compile(r"_\d{5}_\d{5}$")
_ACTION_PREFIXES = ("agent.action_head.", "action_head.")
_COMPONENT_PREFIXES = {
    "q_former": ("q_former.", "scene_embeds"),
    "ego_encoder": ("hist_encoding.",),
    "trajectory_embedding": ("pos_embed.",),
    "scorer_attention": ("scorer_attention.",),
    "factor_heads": ("scorer.",),
}


@dataclass(frozen=True)
class ProposalAuditData:
    tokens: list[str]
    segment_logs: list[str]
    physical_logs: list[str]
    proposals: torch.Tensor
    cached_base_scores: torch.Tensor
    target_factors: torch.Tensor


@dataclass(frozen=True)
class ContextTensors:
    scene_features: torch.Tensor
    ego_features: torch.Tensor


def physical_log_name(log_name: str) -> str:
    """Map a NAVSIM segment directory to its physical log."""

    return _SEGMENT_SUFFIX.sub("", str(log_name))


def _sha256(path: Path, block_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json_dump(payload: Mapping[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_text_dump(value: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(value)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _parse_named_paths(values: Sequence[str], option: str) -> Dict[str, Path]:
    parsed: Dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"{option} entries must use NAME=PATH: {value}")
        name, raw_path = value.split("=", 1)
        if not name or name in parsed:
            raise ValueError(f"Invalid or duplicate {option} name: {name!r}")
        path = Path(raw_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        parsed[name] = path
    if not parsed:
        raise ValueError(f"At least one {option} entry is required")
    return parsed


def _allowed_physical_logs(
    split_manifest: Path, split: str
) -> set[str] | None:
    if split == "all":
        return None
    payload = json.loads(split_manifest.read_text())
    key = f"{split}_physical_logs"
    if key not in payload:
        raise KeyError(f"{split_manifest} lacks {key}")
    return {str(value) for value in payload[key]}


def _chunk_paths(root: Path) -> list[Path]:
    paths = sorted(root.glob("*_shard_*-of-*/chunk_*.pt"))
    if not paths:
        raise RuntimeError(f"No cache chunks found under {root}")
    return paths


def load_proposal_audit_data(
    source_root: Path,
    label_root: Path,
    allowed_logs: set[str] | None,
    *,
    max_scenes: int = 0,
) -> ProposalAuditData:
    """Load fixed proposals and offline labels without exposing labels to models."""

    tokens: list[str] = []
    segment_logs: list[str] = []
    proposal_parts: list[torch.Tensor] = []
    base_score_parts: list[torch.Tensor] = []
    target_parts: list[torch.Tensor] = []
    for source_path in _chunk_paths(source_root):
        label_path = label_root / source_path.relative_to(source_root)
        if not label_path.is_file():
            raise FileNotFoundError(label_path)
        source = torch.load(source_path, map_location="cpu", weights_only=False)
        labels = torch.load(label_path, map_location="cpu", weights_only=False)
        source_tokens = [str(value) for value in source["tokens"]]
        if source_tokens != [str(value) for value in labels["tokens"]]:
            raise RuntimeError(f"Source/label token mismatch in {source_path}")
        if tuple(source["factor_keys"]) != FACTOR_KEYS:
            raise RuntimeError(f"Unexpected scorer factor schema in {source_path}")
        if tuple(labels["target_factor_keys"]) != TARGET_FACTOR_KEYS:
            raise RuntimeError(f"Unexpected target schema in {label_path}")
        logs = [str(value) for value in source["log_names"]]
        valid = labels["valid_mask"].bool()
        keep = torch.tensor(
            [
                bool(valid[index])
                and (
                    allowed_logs is None
                    or physical_log_name(log_name) in allowed_logs
                )
                for index, log_name in enumerate(logs)
            ],
            dtype=torch.bool,
        )
        if not bool(keep.any()):
            continue
        indices = keep.nonzero(as_tuple=False).flatten().tolist()
        tokens.extend(source_tokens[index] for index in indices)
        segment_logs.extend(logs[index] for index in indices)
        proposal_parts.append(source["proposals"][keep].float())
        base_score_parts.append(source["base_scores"][keep].float())
        target_parts.append(labels["target_factors"][keep].float())
        if max_scenes > 0 and len(tokens) >= max_scenes:
            break
    if not tokens:
        raise RuntimeError("No valid proposal scenes matched the requested split")
    proposals = torch.cat(proposal_parts)[: max_scenes or None]
    cached_base_scores = torch.cat(base_score_parts)[: max_scenes or None]
    target_factors = torch.cat(target_parts)[: max_scenes or None]
    tokens = tokens[: max_scenes or None]
    segment_logs = segment_logs[: max_scenes or None]
    if len(set(tokens)) != len(tokens):
        raise RuntimeError("Duplicate proposal tokens")
    expected = len(tokens)
    if proposals.shape != (expected, 64, 8, 3):
        raise RuntimeError(f"Unexpected proposal shape {tuple(proposals.shape)}")
    if cached_base_scores.shape != (expected, 64):
        raise RuntimeError(
            f"Unexpected cached score shape {tuple(cached_base_scores.shape)}"
        )
    if target_factors.shape != (expected, 64, 7):
        raise RuntimeError(f"Unexpected target shape {tuple(target_factors.shape)}")
    return ProposalAuditData(
        tokens=tokens,
        segment_logs=segment_logs,
        physical_logs=[physical_log_name(value) for value in segment_logs],
        proposals=proposals,
        cached_base_scores=cached_base_scores,
        target_factors=target_factors,
    )


def load_navtest_proposal_audit_data(
    proposal_pickle: Path,
    candidate_npz: Path,
    *,
    max_scenes: int = 0,
) -> ProposalAuditData:
    """Load a locked Navtest proposal cache and its offline PDM matrix.

    The proposal pickle contains inference-time model outputs only.  The NPZ
    contains the offline PDM factors and is used only after a score combination
    has selected an index.  Rows are aligned by scene token rather than by
    pickle insertion order.
    """

    with proposal_pickle.open("rb") as stream:
        predictions = pickle.load(stream)
    if not isinstance(predictions, Mapping):
        raise TypeError(f"Expected token mapping in {proposal_pickle}")

    with np.load(candidate_npz, allow_pickle=False) as matrix:
        required = {
            "tokens",
            "log_names",
            "candidate_scores",
            "predicted_scores",
            "candidate_factors",
            "candidate_factor_names",
        }
        missing_keys = sorted(required.difference(matrix.files))
        if missing_keys:
            raise KeyError(f"{candidate_npz} lacks {missing_keys}")
        factor_names = tuple(str(value) for value in matrix["candidate_factor_names"])
        if factor_names != TARGET_FACTOR_KEYS:
            raise RuntimeError(
                f"Unexpected target schema in {candidate_npz}: {factor_names}"
            )
        all_tokens = [str(value) for value in matrix["tokens"]]
        all_logs = [str(value) for value in matrix["log_names"]]
        stop = min(len(all_tokens), max_scenes) if max_scenes > 0 else len(all_tokens)
        tokens = all_tokens[:stop]
        segment_logs = all_logs[:stop]
        target_factors = torch.from_numpy(
            np.asarray(matrix["candidate_factors"][:stop]).copy()
        ).float()
        matrix_scores = torch.from_numpy(
            np.asarray(matrix["candidate_scores"][:stop]).copy()
        ).float()
        matrix_predicted_scores = torch.from_numpy(
            np.asarray(matrix["predicted_scores"][:stop]).copy()
        ).float()

    if len(set(tokens)) != len(tokens):
        raise RuntimeError(f"Duplicate Navtest tokens in {candidate_npz}")
    missing_tokens = [token for token in tokens if token not in predictions]
    if missing_tokens:
        raise RuntimeError(
            f"Proposal cache lacks {len(missing_tokens)} requested tokens"
        )

    proposals = torch.from_numpy(
        np.stack(
            [np.asarray(predictions[token]["proposals"]) for token in tokens]
        ).copy()
    ).float()
    cached_base_scores = torch.from_numpy(
        np.stack(
            [np.asarray(predictions[token]["predicted_scores"]) for token in tokens]
        ).copy()
    ).float()
    expected = len(tokens)
    if proposals.shape != (expected, 64, 8, 3):
        raise RuntimeError(f"Unexpected proposal shape {tuple(proposals.shape)}")
    if cached_base_scores.shape != (expected, 64):
        raise RuntimeError(
            f"Unexpected cached score shape {tuple(cached_base_scores.shape)}"
        )
    if target_factors.shape != (expected, 64, 7):
        raise RuntimeError(f"Unexpected target shape {tuple(target_factors.shape)}")
    target_score_error = float(
        (matrix_scores - target_factors[..., -1]).abs().max()
    )
    if target_score_error > 1e-6:
        raise RuntimeError(
            "candidate_scores do not match the score column of candidate_factors: "
            f"max_abs={target_score_error}"
        )
    predicted_score_error = float(
        (matrix_predicted_scores - cached_base_scores).abs().max()
    )
    if predicted_score_error > 1e-6:
        raise RuntimeError(
            "Proposal cache does not match the locked candidate matrix: "
            f"max_abs={predicted_score_error}"
        )
    return ProposalAuditData(
        tokens=tokens,
        segment_logs=segment_logs,
        physical_logs=[physical_log_name(value) for value in segment_logs],
        proposals=proposals,
        cached_base_scores=cached_base_scores,
        target_factors=target_factors,
    )


def load_aligned_context(root: Path, requested_tokens: Sequence[str]) -> ContextTensors:
    """Load only candidate-independent context and align it to proposal order."""

    requested = {str(token): index for index, token in enumerate(requested_tokens)}
    if len(requested) != len(requested_tokens):
        raise RuntimeError("Requested context tokens are not unique")
    found_tokens: list[str] = []
    scene_parts: list[torch.Tensor] = []
    ego_parts: list[torch.Tensor] = []
    for path in _chunk_paths(root):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        chunk_tokens = [str(value) for value in payload["tokens"]]
        keep_indices = [
            index for index, token in enumerate(chunk_tokens) if token in requested
        ]
        if not keep_indices:
            continue
        index_tensor = torch.tensor(keep_indices, dtype=torch.long)
        found_tokens.extend(chunk_tokens[index] for index in keep_indices)
        scene_parts.append(payload["scene_features"].index_select(0, index_tensor))
        ego_parts.append(payload["ego_features"].index_select(0, index_tensor))
        if len(found_tokens) == len(requested_tokens):
            break
    if len(set(found_tokens)) != len(found_tokens):
        raise RuntimeError(f"Duplicate context tokens under {root}")
    missing = sorted(set(requested).difference(found_tokens))
    extra = sorted(set(found_tokens).difference(requested))
    if missing or extra:
        raise RuntimeError(
            f"Context token mismatch for {root}: missing={len(missing)}, extra={len(extra)}"
        )
    scene = torch.cat(scene_parts)
    ego = torch.cat(ego_parts)
    row_for_token = {token: index for index, token in enumerate(found_tokens)}
    order = torch.tensor(
        [row_for_token[str(token)] for token in requested_tokens], dtype=torch.long
    )
    scene = scene.index_select(0, order)
    ego = ego.index_select(0, order)
    if scene.shape[:2] != (len(requested_tokens), 16) or scene.shape[-1] != 256:
        raise RuntimeError(f"Unexpected scene context shape {tuple(scene.shape)}")
    if ego.shape != (len(requested_tokens), 1, 256):
        raise RuntimeError(f"Unexpected ego context shape {tuple(ego.shape)}")
    return ContextTensors(scene_features=scene, ego_features=ego)


def load_aligned_pickle_context(
    path: Path, requested_tokens: Sequence[str]
) -> ContextTensors:
    """Load current-observation context fields from a Navtest feature pickle."""

    with path.open("rb") as stream:
        payload = pickle.load(stream)
    if not isinstance(payload, Mapping):
        raise TypeError(f"Expected token mapping in {path}")
    missing = [str(token) for token in requested_tokens if str(token) not in payload]
    if missing:
        raise RuntimeError(f"Context cache {path} lacks {len(missing)} tokens")
    scene = torch.from_numpy(
        np.stack(
            [np.asarray(payload[str(token)]["scene_features"]) for token in requested_tokens]
        ).copy()
    ).float()
    ego = torch.from_numpy(
        np.stack(
            [np.asarray(payload[str(token)]["ego_features"]) for token in requested_tokens]
        ).copy()
    ).float()
    if scene.shape != (len(requested_tokens), 16, 256):
        raise RuntimeError(f"Unexpected scene context shape {tuple(scene.shape)}")
    if ego.shape != (len(requested_tokens), 1, 256):
        raise RuntimeError(f"Unexpected ego context shape {tuple(ego.shape)}")
    return ContextTensors(scene_features=scene, ego_features=ego)


def load_context(path: Path, requested_tokens: Sequence[str]) -> ContextTensors:
    """Dispatch between chunked validation caches and Navtest pickle caches."""

    if path.is_file():
        return load_aligned_pickle_context(path, requested_tokens)
    return load_aligned_context(path, requested_tokens)


def _torch_load_checkpoint(path: Path) -> Mapping[str, object]:
    try:
        return torch.load(
            path, map_location="cpu", weights_only=False, mmap=True
        )
    except (TypeError, RuntimeError):
        return torch.load(path, map_location="cpu", weights_only=False)


def action_head_state(path: Path) -> Dict[str, torch.Tensor]:
    payload = _torch_load_checkpoint(path)
    raw = payload.get("state_dict", payload)
    if not isinstance(raw, Mapping):
        raise TypeError(f"Checkpoint {path} has no state dictionary")
    for prefix in _ACTION_PREFIXES:
        state = {
            str(key)[len(prefix) :]: value
            for key, value in raw.items()
            if str(key).startswith(prefix)
        }
        if state:
            return state
    raise RuntimeError(f"Checkpoint {path} has no EpisodeDrive action-head keys")


def load_action_decoder(
    checkpoint: Path, config_path: Path, device: torch.device
) -> ActionDecoder:
    config = OmegaConf.load(config_path)
    model = ActionDecoder(config.action_head_config)
    state = action_head_state(checkpoint)
    model.load_state_dict(state, strict=True)
    model.to(device=device, dtype=torch.float32).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def stack_factor_logits(prediction: Mapping[str, torch.Tensor]) -> torch.Tensor:
    return torch.stack([prediction[key] for key in FACTOR_KEYS], dim=-1)


def aggregate_factor_logits(factor_logits: torch.Tensor) -> torch.Tensor:
    """Apply the released M0 factor aggregation in log space."""

    if factor_logits.shape[-1] != len(FACTOR_KEYS):
        raise ValueError("factor_logits must end with the six M0 factors")
    probability = factor_logits.sigmoid().clamp(1e-7, 1.0 - 1e-7)
    return (
        probability[..., 0].log()
        + probability[..., 1].log()
        + (
            5.0 * probability[..., 3]
            + 5.0 * probability[..., 4]
            + 2.0 * probability[..., 5]
        ).log()
    )


@torch.inference_mode()
def score_module_combinations(
    proposals: torch.Tensor,
    contexts: Mapping[str, ContextTensors],
    models: Mapping[str, ActionDecoder],
    *,
    device: torch.device,
    batch_size: int,
) -> Dict[str, torch.Tensor]:
    """Score all context x attention x head combinations.

    Context is candidate independent.  For each context/attention pair the
    trajectory-conditioned hidden state is computed once, then reused by every
    factor-head checkpoint.
    """

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    output_parts: Dict[str, list[torch.Tensor]] = {
        f"context={context}|attention={attention}|head={head}": []
        for context in contexts
        for attention in models
        for head in models
    }
    scene_count = len(proposals)
    for context_name, context in contexts.items():
        if len(context.scene_features) != scene_count:
            raise RuntimeError(f"Context length mismatch for {context_name}")
        for attention_name, attention_model in models.items():
            for start in range(0, scene_count, batch_size):
                stop = min(start + batch_size, scene_count)
                proposal_batch = proposals[start:stop].to(
                    device=device, dtype=torch.float32, non_blocking=True
                )
                scene_batch = context.scene_features[start:stop].to(
                    device=device, dtype=torch.float32, non_blocking=True
                )
                ego_batch = context.ego_features[start:stop].to(
                    device=device, dtype=torch.float32, non_blocking=True
                )
                batch, candidates = proposal_batch.shape[:2]
                embedded = attention_model.pos_embed(
                    proposal_batch.reshape(batch, candidates, -1).detach()
                )
                candidate_features = attention_model.scorer_attention(
                    embedded, scene_batch
                ) + ego_batch
                for head_name, head_model in models.items():
                    prediction = head_model.scorer(
                        proposal_batch, candidate_features
                    )[0]
                    key = (
                        f"context={context_name}|attention={attention_name}|"
                        f"head={head_name}"
                    )
                    output_parts[key].append(
                        aggregate_factor_logits(
                            stack_factor_logits(prediction)
                        ).float().cpu()
                    )
    return {key: torch.cat(parts) for key, parts in output_parts.items()}


def _log_bootstrap_ci(
    values: np.ndarray,
    physical_logs: Sequence[str],
    *,
    seed: int,
    replicates: int,
) -> tuple[float, float]:
    grouped: Dict[str, list[float]] = {}
    for value, log_name in zip(values, physical_logs):
        grouped.setdefault(str(log_name), []).append(float(value))
    names = sorted(grouped)
    if not names or replicates <= 0:
        return float("nan"), float("nan")
    sums = np.asarray([np.sum(grouped[name]) for name in names])
    counts = np.asarray([len(grouped[name]) for name in names])
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, len(names), size=(replicates, len(names)))
    estimates = sums[samples].sum(axis=1) / counts[samples].sum(axis=1)
    return float(np.quantile(estimates, 0.025)), float(
        np.quantile(estimates, 0.975)
    )


def _pairwise_accuracy(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    minimum_delta: float,
    batch_size: int = 512,
) -> tuple[float, int]:
    candidates = prediction.shape[1]
    left, right = torch.triu_indices(candidates, candidates, offset=1)
    correct = 0
    total = 0
    for start in range(0, len(prediction), batch_size):
        stop = min(start + batch_size, len(prediction))
        target_delta = target[start:stop, left] - target[start:stop, right]
        prediction_delta = (
            prediction[start:stop, left] - prediction[start:stop, right]
        )
        valid = target_delta.abs() >= minimum_delta
        correct += int(
            ((target_delta.sign() == prediction_delta.sign()) & valid).sum()
        )
        total += int(valid.sum())
    return float(correct / max(total, 1)), total


def evaluate_scores(
    predicted_scores: torch.Tensor,
    data: ProposalAuditData,
    *,
    bootstrap_seed: int,
    bootstrap_replicates: int,
    baseline_values: np.ndarray,
) -> Dict[str, object]:
    if predicted_scores.shape != data.cached_base_scores.shape:
        raise ValueError("Predicted score shape does not match proposal bank")
    target_scores = data.target_factors[..., -1]
    selected = predicted_scores.argmax(dim=1)
    oracle = target_scores.argmax(dim=1)
    rows = torch.arange(len(selected))
    selected_values_tensor = target_scores[rows, selected]
    oracle_values_tensor = target_scores[rows, oracle]
    selected_values = selected_values_tensor.numpy()
    delta = selected_values - baseline_values
    ci = _log_bootstrap_ci(
        delta,
        data.physical_logs,
        seed=bootstrap_seed,
        replicates=bootstrap_replicates,
    )
    pairwise, pair_count = _pairwise_accuracy(
        predicted_scores, target_scores, minimum_delta=0.02
    )
    selected_factors = data.target_factors[rows, selected].mean(dim=0)
    return {
        "selected_pdms": float(selected_values_tensor.mean()),
        "selected_pdms_delta_vs_cached_base": float(delta.mean()),
        "selected_pdms_delta_log_bootstrap_95ci": list(ci),
        "best_of_64_pdms": float(oracle_values_tensor.mean()),
        "top1_regret": float(
            (oracle_values_tensor - selected_values_tensor).mean()
        ),
        "pairwise_accuracy_delta_ge_0_02": pairwise,
        "pair_count_delta_ge_0_02": pair_count,
        "selected_index_agreement_with_cached_base": float(
            (selected == data.cached_base_scores.argmax(dim=1)).float().mean()
        ),
        "selected_factors": {
            key: float(selected_factors[index])
            for index, key in enumerate(TARGET_FACTOR_KEYS)
        },
    }


def component_drift(
    reference_state: Mapping[str, torch.Tensor],
    candidate_state: Mapping[str, torch.Tensor],
) -> Dict[str, object]:
    """Measure checkpoint displacement for semantically distinct scorer blocks."""

    result: Dict[str, object] = {}
    for name, prefixes in _COMPONENT_PREFIXES.items():
        keys = sorted(
            key
            for key in reference_state
            if any(key == prefix or key.startswith(prefix) for prefix in prefixes)
        )
        if not keys or any(key not in candidate_state for key in keys):
            raise RuntimeError(f"Incomplete component {name} in compared checkpoint")
        dot = 0.0
        reference_norm = 0.0
        candidate_norm = 0.0
        difference_norm = 0.0
        maximum = 0.0
        parameter_count = 0
        for key in keys:
            reference = reference_state[key].detach().double().reshape(-1)
            candidate = candidate_state[key].detach().double().reshape(-1)
            if reference.shape != candidate.shape:
                raise RuntimeError(f"Shape mismatch for {key}")
            difference = candidate - reference
            dot += float(torch.dot(reference, candidate))
            reference_norm += float(torch.dot(reference, reference))
            candidate_norm += float(torch.dot(candidate, candidate))
            difference_norm += float(torch.dot(difference, difference))
            maximum = max(maximum, float(difference.abs().max()))
            parameter_count += reference.numel()
        result[name] = {
            "tensor_count": len(keys),
            "parameter_count": parameter_count,
            "relative_l2": float(
                np.sqrt(difference_norm) / max(np.sqrt(reference_norm), 1e-12)
            ),
            "cosine_similarity": float(
                dot
                / max(np.sqrt(reference_norm) * np.sqrt(candidate_norm), 1e-12)
            ),
            "max_abs_difference": maximum,
        }
    return result


def _markdown_report(payload: Mapping[str, object]) -> str:
    lines = [
        "# M0 scorer module-swap audit",
        "",
        "The proposal bank is fixed. PDM labels are joined only after each "
        "module combination selects a candidate.",
        "",
        f"Scenes: {payload['scene_count']}; physical logs: {payload['physical_log_count']}",
        "",
        "| Context | Attention | Head | Selected PDMS | Delta vs Base | Pairwise | Regret | Base-index agreement |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    combinations = payload["combinations"]
    for key in sorted(combinations):
        fields = dict(part.split("=", 1) for part in key.split("|"))
        metrics = combinations[key]
        lines.append(
            "| {context} | {attention} | {head} | {selected:.6f} | "
            "{delta:+.6f} | {pairwise:.6f} | {regret:.6f} | {agreement:.4f} |".format(
                **fields,
                selected=metrics["selected_pdms"],
                delta=metrics["selected_pdms_delta_vs_cached_base"],
                pairwise=metrics["pairwise_accuracy_delta_ge_0_02"],
                regret=metrics["top1_regret"],
                agreement=metrics["selected_index_agreement_with_cached_base"],
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "A context swap changes the current-observation representation. An "
            "attention swap changes trajectory/context interaction. A head swap "
            "changes only the six factor classifiers. This audit diagnoses "
            "checkpoint calibration; it does not train on or tune against Navtest.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--proposal-root", type=Path)
    parser.add_argument("--label-root", type=Path)
    parser.add_argument("--proposal-pickle", type=Path)
    parser.add_argument("--candidate-npz", type=Path)
    parser.add_argument("--context", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument("--checkpoint", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument("--baseline-context", required=True)
    parser.add_argument("--baseline-checkpoint", required=True)
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument("--split", choices=("train", "validation", "all"), default="validation")
    parser.add_argument(
        "--config",
        type=Path,
        default=(
            Path(__file__).resolve().parents[1]
            / "navsim/planning/script/config/common/agent/episode_drive.yaml"
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-scenes", type=int, default=0)
    parser.add_argument("--bootstrap-seed", type=int, default=2)
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    contexts_paths = _parse_named_paths(args.context, "context")
    checkpoint_paths = _parse_named_paths(args.checkpoint, "checkpoint")
    if args.baseline_context not in contexts_paths:
        raise KeyError(f"Unknown baseline context {args.baseline_context}")
    if args.baseline_checkpoint not in checkpoint_paths:
        raise KeyError(f"Unknown baseline checkpoint {args.baseline_checkpoint}")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    navtest_mode = args.proposal_pickle is not None or args.candidate_npz is not None
    if navtest_mode:
        if args.proposal_pickle is None or args.candidate_npz is None:
            raise ValueError(
                "--proposal-pickle and --candidate-npz must be provided together"
            )
        if args.split != "all":
            raise ValueError("Navtest pickle mode requires --split all")
        data = load_navtest_proposal_audit_data(
            args.proposal_pickle.resolve(),
            args.candidate_npz.resolve(),
            max_scenes=args.max_scenes,
        )
        proposal_source = str(args.proposal_pickle.resolve())
        label_source = str(args.candidate_npz.resolve())
    else:
        if (
            args.proposal_root is None
            or args.label_root is None
            or args.split_manifest is None
        ):
            raise ValueError(
                "Chunk mode requires --proposal-root, --label-root, and "
                "--split-manifest"
            )
        allowed = _allowed_physical_logs(args.split_manifest, args.split)
        data = load_proposal_audit_data(
            args.proposal_root.resolve(),
            args.label_root.resolve(),
            allowed,
            max_scenes=args.max_scenes,
        )
        proposal_source = str(args.proposal_root.resolve())
        label_source = str(args.label_root.resolve())
    contexts = {
        name: load_context(path, data.tokens)
        for name, path in contexts_paths.items()
    }
    models = {
        name: load_action_decoder(path, args.config.resolve(), device)
        for name, path in checkpoint_paths.items()
    }
    scores = score_module_combinations(
        data.proposals,
        contexts,
        models,
        device=device,
        batch_size=args.batch_size,
    )
    base_index = data.cached_base_scores.argmax(dim=1)
    rows = torch.arange(len(base_index))
    baseline_values = data.target_factors[rows, base_index, -1].numpy()
    combinations = {
        key: evaluate_scores(
            value,
            data,
            bootstrap_seed=args.bootstrap_seed,
            bootstrap_replicates=args.bootstrap_replicates,
            baseline_values=baseline_values,
        )
        for key, value in scores.items()
    }
    exact_key = (
        f"context={args.baseline_context}|attention={args.baseline_checkpoint}|"
        f"head={args.baseline_checkpoint}"
    )
    parity_difference = (
        scores[exact_key] - data.cached_base_scores
    ).abs()
    reference_state = action_head_state(
        checkpoint_paths[args.baseline_checkpoint]
    )
    drift = {
        name: component_drift(reference_state, action_head_state(path))
        for name, path in checkpoint_paths.items()
        if name != args.baseline_checkpoint
    }
    payload: Dict[str, object] = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "proposal_source": proposal_source,
        "label_source": label_source,
        "split": args.split,
        "split_manifest": (
            str(args.split_manifest.resolve())
            if args.split_manifest is not None
            else None
        ),
        "scene_count": len(data.tokens),
        "segment_log_count": len(set(data.segment_logs)),
        "physical_log_count": len(set(data.physical_logs)),
        "candidate_count": int(data.proposals.shape[1]),
        "inference_inputs": [
            "fixed proposals",
            "current-observation scene_features",
            "current ego_features",
        ],
        "forbidden_model_inputs": [
            "PDM targets",
            "official scores",
            "future annotations",
            "future images",
        ],
        "contexts": {name: str(path) for name, path in contexts_paths.items()},
        "checkpoints": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in checkpoint_paths.items()
        },
        "cached_base_selected_pdms": float(baseline_values.mean()),
        "cached_base_best_of_64_pdms": float(
            data.target_factors[..., -1].max(dim=1).values.mean()
        ),
        "baseline_recompute_parity": {
            "combination": exact_key,
            "score_max_abs_error": float(parity_difference.max()),
            "score_mean_abs_error": float(parity_difference.mean()),
            "selected_index_agreement": float(
                (
                    scores[exact_key].argmax(dim=1)
                    == data.cached_base_scores.argmax(dim=1)
                )
                .float()
                .mean()
            ),
        },
        "component_drift_from_baseline": drift,
        "combinations": combinations,
    }
    _atomic_json_dump(payload, args.output_json.resolve())
    _atomic_text_dump(_markdown_report(payload), args.output_md.resolve())
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

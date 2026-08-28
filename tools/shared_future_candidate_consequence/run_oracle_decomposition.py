#!/usr/bin/env python3
"""Five-fold oracle decomposition of static, dynamic-state and risk value."""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata, spearmanr
from torch import nn

from .common import (
    DEFAULT_CACHE_DIR,
    DEFAULT_REPORT_DIR,
    append_command,
    assert_feature_names_safe,
    ensure_dir,
    log_bootstrap_ci,
    require_gate,
    seed_everything,
    update_gate,
    write_json,
    write_markdown,
)


@dataclass
class OracleDataset:
    scene_tokens: np.ndarray
    log_names: np.ndarray
    folds: np.ndarray
    candidate_indices: np.ndarray
    candidate_families: np.ndarray
    trajectory: np.ndarray
    current: np.ndarray
    k_exact: np.ndarray
    static: np.ndarray
    d_state: np.ndarray
    d_risk: np.ndarray
    d_signal: np.ndarray
    recomputed_risk: np.ndarray
    score: np.ndarray
    factors: np.ndarray
    family_names: tuple[str, ...] = ()

    @property
    def scenes(self) -> int:
        return int(self.score.shape[0])

    @property
    def candidates(self) -> int:
        return int(self.score.shape[1])


GROUP_NAMES = tuple(f"O{index}" for index in range(14))


def _completed_prefix_length(completed: np.ndarray) -> int:
    """Accept only a complete prefix followed by an optional failure suffix."""

    completed = np.asarray(completed, dtype=bool)
    if completed.size == 0:
        return 0
    incomplete = np.flatnonzero(~completed)
    if not len(incomplete):
        return len(completed)
    first_incomplete = int(incomplete[0])
    if bool(completed[first_incomplete:].any()):
        raise RuntimeError(
            "Oracle store contains an incomplete hole before a later completed scene; "
            "only a contiguous trailing failure suffix is auditable without copying the store"
        )
    return first_incomplete


def load_oracle_store(store_dir: Path, max_scenes: int = 0) -> OracleDataset:
    """Load the consolidated arrays as read-only memory maps."""

    config_path = store_dir / "store_config.json"
    metadata_path = store_dir / "scene_metadata.parquet"
    if not config_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"Incomplete oracle store: {store_dir}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    metadata = pd.read_parquet(metadata_path).sort_values("scene_index").reset_index(drop=True)
    requested_count = len(metadata) if max_scenes <= 0 else min(max_scenes, len(metadata))
    completed_raw = np.load(store_dir / "completed.npy", mmap_mode="r")[:requested_count]
    count = _completed_prefix_length(completed_raw)
    if count == 0:
        raise RuntimeError("Oracle store has no completed scene prefix")
    metadata = metadata.iloc[:count]

    def array(name: str) -> np.ndarray:
        value = np.load(store_dir / f"{name}.npy", mmap_mode="r")
        return value[:count]

    completed = array("completed")
    if not bool(completed.all()):
        raise RuntimeError("Internal error: trimmed oracle-store prefix is not complete")
    family_names = tuple(config["family_names"])
    family_ids = array("candidate_family_ids")
    # Family strings are small relative to dynamic tensors and simplify the
    # held-out-family audit while stable integer IDs remain on disk.
    families = np.asarray(family_names, dtype="U40")[family_ids]
    return OracleDataset(
        scene_tokens=metadata.scene_token.to_numpy(str),
        log_names=metadata.log_name.to_numpy(str),
        folds=metadata.fold.to_numpy(dtype=np.int64),
        candidate_indices=array("candidate_indices"),
        candidate_families=families,
        trajectory=array("trajectory"),
        current=array("current"),
        k_exact=array("k_exact"),
        static=array("static"),
        d_state=array("d_state"),
        d_risk=array("d_risk"),
        d_signal=array("d_signal"),
        recomputed_risk=array("recomputed_risk"),
        score=array("score"),
        factors=array("factors"),
        family_names=family_names,
    )


class PairwiseRanker(nn.Module):
    def __init__(self, input_dim: int, kind: str, hidden_dim: int | None = None):
        super().__init__()
        if kind == "linear":
            self.body = nn.Identity()
            body_dim = input_dim
        elif kind == "mlp":
            hidden_dim = hidden_dim or matched_hidden_dim(input_dim)
            self.body = nn.Sequential(
                nn.Linear(input_dim, hidden_dim), nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            )
            body_dim = hidden_dim
        else:
            raise ValueError(kind)
        self.score_head = nn.Linear(body_dim, 1)
        self.factor_head = nn.Linear(body_dim, 6)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.body(features)
        return self.score_head(hidden).squeeze(-1), self.factor_head(hidden)


def matched_hidden_dim(input_dim: int, target_parameters: int = 120_000) -> int:
    """Choose the two-layer width with closest total capacity across O0–O13."""

    def count(hidden: int) -> int:
        # two affine body layers plus scalar and six-factor heads
        return (
            input_dim * hidden + hidden
            + hidden * hidden + hidden
            + hidden + 1
            + hidden * 6 + 6
        )

    candidates = range(8, 513)
    return min(candidates, key=lambda hidden: abs(count(hidden) - target_parameters))


def _risk_from_actor(state_summary: np.ndarray, actor: np.ndarray, mask: np.ndarray) -> np.ndarray:
    clearance = state_summary[..., 0]
    collision = (clearance <= 1e-4).astype(np.float32)
    soft = 1.0 / (1.0 + np.exp(np.clip(clearance / 0.75, -30.0, 30.0)))
    rel_x = actor[..., 1]
    rel_y = actor[..., 2]
    rel_vx = actor[..., 3]
    length = actor[..., 6]
    width = actor[..., 7]
    front = rel_x - (5.176 + length) / 2.0
    closing = -rel_vx
    lateral_ok = np.abs(rel_y) <= (2.297 + width) / 2.0 + 0.5
    valid = mask & (front > 0) & (closing > 1e-3) & lateral_ok
    ttc_actor = np.where(valid, front / np.maximum(closing, 1e-3), 10.0)
    ttc = np.minimum(np.min(ttc_actor, axis=-1), 10.0).astype(np.float32)
    risky = (collision > 0.5) | (ttc < 3.0) | (clearance < 1.0)
    onset = np.full(risky.shape[:2], 10.0, dtype=np.float32)
    horizons = np.arange(0.5, 4.1, 0.5, dtype=np.float32)
    for scene in range(len(risky)):
        for candidate in range(risky.shape[1]):
            indices = np.flatnonzero(risky[scene, candidate])
            if len(indices):
                onset[scene, candidate] = horizons[indices[0]]
    return np.stack(
        [collision, ttc, soft, clearance, np.broadcast_to(onset[..., None], clearance.shape)],
        axis=-1,
    ).astype(np.float32)


def load_dataset(
    cache_dir: Path,
    report_dir: Path,
    max_scenes: int = 0,
    store_dir: Path | None = None,
) -> OracleDataset:
    store_dir = store_dir or cache_dir / "oracle_store"
    if (store_dir / "store_config.json").is_file():
        return load_oracle_store(store_dir, max_scenes)
    manifest = pd.read_parquet(cache_dir / "controlled_candidate_manifest.parquet")
    metrics = pd.read_parquet(cache_dir / "candidate_metrics.parquet")
    scene_manifest = pd.read_parquet(report_dir / "balanced_scene_manifest.parquet")
    ordered_tokens = scene_manifest.scene_token.tolist()
    if max_scenes > 0:
        ordered_tokens = ordered_tokens[:max_scenes]
    metric_lookup = metrics.set_index(["scene_token", "candidate_index"])
    manifest_lookup = manifest.set_index(["scene_token", "candidate_index"])

    scene_tokens: list[str] = []
    logs: list[str] = []
    folds: list[int] = []
    candidate_indices: list[np.ndarray] = []
    families: list[np.ndarray] = []
    trajectories: list[np.ndarray] = []
    currents: list[np.ndarray] = []
    k_exact: list[np.ndarray] = []
    static: list[np.ndarray] = []
    d_state: list[np.ndarray] = []
    d_risk: list[np.ndarray] = []
    d_signal: list[np.ndarray] = []
    recomputed: list[np.ndarray] = []
    scores: list[np.ndarray] = []
    factors: list[np.ndarray] = []
    for token in ordered_tokens:
        target_path = cache_dir / "targets_v3" / f"{token}.npz"
        if not target_path.is_file():
            continue
        scene_row = scene_manifest[scene_manifest.scene_token == token].iloc[0]
        with np.load(target_path, allow_pickle=False) as target:
            indices = target["candidate_index"].astype(int)
            keys = [(token, int(index)) for index in indices]
            candidate_rows = manifest_lookup.loc[keys]
            metric_rows = metric_lookup.loc[keys]
            pose = np.stack(
                [
                    np.column_stack([row.pose_x_m, row.pose_y_m, row.pose_heading_rad]).reshape(-1)
                    for row in candidate_rows.itertuples(index=False)
                ],
                axis=0,
            ).astype(np.float32)
            actor = target["D_state_actor"].astype(np.float32)
            actor_mask = target["D_state_actor_mask"].astype(bool)
            actor_masked = actor * actor_mask[..., None]
            state_features = np.concatenate(
                [
                    target["D_state_summary"].reshape(len(indices), -1),
                    actor_masked.reshape(len(indices), -1),
                    actor_mask.astype(np.float32).reshape(len(indices), -1),
                ],
                axis=-1,
            )
            current_vector = np.concatenate(
                [
                    target["current_scene_features"].astype(np.float32),
                    (
                        target["current_actor_state"].astype(np.float32)
                        * target["current_actor_mask"].astype(np.float32)[:, None]
                    ).reshape(-1),
                    target["current_actor_mask"].astype(np.float32),
                ]
            )
            current = np.broadcast_to(
                current_vector[None], (len(indices), len(current_vector))
            ).copy()
            factor = np.column_stack(
                [
                    1.0 - metric_rows.no_at_fault_collision.to_numpy(float),
                    1.0 - metric_rows.ttc.to_numpy(float),
                    1.0 - metric_rows.dac.to_numpy(float),
                    1.0 - metric_rows.ddc.to_numpy(float),
                    1.0 - metric_rows.comfort.to_numpy(float),
                    metric_rows.progress.to_numpy(float),
                ]
            ).astype(np.float32)
            scene_tokens.append(token)
            logs.append(str(scene_row.log_name))
            folds.append(int(scene_row.fold))
            candidate_indices.append(indices)
            families.append(candidate_rows.candidate_family.to_numpy(str))
            trajectories.append(pose)
            currents.append(current.astype(np.float32))
            k_exact.append(target["K_exact"].reshape(len(indices), -1).astype(np.float32))
            static.append(target["S_static"].reshape(len(indices), -1).astype(np.float32))
            d_state.append(state_features.astype(np.float32))
            d_risk.append(target["D_risk"].reshape(len(indices), -1).astype(np.float32))
            d_signal.append(target["D_signal"].reshape(len(indices), -1).astype(np.float32))
            recomputed.append(
                _risk_from_actor(
                    target["D_state_summary"][None], actor[None], actor_mask[None]
                )[0].reshape(len(indices), -1)
            )
            scores.append(metric_rows.aggregate_score.to_numpy(dtype=np.float32))
            factors.append(factor)
    if not scene_tokens:
        raise RuntimeError("No v3 targets were loaded")
    candidate_counts = {len(value) for value in scores}
    if len(candidate_counts) != 1:
        raise RuntimeError(f"Oracle decomposition requires fixed K, got {candidate_counts}")
    return OracleDataset(
        scene_tokens=np.asarray(scene_tokens),
        log_names=np.asarray(logs),
        folds=np.asarray(folds, dtype=np.int64),
        candidate_indices=np.stack(candidate_indices),
        candidate_families=np.stack(families),
        trajectory=np.stack(trajectories),
        current=np.stack(currents),
        k_exact=np.stack(k_exact),
        static=np.stack(static),
        d_state=np.stack(d_state),
        d_risk=np.stack(d_risk),
        d_signal=np.stack(d_signal),
        recomputed_risk=np.stack(recomputed),
        score=np.stack(scores),
        factors=np.stack(factors),
        family_names=tuple(sorted(set(np.stack(families).reshape(-1)))),
    )


def _current_by_candidate(dataset: OracleDataset) -> np.ndarray:
    if dataset.current.ndim == 3:
        return dataset.current
    return np.broadcast_to(
        dataset.current[:, None, :],
        (dataset.scenes, dataset.candidates, dataset.current.shape[-1]),
    )


def _base_parts(dataset: OracleDataset, level: int) -> list[np.ndarray]:
    parts = [dataset.trajectory]
    if level >= 1:
        parts.append(_current_by_candidate(dataset))
    if level >= 2:
        parts.append(dataset.k_exact)
    if level >= 3:
        parts.append(dataset.static)
    return parts


def _concatenate_parts(parts: list[np.ndarray]) -> np.ndarray:
    if len(parts) == 1:
        return np.asarray(parts[0])
    return np.concatenate(parts, axis=-1, dtype=np.float32)


def _controlled_feature_group(
    dataset: OracleDataset,
    group_name: str,
    seed: int,
    chunk_scenes: int = 256,
) -> np.ndarray:
    """Construct high-dimensional controls without full-size temporary arrays."""

    base_parts = _base_parts(dataset, 3)
    base_dim = sum(part.shape[-1] for part in base_parts)
    dynamic_parts = [dataset.d_state, dataset.d_risk, dataset.d_signal]
    dynamic_dim = sum(part.shape[-1] for part in dynamic_parts)
    output = np.empty(
        (dataset.scenes, dataset.candidates, base_dim + dynamic_dim), dtype=np.float32
    )
    offset = 0
    for part in base_parts:
        output[..., offset : offset + part.shape[-1]] = part
        offset += part.shape[-1]
    rng = np.random.default_rng(seed)
    if group_name == "O11":
        permutation = rng.permutation(dataset.scenes)
        if dataset.scenes > 1 and np.any(permutation == np.arange(dataset.scenes)):
            permutation = np.roll(np.arange(dataset.scenes), 1)
    else:
        permutation = None
    repeats = int(np.ceil(dynamic_dim / dataset.static.shape[-1]))
    for start in range(0, dataset.scenes, chunk_scenes):
        stop = min(start + chunk_scenes, dataset.scenes)
        if group_name == "O10":
            order = np.argsort(rng.random((stop - start, dataset.candidates)), axis=1)
            cursor = offset
            for part in dynamic_parts:
                chunk = np.asarray(part[start:stop])
                shuffled = np.take_along_axis(chunk, order[..., None], axis=1)
                output[start:stop, :, cursor : cursor + part.shape[-1]] = shuffled
                cursor += part.shape[-1]
        elif group_name == "O11":
            cursor = offset
            source = permutation[start:stop]
            for part in dynamic_parts:
                output[start:stop, :, cursor : cursor + part.shape[-1]] = part[source]
                cursor += part.shape[-1]
        elif group_name == "O12":
            output[start:stop, :, offset:] = rng.standard_normal(
                (stop - start, dataset.candidates, dynamic_dim), dtype=np.float32
            )
        elif group_name == "O13":
            tiled = np.tile(np.asarray(dataset.static[start:stop]), (1, 1, repeats))
            output[start:stop, :, offset:] = tiled[..., :dynamic_dim]
        else:
            raise ValueError(group_name)
    return output


def feature_group(dataset: OracleDataset, group_name: str, seed: int) -> np.ndarray:
    if group_name not in GROUP_NAMES:
        raise ValueError(group_name)
    assert_feature_names_safe(
        ["trajectory", "current_scene", "K_exact", "S_static", "D_state", "D_risk", "D_signal"]
    )
    if group_name == "O0":
        return _concatenate_parts(_base_parts(dataset, 0))
    if group_name == "O1":
        return _concatenate_parts(_base_parts(dataset, 1))
    if group_name == "O2":
        return _concatenate_parts(_base_parts(dataset, 2))
    base = _base_parts(dataset, 3)
    additions = {
        "O3": [],
        "O4": [dataset.d_state],
        "O5": [dataset.d_risk],
        "O6": [dataset.d_signal],
        "O7": [dataset.d_state, dataset.d_risk],
        "O8": [dataset.d_state, dataset.d_risk, dataset.d_signal],
        "O9": [dataset.d_state, dataset.recomputed_risk],
    }
    if group_name in additions:
        return _concatenate_parts(base + additions[group_name])
    return _controlled_feature_group(dataset, group_name, seed)


def feature_groups(dataset: OracleDataset, seed: int) -> dict[str, np.ndarray]:
    """Compatibility helper for small tests; formal runs build groups lazily."""

    return {name: feature_group(dataset, name, seed) for name in GROUP_NAMES}


def _parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def train_ranker(
    features: np.ndarray,
    score: np.ndarray,
    factors: np.ndarray,
    train_mask: np.ndarray,
    validation_mask: np.ndarray,
    kind: str,
    seed: int,
    epochs: int,
    batch_scenes: int,
    device: str,
    train_candidate_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    seed_everything(seed)
    train_indices = np.flatnonzero(train_mask)
    if train_candidate_mask is None:
        train_candidate_mask = np.ones_like(score, dtype=bool)
    feature_sum = np.zeros(features.shape[-1], dtype=np.float64)
    feature_square_sum = np.zeros(features.shape[-1], dtype=np.float64)
    feature_count = 0
    stats_batch = max(batch_scenes, 128)
    for start in range(0, len(train_indices), stats_batch):
        indices = train_indices[start : start + stats_batch]
        chunk = np.asarray(features[indices], dtype=np.float32)
        # A held-out candidate family must be absent from every fitted
        # statistic, not merely from the supervised pairwise loss.  Otherwise
        # feature normalization would be a small but real transductive leak.
        flat = chunk[train_candidate_mask[indices]]
        feature_sum += flat.sum(axis=0, dtype=np.float64)
        feature_square_sum += np.square(flat, dtype=np.float64).sum(axis=0, dtype=np.float64)
        feature_count += len(flat)
    if feature_count == 0:
        raise RuntimeError("No training candidates remain after applying the candidate-family mask")
    mean64 = feature_sum / max(feature_count, 1)
    variance64 = np.maximum(feature_square_sum / max(feature_count, 1) - mean64**2, 0.0)
    mean = mean64.astype(np.float32)
    std = np.sqrt(variance64).astype(np.float32)
    std[std < 1e-5] = 1.0
    model = PairwiseRanker(features.shape[-1], kind).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=3e-3 if kind == "linear" else 1e-3,
        weight_decay=1e-3,
    )
    rng = np.random.default_rng(seed)
    feature_device: torch.Tensor | None = None
    score_device: torch.Tensor | None = None
    factors_device: torch.Tensor | None = None
    mean_device: torch.Tensor | None = None
    std_device: torch.Tensor | None = None
    # A800 cards have ample memory for the largest formal O8 tensor (~5 GiB).
    # Keeping one raw copy on-device avoids transferring that tensor once per
    # epoch.  Fall back to the original batch transfer path on a busy/OOM card.
    if str(device).startswith("cuda") and features.nbytes <= 8 * 2**30:
        try:
            feature_device = torch.from_numpy(np.asarray(features)).to(device)
            # Score/factor arrays are read-only mmaps and small enough to copy;
            # making them writable avoids PyTorch's undefined-write warning.
            score_device = torch.from_numpy(np.array(score, copy=True)).to(device)
            factors_device = torch.from_numpy(np.array(factors, copy=True)).to(device)
            mean_device = torch.from_numpy(mean).to(device)
            std_device = torch.from_numpy(std).to(device)
        except torch.cuda.OutOfMemoryError:
            del feature_device, score_device, factors_device, mean_device, std_device
            feature_device = score_device = factors_device = mean_device = std_device = None
            torch.cuda.empty_cache()
    model.train()
    final_loss = float("nan")
    for _ in range(epochs):
        rng.shuffle(train_indices)
        losses = []
        for start in range(0, len(train_indices), batch_scenes):
            indices = train_indices[start : start + batch_scenes]
            if feature_device is not None:
                device_indices = torch.as_tensor(indices, dtype=torch.long, device=device)
                x = (feature_device[device_indices] - mean_device) / std_device
                y = score_device[device_indices]
                f = factors_device[device_indices]
            else:
                x = torch.from_numpy((features[indices] - mean) / std).to(device)
                y = torch.from_numpy(score[indices]).to(device)
                f = torch.from_numpy(factors[indices]).to(device)
            prediction, factor_prediction = model(x)
            difference = prediction[:, :, None] - prediction[:, None, :]
            target_difference = y[:, :, None] - y[:, None, :]
            triangle = torch.triu(torch.ones_like(target_difference, dtype=torch.bool), diagonal=1)
            candidate_valid = torch.from_numpy(train_candidate_mask[indices]).to(device)
            valid = (
                triangle
                & (target_difference.abs() > 1e-6)
                & candidate_valid[:, :, None]
                & candidate_valid[:, None, :]
            )
            if not valid.any():
                continue
            sign = torch.sign(target_difference[valid])
            rank_loss = torch.nn.functional.softplus(-sign * difference[valid]).mean()
            binary_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                factor_prediction[..., :5][candidate_valid], f[..., :5][candidate_valid]
            )
            progress_loss = torch.nn.functional.smooth_l1_loss(
                torch.sigmoid(factor_prediction[..., 5][candidate_valid]), f[..., 5][candidate_valid]
            )
            loss = rank_loss + 0.15 * binary_loss + 0.05 * progress_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        final_loss = float(np.mean(losses)) if losses else float("nan")
    model.eval()
    predictions = np.full_like(score, np.nan, dtype=np.float32)
    factor_predictions = np.full_like(factors, np.nan, dtype=np.float32)
    validation_indices = np.flatnonzero(validation_mask)
    with torch.no_grad():
        for start in range(0, len(validation_indices), batch_scenes):
            indices = validation_indices[start : start + batch_scenes]
            if feature_device is not None:
                device_indices = torch.as_tensor(indices, dtype=torch.long, device=device)
                x = (feature_device[device_indices] - mean_device) / std_device
            else:
                x = torch.from_numpy((features[indices] - mean) / std).to(device)
            prediction, factor_prediction = model(x)
            predictions[indices] = prediction.cpu().numpy()
            raw = factor_prediction.cpu().numpy()
            raw[..., :5] = 1.0 / (1.0 + np.exp(-np.clip(raw[..., :5], -30, 30)))
            raw[..., 5] = 1.0 / (1.0 + np.exp(-np.clip(raw[..., 5], -30, 30)))
            factor_predictions[indices] = raw
    return predictions, factor_predictions, {
        "parameter_count": _parameter_count(model),
        "final_train_loss": final_loss,
        "epochs": epochs,
        "input_dim": int(features.shape[-1]),
    }


def _auroc(truth: np.ndarray, probability: np.ndarray) -> float:
    truth = truth.astype(bool)
    positives = int(truth.sum())
    negatives = len(truth) - positives
    if positives == 0 or negatives == 0:
        return float("nan")
    ranks = rankdata(probability)
    return float((ranks[truth].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def _calibration_curve(
    truth: np.ndarray, probability: np.ndarray, bins: int = 10
) -> list[dict[str, float | int]]:
    """Return fixed-width reliability bins without retaining candidate labels."""

    truth = truth.astype(np.float64)
    probability = np.clip(probability.astype(np.float64), 0.0, 1.0)
    edges = np.linspace(0.0, 1.0, bins + 1)
    assignments = np.minimum(np.searchsorted(edges, probability, side="right") - 1, bins - 1)
    assignments = np.maximum(assignments, 0)
    rows: list[dict[str, float | int]] = []
    for index in range(bins):
        mask = assignments == index
        rows.append(
            {
                "bin": index,
                "lower": float(edges[index]),
                "upper": float(edges[index + 1]),
                "count": int(mask.sum()),
                "mean_probability": float(probability[mask].mean()) if mask.any() else float("nan"),
                "observed_rate": float(truth[mask].mean()) if mask.any() else float("nan"),
            }
        )
    return rows


def _ndcg(truth: np.ndarray, prediction: np.ndarray) -> float:
    order = np.argsort(-prediction)
    ideal = np.argsort(-truth)
    discount = 1.0 / np.log2(np.arange(len(truth)) + 2.0)
    dcg = float(np.sum((2.0 ** truth[order] - 1.0) * discount))
    idcg = float(np.sum((2.0 ** truth[ideal] - 1.0) * discount))
    return dcg / idcg if idcg > 0 else 1.0


def ranking_metrics(
    dataset: OracleDataset,
    prediction: np.ndarray,
    factor_prediction: np.ndarray,
    mask: np.ndarray,
) -> tuple[dict[str, Any], pd.DataFrame]:
    thresholds = (0.0, 0.02, 0.05, 0.10)
    pair_correct_count = {threshold: 0 for threshold in thresholds}
    pair_count = {threshold: 0 for threshold in thresholds}
    scene_rows = []
    valid_indices = np.flatnonzero(mask)
    truth_batch = np.asarray(dataset.score[valid_indices])
    prediction_batch = np.asarray(prediction[valid_indices])
    truth_rank = rankdata(truth_batch, axis=1)
    prediction_rank = rankdata(prediction_batch, axis=1)
    truth_rank -= truth_rank.mean(axis=1, keepdims=True)
    prediction_rank -= prediction_rank.mean(axis=1, keepdims=True)
    rank_denominator = np.sqrt(
        np.square(truth_rank).sum(axis=1) * np.square(prediction_rank).sum(axis=1)
    )
    rank_correlation = np.divide(
        (truth_rank * prediction_rank).sum(axis=1),
        rank_denominator,
        out=np.zeros_like(rank_denominator, dtype=np.float64),
        where=rank_denominator > 0,
    )
    for local_index, scene in enumerate(valid_indices):
        truth = dataset.score[scene]
        pred = prediction[scene]
        differences = truth[:, None] - truth[None, :]
        pred_differences = pred[:, None] - pred[None, :]
        triangle = np.triu(np.ones_like(differences, dtype=bool), k=1)
        accuracies = {}
        for threshold in thresholds:
            valid = triangle & (np.abs(differences) > threshold)
            count = int(valid.sum())
            correct = float(np.mean(np.sign(differences[valid]) == np.sign(pred_differences[valid]))) if count else np.nan
            pair_correct_count[threshold] += int(
                np.sum(np.sign(differences[valid]) == np.sign(pred_differences[valid]))
            )
            pair_count[threshold] += count
            accuracies[threshold] = correct
        best = int(np.argmax(truth))
        selected = int(np.argmax(pred))
        predicted_order = np.argsort(-pred)
        scene_rows.append(
            {
                "scene_index": scene,
                "scene_token": dataset.scene_tokens[scene],
                "log_name": dataset.log_names[scene],
                "fold": int(dataset.folds[scene]),
                "pairwise_accuracy": accuracies[0.0],
                "pairwise_accuracy_gt_0p02": accuracies[0.02],
                "pairwise_accuracy_gt_0p05": accuracies[0.05],
                "pairwise_accuracy_gt_0p10": accuracies[0.10],
                "spearman": float(rank_correlation[local_index]),
                "ndcg": _ndcg(truth, pred),
                "top1_hit": float(selected == best),
                "top2_recall": float(best in predicted_order[:2]),
                "top4_recall": float(best in predicted_order[:4]),
                "top1_regret": float(truth[best] - truth[selected]),
            }
        )
    per_scene = pd.DataFrame(scene_rows)
    truth_factors = dataset.factors[mask].reshape(-1, 6)
    pred_factors = factor_prediction[mask].reshape(-1, 6)
    factor_names = ("collision", "ttc", "dac", "ddc", "comfort")
    factor_metrics = {}
    for index, name in enumerate(factor_names):
        truth = truth_factors[:, index]
        probability = pred_factors[:, index]
        binary = truth > 0.5
        predicted_binary = probability >= 0.5
        tp = int(np.sum(binary & predicted_binary))
        fp = int(np.sum(~binary & predicted_binary))
        fn = int(np.sum(binary & ~predicted_binary))
        factor_metrics[name] = {
            "auroc": _auroc(binary, probability),
            "f1": float(2 * tp / max(2 * tp + fp + fn, 1)),
            "positive_rate": float(binary.mean()),
            "calibration": _calibration_curve(binary, probability),
        }
    factor_metrics["progress"] = {
        "mae": float(np.mean(np.abs(truth_factors[:, 5] - pred_factors[:, 5]))),
        "spearman": float(spearmanr(truth_factors[:, 5], pred_factors[:, 5]).statistic),
    }
    metrics = {
        "scene_count": len(per_scene),
        "pairwise_accuracy": pair_correct_count[0.0] / pair_count[0.0] if pair_count[0.0] else float("nan"),
        "pairwise_accuracy_gt_0p02": pair_correct_count[0.02] / pair_count[0.02] if pair_count[0.02] else None,
        "pairwise_accuracy_gt_0p05": pair_correct_count[0.05] / pair_count[0.05] if pair_count[0.05] else None,
        "pairwise_accuracy_gt_0p10": pair_correct_count[0.10] / pair_count[0.10] if pair_count[0.10] else None,
        "pair_counts": {str(key): value for key, value in pair_count.items()},
        "spearman": float(per_scene.spearman.mean()),
        "ndcg": float(per_scene.ndcg.mean()),
        "top1_accuracy": float(per_scene.top1_hit.mean()),
        "top2_recall": float(per_scene.top2_recall.mean()),
        "top4_recall": float(per_scene.top4_recall.mean()),
        "top1_regret": float(per_scene.top1_regret.mean()),
        "factors": factor_metrics,
    }
    return metrics, per_scene


def _aggregate_fold_results(rows: pd.DataFrame) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for (group, model), frame in rows.groupby(["group", "model"], sort=True):
        result[f"{group}:{model}"] = {
            "fold_count": int(len(frame)),
            "pairwise_mean": float(frame.pairwise_accuracy.mean()),
            "pairwise_std": float(frame.pairwise_accuracy.std(ddof=0)),
            "pairwise_worst_fold": float(frame.pairwise_accuracy.min()),
            "ndcg_mean": float(frame.ndcg.mean()),
            "spearman_mean": float(frame.spearman.mean()),
            "top1_accuracy_mean": float(frame.top1_accuracy.mean()),
            "top1_regret_mean": float(frame.top1_regret.mean()),
            "collision_auroc_mean": float(frame.collision_auroc.mean()),
            "ttc_auroc_mean": float(frame.ttc_auroc.mean()),
        }
    return result


def run_heldout_family_audit(
    dataset: OracleDataset,
    folds: list[int],
    seed: int,
    epochs: int,
    batch_scenes: int,
    device: str,
) -> pd.DataFrame:
    """Exclude a candidate family from all train pairs, then test it on unseen logs."""

    features_by_group = {
        "O3": feature_group(dataset, "O3", seed),
        "O8": feature_group(dataset, "O8", seed),
    }
    rows: list[dict[str, Any]] = []
    families = sorted(set(dataset.candidate_families.reshape(-1)) - {"gt"})
    for family in families:
        gains = []
        for fold in folds:
            validation_scene = dataset.folds == fold
            validation_candidate = dataset.candidate_families == family
            train_scene = ~validation_scene
            train_candidate = dataset.candidate_families != family
            values: dict[str, float] = {}
            for group_name in ("O3", "O8"):
                prediction, _, _ = train_ranker(
                    features_by_group[group_name],
                    dataset.score,
                    dataset.factors,
                    train_scene,
                    validation_scene,
                    "linear",
                    seed + fold + sum(ord(char) for char in family),
                    max(4, epochs // 2),
                    batch_scenes,
                    device,
                    train_candidate_mask=train_candidate,
                )
                correct = []
                for scene in np.flatnonzero(validation_scene):
                    family_indices = np.flatnonzero(validation_candidate[scene])
                    other_indices = np.flatnonzero(~validation_candidate[scene])
                    for i in family_indices:
                        for j in other_indices:
                            difference = dataset.score[scene, i] - dataset.score[scene, j]
                            if abs(difference) > 1e-6:
                                correct.append(
                                    np.sign(difference)
                                    == np.sign(prediction[scene, i] - prediction[scene, j])
                                )
                value = float(np.mean(correct)) if correct else float("nan")
                values[group_name] = value
                rows.append(
                    {
                        "family": family,
                        "fold": fold,
                        "group": group_name,
                        "pairwise_accuracy": value,
                    }
                )
            gains.append(values["O8"] - values["O3"])
            print(
                f"heldout family={family} fold={fold} "
                f"O3={values['O3']:.4f} O8={values['O8']:.4f} "
                f"gain={values['O8'] - values['O3']:.4f}",
                flush=True,
            )
        rows.append(
            {
                "family": family,
                "fold": "mean",
                "group": "gain",
                "pairwise_accuracy": float(np.nanmean(gains)),
            }
        )
    del features_by_group
    gc.collect()
    return pd.DataFrame(rows)


def run(args: argparse.Namespace) -> dict[str, Any]:
    require_gate(args.output_dir, "target_v3")
    report_dir = ensure_dir(args.output_dir)
    cache_dir = ensure_dir(args.cache_dir)
    dataset = load_dataset(cache_dir, report_dir, args.num_scenes, args.store_dir)
    requested_groups = [value.strip() for value in args.groups.split(",") if value.strip()]
    unknown_groups = set(requested_groups) - set(GROUP_NAMES)
    if unknown_groups:
        raise ValueError(f"Unknown groups: {unknown_groups}")
    if not requested_groups:
        raise ValueError("At least one feature group is required")
    model_kinds = [value.strip() for value in args.models.split(",") if value.strip()]
    unknown = set(model_kinds) - {"linear", "mlp"}
    if unknown:
        raise ValueError(f"Unknown models: {unknown}")
    available_folds = sorted(set(dataset.folds.tolist()))
    selected_folds = (
        [int(value) for value in args.folds.split(",") if value.strip()]
        if args.folds
        else available_folds
    )
    if set(selected_folds) - set(available_folds):
        raise ValueError(f"Requested folds not in dataset: {selected_folds} vs {available_folds}")
    partial = set(requested_groups) != set(GROUP_NAMES) or set(selected_folds) != set(available_folds)
    if partial:
        job_name = args.job_name or (
            "groups_" + "-".join(requested_groups) + "_folds_" + "-".join(map(str, selected_folds))
        )
        job_root = args.job_root or cache_dir / "oracle_jobs"
        result_dir = ensure_dir(job_root / job_name)
    else:
        result_dir = report_dir
    fold_rows: list[dict[str, Any]] = []
    per_scene_records: list[pd.DataFrame] = []
    calibration_records: list[dict[str, Any]] = []
    epochs = args.epochs or (12 if args.mode == "smoke" else 25 if args.mode == "pilot" else 35)
    for group_position, group_name in enumerate(requested_groups):
        print(
            f"building group {group_name} ({group_position + 1}/{len(requested_groups)})",
            flush=True,
        )
        features = feature_group(dataset, group_name, args.seed)
        for model_kind in model_kinds:
            for fold in selected_folds:
                validation_mask = dataset.folds == fold
                if not validation_mask.any():
                    continue
                train_mask = ~validation_mask
                prediction, factor_prediction, training = train_ranker(
                    features, dataset.score, dataset.factors,
                    train_mask, validation_mask, model_kind,
                    args.seed + fold, epochs, args.batch_scenes, args.device,
                )
                metrics, per_scene = ranking_metrics(dataset, prediction, factor_prediction, validation_mask)
                for factor_name in ("collision", "ttc"):
                    for calibration_bin in metrics["factors"][factor_name]["calibration"]:
                        calibration_records.append(
                            {
                                "group": group_name,
                                "model": model_kind,
                                "fold": fold,
                                "factor": factor_name,
                                **calibration_bin,
                            }
                        )
                fold_rows.append(
                    {
                        "group": group_name,
                        "model": model_kind,
                        "fold": fold,
                        "input_dim": int(features.shape[-1]),
                        "parameter_count": training["parameter_count"],
                        "pairwise_accuracy": metrics["pairwise_accuracy"],
                        "pairwise_accuracy_gt_0p02": metrics["pairwise_accuracy_gt_0p02"],
                        "pairwise_accuracy_gt_0p05": metrics["pairwise_accuracy_gt_0p05"],
                        "pairwise_accuracy_gt_0p10": metrics["pairwise_accuracy_gt_0p10"],
                        "spearman": metrics["spearman"],
                        "ndcg": metrics["ndcg"],
                        "top1_accuracy": metrics["top1_accuracy"],
                        "top2_recall": metrics["top2_recall"],
                        "top4_recall": metrics["top4_recall"],
                        "top1_regret": metrics["top1_regret"],
                        "collision_auroc": metrics["factors"]["collision"]["auroc"],
                        "ttc_auroc": metrics["factors"]["ttc"]["auroc"],
                        "dac_auroc": metrics["factors"]["dac"]["auroc"],
                        "ddc_auroc": metrics["factors"]["ddc"]["auroc"],
                        "comfort_auroc": metrics["factors"]["comfort"]["auroc"],
                        "progress_spearman": metrics["factors"]["progress"]["spearman"],
                    }
                )
                per_scene["group"] = group_name
                per_scene["model"] = model_kind
                per_scene_records.append(per_scene)
                print(
                    f"completed group={group_name} model={model_kind} fold={fold} "
                    f"pairwise={metrics['pairwise_accuracy']:.4f} "
                    f"regret={metrics['top1_regret']:.4f}",
                    flush=True,
                )
        del features
        gc.collect()
    fold_frame = pd.DataFrame(fold_rows)
    fold_frame.to_csv(result_dir / "oracle_fold_results.csv", index=False)
    pd.DataFrame(calibration_records).to_csv(
        result_dir / "oracle_factor_calibration.csv", index=False
    )
    per_scene_frame = pd.concat(per_scene_records, ignore_index=True)
    per_scene_frame.to_parquet(
        result_dir / "oracle_per_scene_results.parquet",
        index=False,
        engine="pyarrow",
        compression="zstd",
    )
    run_heldout = args.heldout or not partial
    heldout = (
        run_heldout_family_audit(
            dataset,
            selected_folds,
            args.seed,
            epochs,
            args.batch_scenes,
            args.device,
        )
        if run_heldout
        else pd.DataFrame(columns=["family", "fold", "group", "pairwise_accuracy"])
    )
    if len(heldout):
        heldout.to_csv(result_dir / "heldout_candidate_type_results.csv", index=False)
    if partial:
        result = {
            "partial": True,
            "job_name": result_dir.name,
            "mode": args.mode,
            "scene_count": dataset.scenes,
            "log_count": int(len(set(dataset.log_names))),
            "candidates_per_scene": dataset.candidates,
            "groups": requested_groups,
            "folds": selected_folds,
            "models": model_kinds,
            "epochs": epochs,
            "heldout_family_audit": run_heldout,
            "fold_result_rows": len(fold_frame),
            "per_scene_result_rows": len(per_scene_frame),
        }
        write_json(result_dir / "job_summary.json", result)
        return result

    aggregated = _aggregate_fold_results(fold_frame)
    primary_model = "mlp" if "mlp" in model_kinds else model_kinds[0]
    primary = {group: aggregated[f"{group}:{primary_model}"] for group in GROUP_NAMES}
    dynamic_gain = primary["O8"]["pairwise_mean"] - primary["O3"]["pairwise_mean"]
    state_gain = primary["O9"]["pairwise_mean"] - primary["O3"]["pairwise_mean"]
    state_retention = state_gain / dynamic_gain if abs(dynamic_gain) > 1e-9 else float("nan")
    regret_reduction = (
        (primary["O3"]["top1_regret_mean"] - primary["O8"]["top1_regret_mean"])
        / max(primary["O3"]["top1_regret_mean"], 1e-9)
    )
    paired = per_scene_frame[
        (per_scene_frame.model == primary_model) & per_scene_frame.group.isin(["O3", "O8"])
    ].pivot(index=["scene_token", "log_name"], columns="group", values="pairwise_accuracy").dropna()
    paired["dynamic_gain"] = paired.O8 - paired.O3
    ci_low, ci_high = log_bootstrap_ci(paired.reset_index(), "dynamic_gain")

    heldout.to_csv(report_dir / "heldout_candidate_type_results.csv", index=False)
    heldout_gain = float(heldout[heldout.group == "gain"].pairwise_accuracy.mean()) if len(heldout) else float("nan")

    controls_limit = primary["O3"]["pairwise_mean"] + 0.01
    factor_improvement = (
        primary["O8"]["collision_auroc_mean"] > primary["O3"]["collision_auroc_mean"] + 0.01
        or primary["O8"]["ttc_auroc_mean"] > primary["O3"]["ttc_auroc_mean"] + 0.01
    )
    criteria = {
        "dynamic_pairwise_gain_at_least_0p03": dynamic_gain >= 0.03,
        "bootstrap_ci_lower_above_zero": ci_low > 0,
        "top1_regret_reduction_at_least_20pct": regret_reduction >= 0.20,
        "collision_or_ttc_improved": factor_improvement,
        "within_scene_shuffle_gain_disappears": primary["O10"]["pairwise_mean"] <= controls_limit,
        "cross_scene_shuffle_gain_disappears": primary["O11"]["pairwise_mean"] <= controls_limit,
        "random_dimension_control_fails": primary["O12"]["pairwise_mean"] <= controls_limit,
        "state_recomputed_risk_retention_at_least_0p40": state_retention >= 0.40,
        "heldout_candidate_types_positive_gain": heldout_gain > 0,
    }
    passed = all(criteria.values())
    status = "PASS" if passed else "FAIL"
    interpretation = (
        "shared future state world-model supervision is supported"
        if passed
        else (
            "evaluation-metric distillation only; shared future state is not supported"
            if primary["O5"]["pairwise_mean"] > primary["O4"]["pairwise_mean"] + 0.02 and state_retention < 0.40
            else "oracle dynamic evidence is incomplete under Gate C1"
        )
    )
    result = {
        "mode": args.mode,
        "scene_count": dataset.scenes,
        "log_count": int(len(set(dataset.log_names))),
        "candidates_per_scene": dataset.candidates,
        "folds": available_folds,
        "models": model_kinds,
        "primary_model": primary_model,
        "aggregated": aggregated,
        "primary": primary,
        "dynamic_gain": dynamic_gain,
        "state_recomputed_risk_gain_retention": state_retention,
        "top1_regret_reduction": regret_reduction,
        "dynamic_gain_log_bootstrap_95ci": [ci_low, ci_high],
        "heldout_candidate_type_gain": heldout_gain,
        "criteria": criteria,
        "gate_c1": status,
        "interpretation": interpretation,
        "leakage_audit": {
            "official_score_in_features": False,
            "official_factor_in_features": False,
            "targets_only": ["aggregate score", "official factor labels"],
        },
    }
    output_name = "oracle_decomposition_results.json" if args.mode == "full" else f"oracle_decomposition_{args.mode}_results.json"
    write_json(report_dir / output_name, result)
    if args.mode == "full":
        update_gate(report_dir, "gate_c1", {"passed": passed, **result})
    else:
        update_gate(report_dir, f"gate_c1_{args.mode}", {"passed": passed, **result})
    rows = []
    for group in GROUP_NAMES:
        item = primary[group]
        rows.append(
            f"| {group} | {item['pairwise_mean']:.4f} ± {item['pairwise_std']:.4f} | "
            f"{item['top1_regret_mean']:.4f} | {item['collision_auroc_mean']:.4f} | {item['ttc_auroc_mean']:.4f} |"
        )
    write_markdown(
        report_dir / "ORACLE_DECOMPOSITION_REPORT.md",
        f"""# Oracle Dynamic-value Decomposition

## Gate C1 ({args.mode}): {status}

| Group | Pairwise accuracy | Top-1 regret | Collision AUROC | TTC AUROC |
|---|---:|---:|---:|---:|
{chr(10).join(rows)}

- Scenes/logs/K: {dataset.scenes} / {len(set(dataset.log_names))} / {dataset.candidates}
- Primary model: `{primary_model}`; both requested capacities: {', '.join(model_kinds)}
- Dynamic gain O8−O3: {dynamic_gain:.4f}
- Log-bootstrap 95% CI: [{ci_low:.4f}, {ci_high:.4f}]
- Top-1 regret reduction: {regret_reduction:.2%}
- State/recomputed-risk retention R_state: {state_retention:.3f}
- Held-out candidate-family gain: {heldout_gain:.4f}
- Interpretation: {interpretation}

`D_risk` consists of physical collision/clearance/TTC relabels and never includes
the official aggregate or factor score. O9 recomputes collision/TTC only from
actor relative state and mask. O10/O11/O12/O13 are permutation, cross-scene,
random-dimensional and repeated-static controls respectively.
""",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("smoke", "pilot", "full"), default="smoke")
    parser.add_argument("--num-scenes", type=int, default=0)
    parser.add_argument("--groups", default=",".join(GROUP_NAMES))
    parser.add_argument("--folds", default="")
    parser.add_argument("--models", default="linear,mlp")
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--batch-scenes", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=20260828)
    parser.add_argument("--job-name", default="")
    parser.add_argument("--store-dir", type=Path)
    parser.add_argument("--job-root", type=Path)
    parser.add_argument(
        "--heldout",
        action="store_true",
        help="Run the expensive held-out candidate-family audit in this partial job.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    args = parser.parse_args()
    run(args)
    append_command(args.output_dir, "python -m tools.shared_future_candidate_consequence.run_oracle_decomposition " + " ".join(sys.argv[1:]))


if __name__ == "__main__":
    main()

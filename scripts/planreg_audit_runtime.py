"""Shared exact-runtime helpers for PlanReg model audits.

This module is intentionally importable by scripts, not part of deployment.
All model audits default to FP32 and restore the complete Lightning checkpoint
strictly into the exact formal training topology.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data._utils.collate import default_collate

from navsim.common.dataclasses import SceneFilter
from navsim.common.dataloader import MetricCacheLoader, SceneLoader
from navsim.planning.training.dataset import Dataset


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_formal_training_agent(
    resolved_config: Path,
    checkpoint: Path,
    *,
    device: torch.device,
    compute_dtype: str = "float32",
):
    os.environ.setdefault("DRIVEVLA_SCORE_RAY", "0")
    os.environ.setdefault("DRIVEVLA_SCORE_PROCESSES", "0")
    os.environ.setdefault("LOCAL_RANK", str(device.index or 0))
    cfg = OmegaConf.load(resolved_config)
    OmegaConf.update(cfg, "agent.vlm_config.compute_dtype", compute_dtype, force_add=True)
    OmegaConf.update(cfg, "agent.vlm_config.gradient_checkpointing", False, force_add=True)
    OmegaConf.update(cfg, "agent.batch_size", 1, force_add=True)
    OmegaConf.update(cfg, "agent.num_gpus", 1, force_add=True)
    OmegaConf.update(cfg, "agent.checkpoint_path", None, force_add=True)
    OmegaConf.update(cfg, "agent.stage1_checkpoint_path", None, force_add=True)
    # The shared-init artifact is intentionally dtype-exact BF16.  A full
    # checkpoint audit in FP32 must construct the identical topology without
    # first applying that BF16 artifact, then strictly restore every tensor
    # from the full checkpoint below.  This is not a different initialization:
    # no forward occurs before strict restoration.
    OmegaConf.update(cfg, "agent.initialization", None, force_add=True)
    agent = instantiate(cfg.agent)
    agent.initialize()
    try:
        payload = torch.load(
            checkpoint, map_location="cpu", weights_only=False, mmap=True
        )
    except TypeError:  # pragma: no cover
        payload = torch.load(checkpoint, map_location="cpu")
    lightning_state = payload.get("state_dict", payload)
    if not lightning_state or not all(name.startswith("agent.") for name in lightning_state):
        raise RuntimeError("Expected an exact AgentLightningModule state_dict with agent.* keys")
    state = {name[len("agent."):]: value for name, value in lightning_state.items()}
    migrated_ema = False
    if agent.ema_register_target is not None:
        migrated_ema = agent.ema_register_target.migrate_legacy_state_dict(
            state, "ema_register_target."
        )
    incompatible = agent.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"Strict checkpoint restoration failed: {incompatible}")
    agent.to(device)
    return cfg, agent, {
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_epoch": int(payload.get("epoch", -1)),
        "checkpoint_global_step": int(payload.get("global_step", -1)),
        "compute_dtype": compute_dtype,
        "strict_checkpoint_restore": True,
        "legacy_bf16_ema_history_unrecoverable": bool(migrated_ema),
    }


def select_representative_tokens(
    candidate_bank: Path,
    metric_cache: Path,
    count: int,
) -> Tuple[List[str], Dict[str, dict]]:
    if count < 4:
        raise ValueError("Representative audit requires at least four scenes")
    with np.load(candidate_bank, allow_pickle=False) as payload:
        tokens = np.asarray(payload["tokens"])
        scores = np.asarray(payload["candidate_scores"], dtype=np.float64)[..., -1]
        selected_indices = np.asarray(payload["selected_indices"], dtype=np.int64)
    rows = np.arange(len(tokens))
    selected = scores[rows, selected_indices]
    oracle = scores.max(axis=1)
    regret = oracle - selected
    loader = MetricCacheLoader(metric_cache)
    token_logs = {
        str(token): Path(loader.metric_cache_paths[str(token)]).relative_to(metric_cache).parts[0]
        for token in tokens
    }
    categories = [
        ("highest_regret", np.argsort(-regret)),
        ("hardest_oracle", np.argsort(oracle)),
        ("strong_selected", np.argsort(-selected)),
        ("typical", np.argsort(np.abs(selected - np.median(selected)))),
    ]
    chosen: List[str] = []
    metadata: Dict[str, dict] = {}
    used_logs = set()
    per_category = max(1, math_ceil_div(count, len(categories)))
    for category, order in categories:
        added = 0
        for index in order:
            token = str(tokens[index])
            log = token_logs[token]
            if token in metadata or log in used_logs:
                continue
            chosen.append(token)
            used_logs.add(log)
            metadata[token] = {
                "category": category,
                "log_name": log,
                "selected_pdms_epoch27": float(selected[index]),
                "offline_oracle_at_64_epoch27": float(oracle[index]),
                "scorer_regret_epoch27": float(regret[index]),
            }
            added += 1
            if added >= per_category or len(chosen) >= count:
                break
        if len(chosen) >= count:
            break
    if len(chosen) < count:
        for index in np.argsort(-regret):
            token = str(tokens[index])
            if token in metadata:
                continue
            chosen.append(token)
            metadata[token] = {
                "category": "fill",
                "log_name": token_logs[token],
                "selected_pdms_epoch27": float(selected[index]),
                "offline_oracle_at_64_epoch27": float(oracle[index]),
                "scorer_regret_epoch27": float(regret[index]),
            }
            if len(chosen) >= count:
                break
    return chosen, metadata


def math_ceil_div(numerator: int, denominator: int) -> int:
    return (numerator + denominator - 1) // denominator


def build_navtest_samples(
    agent,
    tokens: Sequence[str],
    token_metadata: Dict[str, dict],
    *,
    navsim_log_path: Path,
    sensor_blobs_path: Path,
) -> Dict[str, Tuple[dict, dict]]:
    log_names = sorted({token_metadata[token]["log_name"] for token in tokens})
    scene_filter = SceneFilter(
        num_history_frames=4,
        num_future_frames=10,
        frame_interval=1,
        has_route=True,
        log_names=log_names,
        tokens=list(tokens),
    )
    loader = SceneLoader(
        sensor_blobs_path=sensor_blobs_path,
        data_path=navsim_log_path,
        scene_filter=scene_filter,
        sensor_config=agent.get_sensor_config(),
        load_image_path=True,
    )
    dataset = Dataset(
        scene_loader=loader,
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        cache_path=None,
        force_cache_computation=False,
    )
    samples = {}
    for index, token in enumerate(loader.tokens):
        features, targets = dataset[index]
        samples[str(token)] = (features, targets)
    missing = sorted(set(tokens) - set(samples))
    if missing:
        raise RuntimeError(f"Navtest loader did not return selected tokens: {missing}")
    return samples


def collate_samples(
    samples: Iterable[Tuple[dict, dict]],
) -> Tuple[Dict[str, object], Dict[str, object]]:
    sample_list = list(samples)
    if not sample_list:
        raise ValueError("Cannot collate an empty audit batch")
    feature_rows = [dict(features) for features, _ in sample_list]
    path_rows = [row.pop("image_path_tensor") for row in feature_rows]
    features = default_collate(feature_rows)
    features["image_path_tensor"] = pad_sequence(
        path_rows, batch_first=True, padding_value=0
    )
    features["image_path_length"] = torch.as_tensor(
        [len(value) for value in path_rows], dtype=torch.long
    )
    targets = default_collate([targets for _, targets in sample_list])
    return features, targets


def to_device_non_paths(value, device: torch.device):
    """Mirror Lightning transfer while keeping path/prompt fields on host."""
    if isinstance(value, torch.Tensor):
        return value.to(device=device, non_blocking=True)
    if isinstance(value, dict):
        return {name: to_device_non_paths(item, device) for name, item in value.items()}
    if isinstance(value, list):
        return [to_device_non_paths(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(to_device_non_paths(item, device) for item in value)
    return value


__all__ = [
    "build_navtest_samples",
    "collate_samples",
    "load_formal_training_agent",
    "select_representative_tokens",
    "sha256_file",
]

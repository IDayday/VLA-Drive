#!/usr/bin/env python3
"""Run the teacher-oracle and student-inheritance gates for VGGT V3.

This is intentionally a local, action-only threshold test rather than a full
NAVSIM/PDMS evaluation.  It freezes the selected V2 checkpoint, retains all
195 VGGT slots, and trains only the V3 residual reader.  Correct and hard-
shuffled scenes share the exact same flow-matching noise and timestep, so the
reported causal gap cannot be attributed to diffusion sampling noise.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from infer import VLAAgent  # noqa: E402
from starVLA.dataloader.navsim_dataset import NavSimDataset, collate_fn  # noqa: E402
from starVLA.model.modules.vggt_query.planning_heads import (  # noqa: E402
    V3ResidualGeometryFusion,
)


SCHEMA_VERSION = 1


def _atomic_torch_save(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _atomic_json_dump(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _evenly_spaced_indices(length: int, count: int) -> list[int]:
    if not 0 < count <= length:
        raise ValueError(f"sample count must be in [1,{length}], found {count}")
    return np.linspace(0, length - 1, count, dtype=np.int64).tolist()


def _select_indices(length: int, train_count: int, validation_count: int, seed: int):
    validation = _evenly_spaced_indices(length, validation_count)
    validation_set = set(validation)
    candidates = np.asarray(
        [index for index in range(length) if index not in validation_set],
        dtype=np.int64,
    )
    if train_count > len(candidates):
        raise ValueError("not enough samples after reserving validation indices")
    generator = np.random.default_rng(seed)
    train = generator.choice(candidates, size=train_count, replace=False).tolist()
    return train, validation


def _build_dataset(agent: VLAAgent, args) -> NavSimDataset:
    config = deepcopy(agent.model_config)
    config.datasets.video_data.load_2d_data = 0
    config.datasets.gs_data.load_3d_data = 0
    config.datasets.reward_data.load_reward_data = 0
    config.datasets.vla_data.w_neg_traj = None
    config.w_depth = 0
    config.enable_image_aug = 0
    config.framework.vggt.cache.enabled = True
    config.framework.vggt.cache.root = str(args.vggt_cache_root)
    config.framework.vggt.cache.strict = True
    # A global Qwen cache has a different prompt contract.  V3 precomputes its
    # compact experiment-local features from raw images exactly once.
    os.environ.pop("NAVSIM_FEATURE_CACHE_ROOT", None)
    os.environ.pop("NAVSIM_CACHE_COMPONENTS", None)
    os.environ.pop("NAVSIM_AGENT_DINO_CACHE_ROOT", None)
    return NavSimDataset(
        datalist_path=str(args.datalist_path),
        split="train",
        video_data_cfg=config.datasets.video_data,
        gs_data_cfg=config.datasets.gs_data,
        reward_data_cfg=config.datasets.reward_data,
        ver_1225=config.ver_1225,
        dataset_cfg=config.datasets.vla_data,
        all_cfg=config,
        data_root=str(args.data_root),
    )


@torch.inference_mode()
def _extract_batch(model, examples: list[dict]) -> dict[str, Any]:
    instructions = [
        example["lang"] + model._build_action_prompt_suffix() for example in examples
    ]
    (
        input_ids,
        attention_mask,
        position_ids,
        token_positions,
        image_embeds,
        deepstack_embeds,
        image_grid_thw,
    ) = model._build_qwen_batch(examples, instructions)

    with torch.autocast("cuda", dtype=torch.bfloat16):
        text_embeds = model.qwen_vl_interface.model.get_input_embeddings()(input_ids)
    state_device = next(model.action_input_model.parameters()).device
    states = torch.as_tensor(
        np.asarray([example["state"] for example in examples]),
        device=state_device,
        dtype=torch.float32,
    )[:, 0, :]
    states_embed = model.action_input_model(states).to(
        device=text_embeds.device, dtype=text_embeds.dtype
    )
    batch_indices = torch.arange(len(examples), device=text_embeds.device)
    text_embeds[
        batch_indices, token_positions["history"][:, 0], :
    ] = states_embed
    with torch.autocast("cuda", dtype=torch.bfloat16):
        last_hidden = model._qwen_language_forward(
            input_ids=input_ids,
            inputs_embeds=text_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            image_embeds=image_embeds,
            deepstack_embeds=deepstack_embeds,
        )

    action_queries = model._gather_queries(last_hidden, token_positions["action"])
    raw_memory, student_valid = model._build_student_memory(
        last_hidden,
        input_ids=input_ids,
        image_grid_thw=image_grid_thw,
        global_positions=token_positions["vggt"],
    )
    student_memory = model.vggt_aligner.project_student(raw_memory).float()

    teacher_features, teacher_masks = [], []
    for example in examples:
        payload = example["vggt_query_feature_cache"]
        features = payload["features"].float()
        mask = payload["valid_mask"].bool()
        active_mask = payload.get("active_slot_mask")
        if active_mask is not None:
            mask = mask & active_mask.bool()
        teacher_features.append(features)
        teacher_masks.append(mask)
    teacher_memory = F.layer_norm(
        torch.stack(teacher_features).to(device=last_hidden.device),
        (model.vggt_layout.teacher_dim,),
    )
    teacher_valid = torch.stack(teacher_masks).to(
        device=last_hidden.device, dtype=torch.bool
    )
    valid_mask = teacher_valid & student_valid

    return {
        "action_queries": action_queries.detach().to(torch.bfloat16).cpu(),
        "student_memory": student_memory.detach().to(torch.bfloat16).cpu(),
        "teacher_memory": teacher_memory.detach().to(torch.bfloat16).cpu(),
        "valid_mask": valid_mask.detach().cpu(),
        "actions": torch.as_tensor(
            np.asarray([example["action"] for example in examples]),
            dtype=torch.float32,
        ),
        "tokens": [str(example["token"]) for example in examples],
    }


def _concatenate(parts: list[dict[str, Any]]) -> dict[str, Any]:
    tensor_names = (
        "action_queries",
        "student_memory",
        "teacher_memory",
        "valid_mask",
        "actions",
    )
    return {
        **{name: torch.cat([part[name] for part in parts], dim=0) for name in tensor_names},
        "tokens": sum((part["tokens"] for part in parts), []),
    }


def _precompute_split(
    model,
    dataset: NavSimDataset,
    indices: list[int],
    batch_size: int,
    workers: int,
    description: str,
) -> dict[str, Any]:
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        collate_fn=collate_fn,
    )
    parts = []
    for examples in tqdm(loader, desc=description):
        parts.append(_extract_batch(model, examples))
    return _concatenate(parts)


def _feature_identity(agent: VLAAgent, args, train_indices, validation_indices):
    checkpoint = Path(agent.model_path).resolve()
    return {
        "schema_version": SCHEMA_VERSION,
        "checkpoint": str(checkpoint),
        "checkpoint_size": checkpoint.stat().st_size,
        "checkpoint_mtime_ns": checkpoint.stat().st_mtime_ns,
        "base_vlm": str(Path(args.base_vlm).resolve()),
        "datalist": str(Path(args.datalist_path).resolve()),
        "datalist_sha256": _sha256_file(Path(args.datalist_path)),
        "vggt_cache_root": str(Path(args.vggt_cache_root).resolve()),
        "train_indices": train_indices,
        "validation_indices": validation_indices,
    }


def _load_or_precompute_features(agent: VLAAgent, args) -> dict[str, Any]:
    dataset = _build_dataset(agent, args)
    train_indices, validation_indices = _select_indices(
        len(dataset), args.train_samples, args.validation_samples, args.sample_seed
    )
    identity = _feature_identity(agent, args, train_indices, validation_indices)
    feature_path = Path(args.feature_file)
    if feature_path.is_file():
        value = torch.load(feature_path, map_location="cpu", weights_only=False)
        if value.get("identity") == identity:
            print(f"[V3 gate] reusing features: {feature_path}")
            return value
        if not args.overwrite_features:
            raise RuntimeError(
                f"Feature identity changed for {feature_path}; pass --overwrite-features"
            )
    value = {
        "identity": identity,
        "train": _precompute_split(
            agent.model,
            dataset,
            train_indices,
            args.feature_batch_size,
            args.workers,
            "V3 train features",
        ),
        "validation": _precompute_split(
            agent.model,
            dataset,
            validation_indices,
            args.feature_batch_size,
            args.workers,
            "V3 validation features",
        ),
    }
    _atomic_torch_save(value, feature_path)
    print(f"[V3 gate] wrote reusable features: {feature_path}")
    return value


def _hard_negative_indices(actions: torch.Tensor) -> torch.Tensor:
    """Select the closest different trajectory, limiting route-label shortcuts."""

    if actions.shape[0] < 2:
        raise ValueError("hard-negative pairing needs batch size >= 2")
    flattened = actions.float().flatten(1)
    distance = torch.cdist(flattened, flattened).square()
    distance.fill_diagonal_(float("inf"))
    return distance.argmin(dim=1)


def _sample_flow_inputs(head, actions: torch.Tensor, generator: torch.Generator):
    noise = torch.randn(
        actions.shape,
        device=actions.device,
        dtype=actions.dtype,
        generator=generator,
    )
    # sample_time uses torch.distributions, whose sampler has no generator
    # argument.  A local RNG fork keeps the paired calls reproducible.
    beta_sample = torch._standard_gamma(
        torch.full((actions.shape[0],), head.config.noise_beta_alpha, device=actions.device),
        generator=generator,
    )
    beta_other = torch._standard_gamma(
        torch.full((actions.shape[0],), head.config.noise_beta_beta, device=actions.device),
        generator=generator,
    )
    sampled = beta_sample / (beta_sample + beta_other).clamp_min(1e-8)
    t = (head.config.noise_s - sampled) / head.config.noise_s
    t = t.to(dtype=actions.dtype)
    return noise, t


def _flow_loss_per_sample(
    head,
    condition: torch.Tensor,
    actions: torch.Tensor,
    noise: torch.Tensor,
    t: torch.Tensor,
) -> torch.Tensor:
    noisy_trajectory = (1 - t[:, None, None]) * noise + t[:, None, None] * actions
    velocity = actions - noise
    timestep = (t * head.num_timestep_buckets).long()
    action_features = head.action_encoder(noisy_trajectory, timestep)
    if head.config.add_pos_embed:
        positions = torch.arange(
            action_features.shape[1], device=actions.device, dtype=torch.long
        )
        action_features = action_features + head.position_embedding(positions).unsqueeze(0)

    projected = head.qwen_proj(condition.float())
    output = head.model(
        hidden_states=action_features,
        encoder_hidden_states=projected,
        timestep=timestep,
        return_all_hidden_states=False,
    )
    prediction = head.action_decoder(output)
    return (prediction - velocity).square().flatten(1).mean(1)


def _paired_flow_loss(
    head,
    real_condition: torch.Tensor,
    shuffled_condition: torch.Tensor,
    actions: torch.Tensor,
    *,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    noise, t = _sample_flow_inputs(head, actions, generator)
    return (
        _flow_loss_per_sample(head, real_condition, actions, noise, t),
        _flow_loss_per_sample(head, shuffled_condition, actions, noise, t),
    )


@torch.inference_mode()
def _predict_with_noise(head, condition: torch.Tensor, initial_noise: torch.Tensor):
    actions = initial_noise.clone()
    projected = head.qwen_proj(condition.float())
    step_size = 1.0 / head.num_inference_timesteps
    for step in range(head.num_inference_timesteps):
        discrete = int(step / float(head.num_inference_timesteps) * head.num_timestep_buckets)
        timestep = torch.full(
            (actions.shape[0],), discrete, device=actions.device, dtype=torch.long
        )
        action_features = head.action_encoder(actions, timestep)
        if head.config.add_pos_embed:
            positions = torch.arange(
                action_features.shape[1], device=actions.device, dtype=torch.long
            )
            action_features = action_features + head.position_embedding(positions).unsqueeze(0)
        output = head.model(
            hidden_states=action_features,
            encoder_hidden_states=projected,
            timestep=timestep,
            return_all_hidden_states=False,
        )
        actions = actions + step_size * head.action_decoder(output)
    return actions


def _bootstrap_ci(values: np.ndarray, seed: int, draws: int = 2000):
    values = np.asarray(values, dtype=np.float64)
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(values), size=(draws, len(values)))
    means = values[indices].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def _summary(values: Iterable[float], seed: int) -> dict[str, Any]:
    array = np.asarray(list(values), dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "ci95_mean": _bootstrap_ci(array, seed),
        "count": int(array.size),
    }


def _slot_mean(args) -> torch.Tensor:
    component = Path(args.vggt_cache_root) / "vggt_query"
    manifest = json.loads((component / "manifest.json").read_text(encoding="utf-8"))
    statistics_path = component / manifest["slot_statistics_file"]
    if _sha256_file(statistics_path) != manifest["slot_statistics_sha256"]:
        raise RuntimeError("VGGT slot-statistics checksum changed")
    statistics = torch.load(statistics_path, map_location="cpu", weights_only=True)
    return F.layer_norm(statistics["slot_mean"].float(), (1024,))


def _train_gate(agent: VLAAgent, features, args):
    model = agent.model
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    gate = V3ResidualGeometryFusion(
        action_dim=model.qwen_vl_interface.model.config.hidden_size,
        memory_dim=model.vggt_layout.teacher_dim,
        num_heads=args.attention_heads,
        layout=model.vggt_layout,
        minimum_scale=args.minimum_scale,
        maximum_scale=args.maximum_scale,
        initial_scale=args.initial_scale,
        reference_memory=_slot_mean(args),
    ).to(agent.device)
    gate.reader.load_state_dict(model.vggt_waypoint_reader.state_dict())
    gate.train()
    trainable = sum(parameter.numel() for parameter in gate.parameters())
    optimizer = torch.optim.AdamW(
        gate.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.steps, eta_min=args.learning_rate * 0.1
    )
    source = features["train"]
    random_generator = torch.Generator(device=agent.device).manual_seed(args.seed)
    index_generator = torch.Generator().manual_seed(args.seed)
    history = []
    started = time.time()
    for step in range(1, args.steps + 1):
        selected = torch.randperm(
            source["actions"].shape[0], generator=index_generator
        )[: args.batch_size]
        action_queries = source["action_queries"][selected].to(
            agent.device, non_blocking=True
        )
        memory = source["teacher_memory"][selected].to(agent.device, non_blocking=True)
        valid_mask = source["valid_mask"][selected].to(agent.device, non_blocking=True)
        actions = source["actions"][selected].to(agent.device, non_blocking=True)
        negative = _hard_negative_indices(actions)

        real_condition, diagnostics = gate(action_queries, memory, valid_mask)
        shuffled_condition, _ = gate(
            action_queries, memory[negative], valid_mask[negative]
        )
        real_loss, shuffled_loss = _paired_flow_loss(
            model.action_model,
            real_condition,
            shuffled_condition,
            actions,
            generator=random_generator,
        )
        relative_margin = args.margin_fraction * real_loss.detach().clamp_min(0.1)
        ranking = F.relu(relative_margin + real_loss - shuffled_loss)
        loss = real_loss.mean() + args.margin_weight * ranking.mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(gate.parameters(), args.max_grad_norm)
        optimizer.step()
        scheduler.step()
        if step == 1 or step % args.log_interval == 0 or step == args.steps:
            record = {
                "step": step,
                "loss": float(loss.detach()),
                "real_flow": float(real_loss.mean().detach()),
                "shuffled_flow": float(shuffled_loss.mean().detach()),
                "ranking": float(ranking.mean().detach()),
                "alpha": float(gate.residual_scale.detach()),
                "residual_to_action_norm": float(
                    diagnostics["residual_to_action_norm_ratio"]
                ),
                "learning_rate": scheduler.get_last_lr()[0],
                "elapsed_seconds": time.time() - started,
            }
            history.append(record)
            print("[V3 train] " + json.dumps(record, sort_keys=True))
    gate.eval()
    return gate, trainable, history


@torch.inference_mode()
def _evaluate_gate(agent, gate, features, args):
    source = features["validation"]
    head = agent.model.action_model
    slot_mean = _slot_mean(args)
    records = []
    generator = torch.Generator(device=agent.device).manual_seed(args.validation_seed)
    for start in tqdm(
        range(0, source["actions"].shape[0], args.validation_batch_size),
        desc="V3 held-out interventions",
    ):
        stop = min(start + args.validation_batch_size, source["actions"].shape[0])
        selected = slice(start, stop)
        action_queries = source["action_queries"][selected].to(agent.device)
        teacher = source["teacher_memory"][selected].to(agent.device)
        student = source["student_memory"][selected].to(agent.device)
        valid = source["valid_mask"][selected].to(agent.device)
        actions = source["actions"][selected].to(agent.device)
        negative = _hard_negative_indices(actions)
        initial_noise = torch.randn(
            actions.shape,
            device=agent.device,
            dtype=actions.dtype,
            generator=generator,
        )
        memory_variants = {
            "teacher_real": (teacher, valid),
            "teacher_shuffled": (teacher[negative], valid[negative]),
            "teacher_slot_mean": (
                slot_mean.to(agent.device, dtype=teacher.dtype).unsqueeze(0).expand_as(teacher),
                valid,
            ),
            "teacher_zero": (torch.zeros_like(teacher), valid),
            "student_real": (student, valid),
            "student_shuffled": (student[negative], valid[negative]),
        }
        conditions = {"base": action_queries}
        diagnostics = {}
        for name, (memory, mask) in memory_variants.items():
            conditions[name], diagnostic = gate(action_queries, memory, mask)
            diagnostics[name] = diagnostic

        # One shared noise/timestep sequence for every intervention.
        flow_noise, flow_time = _sample_flow_inputs(head, actions, generator)
        flow_losses = {}
        for name, condition in conditions.items():
            flow_losses[name] = _flow_loss_per_sample(
                head, condition, actions, flow_noise, flow_time
            )
        predictions = {
            name: _predict_with_noise(head, condition, initial_noise)
            for name, condition in conditions.items()
        }
        for local_index in range(stop - start):
            truth_xy = actions[local_index, :, :2]
            sample = {
                "token": source["tokens"][start + local_index],
                "flow": {
                    name: float(value[local_index]) for name, value in flow_losses.items()
                },
                "ade": {
                    name: float(
                        (prediction[local_index, :, :2] - truth_xy)
                        .square()
                        .sum(-1)
                        .sqrt()
                        .mean()
                    )
                    for name, prediction in predictions.items()
                },
                "trajectory_l2_from_teacher_real": {
                    name: float(
                        (prediction[local_index] - predictions["teacher_real"][local_index])
                        .square()
                        .sum(-1)
                        .sqrt()
                        .mean()
                    )
                    for name, prediction in predictions.items()
                },
            }
            records.append(sample)
    return records


def _metric(records, getter, seed):
    return _summary((getter(record) for record in records), seed)


def _build_report(agent, gate, trainable, history, records, features, args):
    epsilon = 1e-8
    teacher_flow_gap = _metric(
        records,
        lambda r: (r["flow"]["teacher_shuffled"] - r["flow"]["teacher_real"])
        / max(r["flow"]["teacher_real"], epsilon),
        args.validation_seed + 1,
    )
    student_flow_gap = _metric(
        records,
        lambda r: (r["flow"]["student_shuffled"] - r["flow"]["student_real"])
        / max(r["flow"]["student_real"], epsilon),
        args.validation_seed + 2,
    )
    teacher_trajectory_gap = _metric(
        records,
        lambda r: r["trajectory_l2_from_teacher_real"]["teacher_shuffled"]
        / max(r["ade"]["teacher_real"], epsilon),
        args.validation_seed + 3,
    )
    teacher_wrong_ade = _metric(
        records,
        lambda r: r["ade"]["teacher_shuffled"] - r["ade"]["teacher_real"],
        args.validation_seed + 4,
    )
    teacher_flow_utility = _metric(
        records,
        lambda r: (r["flow"]["base"] - r["flow"]["teacher_real"])
        / max(r["flow"]["base"], epsilon),
        args.validation_seed + 5,
    )
    teacher_ade_utility = _metric(
        records,
        lambda r: r["ade"]["base"] - r["ade"]["teacher_real"],
        args.validation_seed + 6,
    )
    student_teacher_flow = _metric(
        records,
        lambda r: (r["flow"]["student_real"] - r["flow"]["teacher_real"])
        / max(r["flow"]["teacher_real"], epsilon),
        args.validation_seed + 7,
    )
    intervention = {
        "teacher_shuffled_relative_flow_gap": teacher_flow_gap,
        "student_shuffled_relative_flow_gap": student_flow_gap,
        "teacher_shuffled_trajectory_l2_over_real_ade": teacher_trajectory_gap,
        "teacher_shuffled_minus_real_ade": teacher_wrong_ade,
        "teacher_real_relative_flow_gain_over_base": teacher_flow_utility,
        "base_minus_teacher_real_ade": teacher_ade_utility,
        "student_minus_teacher_relative_flow": student_teacher_flow,
        "flow_mean": {
            name: float(np.mean([record["flow"][name] for record in records]))
            for name in records[0]["flow"]
        },
        "ade_mean": {
            name: float(np.mean([record["ade"][name] for record in records]))
            for name in records[0]["ade"]
        },
    }
    thresholds = {
        "teacher_scene_flow_gap_mean_min": args.scene_flow_threshold,
        "teacher_scene_flow_gap_ci_lower_min": 0.0,
        "teacher_trajectory_gap_median_min": args.trajectory_threshold,
        "teacher_wrong_ade_ci_lower_min": 0.0,
        "teacher_flow_utility_mean_min": args.utility_flow_threshold,
        "teacher_flow_utility_ci_lower_min": 0.0,
        "student_teacher_retention_min": args.student_retention_threshold,
        "student_teacher_flow_degradation_max": args.student_flow_degradation_threshold,
    }
    teacher_scene_flow_pass = (
        teacher_flow_gap["mean"] >= args.scene_flow_threshold
        and teacher_flow_gap["ci95_mean"][0] > 0.0
    )
    teacher_trajectory_response_pass = (
        teacher_trajectory_gap["median"] >= args.trajectory_threshold
    )
    teacher_wrong_ade_pass = teacher_wrong_ade["ci95_mean"][0] > 0.0
    oracle_causal = (
        teacher_scene_flow_pass
        and teacher_trajectory_response_pass
        and teacher_wrong_ade_pass
    )
    teacher_utility_flow_pass = (
        teacher_flow_utility["mean"] >= args.utility_flow_threshold
        and teacher_flow_utility["ci95_mean"][0] > 0.0
    )
    teacher_utility_ade_pass = teacher_ade_utility["ci95_mean"][0] > 0.0
    oracle_utility = teacher_utility_flow_pass and teacher_utility_ade_pass
    inherited_ratio = student_flow_gap["mean"] / max(teacher_flow_gap["mean"], epsilon)
    student_scene_flow_pass = student_flow_gap["ci95_mean"][0] > 0.0
    student_retention_pass = inherited_ratio >= args.student_retention_threshold
    student_quality_pass = (
        student_teacher_flow["mean"] <= args.student_flow_degradation_threshold
    )
    student_inheritance = (
        teacher_scene_flow_pass
        and teacher_trajectory_response_pass
        and student_scene_flow_pass
        and student_retention_pass
        and student_quality_pass
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "design": {
            "memory_slots": 195,
            "layout": "15 global + 3 views * 6 rows * 10 columns",
            "planner_entry": (
                "A + bounded_alpha * (CrossAttention(A, structured_R195) - "
                "CrossAttention(A, structured_slot_mean195))"
            ),
            "template_bypass_cancelled": True,
            "extra_context": False,
            "base_model_frozen": True,
            "paired_diffusion_noise_and_timestep": True,
            "hard_negative_policy": "nearest different trajectory within batch",
        },
        "source": {
            "checkpoint": str(Path(agent.model_path).resolve()),
            "checkpoint_step": args.checkpoint_step,
            "base_vlm": str(Path(args.base_vlm).resolve()),
            "vggt_cache": str(Path(args.vggt_cache_root).resolve()),
            "feature_file": str(Path(args.feature_file).resolve()),
        },
        "counts": {
            "train_samples": int(features["train"]["actions"].shape[0]),
            "validation_samples": len(records),
            "v3_trainable_parameters": trainable,
            "frozen_checkpoint_parameters": sum(
                parameter.numel() for parameter in agent.model.parameters()
            ),
        },
        "training": {
            "steps": args.steps,
            "seed": args.seed,
            "history": history,
            "final_residual_scale": float(gate.residual_scale.detach()),
        },
        "interventions": intervention,
        "thresholds": thresholds,
        "gates": {
            "teacher_scene_flow_pass": teacher_scene_flow_pass,
            "teacher_trajectory_response_pass": teacher_trajectory_response_pass,
            "teacher_wrong_memory_ade_pass": teacher_wrong_ade_pass,
            "teacher_oracle_causal": oracle_causal,
            "teacher_utility_flow_pass": teacher_utility_flow_pass,
            "teacher_utility_ade_pass": teacher_utility_ade_pass,
            "teacher_oracle_utility": oracle_utility,
            "student_scene_flow_pass": student_scene_flow_pass,
            "student_teacher_retention_pass": student_retention_pass,
            "student_teacher_quality_pass": student_quality_pass,
            "student_inheritance": student_inheritance,
            "student_teacher_scene_gap_retention": inherited_ratio,
            "ready_for_full_v3_training": bool(
                oracle_causal and oracle_utility and student_inheritance
            ),
        },
        "records": records,
    }


def _validate_paths(args) -> None:
    for path, label in (
        (args.run_dir, "V2 run"),
        (args.base_vlm, "V2 VGGT-token base VLM"),
        (args.vggt_cache_root, "VGGT cache"),
        (args.datalist_path, "training datalist"),
        (args.data_root, "NAVSIM processed data root"),
    ):
        if not Path(path).exists():
            raise FileNotFoundError(f"Missing {label}: {path}")
    if args.batch_size < 2 or args.validation_batch_size < 2:
        raise ValueError("training and validation batch sizes must be >= 2")
    if args.train_samples < args.batch_size:
        raise ValueError("train_samples must be >= batch_size")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-step", type=int, default=75000)
    parser.add_argument("--base-vlm", type=Path, required=True)
    parser.add_argument("--vggt-cache-root", type=Path, required=True)
    parser.add_argument("--datalist-path", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--feature-file", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--train-samples", type=int, default=384)
    parser.add_argument("--validation-samples", type=int, default=96)
    parser.add_argument("--feature-batch-size", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--validation-batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--margin-fraction", type=float, default=0.02)
    parser.add_argument("--margin-weight", type=float, default=1.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--attention-heads", type=int, default=16)
    parser.add_argument("--minimum-scale", type=float, default=0.05)
    parser.add_argument("--maximum-scale", type=float, default=0.50)
    parser.add_argument("--initial-scale", type=float, default=0.10)
    parser.add_argument("--scene-flow-threshold", type=float, default=0.02)
    parser.add_argument("--trajectory-threshold", type=float, default=0.02)
    parser.add_argument("--utility-flow-threshold", type=float, default=0.005)
    parser.add_argument("--student-retention-threshold", type=float, default=0.70)
    parser.add_argument("--student-flow-degradation-threshold", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--sample-seed", type=int, default=20260812)
    parser.add_argument("--validation-seed", type=int, default=20260814)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--overwrite-features", action="store_true")
    parser.add_argument(
        "--precompute-only",
        action="store_true",
        help="write/reuse the frozen feature file and exit before fitting the V3 gate",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _validate_paths(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    # Explicit CLI is the source of truth for this one process; env.local is
    # neither modified nor allowed to select a stale V1 vocabulary bundle.
    os.environ["VGGT_BASE_VLM"] = str(Path(args.base_vlm).resolve())
    os.environ.setdefault("VLM_ATTN_IMPLEMENTATION", "sdpa")
    agent = VLAAgent(
        str(args.run_dir),
        model_iter=args.checkpoint_step,
        device=args.device,
        qwen_forward_mode="auto",
    )
    features = _load_or_precompute_features(agent, args)
    if args.precompute_only:
        print(f"[V3 gate] precompute-only complete: {args.feature_file}")
        return
    gate, trainable, history = _train_gate(agent, features, args)
    records = _evaluate_gate(agent, gate, features, args)
    report = _build_report(agent, gate, trainable, history, records, features, args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_torch_save(
        {
            "schema_version": SCHEMA_VERSION,
            "state_dict": gate.state_dict(),
            "source_checkpoint": str(Path(agent.model_path).resolve()),
            "report_gates": report["gates"],
        },
        args.output_dir / "v3_gate.pt",
    )
    _atomic_json_dump(report, args.output_dir / "report.json")
    print("[V3 gate] " + json.dumps(report["gates"], sort_keys=True))
    print(f"[V3 gate] report: {args.output_dir / 'report.json'}")


if __name__ == "__main__":
    main()

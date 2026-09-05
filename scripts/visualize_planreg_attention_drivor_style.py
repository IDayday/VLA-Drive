#!/usr/bin/env python3
"""DrivOR-style attention audit adapted to single-camera InternViT registers.

The three register-to-patch views are sourced from DrivOR@fc6e5aa:
last-layer head mean, last-three-layer rollout with 0.9 attention + 0.1
identity, and the lowest-entropy head per register.  Our registers live after
CLS inside InternViT, so their indices are 1:17 rather than DrivOR's 0:16.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from navsim.agents.EpisodeDrive.utils.internvl_preprocess import load_image
from planreg_audit_runtime import (
    build_navtest_samples,
    collate_samples,
    load_formal_training_agent,
    select_representative_tokens,
    sha256_file,
)


NUM_REGISTERS = 16
REGISTER_START = 1
REGISTER_STOP = 17


def _qk(attention, hidden_states):
    batch, tokens, dim = hidden_states.shape
    heads = int(attention.num_heads)
    head_dim = dim // heads
    qkv = (
        attention.qkv(hidden_states)
        .reshape(batch, tokens, 3, heads, head_dim)
        .permute(2, 0, 3, 1, 4)
    )
    query, key, _ = qkv.unbind(0)
    if bool(getattr(attention, "qk_normalization", False)):
        query = (
            attention.q_norm(query.transpose(1, 2).flatten(-2, -1))
            .view(batch, tokens, heads, head_dim)
            .transpose(1, 2)
        )
        key = (
            attention.k_norm(key.transpose(1, 2).flatten(-2, -1))
            .view(batch, tokens, heads, head_dim)
            .transpose(1, 2)
        )
    return query, key


@torch.no_grad()
def _register_rows(attention, hidden_states):
    query, key = _qk(attention, hidden_states)
    logits = (query[:, :, REGISTER_START:REGISTER_STOP] * float(attention.scale)) @ key.transpose(-2, -1)
    return logits.softmax(dim=-1)


@torch.no_grad()
def _full_head_mean(attention, hidden_states, chunk_size: int = 96):
    query, key = _qk(attention, hidden_states)
    outputs = []
    tokens = query.shape[-2]
    for start in range(0, tokens, chunk_size):
        stop = min(tokens, start + chunk_size)
        logits = (query[:, :, start:stop] * float(attention.scale)) @ key.transpose(-2, -1)
        row_ids = torch.arange(start, stop, device=logits.device)
        non_register = torch.logical_or(row_ids == 0, row_ids >= REGISTER_STOP)
        if bool(non_register.any()):
            logits[:, :, non_register, REGISTER_START:REGISTER_STOP] = float("-inf")
        outputs.append(logits.softmax(dim=-1).mean(dim=1).cpu())
    return torch.cat(outputs, dim=1)


def _decode_path(path_tensor: torch.Tensor) -> str:
    return "".join(chr(int(value)) for value in path_tensor.tolist() if int(value) != 0)


def _normalize_patch_rows(value: torch.Tensor) -> torch.Tensor:
    return value / value.sum(dim=-1, keepdim=True).clamp_min(1e-12)


def _drivor_maps(register_rows: List[torch.Tensor], full_last_three: Dict[int, torch.Tensor], thumbnail: int):
    final = register_rows[-1][thumbnail]  # [heads, registers, tokens]
    final_patch = _normalize_patch_rows(final[..., REGISTER_STOP:])
    last = _normalize_patch_rows(final_patch.mean(dim=0))
    entropy = -(final_patch.clamp_min(1e-12) * final_patch.clamp_min(1e-12).log()).sum(dim=-1)
    best_heads = entropy.argmin(dim=0)
    best = final_patch[best_heads, torch.arange(NUM_REGISTERS)]

    matrices = []
    for layer in sorted(full_last_three):
        matrix = full_last_three[layer][thumbnail].float()
        identity = torch.eye(matrix.shape[-1], dtype=matrix.dtype)
        matrices.append(0.9 * matrix + 0.1 * identity)
    if len(matrices) != 3:
        raise RuntimeError(f"DrivOR shallow rollout requires exactly three layers, got {len(matrices)}")
    rollout = matrices[0][REGISTER_START:REGISTER_STOP]
    rollout = rollout @ matrices[1]
    rollout = rollout @ matrices[2]
    rollout = _normalize_patch_rows(rollout[:, REGISTER_STOP:])
    patch_count = last.shape[-1]
    grid = math.isqrt(patch_count)
    if grid * grid != patch_count:
        raise RuntimeError(f"InternViT patch count is not square: {patch_count}")
    return {
        "last": last.reshape(NUM_REGISTERS, grid, grid),
        "shallow_rollout": rollout.reshape(NUM_REGISTERS, grid, grid),
        "lowest_entropy_head": best.reshape(NUM_REGISTERS, grid, grid),
    }, best_heads.cpu()


def _plot_register_overlays(image: np.ndarray, maps: torch.Tensor, title: str, output: Path):
    fig, axes = plt.subplots(4, 4, figsize=(15, 10), constrained_layout=True)
    height, width = image.shape[:2]
    for register, axis in enumerate(axes.reshape(-1)):
        heat = maps[register].float().numpy()
        low, high = np.percentile(heat, (1, 99))
        heat = np.clip((heat - low) / max(high - low, 1e-12), 0, 1)
        resized = Image.fromarray(np.uint8(heat * 255)).resize((width, height), Image.Resampling.NEAREST)
        axis.imshow(image)
        axis.imshow(np.asarray(resized), cmap="magma", alpha=0.52, vmin=0, vmax=255)
        axis.set_title(f"register {register}")
        axis.axis("off")
    fig.suptitle(title)
    fig.savefig(output, dpi=170)
    plt.close(fig)


def _plot_patch_energy(image: np.ndarray, maps: torch.Tensor, output: Path):
    energy = maps.mean(dim=0).float().numpy()
    height, width = image.shape[:2]
    energy = (energy - energy.min()) / max(float(np.ptp(energy)), 1e-12)
    resized = Image.fromarray(np.uint8(energy * 255)).resize((width, height), Image.Resampling.NEAREST)
    fig, axis = plt.subplots(figsize=(10, 5))
    axis.imshow(image)
    overlay = axis.imshow(np.asarray(resized), cmap="hot", alpha=0.48, vmin=0, vmax=255)
    axis.set_title("Mean register-to-patch energy (DrivOR-style)")
    axis.axis("off")
    fig.colorbar(overlay, ax=axis, fraction=0.025)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _attention_cosine(maps: torch.Tensor):
    flat = maps.reshape(NUM_REGISTERS, -1).float()
    flat = F.normalize(flat, dim=-1)
    return flat @ flat.T


def _token_geometry(tokens: torch.Tensor) -> dict:
    """Summarize whether nominally distinct slots carry distinct states."""
    if tokens.ndim == 3:
        if tokens.shape[0] != 1:
            raise ValueError(f"Expected a one-scene token tensor, got {tuple(tokens.shape)}")
        tokens = tokens[0]
    value = tokens.detach().float().cpu()
    normalized = F.normalize(value, dim=-1)
    cosine = normalized @ normalized.T
    off_diagonal = cosine[~torch.eye(len(value), dtype=torch.bool)]
    centered = value - value.mean(dim=0, keepdim=True)
    singular = torch.linalg.svdvals(centered)
    energy = singular.square()
    probability = energy / energy.sum().clamp_min(1e-12)
    effective_rank = torch.exp(
        -(probability * probability.clamp_min(1e-12).log()).sum()
    )
    return {
        "effective_rank": float(effective_rank),
        "pairwise_cosine_mean": float(off_diagonal.mean()),
        "pairwise_cosine_p90": float(torch.quantile(off_diagonal, 0.9)),
        "token_feature_std": float(value.std()),
        "centered_rms": float(centered.square().mean().sqrt()),
        "rms": float(value.square().mean().sqrt()),
        "singular_energy_fraction": probability.tolist(),
    }


def _cross_attention_hook(store: dict, name: str):
    @torch.no_grad()
    def hook(module, inputs):
        query = inputs[0]
        key_value = inputs[1]
        _, weights = module.attn(
            query=query,
            key=key_value,
            value=key_value,
            need_weights=True,
            average_attn_weights=False,
        )
        store[name] = weights.detach().float().cpu()
    return hook


def _plot_trajectory_attention(proposals, selected_index, generator_attention, scorer_attention, output: Path):
    fig = plt.figure(figsize=(17, 6), constrained_layout=True)
    grid = fig.add_gridspec(1, 3)
    axis = fig.add_subplot(grid[0, 0])
    dominant = scorer_attention.argmax(dim=-1).numpy()
    colors = plt.get_cmap("tab20")(dominant / 15.0)
    xy = proposals[:, :, :2].numpy()
    for index in range(len(xy)):
        linewidth = 3.0 if index == selected_index else 1.0
        alpha = 1.0 if index == selected_index else 0.35
        axis.plot(xy[index, :, 0], xy[index, :, 1], color=colors[index], alpha=alpha, linewidth=linewidth)
    axis.scatter([0], [0], marker="s", color="black", s=60, label="ego")
    axis.set_aspect("equal")
    axis.set_title(f"64 proposals; selected={selected_index}\ncolor = dominant scene register")
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.grid(alpha=0.25)

    for slot, (name, attention) in enumerate((("generator", generator_attention), ("scorer", scorer_attention)), start=1):
        heat_axis = fig.add_subplot(grid[0, slot])
        image = heat_axis.imshow(attention.numpy(), aspect="auto", cmap="viridis")
        heat_axis.axhline(selected_index, color="red", linewidth=1.5)
        heat_axis.set(xlabel="scene-register slot", ylabel="trajectory query", title=f"Last-layer {name} cross-attention")
        fig.colorbar(image, ax=heat_axis, fraction=0.03)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _plot_composite(image, reg_maps, selected_scorer, selected_generator, output: Path):
    composites = {
        "scorer→scene register→patch (heuristic)": selected_scorer @ reg_maps.reshape(NUM_REGISTERS, -1),
        "generator→scene register→patch (heuristic)": selected_generator @ reg_maps.reshape(NUM_REGISTERS, -1),
    }
    fig, axes = plt.subplots(1, 2, figsize=(16, 5), constrained_layout=True)
    height, width = image.shape[:2]
    for axis, (title, flat) in zip(axes, composites.items()):
        heat = flat.reshape(reg_maps.shape[-2:]).numpy()
        heat = (heat - heat.min()) / max(float(np.ptp(heat)), 1e-12)
        resized = Image.fromarray(np.uint8(heat * 255)).resize((width, height), Image.Resampling.NEAREST)
        axis.imshow(image)
        axis.imshow(np.asarray(resized), cmap="magma", alpha=0.55, vmin=0, vmax=255)
        axis.set_title(title)
        axis.axis("off")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _tile_attention(agent, per_tile_registers, metadata):
    adapter = agent.backbone.planning_register_adapter
    thumbnail_mask = metadata[:, 4] > 0.5
    thumbnail_indices = thumbnail_mask.nonzero(as_tuple=False).flatten()
    if thumbnail_indices.numel() != 1:
        raise RuntimeError("Expected exactly one InternVL thumbnail tile")
    thumbnail_index = int(thumbnail_indices.item())
    crop_mask = ~thumbnail_mask
    crops = per_tile_registers[crop_mask]
    crop_metadata = metadata[crop_mask]
    positioned = crops + adapter.tile_position_mlp(crop_metadata)[:, None]
    thumbnail = per_tile_registers[thumbnail_index]
    logits = torch.einsum("rd,trd->rt", thumbnail, positioned) / math.sqrt(adapter.register_dim)
    return logits.softmax(dim=-1).detach().float().cpu(), thumbnail_index


def _plot_tile_attention(weights: torch.Tensor, metadata: torch.Tensor, gate: torch.Tensor, output: Path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    image = axes[0].imshow(weights.numpy(), aspect="auto", cmap="viridis")
    axes[0].set(xlabel="crop tile", ylabel="register", title="Thumbnail-query tile attention")
    fig.colorbar(image, ax=axes[0], fraction=0.04)
    crop = metadata[metadata[:, 4] < 0.5]
    axes[1].scatter(crop[:, 0], crop[:, 1], s=700, c=np.arange(len(crop)), cmap="tab20", marker="s")
    for index, row in enumerate(crop):
        axes[1].text(float(row[0]), float(row[1]), str(index), ha="center", va="center")
    axes[1].set(xlim=(0, 1), ylim=(1, 0), xlabel="normalized x", ylabel="normalized y", title=f"Crop layout; mean |tanh(tile gate)|={gate.abs().mean():.4f}")
    axes[1].set_aspect("equal")
    fig.savefig(output, dpi=180)
    plt.close(fig)


@torch.no_grad()
def audit_scene(agent, sample, token: str, metadata: dict, output_dir: Path) -> dict:
    features, _ = collate_samples([sample])
    image_path = _decode_path(features["image_path_tensor"][0])
    pixels, tile_metadata = load_image(image_path, return_tile_metadata=True)
    features["pixel_values"] = pixels.unsqueeze(0)
    features["tile_metadata"] = tile_metadata.unsqueeze(0)

    register_rows: List[torch.Tensor] = [None] * 24
    full_last_three: Dict[int, torch.Tensor] = {}
    cross: Dict[str, torch.Tensor] = {}
    adapter_output = {}
    handles = []

    def vision_hook(layer_index):
        def hook(module, inputs):
            hidden = inputs[0]
            register_rows[layer_index] = _register_rows(module, hidden).detach().float().cpu()
            if layer_index >= 21:
                full_last_three[layer_index] = _full_head_mean(module, hidden)
        return hook

    for index, block in enumerate(agent.backbone.model.vision_model.encoder.layers):
        handles.append(block.attn.register_forward_pre_hook(vision_hook(index)))
    for index, block in enumerate(agent.action_head.trajectory_decoder.layers):
        handles.append(block.cross_attn.register_forward_pre_hook(_cross_attention_hook(cross, f"generator_{index}")))
    for index, block in enumerate(agent.action_head.scorer_attention.layers):
        handles.append(block.cross_attn.register_forward_pre_hook(_cross_attention_hook(cross, f"scorer_{index}")))

    def adapter_hook(_module, _inputs, output):
        adapter_output["per_tile_registers"] = output.per_tile_registers.detach()

    handles.append(agent.backbone.planning_register_adapter.register_forward_hook(adapter_hook))
    try:
        prediction = agent.forward(features)
    finally:
        for handle in handles:
            handle.remove()
    if any(value is None for value in register_rows):
        raise RuntimeError("Failed to capture every InternViT layer")
    thumbnail_weights, thumbnail_index = _tile_attention(
        agent,
        adapter_output["per_tile_registers"],
        tile_metadata.to(device=adapter_output["per_tile_registers"].device),
    )
    maps, best_heads = _drivor_maps(register_rows, full_last_three, thumbnail_index)
    image = np.asarray(Image.open(image_path).convert("RGB"))
    scene_dir = output_dir / token
    scene_dir.mkdir(parents=True, exist_ok=True)
    for name, value in maps.items():
        _plot_register_overlays(image, value, f"{token}: {name}", scene_dir / f"register_to_patch_{name}.png")
    _plot_patch_energy(image, maps["last"], scene_dir / "patch_energy.png")

    proposals = prediction["proposals"][0].detach().float().cpu()
    selected_index = int(prediction["pdm_score"][0].argmax())
    generator_attention = cross["generator_3"][0].mean(dim=0)
    scorer_attention = cross["scorer_3"][0].mean(dim=0)
    _plot_trajectory_attention(
        proposals,
        selected_index,
        generator_attention,
        scorer_attention,
        scene_dir / "trajectory_register_attention.png",
    )
    _plot_composite(
        image,
        maps["last"],
        scorer_attention[selected_index],
        generator_attention[selected_index],
        scene_dir / "selected_trajectory_composite_attention.png",
    )
    tile_gate = torch.tanh(agent.backbone.planning_register_adapter.tile_gate.detach().float().cpu()).squeeze()
    _plot_tile_attention(thumbnail_weights, tile_metadata.cpu(), tile_gate, scene_dir / "tile_attention.png")

    planning = agent.action_head.planning_norm(prediction["planning_scene_features"])
    semantic = agent.action_head.semantic_norm(prediction["semantic_scene_features"])
    semantic_context, semantic_weights = agent.action_head.semantic_cross_attention(
        query=planning,
        key=semantic,
        value=semantic,
        need_weights=True,
        average_attn_weights=False,
    )
    semantic_weights = semantic_weights[0].float().cpu()
    semantic_gate = torch.sigmoid(
        agent.action_head.semantic_gate.detach().float()
    )
    gated_semantic_context = semantic_gate.to(semantic_context.device) * semantic_context
    fused_scene = agent.action_head.output_norm(planning + gated_semantic_context)
    semantic_entropy = -(
        semantic_weights.clamp_min(1e-12)
        * semantic_weights.clamp_min(1e-12).log()
    ).sum(dim=-1)
    uniform_probability = 1.0 / semantic_weights.shape[-1]
    semantic_uniform_deviation = (
        semantic_weights - uniform_probability
    ).abs()

    adapter = agent.backbone.planning_register_adapter
    thumbnail_mask = tile_metadata[:, 4] > 0.5
    crop_mask = ~thumbnail_mask
    per_tile_registers = adapter_output["per_tile_registers"]
    thumbnail_registers = per_tile_registers[thumbnail_index]
    crop_metadata_device = tile_metadata[crop_mask].to(
        device=per_tile_registers.device,
        dtype=per_tile_registers.dtype,
    )
    positioned_crops = (
        per_tile_registers[crop_mask]
        + adapter.tile_position_mlp(crop_metadata_device)[:, None]
    )
    tile_probabilities_device = thumbnail_weights.to(
        device=per_tile_registers.device,
        dtype=per_tile_registers.dtype,
    )
    tile_residual = torch.einsum(
        "rt,trd->rd", tile_probabilities_device, positioned_crops
    )
    gated_tile_residual = (
        torch.tanh(adapter.tile_gate.squeeze(0)) * tile_residual
    )
    layer_similarity = []
    layer_entropy = []
    patch_count = register_rows[-1].shape[-1] - REGISTER_STOP
    for rows in register_rows:
        patch = _normalize_patch_rows(rows[thumbnail_index, :, :, REGISTER_STOP:]).mean(dim=0)
        similarity = _attention_cosine(patch)
        mask = ~torch.eye(NUM_REGISTERS, dtype=torch.bool)
        layer_similarity.append(float(similarity[mask].mean()))
        entropy = -(patch.clamp_min(1e-12) * patch.clamp_min(1e-12).log()).sum(dim=-1)
        layer_entropy.append(float((entropy / math.log(patch_count)).mean()))

    method_similarity = {
        left + "_vs_" + right: float(
            F.cosine_similarity(
                maps[left].reshape(NUM_REGISTERS, -1),
                maps[right].reshape(NUM_REGISTERS, -1),
                dim=-1,
            ).mean()
        )
        for left, right in (("last", "shallow_rollout"), ("last", "lowest_entropy_head"), ("shallow_rollout", "lowest_entropy_head"))
    }
    selected_scorer_attention = scorer_attention[selected_index]
    selected_generator_attention = generator_attention[selected_index]
    result = {
        "token": token,
        "category": metadata["category"],
        "image_path": image_path,
        "tile_count": int(len(tile_metadata)),
        "thumbnail_index": thumbnail_index,
        "patch_grid": list(maps["last"].shape[-2:]),
        "selected_query_index": selected_index,
        "selected_predicted_pdms": float(prediction["pred_pdms"][0, selected_index]),
        "epoch27_stratification": metadata,
        "best_heads_per_register": best_heads.tolist(),
        "layer_register_attention_pairwise_cosine": layer_similarity,
        "layer_normalized_attention_entropy": layer_entropy,
        "method_map_cosine": method_similarity,
        "last_map_register_cosine": _attention_cosine(maps["last"]).tolist(),
        "selected_scorer_attention_entropy": float(-(selected_scorer_attention.clamp_min(1e-12) * selected_scorer_attention.clamp_min(1e-12).log()).sum()),
        "selected_scorer_effective_registers": float(torch.exp(-(selected_scorer_attention.clamp_min(1e-12) * selected_scorer_attention.clamp_min(1e-12).log()).sum())),
        "selected_generator_attention_entropy": float(-(selected_generator_attention.clamp_min(1e-12) * selected_generator_attention.clamp_min(1e-12).log()).sum()),
        "semantic_cross_attention_entropy_mean": float(
            semantic_entropy.mean()
        ),
        "semantic_cross_attention_normalized_entropy_mean": float(
            semantic_entropy.mean() / math.log(semantic_weights.shape[-1])
        ),
        "semantic_cross_attention_max_probability_mean": float(
            semantic_weights.max(dim=-1).values.mean()
        ),
        "semantic_cross_attention_abs_deviation_from_uniform_mean": float(
            semantic_uniform_deviation.mean()
        ),
        "semantic_cross_attention_per_head_normalized_entropy": (
            semantic_entropy.mean(dim=-1) / math.log(semantic_weights.shape[-1])
        ).tolist(),
        "semantic_gate_probability": float(semantic_gate),
        "planning_token_geometry": _token_geometry(planning),
        "semantic_token_geometry": _token_geometry(semantic),
        "semantic_context_geometry": _token_geometry(semantic_context),
        "fused_scene_token_geometry": _token_geometry(fused_scene),
        "planning_rms": float(planning.float().square().mean().sqrt()),
        "semantic_context_rms": float(
            semantic_context.float().square().mean().sqrt()
        ),
        "gated_semantic_context_rms": float(
            gated_semantic_context.float().square().mean().sqrt()
        ),
        "gated_semantic_to_planning_rms_ratio": float(
            gated_semantic_context.float().square().mean().sqrt()
            / planning.float().square().mean().sqrt().clamp_min(1e-12)
        ),
        "tile_gate_tanh_mean": float(tile_gate.mean()),
        "tile_gate_tanh_abs_mean": float(tile_gate.abs().mean()),
        "thumbnail_register_rms": float(
            thumbnail_registers.float().square().mean().sqrt()
        ),
        "tile_attention_residual_rms": float(
            tile_residual.float().square().mean().sqrt()
        ),
        "gated_tile_residual_rms": float(
            gated_tile_residual.float().square().mean().sqrt()
        ),
        "gated_tile_to_thumbnail_rms_ratio": float(
            gated_tile_residual.float().square().mean().sqrt()
            / thumbnail_registers.float().square().mean().sqrt().clamp_min(1e-12)
        ),
        "tile_attention_weights": thumbnail_weights.tolist(),
        "visualization_scope": "single front camera; thumbnail tile is overlaid on the original front image",
        "composite_attention_caveat": "Scorer/generator-to-scene attention multiplied by register-to-patch attention is a heuristic attribution, not a causal gradient explanation.",
    }
    (scene_dir / "diagnostics.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolved-config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--candidate-bank", type=Path, required=True)
    parser.add_argument("--metric-cache", type=Path, required=True)
    parser.add_argument("--navsim-log-path", type=Path, required=True)
    parser.add_argument("--sensor-blobs-path", type=Path, required=True)
    parser.add_argument("--scene-count", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    device = torch.device(args.device)
    _, agent, checkpoint = load_formal_training_agent(
        args.resolved_config, args.checkpoint, device=device, compute_dtype="float32"
    )
    agent.remove_training_only_world_model()
    agent.world_model_enabled = False
    agent.future_mode = "disabled"
    agent.eval()
    tokens, token_metadata = select_representative_tokens(
        args.candidate_bank, args.metric_cache, args.scene_count
    )
    samples = build_navtest_samples(
        agent,
        tokens,
        token_metadata,
        navsim_log_path=args.navsim_log_path,
        sensor_blobs_path=args.sensor_blobs_path,
    )
    results = []
    for token in tokens:
        results.append(audit_scene(agent, samples[token], token, token_metadata[token], args.output_dir))
        torch.cuda.empty_cache()

    layer_similarity = np.asarray([row["layer_register_attention_pairwise_cosine"] for row in results])
    layer_entropy = np.asarray([row["layer_normalized_attention_entropy"] for row in results])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    x = np.arange(24)
    for axis, values, ylabel, title in (
        (axes[0], layer_similarity, "mean off-diagonal cosine", "Register attention specialization vs depth"),
        (axes[1], layer_entropy, "normalized entropy", "Register attention sharpness vs depth"),
    ):
        mean = values.mean(axis=0)
        std = values.std(axis=0)
        axis.plot(x, mean, marker="o", markersize=3)
        axis.fill_between(x, mean - std, mean + std, alpha=0.2)
        axis.set(xlabel="InternViT layer", ylabel=ylabel, title=title)
        axis.grid(alpha=0.25)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_dir / "attention_depth_summary.png", dpi=180)
    plt.close(fig)
    report = {
        "schema_version": 1,
        "checkpoint": checkpoint,
        "candidate_bank": str(args.candidate_bank.resolve()),
        "candidate_bank_sha256": sha256_file(args.candidate_bank),
        "drivor_visualization_source": {
            "repo": "valeoai/DrivoR",
            "commit": "fc6e5aa144bbcb5a046e22c18f1bd5cf3af8634a",
            "files": ["scripts/viz/attention_maps_viz.ipynb", "scripts/viz/build_cosine_similarity_maps.py"],
            "methods": ["last_layer_average_heads", "last_3_layer_rollout_alpha_0p9", "per_register_lowest_entropy_head"],
        },
        "adaptation": "InternViT registers are indices 1:17 after CLS; single front-camera thumbnail replaces DrivOR's four-camera visualization.",
        "scene_count": len(results),
        "scenes": results,
        "mean_layer_register_attention_pairwise_cosine": layer_similarity.mean(axis=0).tolist(),
        "mean_layer_normalized_attention_entropy": layer_entropy.mean(axis=0).tolist(),
        "model_inference_precision": "FP32",
        "inference_uses_future_inputs": False,
    }
    (args.output_dir / "attention_audit.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"checkpoint": checkpoint, "scene_count": len(results), "output": str(args.output_dir.resolve())}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Numerically faithful DrivoR decomposition for arbitrary proposal sets.

The upstream DrivoR checkout is intentionally left untouched.  These functions
mirror ``DrivoRModel.forward`` after scene encoding and are parity-gated against
that method.  Crucially, all candidates in a union are passed through the
scorer in one call because ``TransformerDecoderScorer`` contains self-attention.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Mapping, Sequence, Tuple

import torch
from omegaconf import OmegaConf

from .schema import PREDICTED_FACTOR_NAMES


def load_drivor_model(
    config_path: Path,
    checkpoint_path: Path,
    dino_weights: Path,
    device: torch.device,
):
    from navsim.agents.drivoR.drivor_model import DrivoRModel

    full_config = OmegaConf.load(config_path)
    config = full_config["config"] if "config" in full_config else full_config
    config.image_backbone.model_weights = str(dino_weights)
    model = DrivoRModel(config)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    raw_state = checkpoint["state_dict"]
    prefixes = ("agent._drivor_model.", "_drivor_model.")
    state = None
    for prefix in prefixes:
        candidate = {
            key[len(prefix) :]: value
            for key, value in raw_state.items()
            if key.startswith(prefix)
        }
        if len(candidate) == len(raw_state):
            state = candidate
            break
    if state is None:
        raise RuntimeError("Checkpoint does not contain one exact DrivoRModel state dict")
    model.load_state_dict(state, strict=True)
    model.to(device=device, dtype=torch.float32).eval()
    return model, config


def encode_scene(model, features: Mapping[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (scene_features, ego_token, raw current ego status)."""

    if model._config.full_history_status:
        ego_status = features["ego_status"].flatten(-2)
    else:
        ego_status = features["ego_status"][:, -1]
    ego_token = model.hist_encoding(ego_status)[:, None]
    scene_parts = []
    if model.num_cams:
        image = features.get("image", features.get("camera_feature"))
        if image is None:
            raise ValueError("DrivoR camera feature is absent")
        scene_tokens = model.scene_embeds.expand(image.shape[0], -1, -1, -1)
        scene_parts.append(model.image_backbone(image, scene_tokens))
    if model.num_lidar:
        image = features["lidar_feature"]
        scene_tokens = model.lidar_scene_embeds.expand(image.shape[0], -1, -1, -1)
        scene_parts.append(model.lidar_backbone(image, scene_tokens))
    return torch.cat(scene_parts, dim=1), ego_token, ego_status


def context_from_cache(model, visual_tokens: torch.Tensor, status_feature: torch.Tensor):
    """Build exactly the post-encoder context from the immutable FP32 replay."""

    if model._config.full_history_status:
        raise RuntimeError("The deployed checkpoint expects current 11-D status, not flattened history")
    ego_token = model.hist_encoding(status_feature)[:, None]
    return visual_tokens, ego_token


def decode_base_proposals(model, scene_features: torch.Tensor, ego_token: torch.Tensor):
    """Mirror the proposal-generation half of upstream ``forward``."""

    traj_tokens = ego_token + model.init_feature.weight[None]
    proposals = model.traj_head[0](traj_tokens).reshape(
        traj_tokens.shape[0], -1, model.poses_num, model.state_size
    )
    stages = [proposals]
    token_list = model.trajectory_decoder(traj_tokens, scene_features)
    for index in range(model._config.ref_num):
        tokens = token_list[index]
        proposals = model.traj_head[index + 1](tokens).reshape(
            tokens.shape[0], -1, model.poses_num, model.state_size
        )
        stages.append(proposals)
    return stages


def stack_factor_logits(pred_logit: Mapping[str, torch.Tensor]) -> torch.Tensor:
    return torch.stack([pred_logit[key] for key in PREDICTED_FACTOR_NAMES], dim=-1)


def aggregate_score(factor_logits: torch.Tensor, config) -> torch.Tensor:
    probability = factor_logits.sigmoid()
    return (
        float(config.noc) * probability[..., 0].log()
        + float(config.dac) * probability[..., 1].log()
        + float(config.ddc) * probability[..., 2].log()
        + (
            float(config.ttc) * probability[..., 3]
            + float(config.ep) * probability[..., 4]
            + float(config.comfort) * probability[..., 5]
        ).log()
    )


def score_proposals(
    model,
    config,
    proposals: torch.Tensor,
    scene_features: torch.Tensor,
    ego_token: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Jointly score one complete candidate set with no source-ID input."""

    if proposals.ndim != 4 or proposals.shape[-2:] != (model.poses_num, model.state_size):
        raise ValueError(f"Expected [B,K,{model.poses_num},3], got {tuple(proposals.shape)}")
    batch_size, candidate_count = proposals.shape[:2]
    embedded = model.pos_embed(
        proposals.reshape(batch_size, candidate_count, -1).detach()
    )
    candidate_features = model.scorer_attention(embedded, scene_features) + ego_token
    pred_logit = model.scorer(proposals, candidate_features)[0]
    factor_logits = stack_factor_logits(pred_logit)
    scores = aggregate_score(factor_logits, config)
    selected = scores.argmax(dim=1)
    trajectory = proposals[torch.arange(batch_size, device=proposals.device), selected]
    return {
        "factor_logits": factor_logits,
        "factor_probabilities": factor_logits.sigmoid(),
        "pdm_score": scores,
        "selected_index": selected,
        "trajectory": trajectory,
    }


def forward_from_context(model, config, scene_features: torch.Tensor, ego_token: torch.Tensor):
    stages = decode_base_proposals(model, scene_features, ego_token)
    scored = score_proposals(model, config, stages[-1], scene_features, ego_token)
    return {"proposal_list": stages, "proposals": stages[-1], **scored}


def max_abs(left: torch.Tensor, right: torch.Tensor) -> float:
    if tuple(left.shape) != tuple(right.shape):
        raise ValueError(f"Shape mismatch {tuple(left.shape)} != {tuple(right.shape)}")
    return float((left.detach().float() - right.detach().float()).abs().max().item())

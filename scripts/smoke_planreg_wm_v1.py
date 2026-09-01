#!/usr/bin/env python3
"""CPU-friendly end-to-end synthetic smoke for PlanReg-WM-V1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from omegaconf import OmegaConf
import torch
from torch import nn


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from navsim.agents.EpisodeDrive.action_decoder import ActionDecoder  # noqa: E402
from navsim.agents.EpisodeDrive.drivevla_base_agent import DriveVLABaseAgent  # noqa: E402
from navsim.agents.EpisodeDrive.layers.losses.episode_drive_loss import (  # noqa: E402
    EpisodeDriveLoss,
)
from navsim.agents.EpisodeDrive.layers.planning_registers import (  # noqa: E402
    InternVLPlanningRegisters,
    freeze_vision_except_qv_lora,
    inject_internvit_qv_lora,
)
from navsim.agents.EpisodeDrive.layers.world_model import (  # noqa: E402
    EMARegisterTarget,
    FutureRegisterPredictor,
)


class _Attention(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.qkv = nn.Linear(dim, 3 * dim)
        self.output = nn.Linear(dim, dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        query, _key, value = self.qkv(inputs).chunk(3, dim=-1)
        return self.output(torch.tanh(query + value))


class _VisionBlock(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.attn = _Attention(dim)
        self.norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = inputs + self.attn(self.norm(inputs))
        return hidden + self.mlp(self.norm(hidden))


class _Embeddings(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.patch_projection = nn.Linear(3, dim)
        self.cls = nn.Parameter(torch.randn(1, 1, dim))

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        patches = pixels.permute(0, 2, 3, 1).reshape(pixels.shape[0], -1, 3)
        return torch.cat(
            (self.cls.expand(pixels.shape[0], -1, -1), self.patch_projection(patches)),
            dim=1,
        )


class _Encoder(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_VisionBlock(dim), _VisionBlock(dim)])

    def forward(self, inputs_embeds, output_hidden_states=False, return_dict=True):
        hidden = inputs_embeds
        for layer in self.layers:
            hidden = layer(hidden)
        return SimpleNamespace(last_hidden_state=hidden)


class _Vision(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=dim)
        self.embeddings = _Embeddings(dim)
        self.encoder = _Encoder(dim)


class _SyntheticInternVL(nn.Module):
    def __init__(self, vision_dim: int) -> None:
        super().__init__()
        self.vision_model = _Vision(vision_dim)
        self.select_layer = -1
        self.downsample_ratio = 1.0
        self.mlp1 = nn.Linear(vision_dim, 1536)

    @staticmethod
    def pixel_shuffle(inputs: torch.Tensor, scale_factor: float):
        if scale_factor != 1.0:
            raise AssertionError("Synthetic smoke uses scale_factor=1")
        return inputs


class _StudentBackbone(nn.Module):
    def __init__(self, model: nn.Module, adapter: nn.Module) -> None:
        super().__init__()
        self.model = model
        self.planning_register_adapter = adapter


def _action_config():
    return OmegaConf.create(
        {
            "b2d": False,
            "num_poses": 8,
            "tf_d_model": 256,
            "tf_d_ffn": 256,
            "num_scene_tokens": 16,
            "proposal_num": 64,
            "ref_num": 4,
            "scorer_ref_num": 4,
            "one_token_per_traj": True,
            "full_history_status": False,
            "cam_f0": [3],
            "cam_l0": [],
            "cam_l1": [],
            "cam_l2": [],
            "cam_r0": [],
            "cam_r1": [],
            "cam_r2": [],
            "cam_b0": [],
            "lidar_pc": [],
            "double_score": False,
            "agent_pred": False,
            "area_pred": False,
            "bev_map": False,
            "bev_agent": False,
            "refiner_num_heads": 1,
            "refiner_ls_values": 0.0,
            "noc": 1.0,
            "dac": 1.0,
            "ddc": 0.0,
            "ttc": 5.0,
            "ep": 5.0,
            "comfort": 2.0,
        }
    )


def _synthetic_scoring(targets, proposals, test=False):
    batch_size, proposal_count = proposals.shape[:2]
    component_targets = torch.ones(
        batch_size,
        proposal_count,
        7,
        dtype=torch.float64,
        device=proposals.device,
    )
    component_targets[..., -1] = 0.5
    component_targets[0, 0, 3] = 2.0  # TTC invalid sentinel exercises the mask.
    final_scores = component_targets[..., -1]
    return (
        final_scores,
        final_scores.amax(dim=1),
        component_targets,
        None,
        None,
        None,
    )


def run_smoke(seed: int, device: torch.device) -> dict:
    torch.manual_seed(seed)
    batch_size = 2
    vision_dim = 32
    model = _SyntheticInternVL(vision_dim).to(device)
    injected = inject_internvit_qv_lora(model.vision_model, rank=32, dropout=0.0)
    adapter = InternVLPlanningRegisters(
        vision_hidden_dim=vision_dim,
        num_registers=16,
        register_dim=256,
        tile_aggregation="mean",
        device=device,
    )
    student = _StudentBackbone(model, adapter)
    freeze_vision_except_qv_lora(model.vision_model)
    for parameter in adapter.parameters():
        parameter.requires_grad = True
    teacher = EMARegisterTarget(student).to(device)

    current_pixels = torch.randn(batch_size, 3, 2, 2, device=device)
    current_vision = adapter(model, current_pixels, [1, 1])
    current_vision.scene_registers.retain_grad()
    action = ActionDecoder(
        _action_config(),
        scene_fusion_config=OmegaConf.create(
            {
                "mode": "planning_plus_semantic",
                "transition_fraction": 0.10,
                "semantic_gate_init": 0.0,
            }
        ),
        total_optimizer_steps=100,
    ).to(device)
    action.train()
    action.set_optimizer_step(100)
    status = torch.randn(batch_size, 8, device=device)
    predictions = action(
        {
            "last_hidden_state": current_vision.patch_features,
            "status_feature": status,
            "planning_registers": current_vision.scene_registers,
        }
    )

    future_pixels = torch.randn(batch_size, 3, 3, 2, 2, device=device)
    teacher_images = []
    for batch_index in range(batch_size):
        teacher_images.append(current_pixels[batch_index:batch_index + 1])
        teacher_images.extend(
            future_pixels[batch_index, horizon:horizon + 1]
            for horizon in range(3)
        )
    teacher_pixels = torch.cat(teacher_images, dim=0)
    with torch.no_grad():
        teacher_registers = teacher(teacher_pixels, [1] * (batch_size * 4))
    teacher_registers = teacher_registers.reshape(batch_size, 4, 16, 256)

    target_trajectory = torch.randn(batch_size, 8, 3, device=device)
    targets = {"trajectory": target_trajectory}
    base_loss = EpisodeDriveLoss()(targets, predictions, _action_config(), _synthetic_scoring)

    loss_harness = DriveVLABaseAgent.__new__(DriveVLABaseAgent)
    nn.Module.__init__(loss_harness)
    loss_harness.world_model_config = SimpleNamespace(
        predictor_only=False,
        horizons_sec=(0.5, 1.5, 3.0),
        abs_weight=1.0,
        delta_weight=0.25,
    )
    loss_harness.future_mode = "correct"
    loss_harness.future_register_predictor = FutureRegisterPredictor(
        hidden_dim=256,
        predictor_layers=2,
    ).to(device)
    wm = loss_harness._compute_world_model_loss_from_registers(
        current_vision.scene_registers,
        target_trajectory,
        teacher_registers[:, 0],
        teacher_registers[:, 1:],
        torch.ones(batch_size, 3, dtype=torch.bool, device=device),
    )
    complete_loss = base_loss["loss"] + 0.25 * wm["wm_loss"]
    complete_loss.backward()

    qv_grad = torch.stack(
        [
            parameter.grad.detach().float().square().sum()
            for name, parameter in model.vision_model.named_parameters()
            if ("q_lora" in name or "v_lora" in name) and parameter.grad is not None
        ]
    ).sum().sqrt()
    register_grad = adapter.planning_registers.grad.detach().float().norm()

    # Pure current-frame inference: no future keys, no teacher, no predictor.
    action.eval()
    inference_inputs = {
        "last_hidden_state": current_vision.patch_features.detach(),
        "status_feature": status,
        "planning_registers": current_vision.scene_registers.detach(),
    }
    with torch.no_grad():
        inference = action(inference_inputs)

    report = {
        "seed": seed,
        "device": str(device),
        "injected_visual_layers": len(injected),
        "patch_features": list(current_vision.patch_features.shape),
        "per_tile_registers": list(current_vision.per_tile_registers.shape),
        "planning_registers": list(current_vision.scene_registers.shape),
        "proposals": list(predictions["proposals"].shape),
        "predicted_future_registers": list(wm["predicted_future_registers"].shape),
        "teacher_current": list(teacher_registers[:, 0].shape),
        "teacher_future": list(teacher_registers[:, 1:].shape),
        "complete_loss": float(complete_loss.detach().cpu()),
        "vision_qv_lora_grad_norm": float(qv_grad.cpu()),
        "planning_register_grad_norm": float(register_grad.cpu()),
        "inference_trajectory": list(inference["trajectory"].shape),
        "inference_used_future_keys": False,
    }
    if report["planning_registers"] != [2, 16, 256]:
        raise AssertionError(report)
    if report["proposals"] != [2, 64, 8, 3]:
        raise AssertionError(report)
    if report["predicted_future_registers"] != [2, 1, 3, 16, 256]:
        raise AssertionError(report)
    if qv_grad.item() <= 0.0 or register_grad.item() <= 0.0:
        raise AssertionError(f"Gradient routing failed: {report}")
    if report["inference_trajectory"] != [2, 8, 3]:
        raise AssertionError(report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    report = run_smoke(args.seed, torch.device(args.device))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

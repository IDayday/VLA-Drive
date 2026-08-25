#!/usr/bin/env python3
"""Benchmark Flow K64/NFE10 against one-pass Register64 after Qwen hidden states."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from starVLA.model.modules.action_model.GR00T_ActionHeader import FlowmatchingActionHead
from starVLA.model.modules.register_planner.generator import RegisterTrajectoryGenerator
from starVLA.model.modules.register_planner.losses import RegisterTrajectoryLoss
from starVLA.model.modules.scene_encoder import GlobalSceneQFormer
from starVLA.model.modules.trajectory_scorer.drivor_dynamic_scorer import (
    DrivoRDynamicScorer,
)
from starVLA.model.modules.trajectory_scorer.losses import DrivoRMetricLoss
from starVLA.training.config_loader import load_training_config
from starVLA.training.register_stage_utils import atomic_json


def _time(callable_, *, warmup: int, iterations: int, device: torch.device) -> float:
    for _ in range(warmup):
        callable_()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    for _ in range(iterations):
        callable_()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return (time.perf_counter() - start) / iterations


def _measure(callable_, *, warmup: int, iterations: int, device: torch.device):
    baseline = 0
    if device.type == "cuda":
        torch.cuda.empty_cache()
        baseline = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
    latency = _time(callable_, warmup=warmup, iterations=iterations, device=device)
    peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    return latency, int(peak), int(max(0, peak - baseline))


def _build_flow(path: str, device, dtype) -> FlowmatchingActionHead:
    config = load_training_config(path)
    action = config.framework.action_model
    action.DiTConfig = {
        "num_layers": int(action.diffusion_model_cfg.num_layers),
        "input_embedding_dim": int(action.hidden_size),
        "attention_head_dim": 64,
        "num_attention_heads": int(action.hidden_size) // 64,
    }
    action.num_inference_timesteps = 10
    return FlowmatchingActionHead(config).to(device=device, dtype=dtype).eval()


def _build_register(path: str, device, dtype) -> RegisterTrajectoryGenerator:
    config = load_training_config(path).framework.register_generator
    return RegisterTrajectoryGenerator(
        proposal_num=64,
        num_poses=8,
        state_dim=3,
        model_dim=int(config.model_dim),
        ffn_dim=int(config.ffn_dim),
        num_layers=int(config.num_layers),
        num_heads=int(config.num_heads),
        proj_drop=float(config.proj_drop),
        drop_path=float(config.drop_path),
        layer_scale_init=float(config.layer_scale_init),
        ego_state_dim=4,
    ).to(device=device, dtype=dtype).eval()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--flow-config",
        default="starVLA/config/training/qwenpi_drivor_suprim.yaml",
    )
    parser.add_argument(
        "--register-config",
        default="starVLA/config/training/qwen_register64_generator.yaml",
    )
    parser.add_argument("--register-checkpoint")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--qwen-sequence-length", type=int, default=128)
    parser.add_argument("--candidate-chunk-size", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--train-iterations", type=int, default=10)
    parser.add_argument("--steps-per-epoch", type=int, default=1000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--output", default="register_generator_benchmark.json")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA benchmark requested but no GPU is available")
    dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[args.dtype]
    flow = _build_flow(args.flow_config, device, dtype)
    register = _build_register(args.register_config, device, dtype)
    if args.register_checkpoint:
        payload = torch.load(
            args.register_checkpoint, map_location="cpu", weights_only=False
        )
        register.load_state_dict(
            payload["state_dict"]["register_generator"], strict=True
        )
    batch = args.batch_size
    action_hidden = torch.randn(batch, 8, 2048, device=device, dtype=dtype)
    scene_tokens = torch.randn(batch, 16, 256, device=device, dtype=dtype)
    ego_state = torch.randn(batch, 4, device=device, dtype=dtype)
    initial_noise = torch.randn(batch, 64, 8, 4, device=device, dtype=dtype)

    @torch.inference_mode()
    def flow_forward():
        return flow.predict_multi_action(
            action_hidden,
            global_scene_tokens=scene_tokens,
            num_candidates=64,
            candidate_chunk_size=args.candidate_chunk_size,
            initial_noise=initial_noise,
        )

    @torch.inference_mode()
    def register_forward():
        return register(scene_tokens, ego_state).proposals

    flow_latency, flow_peak, flow_incremental = _measure(
        flow_forward, warmup=args.warmup, iterations=args.iterations, device=device
    )
    register_latency, register_peak, register_incremental = _measure(
        register_forward, warmup=args.warmup, iterations=args.iterations, device=device
    )

    qformer = GlobalSceneQFormer(
        input_dim=2048,
        hidden_dim=256,
        output_dim=256,
        num_queries=16,
        num_layers=4,
        num_heads=8,
        ffn_dim=1024,
        dropout=0.0,
        detach_qwen_input=False,
    ).to(device=device, dtype=dtype).eval()
    qwen_hidden = torch.randn(
        batch, args.qwen_sequence_length, 2048, device=device, dtype=dtype
    )
    attention_mask = torch.ones(
        batch, args.qwen_sequence_length, device=device, dtype=torch.bool
    )

    @torch.inference_mode()
    def flow_post_qwen():
        scene = qformer(qwen_hidden, attention_mask)
        return flow.predict_multi_action(
            qwen_hidden[:, :8],
            global_scene_tokens=scene.global_tokens,
            num_candidates=64,
            candidate_chunk_size=args.candidate_chunk_size,
            initial_noise=initial_noise,
        )

    @torch.inference_mode()
    def register_post_qwen():
        scene = qformer(qwen_hidden, attention_mask)
        return register(scene.global_tokens, ego_state).proposals

    flow_end_to_end = _time(
        flow_post_qwen, warmup=args.warmup, iterations=args.iterations, device=device
    )
    register_end_to_end = _time(
        register_post_qwen,
        warmup=args.warmup,
        iterations=args.iterations,
        device=device,
    )

    # Separate generator-training projection: one forward, one WTA backward.
    register.train()
    loss_fn = RegisterTrajectoryLoss()
    gt = torch.randn(batch, 8, 3, device=device, dtype=dtype)

    def register_train_step():
        register.zero_grad(set_to_none=True)
        output = register(scene_tokens, ego_state)
        loss = loss_fn(output.proposal_list, gt).loss
        loss.backward()

    register_train_latency = _time(
        register_train_step,
        warmup=1,
        iterations=args.train_iterations,
        device=device,
    )
    register.eval()

    scorer = DrivoRDynamicScorer(
        scene_dim=256,
        model_dim=256,
        ffn_dim=1024,
        num_layers=4,
        num_heads=1,
        decoder_style="donor_register",
        proj_drop=0.1,
        drop_path=0.2,
    ).to(device=device, dtype=dtype).train()
    scorer_loss = DrivoRMetricLoss()
    # Materialize a normal detached tensor; tensors created under
    # inference_mode cannot be saved by the scorer's backward graph.
    proposals = register_forward().detach().clone()
    targets = {
        name: torch.rand(batch, 64, device=device, dtype=dtype)
        for name in (
            "no_at_fault_collisions",
            "drivable_area_compliance",
            "time_to_collision_within_bound",
            "ego_progress",
            "driving_direction_compliance",
            "comfort",
        )
    }

    def scorer_train_step():
        scorer.zero_grad(set_to_none=True)
        output = scorer(proposals, scene_tokens, ego_state, topm=32)
        loss = scorer_loss(output.metric_logits, targets)[0]
        loss.backward()

    scorer_train_latency = _time(
        scorer_train_step,
        warmup=1,
        iterations=args.train_iterations,
        device=device,
    )
    steps_per_epoch = args.steps_per_epoch
    result = {
        "device": str(device),
        "device_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
        ),
        "dtype": args.dtype,
        "batch_size": batch,
        "warmup_iterations": args.warmup,
        "measured_iterations": args.iterations,
        "training_iterations": args.train_iterations,
        "projection_steps_per_epoch": args.steps_per_epoch,
        "flow": {
            "candidates": 64,
            "nfe": 10,
            "candidate_chunk_size": args.candidate_chunk_size,
            "generator_forward_seconds": flow_latency,
            "peak_allocated_bytes": flow_peak,
            "peak_incremental_bytes": flow_incremental,
            "proposals_per_second": batch * 64 / flow_latency,
            "post_qwen_end_to_end_seconds": flow_end_to_end,
        },
        "register64": {
            "decoder_layers": 4,
            "proposal_head_style": register.proposal_head_style,
            "parameter_count": sum(
                parameter.numel() for parameter in register.parameters()
            ),
            "generator_forward_seconds": register_latency,
            "peak_allocated_bytes": register_peak,
            "peak_incremental_bytes": register_incremental,
            "proposals_per_second": batch * 64 / register_latency,
            "post_qwen_end_to_end_seconds": register_end_to_end,
            "training_sec_per_step": register_train_latency,
            "training_samples_per_second": batch / register_train_latency,
            "projected_epoch_seconds": register_train_latency * steps_per_epoch,
            "projected_25_epoch_seconds": register_train_latency * steps_per_epoch * 25,
        },
        "drivor_scorer": {
            "training_sec_per_step": scorer_train_latency,
            "projected_5_epoch_seconds": scorer_train_latency * steps_per_epoch * 5,
            "projected_10_epoch_seconds": scorer_train_latency * steps_per_epoch * 10,
        },
        "speedup": {
            "generator_forward": flow_latency / register_latency,
            "post_qwen_end_to_end": flow_end_to_end / register_end_to_end,
        },
        "scope": "Qwen hidden states are inputs; common Qwen backbone latency is excluded",
    }
    atomic_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

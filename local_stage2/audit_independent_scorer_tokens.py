"""Audit current-observation token choices before training a private scorer."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict

import torch
from hydra.utils import instantiate
from torch.utils.data import DataLoader

from local_stage2.export_public_base_scorer_cache import (
    _CacheNameBuilder,
    _compose_agent_config,
)
from navsim.planning.training.dataset import CacheOnlyDataset, drivevla_cached_collate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--vlm-path", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if not args.vlm_path.is_dir():
        raise FileNotFoundError(args.vlm_path)
    os.environ["DRIVEVLA_VLM_CONFIG"] = str(args.vlm_path.resolve())
    cfg = _compose_agent_config(args.repo_root.resolve(), args.checkpoint.resolve())
    agent = instantiate(cfg.agent)
    agent.initialize()
    agent.cuda().eval()
    for parameter in agent.parameters():
        parameter.requires_grad_(False)

    logs = [str(value) for value in cfg.train_logs + cfg.val_logs]
    dataset = CacheOnlyDataset(
        cache_path=str(args.feature_cache),
        feature_builders=[_CacheNameBuilder("internvl_feature")],
        target_builders=[],
        log_names=logs,
        append_token_to_batch=True,
        preprocess_images=True,
        preprocess_image_dtype="bfloat16",
        pretokenize_inputs=True,
        tokenizer=agent.backbone.tokenizer,
    )
    dataset.tokens = sorted(dataset.tokens)[:1]
    loader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=0,
        collate_fn=drivevla_cached_collate,
    )
    features, _targets, tokens = next(iter(loader))
    pixel_values = features["pixel_values"]
    if pixel_values.ndim != 5:
        raise RuntimeError(f"Expected batched crop tensor, got {pixel_values.shape}")
    input_ids = features["input_ids"]
    attention_mask = features["attention_mask"]

    captured: Dict[str, torch.Tensor] = {}

    def capture_qformer_input(_module, inputs):
        captured["qformer_scene_queries"] = inputs[0].detach().cpu()
        captured["qformer_observation_tokens"] = inputs[1].detach().cpu()

    handle = agent.action_head.q_former.register_forward_pre_hook(
        capture_qformer_input
    )
    torch.cuda.synchronize()
    started = time.perf_counter()
    try:
        with torch.inference_mode():
            prediction = agent.forward(features)
        torch.cuda.synchronize()
    finally:
        handle.remove()
    full_forward_seconds = time.perf_counter() - started

    flattened_pixels = pixel_values.cuda(non_blocking=True).flatten(0, 1)
    visual_model = agent.backbone
    visual_model_chain = []
    for _ in range(6):
        visual_model_chain.append(
            f"{type(visual_model).__module__}.{type(visual_model).__name__}"
        )
        if callable(getattr(visual_model, "extract_feature", None)):
            break
        visual_model = getattr(visual_model, "model", None)
        if visual_model is None:
            raise RuntimeError(
                "Could not resolve InternVL extract_feature through wrapper chain"
            )
    else:
        raise RuntimeError("InternVL wrapper chain exceeded audit depth")
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.inference_mode():
        raw_visual_tokens = visual_model.extract_feature(flattened_pixels)
    torch.cuda.synchronize()
    visual_forward_seconds = time.perf_counter() - started

    qformer_tokens = captured["qformer_observation_tokens"]
    image_context_id = int(agent.backbone.img_context_token_id)
    image_mask = input_ids == image_context_id
    scene_count = 103288

    def tebibytes(shape, dtype_bytes: int = 2) -> float:
        elements = 1
        for value in shape:
            elements *= int(value)
        return float(elements * dtype_bytes * scene_count / (1024**4))

    payload = {
        "scene_token": str(tokens[0]),
        "resolved_vlm_path": str(args.vlm_path.resolve()),
        "input_ids_shape": list(input_ids.shape),
        "valid_input_token_count": int(attention_mask.sum()),
        "image_context_token_count": int(image_mask.sum()),
        "qformer_observation_shape": list(qformer_tokens.shape),
        "qformer_observation_dtype": str(qformer_tokens.dtype),
        "qformer_query_shape": list(captured["qformer_scene_queries"].shape),
        "raw_visual_feature_shape": list(raw_visual_tokens.shape),
        "raw_visual_feature_dtype": str(raw_visual_tokens.dtype),
        "visual_model_wrapper_chain": visual_model_chain,
        "pixel_crop_shape": list(pixel_values.shape),
        "released_scene_feature_shape": list(prediction["language_feature"].shape),
        "released_ego_feature_shape": list(prediction["ego_feature"].shape),
        "proposal_shape": list(prediction["proposals"].shape),
        "full_base_forward_seconds": full_forward_seconds,
        "standalone_visual_forward_seconds": visual_forward_seconds,
        "estimated_full_navtrain_qformer_input_fp16_tib": tebibytes(
            qformer_tokens.shape[1:]
        ),
        "estimated_full_navtrain_raw_visual_fp16_tib": tebibytes(
            raw_visual_tokens.shape
        ),
        "interpretation": {
            "qformer_input": (
                "final language-model sequence; contains image-context and prompt tokens"
            ),
            "raw_visual_feature": (
                "frozen InternVL projected crop/patch tokens before language-model mixing"
            ),
            "training_decision": (
                "do not materialize a full-dataset raw-token cache; run the frozen visual "
                "encoder and train scorer-private query compression online"
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

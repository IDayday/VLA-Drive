"""Distributed NAVSIM frozen-feature precomputation.

Run this module through ``pre_cache.sh``.  Every distributed rank writes its
own LMDB under a durable shared path; no model checkpoint or feature payload is
written to container-local storage.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import torch
import torch.distributed as dist
import torchvision.transforms.functional as TF
from einops import rearrange
from omegaconf import OmegaConf
from PIL import Image

from starVLA.cache.navsim_feature_cache import (
    CACHE_SCHEMA_VERSION,
    GS_QUERY_TOKENS,
    MINE_AGENT_QUERY_TOKENS,
    REWARD_QUERY_TOKENS,
    RGB_QUERY_TOKENS,
    ROBOT_HISTORY_TOKEN,
    RankCacheWriter,
    action_query_tokens,
    append_world_action_tokens,
    write_manifest,
    write_rank_completion,
)


def distributed_context():
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    torch.cuda.set_device(local_rank)
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    return rank, world_size, local_rank, torch.device("cuda", local_rank)


def file_tree_signature(path_value: str) -> str:
    """Cheap deterministic model provenance signature (paths, sizes, mtimes)."""
    path = Path(path_value).resolve()
    digest = hashlib.sha256()
    digest.update(str(path).encode("utf-8"))
    files = [path] if path.is_file() else sorted(item for item in path.rglob("*") if item.is_file())
    for item in files:
        stat = item.stat()
        digest.update(str(item.relative_to(path.parent if path.is_file() else path)).encode("utf-8"))
        digest.update(str(stat.st_size).encode("ascii"))
        digest.update(str(stat.st_mtime_ns).encode("ascii"))
    return digest.hexdigest()


def file_sha256(path_value: str) -> str:
    digest = hashlib.sha256()
    with open(path_value, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_cpu(value: torch.Tensor, dtype=None) -> torch.Tensor:
    value = value.detach()
    if dtype is not None:
        value = value.to(dtype=dtype)
    return value.contiguous().cpu()


def token_positions(
    input_ids: torch.Tensor,
    tokenizer,
    act_token_count: int,
    with_mine_agent: bool = False,
) -> Dict[str, torch.Tensor]:
    groups = {
        "history": (ROBOT_HISTORY_TOKEN,),
        "rgb": RGB_QUERY_TOKENS,
        "gs": GS_QUERY_TOKENS,
    }
    if with_mine_agent:
        groups["mine_agent"] = MINE_AGENT_QUERY_TOKENS
    groups.update(
        {
            "action": action_query_tokens(act_token_count),
            "reward": REWARD_QUERY_TOKENS,
        }
    )
    result = {}
    for name, tokens in groups.items():
        ids = torch.as_tensor(tokenizer.convert_tokens_to_ids(list(tokens)), dtype=input_ids.dtype)
        matches = input_ids.cpu().unsqueeze(-1).eq(ids.view(1, -1))
        counts = matches.sum(dim=0)
        if not torch.all(counts == 1):
            raise RuntimeError(f"Expected one occurrence of each {name} token, found counts={counts.tolist()}")
        result[f"{name}_positions"] = matches.to(torch.int8).argmax(dim=0).long()
    return result


def make_dataset(cfg, component: str, max_samples: int | None):
    # The pre-cache process must never recursively read an older feature cache.
    os.environ.pop("NAVSIM_FEATURE_CACHE_ROOT", None)
    from starVLA.dataloader.navsim_dataset import NavSimDataset

    dataset_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    if component == "qwen":
        dataset_cfg.datasets.video_data.load_2d_data = 0
        dataset_cfg.w_depth = 0
    elif component == "wan":
        dataset_cfg.datasets.video_data.load_2d_data = 1
        dataset_cfg.w_depth = 0
    elif component == "ppd":
        dataset_cfg.datasets.video_data.load_2d_data = 0
        dataset_cfg.w_depth = 1
    else:
        raise ValueError(component)

    return NavSimDataset(
        datalist_path=dataset_cfg.datasets.vla_data.datalist_path,
        split=dataset_cfg.datasets.vla_data.split,
        video_data_cfg=dataset_cfg.datasets.video_data,
        gs_data_cfg=dataset_cfg.datasets.gs_data,
        reward_data_cfg=dataset_cfg.datasets.reward_data,
        ver_1225=dataset_cfg.ver_1225,
        dataset_cfg=dataset_cfg.datasets.vla_data,
        all_cfg=dataset_cfg,
        max_samples=max_samples,
    )


class QwenPrecomputer:
    def __init__(self, cfg, device: torch.device):
        from starVLA.model.modules.vlm.QWen3 import _QWen3_VL_Interface

        self.cfg = cfg
        self.device = device
        self.interface = _QWen3_VL_Interface(config=cfg).eval()
        self.interface.requires_grad_(False)
        self.tokenizer = self.interface.processor.tokenizer
        self.act_token_count = int(cfg.act_tok)
        self.w_depth = bool(cfg.w_depth)
        self.action_prompt_mode = str(getattr(cfg.framework, "action_prompt_mode", "full")).lower()

    @torch.inference_mode()
    def __call__(self, samples: List[dict]) -> List[Dict[str, torch.Tensor]]:
        instructions = [
            append_world_action_tokens(
                sample["lang"],
                self.act_token_count,
                self.w_depth,
                with_mine_agent=(self.action_prompt_mode == "minimal_agent"),
            )
            for sample in samples
        ]
        inputs = self.interface.build_qwenvl_inputs(
            images=[sample["image"] for sample in samples],
            instructions=instructions,
        )
        base = self.interface.model.model
        position_ids, _ = base.get_rope_index(
            input_ids=inputs["input_ids"],
            image_grid_thw=inputs["image_grid_thw"],
            video_grid_thw=inputs.get("video_grid_thw", None),
            attention_mask=inputs["attention_mask"],
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            image_parts, deepstack = base.get_image_features(
                inputs["pixel_values"],
                inputs["image_grid_thw"],
            )

        images_per_sample = len(samples[0]["image"])
        if any(len(sample["image"]) != images_per_sample for sample in samples):
            raise RuntimeError("Qwen pre-cache batch has inconsistent image counts")
        sample_image_embeds = [
            torch.cat(image_parts[index:index + images_per_sample], dim=0)
            for index in range(0, len(image_parts), images_per_sample)
        ]
        visual_counts = [value.shape[0] for value in sample_image_embeds]
        placeholder_counts = (
            inputs["input_ids"]
            .eq(self.interface.model.config.image_token_id)
            .sum(dim=1)
            .detach()
            .cpu()
            .tolist()
        )
        if placeholder_counts != visual_counts:
            raise RuntimeError(
                "Qwen visual placeholder mismatch: "
                f"prompt={placeholder_counts} embeddings={visual_counts}"
            )
        split_deepstack = [torch.split(value, visual_counts, dim=0) for value in deepstack]

        outputs = []
        for batch_index, sample_embeds in enumerate(sample_image_embeds):
            valid = inputs["attention_mask"][batch_index].bool()
            active_ids = inputs["input_ids"][batch_index, valid]
            active_attention = inputs["attention_mask"][batch_index, valid]
            active_positions = position_ids[:, batch_index, valid]
            payload = {
                "input_ids": tensor_cpu(active_ids, torch.long),
                "attention_mask": tensor_cpu(active_attention, torch.long),
                "position_ids": tensor_cpu(active_positions, torch.long),
                "image_grid_thw": tensor_cpu(
                    inputs["image_grid_thw"][
                        batch_index * images_per_sample:(batch_index + 1) * images_per_sample
                    ],
                    torch.long,
                ),
                "image_embeds": tensor_cpu(sample_embeds, torch.bfloat16),
            }
            payload.update(
                token_positions(
                    active_ids,
                    self.tokenizer,
                    self.act_token_count,
                    with_mine_agent=(self.action_prompt_mode == "minimal_agent"),
                )
            )
            for layer_index, layer_values in enumerate(split_deepstack):
                payload[f"deepstack_{layer_index}"] = tensor_cpu(
                    layer_values[batch_index],
                    torch.bfloat16,
                )
            outputs.append(payload)
        return outputs


class WanPrecomputer:
    def __init__(self, cfg, device: torch.device):
        from starVLA.model.modules.video_model.wan_i2v_header import resize_mask
        from starVLA.model.modules.video_model.videox_fun.models import AutoencoderKLWan, CLIPModel

        self.resize_mask = resize_mask
        self.device = device
        wan_cfg = OmegaConf.load(cfg.framework.video_model.config_path)
        model_root = cfg.framework.video_model.model_name
        self.vae = AutoencoderKLWan.from_pretrained(
            os.path.join(model_root, wan_cfg.vae_kwargs.get("vae_subpath", "vae")),
            additional_kwargs=OmegaConf.to_container(wan_cfg.vae_kwargs),
        ).to(device=device, dtype=torch.bfloat16).eval()
        self.clip = CLIPModel.from_pretrained(
            os.path.join(model_root, wan_cfg.image_encoder_kwargs.get("image_encoder_subpath", "image_encoder")),
        ).to(device=device, dtype=torch.bfloat16).eval()
        self.vae.requires_grad_(False)
        self.clip.requires_grad_(False)

    @torch.inference_mode()
    def __call__(self, samples: List[dict]) -> List[Dict[str, torch.Tensor]]:
        items = [sample["2d_gen_data"] for sample in samples]
        pixel_values = torch.stack([item["pixel_values"] for item in items]).to(
            self.device, dtype=torch.bfloat16
        )
        mask_pixel_values = torch.stack([item["mask_pixel_values"] for item in items]).to(
            self.device, dtype=torch.bfloat16
        )
        raw_mask = torch.stack([item["mask"] for item in items]).to(self.device)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            posterior = self.vae.encode(rearrange(pixel_values, "b f c h w -> b c f h w"))[0]
            mask_posterior = self.vae.encode(
                rearrange(mask_pixel_values, "b f c h w -> b c f h w")
            )[0]

        latent_reference = posterior.mean
        latent_mask = rearrange(raw_mask, "b f c h w -> b c f h w").to(torch.bfloat16)
        latent_mask = torch.concat(
            [
                torch.repeat_interleave(latent_mask[:, :, 0:1], repeats=4, dim=2),
                latent_mask[:, :, 1:],
            ],
            dim=2,
        )
        latent_mask = latent_mask.view(
            latent_mask.shape[0], latent_mask.shape[2] // 4, 4,
            latent_mask.shape[3], latent_mask.shape[4]
        ).transpose(1, 2)
        latent_mask = self.resize_mask(1 - latent_mask, latent_reference)

        clip_images = []
        for item in items:
            # Preserve the released BF16 -> uint8 -> PIL preprocessing exactly.
            clip_pixels = item["clip_pixel_values"].to(torch.bfloat16)
            clip_image = Image.fromarray(np.uint8(clip_pixels.float().cpu().numpy()))
            clip_image = TF.to_tensor(clip_image).sub_(0.5).div_(0.5)
            clip_images.append(clip_image[:, None].to(self.device, dtype=torch.bfloat16))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            clip_context = self.clip(clip_images)

        outputs = []
        for index in range(len(samples)):
            outputs.append(
                {
                    "latent_parameters": tensor_cpu(posterior.parameters[index], torch.bfloat16),
                    "mask_latent_parameters": tensor_cpu(
                        mask_posterior.parameters[index], torch.bfloat16
                    ),
                    "latent_mask": tensor_cpu(latent_mask[index], torch.bfloat16),
                    "clip_context": tensor_cpu(clip_context[index], torch.bfloat16),
                    "mask_is_all_ones": tensor_cpu(
                        raw_mask[index].eq(1).all().reshape(1), torch.bool
                    ),
                }
            )
        return outputs


class PPDPrecomputer:
    def __init__(self, cfg, device: torch.device):
        from starVLA.model.modules.depth_model.models.depth_anything_v2.dpt import DepthAnythingV2

        ppd_cfg = OmegaConf.load("starVLA/model/modules/depth_model/configs/train_finetune.yaml")
        semantics_path = cfg.cache_depth_model
        self.device = device
        self.encoder = DepthAnythingV2(
            encoder="vitl",
            features=256,
            out_channels=[256, 512, 1024, 1024],
        )
        self.encoder.load_state_dict(torch.load(semantics_path, map_location="cpu"), strict=False)
        self.encoder = self.encoder.to(device=device, dtype=torch.bfloat16).eval()
        self.encoder.requires_grad_(False)

    @torch.inference_mode()
    def __call__(self, samples: List[dict]) -> List[Dict[str, torch.Tensor]]:
        images = torch.stack([sample["depth_data"]["image"] for sample in samples])
        batch_size, views = images.shape[:2]
        flat_images = images.reshape(batch_size * views, *images.shape[2:]).to(self.device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            semantics = self.encoder.forward_semantics(flat_images)
        semantics = semantics.reshape(batch_size, views, *semantics.shape[1:])

        outputs = []
        for index, sample in enumerate(samples):
            depth_data = sample["depth_data"]
            image_uint8 = depth_data["image"].mul(255.0).round().clamp_(0, 255).to(torch.uint8)
            outputs.append(
                {
                    "image_uint8": tensor_cpu(image_uint8, torch.uint8),
                    "depth": tensor_cpu(depth_data["depth"], torch.float32),
                    "mask": tensor_cpu(depth_data["mask"], torch.bool),
                    "semantics": tensor_cpu(semantics[index], torch.bfloat16),
                }
            )
        return outputs


def load_config(args):
    cfg = OmegaConf.load(args.config_yaml)
    cfg.framework.name = "QwenOFT"
    cfg.framework.qwenvl.base_vlm = args.base_vlm
    cfg.framework.qwenvl.attn_implementation = args.attn_implementation
    cfg.framework.video_model.model_name = args.video_model
    cfg.framework.video_model.config_path = args.video_config
    cfg.datasets.vla_data.datalist_path = args.datalist
    cfg.datasets.vla_data.data_root = args.data_root
    cfg.datasets.vla_data.split = args.split
    cfg.datasets.video_data.rgb_meta_dir = os.path.join(args.data_root, "navsim_video")
    cfg.datasets.video_data.load_2d_data = 1
    cfg.datasets.gs_data.load_3d_data = 0
    cfg.datasets.reward_data.load_reward_data = 0
    cfg.w_depth = 1
    cfg.rgb_query_loss = 1
    cfg.gs_query_loss = 1
    cfg.cache_depth_model = args.depth_model
    return cfg


def chunks(values: List[int], size: int) -> Iterable[List[int]]:
    for offset in range(0, len(values), size):
        yield values[offset:offset + size]


def main(args):
    rank, world_size, local_rank, device = distributed_context()
    cfg = load_config(args)
    dataset = make_dataset(cfg, args.component, args.max_samples)
    sample_count = len(dataset)
    owned_indices = list(range(rank, sample_count, world_size))
    map_size_bytes = int(args.map_size_gb * 1024**3)
    constructor = {
        "qwen": QwenPrecomputer,
        "wan": WanPrecomputer,
        "ppd": PPDPrecomputer,
    }[args.component]
    precomputer = constructor(cfg, device)

    started = time.time()
    with RankCacheWriter(
        cache_root=args.cache_root,
        component=args.component,
        rank=rank,
        map_size_bytes=map_size_bytes,
        commit_interval=args.commit_interval,
    ) as writer:
        processed = 0
        for batch_indices in chunks(owned_indices, args.batch_size):
            pending = []
            for sample_index in batch_indices:
                token = dataset.raw_list[sample_index]
                if not args.overwrite and writer.contains(token):
                    writer.skipped += 1
                else:
                    pending.append((sample_index, token))
            if not pending:
                continue
            samples = [dataset[sample_index] for sample_index, _ in pending]
            payloads = precomputer(samples)
            for (_, token), payload in zip(pending, payloads):
                writer.put(token, payload, overwrite=args.overwrite)
                processed += 1
            if rank == 0 and processed % args.log_interval < len(pending):
                elapsed = time.time() - started
                rate = processed / max(elapsed, 1e-6)
                print(
                    f"[pre-cache] component={args.component} rank=0 "
                    f"processed={processed}/{len(owned_indices)} rate={rate:.2f} samples/s",
                    flush=True,
                )
        completion = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "component": args.component,
            "rank": rank,
            "world_size": world_size,
            "sample_count": sample_count,
            "owned_samples": len(owned_indices),
            "written": writer.written,
            "skipped": writer.skipped,
            "elapsed_seconds": time.time() - started,
        }

    write_rank_completion(args.cache_root, args.component, rank, completion)
    if dist.is_initialized():
        dist.barrier()

    if rank == 0:
        component_path = Path(args.cache_root) / args.component
        completions = []
        for owner_rank in range(world_size):
            path = component_path / f"rank_{owner_rank:05d}.complete.json"
            if not path.is_file():
                raise FileNotFoundError(f"Missing rank completion marker: {path}")
            completions.append(json.loads(path.read_text(encoding="utf-8")))
        model_paths = {
            "qwen": [args.base_vlm],
            "wan": [args.video_model],
            "ppd": [args.ppd_model, args.depth_model],
        }[args.component]
        manifest = {
            "component": args.component,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "world_size": world_size,
            "sample_count": sample_count,
            "split": args.split,
            "datalist": str(Path(args.datalist).resolve()),
            "datalist_sha256": file_sha256(args.datalist),
            "model_paths": [str(Path(value).resolve()) for value in model_paths],
            "model_signatures": {
                str(Path(value).resolve()): file_tree_signature(value)
                for value in model_paths
            },
            "config_yaml": str(Path(args.config_yaml).resolve()),
            "cache_contract": {
                "ver_1225": int(cfg.ver_1225),
                "act_tok": int(cfg.act_tok),
                "w_depth": int(cfg.w_depth),
                "enable_image_aug": int(cfg.enable_image_aug),
                "video_source": os.environ.get("NAVSIM_VIDEO_SOURCE", "images"),
                "video_text_input": int(cfg.datasets.video_data.text_input),
                "load_2d_data": 1,
                "load_3d_data": 0,
                "load_reward_data": 0,
            },
            "batch_size_per_rank": args.batch_size,
            "rank_completions": completions,
        }
        path = write_manifest(args.cache_root, args.component, manifest)
        total_elapsed = max(value["elapsed_seconds"] for value in completions)
        print(
            f"[pre-cache] COMPLETE component={args.component} samples={sample_count} "
            f"world_size={world_size} elapsed={total_elapsed:.1f}s manifest={path}",
            flush=True,
        )

    del precomputer
    gc.collect()
    torch.cuda.empty_cache()
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--component", choices=("qwen", "wan", "ppd"), required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--config-yaml", default="starVLA/config/training/cfg_yaw_1225.yaml")
    parser.add_argument("--base-vlm", required=True)
    parser.add_argument("--video-model", required=True)
    parser.add_argument("--ppd-model", required=True)
    parser.add_argument("--depth-model", required=True)
    parser.add_argument("--video-config", default="starVLA/model/modules/video_model/config/wan2.1/wan_civitai.yaml")
    parser.add_argument("--datalist", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--map-size-gb", type=int, default=256)
    parser.add_argument("--commit-interval", type=int, default=16)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--overwrite", action="store_true")
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())

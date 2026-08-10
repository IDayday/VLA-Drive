"""
Evaluate DriveDreamer-Policy world heads on NAVSIM samples.

The script loads a trained QwenOFT checkpoint, runs the 2D Wan future-video
head and/or the PPD depth head, saves sparse visualizations, and writes
aggregate metrics to metrics.json. Metrics are computed for every evaluated
sample; visual files are saved only when sample_index % --save_visual_every == 0
(default: 100) to avoid large storage usage on full test.

Typical full-test command:

    BASE_VLM=$PWD/weights/derived/Qwen3-VL-2B-WorldAction \
    OPENSCENE_DATA_ROOT=/mnt/workspace/Public_Space/navsim \
    NAVSIM_SENSOR_BLOBS_ROOT=/mnt/workspace/Public_Space/navsim/test_sensor_blobs \
    NAVSIM_VIDEO_SOURCE=images \
    CUDA_VISIBLE_DEVICES=0 \
    python eval_world_heads.py \
      --ckpt_dir navsim_exp/navsim-v1-formal-16gpu-bz2-ga1-flashattn-20260801_073617/final_model/pytorch_model.pt \
      --datalist_path test_meta.json \
      --data_root navsim_dataset \
      --out_dir artifacts/world_head_eval_test \
      --split test \
      --tasks video,depth \
      --max_samples 0 \
      --batch_size 1 \
      --num_workers 0 \
      --save_visual_every 100 \
      --depth_semantics_pth /mnt/workspace/VLA_Group/LLM_weight/depth-anything/Depth-Anything-V2-Large/depth_anything_v2_vitl.pth \
      --depth_ppd_pth /mnt/workspace/VLA_Group/LLM_weight/gangweix/Pixel-Perfect-Depth/ppd.pth \
      --wan_model_path /mnt/workspace/VLA_Group/LLM_weight/alibaba-pai/Wan2.1-Fun-V1.1-1.3B-InP \
      --video_root navsim_dataset/navsim_video

Key parameters:
  --ckpt_dir             Experiment directory or pytorch_model.pt checkpoint.
  --datalist_path        JSON token list, e.g. test_meta.json.
  --data_root            Processed NAVSIM root containing meta/ and navsim_video/.
  --out_dir              Directory for metrics JSON and sparse visualizations.
  --split                Dataset split passed to NavSimDataset.
  --tasks                Comma list: video, depth, or video,depth.
  --max_samples          0 means all samples; positive values run a prefix only.
  --save_visual_every    Save mp4/jpg every N evaluated global samples; 0 disables visuals.
  --rank/--world_size    Optional dataset sharding for multi-process evaluation.
  --depth_semantics_pth  Depth-Anything-V2 ViT-L checkpoint for PPD.
  --depth_ppd_pth        Pixel-Perfect-Depth checkpoint.
  --wan_model_path       Wan2.1-Fun base model directory.
  --video_root           Root containing navsim_video/<split>/ when using mp4 source.

Depth metrics are computed in the PPD normalized log-depth latent space, matching
the training target. Video metrics are computed on generated future frames against
GT RGB frames after the first conditioning frame.
"""

import argparse
import json
import math
import os
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from infer import to_device
from starVLA.cache.navsim_feature_cache import append_world_action_tokens
from starVLA.dataloader.navsim_dataset import NavSimDataset, collate_fn
from starVLA.model.framework.QwenOFT import Qwenvl_OFT


def atomic_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def resolve_config_path(ckpt: str) -> Path:
    p = Path(ckpt)
    if p.suffix == ".pt":
        candidates = [p.parent.parent / "config.yaml", p.parent.parent.parent / "config.yaml"]
    else:
        candidates = [p / "config.yaml"]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"config.yaml not found for checkpoint: {ckpt}")


def resolve_weight_path(ckpt: str, model_iter: Optional[int]) -> Path:
    p = Path(ckpt)
    if p.suffix == ".pt":
        return p
    if model_iter is not None:
        candidate = p / "checkpoints" / f"steps_{model_iter}_pytorch_model.pt"
        if not candidate.is_file():
            raise FileNotFoundError(str(candidate))
        return candidate
    for candidate in [p / "final_model" / "pytorch_model.pt", p / "pytorch_model.pt"]:
        if candidate.is_file():
            return candidate
    ckpt_dir = p / "checkpoints"
    pat = re.compile(r"steps_(\d+)_pytorch_model\.pt")
    files = [x for x in ckpt_dir.iterdir() if x.is_file() and pat.search(x.name)]
    if not files:
        raise FileNotFoundError(f"No checkpoint weights found under {p}")
    return sorted(files, key=lambda x: int(pat.search(x.name).group(1)))[-1]


def load_model(
    ckpt: str,
    device: torch.device,
    model_iter: Optional[int],
    enable_video: bool,
    enable_depth: bool,
    qwen_forward_mode: str,
) -> Tuple[Qwenvl_OFT, Any, Path, Path]:
    cfg_path = resolve_config_path(ckpt)
    weight_path = resolve_weight_path(ckpt, model_iter)
    cfg = OmegaConf.load(cfg_path)

    if os.environ.get("BASE_VLM"):
        OmegaConf.update(cfg, "framework.qwenvl.base_vlm", os.environ["BASE_VLM"], force_add=True)
    if os.environ.get("WAN_MODEL_PATH"):
        OmegaConf.update(cfg, "framework.video_model.model_name", os.environ["WAN_MODEL_PATH"], force_add=True)
    if os.environ.get("NAVSIM_VIDEO_ROOT"):
        OmegaConf.update(cfg, "datasets.video_data.rgb_meta_dir", os.environ["NAVSIM_VIDEO_ROOT"], force_add=True)
    OmegaConf.update(
        cfg,
        "framework.qwenvl.attn_implementation",
        os.environ.get("VLM_ATTN_IMPLEMENTATION", "sdpa"),
        force_add=True,
    )
    OmegaConf.update(cfg, "datasets.video_data.load_2d_data", int(enable_video), force_add=True)
    OmegaConf.update(cfg, "w_depth", int(enable_depth), force_add=True)
    OmegaConf.update(cfg, "datasets.gs_data.load_3d_data", 0, force_add=True)
    OmegaConf.update(cfg, "datasets.reward_data.load_reward_data", 0, force_add=True)
    OmegaConf.update(cfg, "enable_image_aug", 0, force_add=True)
    OmegaConf.update(cfg, "datasets.vla_data.w_neg_traj", None, force_add=True)

    model = Qwenvl_OFT(cfg, infer_not_load_wan=0 if enable_video else 1)
    model._inference_qwen_forward_mode = qwen_forward_mode
    state = torch.load(weight_path, map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[WorldEval] config:  {cfg_path}")
    print(f"[WorldEval] weights: {weight_path}")
    print(f"[WorldEval] missing keys: {len(missing)} {missing[:20]}")
    print(f"[WorldEval] unexpected keys: {len(unexpected)} {unexpected[:20]}")
    model.to(device).eval()
    if enable_video and hasattr(model, "rgb_model"):
        model.rgb_model.accelerator = SimpleNamespace(device=device)
    return model, cfg, cfg_path, weight_path


@torch.no_grad()
def qwen_world_queries(model: Qwenvl_OFT, examples: List[dict]) -> Dict[str, torch.Tensor]:
    instructions = [
        append_world_action_tokens(example["lang"], model.act_tok, bool(model.w_depth))
        for example in examples
    ]
    (
        input_ids,
        attention_mask,
        position_ids,
        token_positions,
        image_embeds,
        deepstack_embeds,
    ) = model._build_qwen_batch(examples, instructions)

    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
        text_embeds = model.qwen_vl_interface.model.get_input_embeddings()(input_ids)

    states = torch.as_tensor(
        np.asarray([example["state"] for example in examples]),
        device=text_embeds.device,
        dtype=torch.float32,
    )[:, 0, :]
    with torch.autocast("cuda", dtype=torch.float32, enabled=torch.cuda.is_available()):
        states_embed = model.action_input_model(states)
    states_embed = states_embed.to(dtype=text_embeds.dtype)

    bsz, _, hidden = text_embeds.shape
    batch_idx = torch.arange(bsz, device=text_embeds.device)
    text_embeds[batch_idx, token_positions["history"][:, 0], :] = states_embed
    if model.config.datasets.video_data.load_2d_data:
        text_embeds[batch_idx[:, None], token_positions["rgb"], :] = model.rgb_query.unsqueeze(0).to(text_embeds.dtype)
    if model.config.datasets.gs_data.load_3d_data or model.w_depth:
        text_embeds[batch_idx[:, None], token_positions["gs"], :] = model.gs_query.unsqueeze(0).to(text_embeds.dtype)

    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
        if getattr(model, "_inference_qwen_forward_mode", "optimized") == "optimized":
            last_hidden = model._qwen_language_forward(
                input_ids=input_ids,
                inputs_embeds=text_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                image_embeds=image_embeds,
                deepstack_embeds=deepstack_embeds,
            )
        else:
            out = model.qwen_vl_interface(
                inputs_embeds=text_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                pixel_values=None,
                image_grid_thw=None,
                output_hidden_states=True,
                return_dict=True,
            )
            last_hidden = out.hidden_states[-1]

    result = {}
    if model.config.datasets.video_data.load_2d_data:
        idx = token_positions["rgb"].unsqueeze(-1).expand(-1, -1, hidden)
        result["rgb"] = last_hidden.gather(1, idx)
    if model.config.datasets.gs_data.load_3d_data or model.w_depth:
        idx = token_positions["gs"].unsqueeze(-1).expand(-1, -1, hidden)
        result["gs"] = last_hidden.gather(1, idx)
    return result


def video_to_uint8_frames(video: Any) -> np.ndarray:
    arr = video.detach().float().cpu().numpy() if torch.is_tensor(video) else np.asarray(video)
    if arr.ndim != 4:
        raise ValueError(f"Expected video with 4 dims, got {arr.shape}")
    if arr.shape[0] in {1, 3, 4}:
        arr = np.transpose(arr[:3], (1, 2, 3, 0))
    elif arr.shape[-1] in {1, 3, 4}:
        arr = arr[..., :3]
    else:
        raise ValueError(f"Cannot infer video layout: {arr.shape}")
    if arr.max() <= 1.5:
        arr = arr * 255.0
    return np.clip(arr, 0, 255).astype(np.uint8)


def gt_video_uint8(sample: dict) -> np.ndarray:
    gt = sample["2d_gen_data"]["pixel_values"].detach().float().cpu()
    gt = (gt * 0.5 + 0.5).clamp(0, 1)
    gt = gt.permute(0, 2, 3, 1).numpy() * 255.0
    return np.clip(gt, 0, 255).astype(np.uint8)


def write_mp4(path: Path, frames: np.ndarray, fps: int = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames.shape[1:3]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {path}")
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


def write_strip(path: Path, frames: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    strip = np.concatenate(list(frames), axis=1)
    cv2.imwrite(str(path), cv2.cvtColor(strip, cv2.COLOR_RGB2BGR))


def psnr_from_mse(mse: float) -> float:
    return 99.0 if mse <= 1e-12 else 20.0 * math.log10(255.0 / math.sqrt(mse))


def video_metrics(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    n = min(len(pred), len(gt))
    pred_f = pred[:n].astype(np.float32)
    gt_f = gt[:n].astype(np.float32)
    diff = pred_f - gt_f
    mse = float(np.mean(diff ** 2))
    out = {
        "mae": float(np.mean(np.abs(diff))),
        "mse": mse,
        "psnr": psnr_from_mse(mse),
    }
    try:
        from skimage.metrics import structural_similarity

        scores = [
            structural_similarity(gt_f[i], pred_f[i], channel_axis=2, data_range=255)
            for i in range(n)
        ]
        out["ssim"] = float(np.mean(scores))
    except Exception:
        pass
    return out


def depth_color(depth: np.ndarray, valid: Optional[np.ndarray] = None) -> np.ndarray:
    d = np.asarray(depth, dtype=np.float32)
    if valid is None:
        valid = np.isfinite(d)
    if valid.any():
        lo, hi = np.quantile(d[valid], [0.02, 0.98])
    else:
        lo, hi = float(np.nanmin(d)), float(np.nanmax(d))
    if hi <= lo:
        hi = lo + 1e-6
    x = np.clip((d - lo) / (hi - lo), 0, 1)
    colored = cv2.applyColorMap((x * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    colored[~valid] = 0
    return colored


def depth_target_norm(model: Qwenvl_OFT, depth_batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    latent, mask = model.gs_model.get_gt(depth_batch)
    return latent + 0.5, mask.bool()


@torch.no_grad()
def predict_depth(model: Qwenvl_OFT, depth_batch: Dict[str, torch.Tensor], gs_queries: torch.Tensor) -> torch.Tensor:
    image = depth_batch["image"]
    cond = model.gs_model.get_cond(image)
    semantics = depth_batch.get("semantics")
    if semantics is None:
        semantics = model.gs_model.semantics_prompt(image)
    else:
        semantics = semantics.to(image.device)
    latent = torch.randn(
        size=[cond.shape[0], 1, cond.shape[2], cond.shape[3]],
        device=image.device,
        dtype=cond.dtype,
    )
    qwen_tokens = gs_queries.to(device=image.device, dtype=torch.float32).repeat_interleave(3, dim=0)
    for timestep in model.gs_model.sampling_timesteps:
        x = torch.cat([latent, cond], dim=1)
        pred = model.gs_model.dit(
            x=x,
            semantics=semantics,
            timestep=timestep,
            qwen_tokens=qwen_tokens,
        )
        latent = model.gs_model.sampler.step(pred=pred, x_t=latent, t=timestep)
    return latent + 0.5


def depth_metrics(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> Dict[str, float]:
    p = pred.float()[mask]
    t = target.float()[mask]
    diff = p - t
    mse = torch.mean(diff ** 2)
    return {
        "mae": float(torch.mean(torch.abs(diff)).detach().cpu()),
        "rmse": float(torch.sqrt(mse).detach().cpu()),
        "abs_rel": float(torch.mean(torch.abs(diff) / torch.clamp(t.abs(), min=1e-3)).detach().cpu()),
    }


def mean_dict(items: List[Dict[str, float]]) -> Dict[str, float]:
    keys = sorted(
        {
            k
            for item in items
            for k, value in item.items()
            if isinstance(value, (int, float, np.integer, np.floating))
        }
    )
    return {k: float(np.mean([item[k] for item in items if k in item])) for k in keys}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("Evaluate DriveDreamer world heads")
    p.add_argument("--ckpt_dir", required=True)
    p.add_argument("--datalist_path", required=True)
    p.add_argument("--data_root", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--split", default="mini")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--max_samples", type=int, default=8)
    p.add_argument("--model_iter", type=int, default=None)
    p.add_argument("--device", default="cuda")
    p.add_argument("--tasks", default="video,depth", help="Comma list: video,depth")
    p.add_argument("--qwen_forward_mode", choices=("optimized", "legacy"), default="optimized")
    p.add_argument("--save_visual_every", type=int, default=100)
    p.add_argument("--rank", type=int, default=0)
    p.add_argument("--world_size", type=int, default=1)
    p.add_argument("--depth_semantics_pth", default=None)
    p.add_argument("--depth_ppd_pth", default=None)
    p.add_argument("--wan_model_path", default=None)
    p.add_argument("--video_root", default=None)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    tasks = {x.strip() for x in args.tasks.split(",") if x.strip()}
    enable_video = "video" in tasks
    enable_depth = "depth" in tasks
    if not enable_video and not enable_depth:
        raise ValueError("--tasks must include video and/or depth")
    if args.depth_semantics_pth:
        os.environ["DEPTH_ANYTHING_V2_VITL_PATH"] = args.depth_semantics_pth
    if args.depth_ppd_pth:
        os.environ["PPD_CKPT_PATH"] = args.depth_ppd_pth
    if args.wan_model_path:
        os.environ["WAN_MODEL_PATH"] = args.wan_model_path
    if args.video_root:
        os.environ["NAVSIM_VIDEO_ROOT"] = args.video_root

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, cfg, cfg_path, weight_path = load_model(
        args.ckpt_dir,
        device,
        args.model_iter,
        enable_video,
        enable_depth,
        args.qwen_forward_mode,
    )

    data_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    data_cfg.datasets.video_data.load_2d_data = int(enable_video)
    data_cfg.datasets.gs_data.load_3d_data = 0
    data_cfg.datasets.reward_data.load_reward_data = 0
    data_cfg.datasets.vla_data.w_neg_traj = None
    data_cfg.w_depth = int(enable_depth)
    data_cfg.enable_image_aug = 0

    dataset = NavSimDataset(
        datalist_path=args.datalist_path,
        split=args.split,
        video_data_cfg=data_cfg.datasets.video_data,
        gs_data_cfg=data_cfg.datasets.gs_data,
        reward_data_cfg=data_cfg.datasets.reward_data,
        ver_1225=OmegaConf.select(data_cfg, "ver_1225", default=False),
        dataset_cfg=data_cfg.datasets.vla_data,
        all_cfg=data_cfg,
        data_root=args.data_root,
    )
    if args.world_size < 1 or not 0 <= args.rank < args.world_size:
        raise ValueError("--rank must be in [0, --world_size)")
    indices = list(range(args.rank, len(dataset), args.world_size))
    if args.max_samples > 0:
        indices = indices[: args.max_samples]
    if args.world_size > 1 or args.max_samples > 0:
        dataset = Subset(dataset, indices)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=False,
    )

    out_dir = Path(args.out_dir)
    video_scores: List[Dict[str, float]] = []
    depth_scores: List[Dict[str, float]] = []

    sample_offset = 0
    for batch in tqdm(loader, desc=f"WorldEval {args.rank}/{args.world_size}"):
        batch_dev = to_device(batch, device)
        queries = qwen_world_queries(model, batch_dev)
        tokens = [item["token"] for item in batch]
        global_indices = [indices[sample_offset + i] for i in range(len(batch))]
        sample_offset += len(batch)

        if enable_video:
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
                pred_videos = model.rgb_model.predict_rgb(
                    [item["2d_gen_data"] for item in batch_dev],
                    queries["rgb"],
                )
            for i, token in enumerate(tokens):
                pred = video_to_uint8_frames(pred_videos[i])
                gt = gt_video_uint8(batch[i])
                score = video_metrics(pred[1:], gt[1:])
                score["token"] = token
                video_scores.append(score)
                save_visual = args.save_visual_every > 0 and global_indices[i] % args.save_visual_every == 0
                if save_visual:
                    write_mp4(out_dir / "video" / f"{token}_pred.mp4", pred)
                    write_mp4(out_dir / "video" / f"{token}_gt.mp4", gt)
                    write_strip(out_dir / "video" / f"{token}_pred_strip.jpg", pred)
                    write_strip(out_dir / "video" / f"{token}_gt_strip.jpg", gt)

        if enable_depth:
            depth_data = [item["depth_data"] for item in batch_dev]
            image = torch.stack([x["image"] for x in depth_data])
            gt_depth = torch.stack([x["depth"] for x in depth_data])
            mask = torch.stack([x["mask"] for x in depth_data])
            bsz, views = image.shape[:2]
            depth_batch = {
                "image": image.reshape(bsz * views, *image.shape[2:]),
                "depth": gt_depth.reshape(bsz * views, *gt_depth.shape[2:]),
                "mask": mask.reshape(bsz * views, *mask.shape[2:]),
            }
            if all("semantics" in x for x in depth_data):
                sem = torch.stack([x["semantics"] for x in depth_data])
                depth_batch["semantics"] = sem.reshape(bsz * views, *sem.shape[2:])
            pred = predict_depth(model, depth_batch, queries["gs"])
            target, valid = depth_target_norm(model, depth_batch)
            score = depth_metrics(pred, target, valid)
            for i, token in enumerate(tokens):
                per_pred = pred[i * views : (i + 1) * views].detach().float().cpu().numpy()[:, 0]
                per_tgt = target[i * views : (i + 1) * views].detach().float().cpu().numpy()[:, 0]
                per_mask = valid[i * views : (i + 1) * views].detach().cpu().numpy()[:, 0]
                per_score = depth_metrics(
                    pred[i * views : (i + 1) * views],
                    target[i * views : (i + 1) * views],
                    valid[i * views : (i + 1) * views],
                )
                per_score["token"] = token
                depth_scores.append(per_score)
                save_visual = args.save_visual_every > 0 and global_indices[i] % args.save_visual_every == 0
                if save_visual:
                    for view_idx, view_name in enumerate(["cam_l0", "cam_f0", "cam_r0"][:views]):
                        pred_color = depth_color(per_pred[view_idx], per_mask[view_idx])
                        tgt_color = depth_color(per_tgt[view_idx], per_mask[view_idx])
                        strip = np.concatenate([pred_color, tgt_color], axis=1)
                        path = out_dir / "depth" / f"{token}_{view_name}_pred_gt.jpg"
                        path.parent.mkdir(parents=True, exist_ok=True)
                        cv2.imwrite(str(path), cv2.cvtColor(strip, cv2.COLOR_RGB2BGR))

    metrics = {
        "checkpoint": str(Path(args.ckpt_dir).resolve()),
        "config": str(cfg_path.resolve()),
        "weights": str(weight_path.resolve()),
        "split": args.split,
        "rank": args.rank,
        "world_size": args.world_size,
        "save_visual_every": args.save_visual_every,
        "num_video_samples": len(video_scores),
        "num_depth_samples": len(depth_scores),
        "video": mean_dict(video_scores) if video_scores else {},
        "depth_normalized_log": mean_dict(depth_scores) if depth_scores else {},
        "note": "Depth metrics are computed in PPD normalized log-depth latent space.",
    }
    atomic_json(out_dir / "metrics.json", metrics)
    if video_scores:
        atomic_json(out_dir / "video_metrics_per_sample.json", {"samples": video_scores})
    if depth_scores:
        atomic_json(out_dir / "depth_metrics_per_sample.json", {"samples": depth_scores})
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

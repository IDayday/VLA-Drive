#!/usr/bin/env python3
"""Visualize whether agent-query aligned features activate inside teacher bboxes.

Example:
  BASE_VLM=/mnt/workspace/VLA-Drive/weights/derived/Qwen3-VL-2B-WorldAction \
  python3 tools/visualize_agent_query_dino_heatmap.py \
    --ckpt-dir /mnt/workspace/VLA-Drive/navsim_exp/xxx \
    --datalist-path /mnt/workspace/VLA-Drive/train_meta.json \
    --split train \
    --data-root /mnt/workspace/VLA-Drive/navsim_dataset \
    --agent-cache-root /mnt/workspace/VLA-Drive/navsim_feature_cache/agent_dino_vits14_train_top4_multiview \
    --sensor-dir /mnt/data/navsim/trainval_sensor_blobs/trainval \
    --output-dir /mnt/workspace/VLA-Drive/navsim_exp/agent_query_dino_heatmaps \
    --max-samples 16 \
    --query-indices 0 1 2 3 \
    --views best
"""

from __future__ import annotations

import argparse
import copy
import io
import itertools
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from infer import VLAAgent, to_device
from starVLA.dataloader.navsim_dataset import NavSimDataset, collate_fn
from tools.precompute_agent_dino_cache import VIEW_TO_ID, load_local_dino, split_sensor_suffix


def build_dataset(agent: VLAAgent, args: argparse.Namespace) -> NavSimDataset:
    cfg = agent.model_config
    data_cfg = copy.deepcopy(cfg)
    data_cfg.datasets.video_data.load_2d_data = 0
    data_cfg.datasets.video_data.load_3d_data = 0
    data_cfg.datasets.gs_data.load_3d_data = 0
    data_cfg.datasets.reward_data.load_reward_data = 0
    data_cfg.datasets.vla_data.w_neg_traj = None
    data_cfg.w_depth = 0
    data_cfg.enable_image_aug = 0
    return NavSimDataset(
        datalist_path=args.datalist_path,
        split=args.split,
        video_data_cfg=data_cfg.datasets.video_data,
        gs_data_cfg=data_cfg.datasets.gs_data,
        reward_data_cfg=data_cfg.datasets.reward_data,
        ver_1225=OmegaConf.select(cfg, "ver_1225", default=False),
        dataset_cfg=data_cfg.datasets.vla_data,
        all_cfg=data_cfg,
        max_samples=getattr(args, "dataset_max_samples", None) if getattr(args, "dataset_max_samples", None) is not None else args.max_samples,
        data_root=args.data_root,
    )


def decode_metadata(payload: Dict[str, Any]) -> Dict[str, Any]:
    value = payload.get("metadata_json_uint8")
    if value is None:
        return {}
    if torch.is_tensor(value):
        raw = bytes(value.detach().cpu().tolist())
    else:
        raw = bytes(value)
    return json.loads(raw.decode("utf-8"))


def resolve_image_path(image_path: str, sensor_dir: Optional[str]) -> str:
    path = Path(image_path)
    if path.is_file():
        return str(path)
    suffix = split_sensor_suffix(image_path)
    if suffix and sensor_dir:
        candidate = Path(sensor_dir) / suffix
        if candidate.is_file():
            return str(candidate)
    raise FileNotFoundError(f"image not found: {image_path}; sensor_dir={sensor_dir}")


@torch.inference_mode()
def extract_agent_query_predictions(model, examples: List[dict]) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    suffix = model._build_action_prompt_suffix()
    instructions = [example["lang"] + suffix for example in examples]
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

    state_device = next(model.action_input_model.parameters()).device
    states = torch.as_tensor(np.asarray([example["state"] for example in examples]), dtype=torch.float32, device=state_device)[:, 0, :]
    with torch.autocast("cuda", dtype=torch.float32, enabled=torch.cuda.is_available()):
        states_embed = model.action_input_model(states)
    states_embed = states_embed.to(dtype=text_embeds.dtype, device=text_embeds.device)

    batch_size, _, hidden = text_embeds.shape
    batch_indices = torch.arange(batch_size, device=text_embeds.device)
    text_embeds[batch_indices, token_positions["history"][:, 0], :] = states_embed

    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
        last_hidden = model._qwen_language_forward(
            input_ids=input_ids,
            inputs_embeds=text_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            image_embeds=image_embeds,
            deepstack_embeds=deepstack_embeds,
        )

    mine_agent_g_idx = token_positions["mine_agent"].unsqueeze(-1).expand(-1, -1, hidden)
    mine_agent_queries = last_hidden.gather(dim=1, index=mine_agent_g_idx)
    if next(model.agent_dino_head.parameters()).device != mine_agent_queries.device:
        model.agent_dino_head = model.agent_dino_head.to(mine_agent_queries.device)
    pred_features = model.agent_dino_head(mine_agent_queries.float()).float()
    pred_bboxes = None
    if hasattr(model, "agent_bbox_head"):
        if next(model.agent_bbox_head.parameters()).device != mine_agent_queries.device:
            model.agent_bbox_head = model.agent_bbox_head.to(mine_agent_queries.device)
        pred_bbox_raw = model.agent_bbox_head(mine_agent_queries.float()).view(
            mine_agent_queries.shape[0], mine_agent_queries.shape[1], model.agent_view_count, 4
        )
        pred_bboxes = xywh_to_xyxy(pred_bbox_raw).float()
    return pred_features, pred_bboxes


def xywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    center = boxes[..., :2]
    size = boxes[..., 2:].clamp(min=1e-4)
    top_left = center - size * 0.5
    bottom_right = center + size * 0.5
    return torch.cat([top_left, bottom_right], dim=-1).clamp(0.0, 1.0)


def denormalize_bbox(bbox: Iterable[float], width: int, height: int) -> List[float]:
    x1, y1, x2, y2 = [float(v) for v in bbox]
    return [x1 * width, y1 * height, x2 * width, y2 * height]


def top_response_component_bbox(heatmap: np.ndarray, top_patch_percent: float) -> Optional[List[int]]:
    if top_patch_percent <= 0:
        return None
    thresh = np.percentile(heatmap, max(0.0, 100.0 - top_patch_percent))
    mask = heatmap >= thresh
    if not bool(mask.any()):
        return None

    try:
        from scipy import ndimage

        labels, count = ndimage.label(mask)
        if count <= 0:
            return None
        best_label = None
        best_score = None
        for label_index in range(1, count + 1):
            component = labels == label_index
            values = heatmap[component]
            # Prefer the region with the strongest peak; mean response and area are tie-breakers.
            score = (float(values.max()), float(values.mean()), int(component.sum()))
            if best_score is None or score > best_score:
                best_score = score
                best_label = label_index
        ys, xs = np.where(labels == best_label)
    except Exception:
        max_y, max_x = np.unravel_index(int(np.argmax(heatmap)), heatmap.shape)
        target = mask[max_y, max_x]
        if not bool(target):
            return None
        h, w = mask.shape
        seen = np.zeros_like(mask, dtype=bool)
        stack = [(int(max_y), int(max_x))]
        seen[max_y, max_x] = True
        ys_list, xs_list = [], []
        while stack:
            y, x = stack.pop()
            ys_list.append(y)
            xs_list.append(x)
            for ny in (y - 1, y, y + 1):
                for nx in (x - 1, x, x + 1):
                    if ny == y and nx == x:
                        continue
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
        ys = np.asarray(ys_list)
        xs = np.asarray(xs_list)

    if len(xs) == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def linear_sum_assignment(cost: np.ndarray) -> Tuple[List[int], List[int]]:
    try:
        from scipy.optimize import linear_sum_assignment as scipy_lsa

        rows, cols = scipy_lsa(cost)
        return list(map(int, rows)), list(map(int, cols))
    except Exception:
        rows_n, cols_n = cost.shape
        if rows_n == 0 or cols_n == 0:
            return [], []
        if rows_n <= cols_n:
            best = None
            best_cols = None
            for cols in itertools.permutations(range(cols_n), rows_n):
                score = sum(float(cost[row, col]) for row, col in enumerate(cols))
                if best is None or score < best:
                    best = score
                    best_cols = cols
            return list(range(rows_n)), list(best_cols or [])
        best = None
        best_rows = None
        for rows in itertools.permutations(range(rows_n), cols_n):
            score = sum(float(cost[row, col]) for col, row in enumerate(rows))
            if best is None or score < best:
                best = score
                best_rows = rows
        return list(best_rows or []), list(range(cols_n))


def valid_teacher_indices(payload: Dict[str, Any], agents: List[Dict[str, Any]]) -> List[int]:
    feature_count = int(payload["agent_features"].shape[0]) if "agent_features" in payload else len(agents)
    count = min(feature_count, len(agents))
    mask = payload.get("agent_valid_mask")
    if mask is None:
        return list(range(count))
    mask = mask[:count].detach().cpu().bool().tolist() if torch.is_tensor(mask) else list(mask)[:count]
    return [idx for idx, is_valid in enumerate(mask) if bool(is_valid)]


def match_query_teacher_pairs(
    pred_features: torch.Tensor,
    payload: Dict[str, Any],
    agents: List[Dict[str, Any]],
    query_indices: List[int],
    strategy: str,
) -> List[Tuple[int, int, float]]:
    teacher_indices = valid_teacher_indices(payload, agents)
    query_indices = [idx for idx in query_indices if idx < pred_features.shape[0]]
    if not query_indices or not teacher_indices:
        return []
    if strategy == "ordered":
        count = min(len(query_indices), len(teacher_indices))
        return [(int(query_indices[i]), int(teacher_indices[i]), 0.0) for i in range(count)]
    teacher_features = payload["agent_features"].detach().cpu().float()
    q = pred_features[query_indices].float()
    t = teacher_features[teacher_indices].float()
    cost = F.smooth_l1_loss(
        q[:, None, :].expand(-1, t.shape[0], -1),
        t[None, :, :].expand(q.shape[0], -1, -1),
        reduction="none",
    ).mean(dim=-1)
    rows, cols = linear_sum_assignment(cost.numpy())
    return [(int(query_indices[row]), int(teacher_indices[col]), float(cost[row, col])) for row, col in zip(rows, cols)]


class DinoPatchHeatmap:
    def __init__(self, backbone: str, cache_dir: str, device: torch.device) -> None:
        self.device = device
        self.model = load_local_dino(backbone, cache_dir).to(device).eval().requires_grad_(False)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.grid = 16

    @torch.inference_mode()
    def patch_tokens(self, image: Image.Image) -> torch.Tensor:
        tensor = self.transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
            feats = self.model.forward_features(tensor)["x_norm_patchtokens"]
        return feats[0].float()

    @torch.inference_mode()
    def heatmap(self, image: Image.Image, query_feature: torch.Tensor) -> np.ndarray:
        tokens = self.patch_tokens(image)
        q = F.normalize(query_feature.to(self.device).float(), dim=0)
        p = F.normalize(tokens, dim=-1)
        sim = (p @ q).reshape(1, 1, self.grid, self.grid)
        sim = F.interpolate(sim, size=(image.height, image.width), mode="bilinear", align_corners=False)[0, 0]
        arr = sim.detach().cpu().numpy()
        lo, hi = np.percentile(arr, [5, 99])
        return np.clip((arr - lo) / max(hi - lo, 1e-6), 0.0, 1.0)


def overlay_heatmap(image: Image.Image, heatmap: np.ndarray, alpha: float) -> Image.Image:
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    color = (cm.jet(heatmap)[..., :3] * 255.0).astype(np.float32)
    mask = heatmap[..., None].astype(np.float32)
    blended = rgb * (1.0 - alpha * mask) + color * (alpha * mask)
    return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8))


def draw_annotations(
    image: Image.Image,
    bbox: Iterable[float],
    label: str,
    heatmap: np.ndarray,
    top_patch_percent: float,
    pred_bbox: Optional[Iterable[float]] = None,
) -> Image.Image:
    out = image.copy()
    draw = ImageDraw.Draw(out)
    x1, y1, x2, y2 = [float(v) for v in bbox]
    for offset in range(3):
        draw.rectangle([x1 - offset, y1 - offset, x2 + offset, y2 + offset], outline=(0, 255, 0))

    if pred_bbox is not None:
        px1, py1, px2, py2 = [float(v) for v in pred_bbox]
        for offset in range(3):
            draw.rectangle([px1 - offset, py1 - offset, px2 + offset, py2 + offset], outline=(255, 0, 0))

    response_bbox = top_response_component_bbox(heatmap, top_patch_percent)
    if response_bbox is not None:
        draw.rectangle(response_bbox, outline=(255, 255, 255), width=2)

    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 18)
    except Exception:
        font = ImageFont.load_default()
    text_bbox = draw.textbbox((x1, y1), label, font=font)
    text_w = text_bbox[2] - text_bbox[0]
    text_h = text_bbox[3] - text_bbox[1]
    label_y = max(0.0, y1 - text_h - 6)
    draw.rectangle([x1, label_y, x1 + text_w + 8, label_y + text_h + 6], fill=(0, 128, 0))
    draw.text((x1 + 4, label_y + 3), label, fill=(255, 255, 255), font=font)
    return out


def select_projections(agent_meta: Dict[str, Any], views: str) -> List[Dict[str, Any]]:
    projections = list(agent_meta.get("camera_projections") or [])
    if not projections and agent_meta.get("image_path") and agent_meta.get("bbox_xyxy") is not None:
        projections = [agent_meta]
    if views == "best":
        return projections[:1]
    return projections


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt-dir", required=True)
    parser.add_argument("--datalist-path", default="train_meta.json")
    parser.add_argument("--split", default="train")
    parser.add_argument("--data-root", default="navsim_dataset")
    parser.add_argument("--agent-cache-root", required=True)
    parser.add_argument("--sensor-dir", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-samples", type=int, default=16, help="Number of non-skipped dataset samples to process.")
    parser.add_argument("--dataset-max-samples", type=int, default=None, help="Optional cap for scanned dataset samples before skipping.")
    parser.add_argument("--skip-tokens-file", default=None, help="Newline-separated sample tokens to skip.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--qwen-forward-mode", default="auto", choices=("auto", "legacy", "optimized"))
    parser.add_argument("--dino-backbone", default="dinov2_vits14")
    parser.add_argument("--dino-cache-dir", default="/mnt/workspace/VLA-Drive/weights/derived")
    parser.add_argument("--query-indices", nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument("--views", choices=("best", "all"), default="best")
    parser.add_argument("--alpha", type=float, default=0.55)
    parser.add_argument("--top-patch-percent", type=float, default=5.0)
    parser.add_argument("--draw-pred-bbox", action="store_true", help="Draw bbox decoded from agent_bbox_head in red.")
    parser.add_argument("--match-strategy", choices=("feature", "ordered"), default="ordered", help="Map queries to teacher agents before drawing.")
    args = parser.parse_args()

    os.environ["NAVSIM_AGENT_DINO_CACHE_ROOT"] = args.agent_cache_root
    os.environ.setdefault("NAVSIM_AGENT_DINO_CACHE_STRICT", "1")
    os.environ["NAVSIM_USE_FEATURE_CACHE"] = "0"
    os.environ.pop("NAVSIM_FEATURE_CACHE_ROOT", None)

    agent = VLAAgent(args.ckpt_dir, device=args.device, qwen_forward_mode=args.qwen_forward_mode)
    dataset = build_dataset(agent, args)
    loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_fn, num_workers=args.num_workers, shuffle=False)
    dino = DinoPatchHeatmap(args.dino_backbone, args.dino_cache_dir, agent.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    skip_tokens = set()
    if args.skip_tokens_file:
        skip_tokens = {line.strip() for line in Path(args.skip_tokens_file).read_text().splitlines() if line.strip()}

    written = 0
    processed_samples = 0
    skipped_samples = 0
    for batch in tqdm(loader, desc="agent-query heatmap"):
        examples = to_device(batch, agent.device)
        pred_features, pred_bboxes = extract_agent_query_predictions(agent.model, examples)
        pred_features = pred_features.detach().cpu()
        if pred_bboxes is not None:
            pred_bboxes = pred_bboxes.detach().cpu()
        for batch_index, example in enumerate(examples):
            payload = example.get("agent_dino_feature_cache")
            if payload is None:
                continue
            metadata = decode_metadata(payload)
            token = str(example.get("token", metadata.get("token", f"sample_{written}")))
            if token in skip_tokens:
                skipped_samples += 1
                continue
            processed_samples += 1
            agents = metadata.get("agents", [])
            pairs = match_query_teacher_pairs(pred_features[batch_index], payload, agents, args.query_indices, args.match_strategy)
            for query_index, teacher_index, match_cost in pairs:
                if teacher_index >= len(agents):
                    continue
                agent_meta = agents[teacher_index]
                projections = select_projections(agent_meta, args.views)
                for projection_index, projection in enumerate(projections):
                    image_path = resolve_image_path(str(projection.get("image_path", agent_meta.get("image_path", ""))), args.sensor_dir)
                    bbox = projection.get("bbox_xyxy") or agent_meta.get("bbox_xyxy")
                    if bbox is None:
                        continue
                    image = Image.open(image_path).convert("RGB")
                    heatmap = dino.heatmap(image, pred_features[batch_index, query_index])
                    overlay = overlay_heatmap(image, heatmap, args.alpha)
                    view = str(projection.get("view", agent_meta.get("view", f"view{projection_index}")))
                    label = f"{token} q{query_index}->t{teacher_index} rank={agent_meta.get('rank', teacher_index)} {agent_meta.get('class_name', '')} {view}"
                    pred_bbox_px = None
                    if args.draw_pred_bbox and pred_bboxes is not None:
                        view_id = VIEW_TO_ID.get(view)
                        agent_view_ids = tuple(getattr(agent.model, "agent_view_ids", (0, 1, 2)))
                        if view_id in agent_view_ids:
                            view_slot = agent_view_ids.index(view_id)
                            pred_bbox_px = denormalize_bbox(
                                pred_bboxes[batch_index, query_index, view_slot].tolist(), image.width, image.height
                            )
                    annotated = draw_annotations(overlay, bbox, label, heatmap, args.top_patch_percent, pred_bbox=pred_bbox_px)
                    out_path = out_dir / f"{token}_q{query_index:02d}_t{teacher_index:02d}_{view}.jpg"
                    annotated.save(out_path, quality=92)
                    written += 1
        if args.max_samples is not None and processed_samples >= args.max_samples:
            break
    print(f"wrote {written} heatmaps to {out_dir}; processed_samples={processed_samples}; skipped_samples={skipped_samples}")


if __name__ == "__main__":
    main()

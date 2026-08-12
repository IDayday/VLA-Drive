"""Load a DriveVLA Retrieve Model checkpoint and save map/agent BEV visuals."""

from __future__ import annotations

import argparse
import json
import time
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from PIL import Image

from navsim.agents.EpisodeDrive.retrieve_model.retrieve_features import (
    RetrieveFeatureBuilder,
    RetrieveTargetBuilder,
    get_agent_class_names,
)
from navsim.agents.EpisodeDrive.retrieve_model.retrieve_loss import RetrieveLoss
from navsim.agents.EpisodeDrive.retrieve_model.retrieve_model import RetrieveModelV1
from navsim.agents.EpisodeDrive.retrieve_model.retrieve_visualization import (
    AGENT_PALETTE,
    MAP_PALETTE,
    semantic_labels_to_rgb,
    semantic_logits_to_rgb,
)
from navsim.common.dataloader import SceneLoader
from navsim.common.dataclasses import SceneFilter, SensorConfig


TensorDict = Dict[str, torch.Tensor]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    repo_root = Path(__file__).resolve().parents[1]
    default_run = (
        repo_root
        / "verification"
        / "retrieve_model"
        / datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=repo_root / "configs" / "retrieve_model_vehicle_only_multiview_agentw5.yaml",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Overrides checkpoint_path in the config.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("/GPU_JDTest_fs01/home/zdhs0164/navsim_data/openscene/openscene-v1.1"),
    )
    parser.add_argument("--split", default="mini")
    parser.add_argument("--sample-offset", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-visualizations", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, default=default_run)
    return parser.parse_args()


def to_plain(config):
    return OmegaConf.to_container(config, resolve=True)


def sensor_config_from_cameras(camera_names: Iterable[str]) -> SensorConfig:
    names = set(camera_names)
    history_iteration = 3
    return SensorConfig(
        cam_f0=[history_iteration] if "cam_f0" in names else [],
        cam_l0=[history_iteration] if "cam_l0" in names else [],
        cam_l1=[],
        cam_l2=[],
        cam_r0=[history_iteration] if "cam_r0" in names else [],
        cam_r1=[],
        cam_r2=[],
        cam_b0=[history_iteration] if "cam_b0" in names else [],
        lidar_pc=[],
    )


def load_samples(
    config,
    data_root: Path,
    split: str,
    sample_offset: int,
    num_samples: int,
) -> List[Tuple[str, TensorDict, TensorDict]]:
    feature_config = to_plain(config.feature_config)
    target_config = to_plain(config.target_config)
    camera_names = feature_config.get("camera_names", ["cam_f0"])
    loader = SceneLoader(
        data_path=data_root / "meta_datas" / split,
        sensor_blobs_path=data_root / "sensor_blobs" / split,
        scene_filter=SceneFilter(max_scenes=sample_offset + num_samples),
        sensor_config=sensor_config_from_cameras(camera_names),
    )
    selected_tokens = loader.tokens[sample_offset : sample_offset + num_samples]
    if len(selected_tokens) != num_samples:
        raise RuntimeError(
            f"Requested {num_samples} samples at offset {sample_offset}, "
            f"found {len(selected_tokens)}."
        )

    feature_builder = RetrieveFeatureBuilder(**feature_config)
    target_builder = RetrieveTargetBuilder(**target_config)
    samples = []
    for token in selected_tokens:
        scene = loader.get_scene_from_token(token)
        features = feature_builder.compute_features(scene.get_agent_input())
        targets = target_builder.compute_targets(scene)
        features = {
            key: value
            for key, value in features.items()
            if key in {"image", "camera_ids"}
        }
        samples.append((token, features, targets))
    return samples


def extract_model_state_dict(payload: Dict) -> Dict[str, torch.Tensor]:
    state_dict = payload.get("state_dict", payload.get("model_state_dict", payload))
    for prefix in ("agent.model.", "model."):
        if any(key.startswith(prefix) for key in state_dict):
            return {
                key[len(prefix) :]: value
                for key, value in state_dict.items()
                if key.startswith(prefix)
            }
    return dict(state_dict)


def load_model(checkpoint: Path, model_config, device: torch.device) -> RetrieveModelV1:
    model = RetrieveModelV1(**to_plain(model_config)).to(device)
    payload = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(extract_model_state_dict(payload), strict=True)
    model.eval()
    return model


def make_batch(samples, indices, device: torch.device):
    features = {
        key: torch.stack([samples[index][1][key] for index in indices]).to(device)
        for key in ("image", "camera_ids")
    }
    targets = {
        key: torch.stack([samples[index][2][key] for index in indices]).to(device)
        for key in ("map_target", "agent_target")
    }
    return features, targets


@torch.inference_mode()
def evaluate(model, loss_module, samples, device: torch.device, batch_size: int):
    logits = {"map_bev_logits": [], "agent_bev_logits": []}
    targets = {"map_target": [], "agent_target": []}
    autocast_context = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else nullcontext()
    )
    for start in range(0, len(samples), batch_size):
        indices = list(range(start, min(start + batch_size, len(samples))))
        features, batch_targets = make_batch(samples, indices, device)
        with autocast_context:
            predictions = model(features)
        for key in logits:
            logits[key].append(predictions[key].float())
        for key in targets:
            targets[key].append(batch_targets[key])
    logits = {key: torch.cat(value, dim=0) for key, value in logits.items()}
    targets = {key: torch.cat(value, dim=0) for key, value in targets.items()}
    metrics = loss_module(logits, targets)
    return {key: float(value.detach().cpu()) for key, value in metrics.items()}


def save_visualizations(model, samples, device: torch.device, output_dir: Path, count: int):
    output_dir.mkdir(parents=True, exist_ok=True)
    autocast_context = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else nullcontext()
    )
    saved = []
    for index in range(min(count, len(samples))):
        token, _, targets = samples[index]
        features, _ = make_batch(samples, [index], device)
        with torch.inference_mode(), autocast_context:
            predictions = model(features)
        sample_dir = output_dir / f"{index:02d}_{token}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        images = {
            "map_target_color": semantic_labels_to_rgb(targets["map_target"], MAP_PALETTE),
            "map_prediction_color": semantic_logits_to_rgb(
                predictions["map_bev_logits"][0],
                MAP_PALETTE,
            ),
            "agent_target_color": semantic_labels_to_rgb(
                targets["agent_target"],
                AGENT_PALETTE,
            ),
            "agent_prediction_color": semantic_logits_to_rgb(
                predictions["agent_bev_logits"][0],
                AGENT_PALETTE,
            ),
        }
        for name, image in images.items():
            Image.fromarray(image).save(sample_dir / f"{name}.png")
        overview = Image.new("RGB", (256 * 2, 128 * 2), "white")
        overview.paste(Image.fromarray(images["map_target_color"]), (0, 0))
        overview.paste(Image.fromarray(images["map_prediction_color"]), (256, 0))
        overview.paste(Image.fromarray(images["agent_target_color"]), (0, 128))
        overview.paste(Image.fromarray(images["agent_prediction_color"]), (256, 128))
        overview.save(sample_dir / "map_agent_overview.png")
        saved.append(str(sample_dir))
    return saved


def main() -> None:
    args = parse_args()
    config = OmegaConf.load(args.config)
    checkpoint = args.checkpoint or Path(OmegaConf.to_container(config, resolve=True)["checkpoint_path"])
    stats_path = Path(OmegaConf.to_container(config.loss_config, resolve=True)["stats_path"])
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    start = time.perf_counter()

    samples = load_samples(
        config,
        args.data_root,
        args.split,
        args.sample_offset,
        args.num_samples,
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = load_model(checkpoint, config.model_config, device)
    target_config = to_plain(config.target_config)
    loss_module = RetrieveLoss(
        map_class_weights=stats["map_class_weights"],
        agent_class_weights=stats["agent_class_weights"],
        alpha=float(config.loss_config.alpha),
        class_weight_cap=float(config.loss_config.class_weight_cap),
        map_num_classes=int(config.model_config.map_num_classes),
        agent_num_classes=int(config.model_config.agent_num_classes),
        bev_pixel_height=int(target_config["bev_pixel_height"]),
        bev_pixel_width=int(target_config["bev_pixel_width"]),
        bev_pixel_size=float(target_config["bev_pixel_size"]),
    ).to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    metrics = evaluate(model, loss_module, samples, device, args.batch_size)
    visual_dirs = save_visualizations(
        model,
        samples,
        device,
        args.output_dir / "visualizations",
        args.num_visualizations,
    )
    report = {
        "status": "ok",
        "checkpoint": str(checkpoint),
        "config": str(args.config),
        "stats": str(stats_path),
        "data_root": str(args.data_root),
        "split": args.split,
        "sample_offset": args.sample_offset,
        "num_samples": len(samples),
        "batch_size": args.batch_size,
        "metrics": metrics,
        "tokens": [sample[0] for sample in samples],
        "visualization_dirs": visual_dirs,
        "agent_class_names": list(
            get_agent_class_names(
                bool(target_config.get("include_pedestrians", True))
            )
        ),
        "elapsed_seconds": time.perf_counter() - start,
    }
    if device.type == "cuda":
        report["peak_cuda_memory_gib"] = torch.cuda.max_memory_allocated(device) / (1024**3)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

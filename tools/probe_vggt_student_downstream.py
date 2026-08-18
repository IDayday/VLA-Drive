#!/usr/bin/env python3
"""Compare frozen VGGT teacher/student memories on concrete geometry tasks.

The probe reports three distinct questions instead of treating cosine as a
complete representation metric:

1. Can an identical-capacity ridge decoder recover the official VGGT
   depth/point-derived geometry targets from each frozen representation?
2. Does a decoder fitted on teacher coordinates work on student coordinates
   without refitting (zero-shot decoder transfer)?
3. Can the representations decode sensor-grounded front-camera LiDAR depth?

It also evaluates the V2 geometry head saved in the checkpoint and writes
held-out geometry maps for visual inspection.  No VGGT model is imported and
no network access or weight download is attempted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

import matplotlib
import torch
import torch.nn.functional as F

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from starVLA.cache.navsim_feature_cache import NavsimFeatureCacheReader  # noqa: E402
from starVLA.model.modules.vggt_query.geometry_probe import (  # noqa: E402
    apply_slot_residualization,
    regression_metrics,
)
from starVLA.model.modules.vggt_query.planning_heads import (  # noqa: E402
    PhysicalGeometryHead,
)
from tools.probe_vggt_geometry_signal import (  # noqa: E402
    fit_ridge_probe,
    load_lidar_target,
    load_metadata,
    scene_image_paths,
    scene_name_from_metadata,
)


SCHEMA_VERSION = 1
SPATIAL_START = 15
SPATIAL_PER_VIEW = 60
GEOMETRY_NAMES = ("x_over_z", "y_over_z", "log_depth_over_scene_median")
VIEWS = ("cam_f0", "cam_l0", "cam_r0")


def _sha256(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(value: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _validate_identity(features: Mapping[str, Any], args) -> list[str]:
    identity = features["identity"]
    datalist_path = Path(args.datalist_path).resolve()
    if identity["datalist_sha256"] != _sha256(datalist_path):
        raise RuntimeError("feature file and selected datalist hashes differ")
    tokens = json.loads(datalist_path.read_text(encoding="utf-8"))
    for split_name, index_name in (
        ("train", "train_indices"),
        ("validation", "validation_indices"),
    ):
        expected = [tokens[index] for index in identity[index_name]]
        if expected != features[split_name]["tokens"]:
            raise RuntimeError(f"{split_name} token/index identity mismatch")
    return tokens


def _load_geometry_targets(features, tokens, cache_root: Path):
    reader = NavsimFeatureCacheReader(
        cache_root=cache_root, components=("vggt_query",), strict=True
    )
    output = {}
    for split_name, index_name in (
        ("train", "train_indices"),
        ("validation", "validation_indices"),
    ):
        targets, confidences, masks = [], [], []
        for sample_index, token in zip(
            features["identity"][index_name], features[split_name]["tokens"]
        ):
            if tokens[sample_index] != token:
                raise RuntimeError("cache sample index does not identify the expected token")
            payload = reader.get("vggt_query", sample_index, token)
            targets.append(payload["geometry_target"].float())
            confidences.append(payload["geometry_confidence"].float())
            masks.append(payload["geometry_valid_mask"].bool())
        feature_valid = features[split_name]["valid_mask"][:, SPATIAL_START:]
        output[split_name] = {
            "target": torch.stack(targets),
            "confidence": torch.stack(confidences),
            "valid": torch.stack(masks) & feature_valid,
        }
    return output


def _scene_names(feature_split, data_root: Path) -> list[str]:
    names = []
    for token in feature_split["tokens"]:
        raw = load_metadata(data_root, "train", token)
        names.append(scene_name_from_metadata(raw, 3))
    return names


def _filter_train_scene_leakage(features, targets, data_root: Path):
    train_scenes = _scene_names(features["train"], data_root)
    validation_scenes = _scene_names(features["validation"], data_root)
    heldout = set(validation_scenes)
    keep = torch.tensor([scene not in heldout for scene in train_scenes], dtype=torch.bool)
    if not keep.any():
        raise RuntimeError("scene-disjoint filtering removed every training sample")
    filtered_features = dict(features)
    filtered_features["train"] = {
        key: value[keep] if torch.is_tensor(value) else [v for v, flag in zip(value, keep) if flag]
        for key, value in features["train"].items()
    }
    filtered_targets = dict(targets)
    filtered_targets["train"] = {
        key: value[keep] for key, value in targets["train"].items()
    }
    split_stats = {
        "train_samples_before_scene_filter": len(train_scenes),
        "train_samples_after_scene_filter": int(keep.sum()),
        "train_unique_scenes_after_filter": len(
            {scene for scene, flag in zip(train_scenes, keep) if flag}
        ),
        "validation_samples": len(validation_scenes),
        "validation_unique_scenes": len(set(validation_scenes)),
        "scene_overlap_after_filter": 0,
    }
    return filtered_features, filtered_targets, split_stats


def _collect_lidar(features, args):
    output = {}
    for split_name in ("train", "validation"):
        values, masks = [], []
        for token in features[split_name]["tokens"]:
            raw = load_metadata(Path(args.data_root), "train", token)
            paths = scene_image_paths(
                raw,
                VIEWS,
                3,
                Path(args.sensor_root),
            )
            target, valid = load_lidar_target(
                raw,
                token=token,
                front_path=paths[0],
                sensor_root=Path(args.sensor_root),
                frame_index=3,
                grid_size=(6, 10),
                min_points=args.lidar_min_points,
            )
            values.append(target)
            masks.append(valid)
        feature_valid = features[split_name]["valid_mask"][
            :, SPATIAL_START : SPATIAL_START + SPATIAL_PER_VIEW
        ]
        output[split_name] = {
            "target": torch.stack(values),
            "valid": torch.stack(masks) & feature_valid,
        }
    return output


def _predict_ridge(features, targets, valid, residualizer, model):
    x, _y, flat_valid = apply_slot_residualization(
        features, targets, valid, residualizer
    )
    residual = torch.from_numpy(model.predict(x.numpy())).float()
    baseline = residualizer.target_mean.unsqueeze(0).expand(targets.shape[0], -1, -1)
    prediction = baseline + residual.reshape_as(targets)
    return prediction, baseline, flat_valid.reshape(valid.shape)


def _metrics(prediction, target, baseline, valid, names):
    flattened_prediction = prediction[valid]
    flattened_target = target[valid]
    flattened_baseline = baseline[valid]
    overall = regression_metrics(
        flattened_prediction, flattened_target, flattened_baseline
    )
    per_channel = {}
    for index, name in enumerate(names):
        per_channel[name] = regression_metrics(
            flattened_prediction[:, index : index + 1],
            flattened_target[:, index : index + 1],
            flattened_baseline[:, index : index + 1],
        )
        per_channel[name]["mae"] = float(
            (flattened_prediction[:, index] - flattened_target[:, index]).abs().mean()
        )
    overall["per_channel"] = per_channel
    return overall


def _fit_and_compare(train_features, validation_features, train_target, validation_target,
                     train_valid, validation_valid, names, alpha):
    probes = {}
    predictions = {}
    for representation, values in (
        ("teacher", train_features["teacher"]),
        ("student", train_features["student"]),
    ):
        residualizer, model = fit_ridge_probe(
            values, train_target, train_valid, alpha=alpha
        )
        probes[representation] = (residualizer, model)
        prediction, baseline, valid = _predict_ridge(
            validation_features[representation],
            validation_target,
            validation_valid,
            residualizer,
            model,
        )
        predictions[f"{representation}_fitted"] = prediction
    teacher_residualizer, teacher_model = probes["teacher"]
    transferred, baseline, valid = _predict_ridge(
        validation_features["student"],
        validation_target,
        validation_valid,
        teacher_residualizer,
        teacher_model,
    )
    predictions["teacher_decoder_on_student"] = transferred
    metrics = {
        name: _metrics(prediction, validation_target, baseline, valid, names)
        for name, prediction in predictions.items()
    }
    teacher_r2 = metrics["teacher_fitted"]["r2_vs_constant"]
    student_r2 = metrics["student_fitted"]["r2_vs_constant"]
    transfer_r2 = metrics["teacher_decoder_on_student"]["r2_vs_constant"]
    metrics["retention"] = {
        "student_fitted_r2_over_teacher": (
            student_r2 / teacher_r2 if teacher_r2 > 0 else None
        ),
        "teacher_decoder_on_student_r2_over_teacher": (
            transfer_r2 / teacher_r2 if teacher_r2 > 0 else None
        ),
        "note": "ratios are undefined when the teacher probe has non-positive R2",
    }
    return metrics, predictions


def _fit_mlp(
    train_features,
    validation_features,
    train_target,
    validation_target,
    train_valid,
    validation_valid,
    *,
    device,
    hidden_dim,
    steps,
    batch_size,
    seed,
):
    torch.manual_seed(seed)
    model = PhysicalGeometryHead(memory_dim=1024, hidden_dim=hidden_dim).to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    flattened_feature = train_features.reshape(-1, train_features.shape[-1])
    flattened_target = train_target.reshape(-1, train_target.shape[-1])
    flattened_valid = train_valid.reshape(-1)
    valid_indices = flattened_valid.nonzero(as_tuple=False).squeeze(1)
    generator = torch.Generator().manual_seed(seed)
    history = []
    for step in range(1, steps + 1):
        selected = valid_indices[
            torch.randint(len(valid_indices), (batch_size,), generator=generator)
        ]
        feature = flattened_feature[selected].to(device, non_blocking=True).float()
        target = flattened_target[selected].to(device, non_blocking=True).float()
        prediction = model.head(feature)
        loss = F.smooth_l1_loss(prediction, target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 1 or step % 100 == 0 or step == steps:
            history.append({"step": step, "smooth_l1": float(loss.detach())})
    model.eval()
    predictions = []
    with torch.inference_mode():
        for start in range(0, validation_features.shape[0], 16):
            predictions.append(
                model.head(validation_features[start : start + 16].to(device).float()).cpu()
            )
    prediction = torch.cat(predictions)
    count = train_valid.sum(dim=0).clamp_min(1).unsqueeze(-1)
    template = (train_target * train_valid.unsqueeze(-1)).sum(dim=0) / count
    baseline = template.unsqueeze(0).expand_as(validation_target)
    metrics = _metrics(
        prediction,
        validation_target,
        baseline,
        validation_valid,
        GEOMETRY_NAMES,
    )
    metrics["history"] = history
    return model, metrics, prediction, baseline


def _fit_mlp_geometry_probes(features, targets, args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    models, metrics, predictions = {}, {}, {}
    for representation in ("teacher", "student"):
        model, result, prediction, baseline = _fit_mlp(
            features["train"][f"{representation}_memory"][:, SPATIAL_START:],
            features["validation"][f"{representation}_memory"][:, SPATIAL_START:],
            targets["train"]["target"],
            targets["validation"]["target"],
            targets["train"]["valid"],
            targets["validation"]["valid"],
            device=device,
            hidden_dim=args.mlp_hidden_dim,
            steps=args.mlp_steps,
            batch_size=args.mlp_batch_size,
            seed=args.seed,
        )
        models[representation] = model
        metrics[f"{representation}_fitted"] = result
        predictions[f"{representation}_fitted"] = prediction
    transferred = []
    teacher_model = models["teacher"]
    with torch.inference_mode():
        student_validation = features["validation"]["student_memory"][:, SPATIAL_START:]
        for start in range(0, student_validation.shape[0], 16):
            transferred.append(
                teacher_model.head(student_validation[start : start + 16].to(device).float()).cpu()
            )
    transferred = torch.cat(transferred)
    predictions["teacher_decoder_on_student"] = transferred
    metrics["teacher_decoder_on_student"] = _metrics(
        transferred,
        targets["validation"]["target"],
        baseline,
        targets["validation"]["valid"],
        GEOMETRY_NAMES,
    )
    teacher_r2 = metrics["teacher_fitted"]["r2_vs_constant"]
    student_r2 = metrics["student_fitted"]["r2_vs_constant"]
    transfer_r2 = metrics["teacher_decoder_on_student"]["r2_vs_constant"]
    metrics["retention"] = {
        "student_fitted_r2_over_teacher": student_r2 / max(teacher_r2, 1e-8),
        "teacher_decoder_on_student_r2_over_teacher": transfer_r2 / max(teacher_r2, 1e-8),
    }
    return metrics, predictions


def _checkpoint_geometry_head(checkpoint_path: Path) -> PhysicalGeometryHead:
    state = torch.load(
        checkpoint_path, map_location="cpu", weights_only=True, mmap=True
    )
    prefix = "vggt_geometry_probe."
    selected = {
        key[len(prefix) :]: value.float()
        for key, value in state.items()
        if key.startswith(prefix)
    }
    if not selected:
        raise RuntimeError("checkpoint has no V2 physical geometry head")
    head = PhysicalGeometryHead(memory_dim=1024, hidden_dim=512).float().eval()
    head.load_state_dict(selected, strict=True)
    return head


@torch.inference_mode()
def _evaluate_checkpoint_head(features, targets, checkpoint_path: Path):
    head = _checkpoint_geometry_head(checkpoint_path)
    target = targets["validation"]["target"]
    valid = targets["validation"]["valid"]
    count = valid.sum(dim=0).clamp_min(1).unsqueeze(-1)
    template = (
        targets["train"]["target"] * targets["train"]["valid"].unsqueeze(-1)
    ).sum(dim=0) / targets["train"]["valid"].sum(dim=0).clamp_min(1).unsqueeze(-1)
    baseline = template.unsqueeze(0).expand_as(target)
    result, predictions = {}, {}
    del count
    for name in ("teacher", "student"):
        memory = features["validation"][f"{name}_memory"][:, SPATIAL_START:].float()
        prediction = head.head(memory)
        predictions[name] = prediction
        result[name] = _metrics(prediction, target, baseline, valid, GEOMETRY_NAMES)
    return result, predictions


def _representation_metrics(features):
    teacher = F.normalize(features["teacher_memory"].float(), dim=-1)
    student = F.normalize(features["student_memory"].float(), dim=-1)
    shuffled = teacher.roll(1, dims=0)
    correct = (student * teacher).sum(-1)
    wrong = (student * shuffled).sum(-1)
    student_scene = F.normalize(student.mean(1), dim=-1)
    teacher_scene = F.normalize(teacher.mean(1), dim=-1)
    similarity = student_scene @ teacher_scene.T
    expected = torch.arange(similarity.shape[0])
    return {
        "token_cosine_correct": float(correct.mean()),
        "token_cosine_shuffled": float(wrong.mean()),
        "correct_minus_shuffled": float((correct - wrong).mean()),
        "scene_retrieval_top1": float(similarity.argmax(1).eq(expected).float().mean()),
        "scene_retrieval_top5": float(
            similarity.topk(min(5, similarity.shape[1]), dim=1)
            .indices.eq(expected[:, None]).any(1).float().mean()
        ),
    }


def _write_geometry_figures(features, targets, predictions, output_dir: Path, count: int):
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = (
        ("target", targets["validation"]["target"]),
        ("teacher fitted", predictions["teacher_fitted"]),
        ("student fitted", predictions["student_fitted"]),
        ("teacher decoder on student", predictions["teacher_decoder_on_student"]),
    )
    paths = []
    for sample_index in range(min(count, len(features["validation"]["tokens"]))):
        figure, axes = plt.subplots(len(rows), 3, figsize=(12, 10), constrained_layout=True)
        for row_index, (row_name, values) in enumerate(rows):
            for channel_index, channel_name in enumerate(GEOMETRY_NAMES):
                image = values[sample_index, :SPATIAL_PER_VIEW, channel_index].reshape(6, 10)
                artist = axes[row_index, channel_index].imshow(image, cmap="coolwarm")
                axes[row_index, channel_index].set_title(f"{row_name}: {channel_name}")
                axes[row_index, channel_index].axis("off")
                figure.colorbar(artist, ax=axes[row_index, channel_index], fraction=0.046)
        token = features["validation"]["tokens"][sample_index]
        path = output_dir / f"{token}.png"
        figure.savefig(path, dpi=140)
        plt.close(figure)
        paths.append(str(path.resolve()))
    return paths


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-file", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--datalist-path", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--sensor-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figure-dir", type=Path, required=True)
    parser.add_argument("--figure-count", type=int, default=3)
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    parser.add_argument("--lidar-min-points", type=int, default=3)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--mlp-hidden-dim", type=int, default=512)
    parser.add_argument("--mlp-steps", type=int, default=1000)
    parser.add_argument("--mlp-batch-size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=20260813)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path, label in (
        (args.feature_file, "frozen feature file"),
        (args.cache_root, "VGGT cache"),
        (args.datalist_path, "NAVSIM datalist"),
        (args.data_root, "processed NAVSIM root"),
        (args.sensor_root, "NAVSIM sensor root"),
    ):
        if not path.exists():
            raise FileNotFoundError(f"Missing {label}: {path}")
    features = torch.load(args.feature_file, map_location="cpu", weights_only=False)
    tokens = _validate_identity(features, args)
    checkpoint_path = (
        args.checkpoint.resolve()
        if args.checkpoint is not None
        else Path(features["identity"]["checkpoint"]).resolve()
    )
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing source checkpoint: {checkpoint_path}")
    geometry = _load_geometry_targets(features, tokens, args.cache_root)
    features, geometry, split_stats = _filter_train_scene_leakage(
        features, geometry, args.data_root
    )
    lidar = _collect_lidar(features, args)

    geometry_features = {
        split: {
            "teacher": features[split]["teacher_memory"][:, SPATIAL_START:].float(),
            "student": features[split]["student_memory"][:, SPATIAL_START:].float(),
        }
        for split in ("train", "validation")
    }
    geometry_metrics, geometry_predictions = _fit_and_compare(
        geometry_features["train"],
        geometry_features["validation"],
        geometry["train"]["target"],
        geometry["validation"]["target"],
        geometry["train"]["valid"],
        geometry["validation"]["valid"],
        GEOMETRY_NAMES,
        args.ridge_alpha,
    )
    lidar_features = {
        split: {
            "teacher": features[split]["teacher_memory"][
                :, SPATIAL_START : SPATIAL_START + SPATIAL_PER_VIEW
            ].float(),
            "student": features[split]["student_memory"][
                :, SPATIAL_START : SPATIAL_START + SPATIAL_PER_VIEW
            ].float(),
        }
        for split in ("train", "validation")
    }
    lidar_metrics, _ = _fit_and_compare(
        lidar_features["train"],
        lidar_features["validation"],
        lidar["train"]["target"],
        lidar["validation"]["target"],
        lidar["train"]["valid"],
        lidar["validation"]["valid"],
        ("log_lidar_depth",),
        args.ridge_alpha,
    )
    checkpoint_metrics, _checkpoint_predictions = _evaluate_checkpoint_head(
        features, geometry, checkpoint_path
    )
    mlp_metrics, mlp_predictions = _fit_mlp_geometry_probes(features, geometry, args)
    figures = _write_geometry_figures(
        features, geometry, mlp_predictions, args.figure_dir, args.figure_count
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "source": {
            "feature_file": str(args.feature_file.resolve()),
            "checkpoint": str(checkpoint_path),
            "cache_root": str(args.cache_root.resolve()),
            "datalist_sha256": _sha256(args.datalist_path),
        },
        "split": split_stats,
        "representation": _representation_metrics(features["validation"]),
        "frozen_ridge_vggt_geometry": geometry_metrics,
        "frozen_mlp_vggt_geometry": mlp_metrics,
        "frozen_ridge_lidar_depth": lidar_metrics,
        "checkpoint_trained_geometry_head": checkpoint_metrics,
        "visualizations": figures,
        "interpretation_contract": {
            "teacher_fitted": "same-capacity decoder upper bound for cached layer-11 teacher memory",
            "student_fitted": "information retained up to a separately learned linear coordinate map",
            "teacher_decoder_on_student": "strict coordinate-compatible transfer without student refitting",
            "vggt_geometry_target": (
                "pooled x/z, y/z and relative log-depth generated by official VGGT point/depth heads"
            ),
            "lidar_target": "sensor-grounded front-camera log-depth, not a VGGT pseudo-label",
            "scope_limit": (
                "camera pose and tracking are not cached; this report does not claim those capabilities"
            ),
        },
    }
    _atomic_json(report, args.output)
    print(json.dumps({
        "representation": report["representation"],
        "geometry_retention": geometry_metrics["retention"],
        "geometry_mlp_retention": mlp_metrics["retention"],
        "lidar_retention": lidar_metrics["retention"],
        "output": str(args.output.resolve()),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

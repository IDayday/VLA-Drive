#!/usr/bin/env python3
"""Train factual-only structured-future probes on frozen scene features."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
from typing import Any

import numpy as np
import torch
import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from research.action_effect.cache_io import (  # noqa: E402
    CacheConflictError,
    content_hash,
    file_sha256,
    read_manifest,
    write_json,
    write_jsonl,
    write_npz,
)
from research.action_effect.losses import StructuredFutureLoss  # noqa: E402
from research.action_effect.probe_data import (  # noqa: E402
    HARD_TARGET_FIELDS,
    SOFT_TARGET_FIELDS,
    load_probe_arrays,
    load_structured_targets,
)
from research.action_effect.world_probe import ActionEffectWorldProbe, count_parameters  # noqa: E402


CONTROL_TO_MODE = {
    "scene_only_probe": "scene_only",
    "trajectory_only_probe": "trajectory_only",
    "scene_action_probe": "scene_action",
    "shuffled_action_probe": "scene_action",
    "same_parameter_no_action": "zero_action",
}


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise TypeError(f"configuration must be a mapping: {path}")
    return value


def _root(variable: str) -> Path:
    value = os.environ.get(variable, "").strip()
    if not value:
        raise ValueError(f"source load_env.sh or set {variable}")
    return Path(value).resolve()


def _cache(explicit: Path | None, relative: str) -> Path:
    return explicit.resolve() if explicit is not None else _root("ACTION_EFFECT_CACHE_ROOT") / relative


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _model(config: dict[str, Any], mode: str) -> ActionEffectWorldProbe:
    probe = config["probe"]
    return ActionEffectWorldProbe(
        scene_input_dim=int(probe["scene_input_dim"]),
        consequence_dim=len(HARD_TARGET_FIELDS) + len(SOFT_TARGET_FIELDS),
        latent_dim=int(probe["latent_dim"]),
        trajectory_input_dim=int(probe["trajectory_input_dim"]),
        trajectory_token_dim=int(probe["trajectory_token_dim"]),
        dropout=float(probe["dropout"]),
        structured_future_shape=tuple(int(value) for value in probe["structured_future_shape"]),
        input_mode=mode,
    )


def _atomic_torch_save(path: Path, value: Any) -> None:
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(handle)
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _loss_on_indices(
    model,
    loss_fn,
    indices: np.ndarray,
    *,
    arrays,
    scene_features: np.ndarray,
    target: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> float:
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            batch = indices[start : start + batch_size]
            scene = torch.from_numpy(scene_features[arrays.scene_feature_indices[batch]]).to(device)
            trajectory = torch.from_numpy(arrays.trajectories[batch]).to(device)
            label = torch.from_numpy(target[batch].astype(np.float32)).to(device)
            prediction = model(scene, trajectory)["structured_future_prediction"]
            assert isinstance(prediction, torch.Tensor)
            value = loss_fn(prediction, label)["total"]
            total += float(value.item()) * len(batch)
            count += len(batch)
    return total / count


@torch.no_grad()
def _predict_all(
    model,
    *,
    arrays,
    scene_features: np.ndarray,
    shape: tuple[int, ...],
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    result = np.empty((len(arrays.scene_ids), *shape), dtype=np.float16)
    for start in range(0, len(result), batch_size):
        stop = min(start + batch_size, len(result))
        scene = torch.from_numpy(scene_features[arrays.scene_feature_indices[start:stop]]).to(device)
        trajectory = torch.from_numpy(arrays.trajectories[start:stop]).to(device)
        prediction = model(scene, trajectory)["structured_future_prediction"]
        assert isinstance(prediction, torch.Tensor)
        result[start:stop] = prediction.float().cpu().numpy().astype(np.float16)
    return result


def _train_one(
    *,
    config: dict[str, Any],
    control: str,
    seed: int,
    arrays,
    scene_features: np.ndarray,
    target: np.ndarray,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    positive_weight: np.ndarray,
    output_dir: Path,
    max_epochs: int | None,
) -> dict[str, Any]:
    training = config["training"]
    requested = str(training["device"])
    device = torch.device(requested if torch.cuda.is_available() else "cpu")
    _seed(seed)
    model = _model(config, CONTROL_TO_MODE[control]).to(device)
    loss_fn = StructuredFutureLoss(torch.from_numpy(positive_weight)).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    epochs = int(max_epochs if max_epochs is not None else training["epochs"])
    patience = int(training["patience"])
    batch_size = int(training["batch_size"])
    rng = np.random.default_rng(seed)
    best_loss, best_epoch, stale = float("inf"), -1, 0
    best_state = None
    history: list[dict[str, Any]] = []
    for epoch in range(epochs):
        model.train()
        order = rng.permutation(train_indices)
        sums = {name: 0.0 for name in ("total", "binary", "velocity", "clearance")}
        seen = 0
        for start in range(0, len(order), batch_size):
            batch = order[start : start + batch_size]
            scene = torch.from_numpy(scene_features[arrays.scene_feature_indices[batch]]).to(device)
            trajectory = torch.from_numpy(arrays.trajectories[batch]).to(device)
            if control == "shuffled_action_probe" and len(batch) > 1:
                trajectory = torch.roll(trajectory, 1, 0)
            label = torch.from_numpy(target[batch].astype(np.float32)).to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(scene, trajectory)["structured_future_prediction"]
            assert isinstance(prediction, torch.Tensor)
            losses = loss_fn(prediction, label)
            losses["total"].backward()
            norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(training["gradient_clip_norm"])
            )
            if not torch.isfinite(norm):
                raise FloatingPointError("non-finite structured-probe gradient")
            optimizer.step()
            for name in sums:
                sums[name] += float(losses[name].item()) * len(batch)
            seen += len(batch)
        validation_loss = _loss_on_indices(
            model,
            loss_fn,
            validation_indices,
            arrays=arrays,
            scene_features=scene_features,
            target=target,
            device=device,
            batch_size=batch_size,
        )
        history.append(
            {"epoch": epoch, **{f"train_{name}": value / seen for name, value in sums.items()}, "validation_loss": validation_loss}
        )
        if validation_loss < best_loss - 1.0e-6:
            best_loss, best_epoch, stale = validation_loss, epoch, 0
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        else:
            stale += 1
        if stale >= patience:
            break
    if best_state is None:
        raise RuntimeError("structured probe produced no checkpoint")
    model.load_state_dict(best_state)
    shape = tuple(int(value) for value in config["probe"]["structured_future_shape"])
    prediction = _predict_all(
        model,
        arrays=arrays,
        scene_features=scene_features,
        shape=shape,
        device=device,
        batch_size=int(training["evaluation_batch_size"]),
    )
    write_npz(output_dir / "predictions.npz", structured_future_prediction=prediction)
    write_jsonl(output_dir / "history.jsonl", history)
    _atomic_torch_save(
        output_dir / "probe.pt",
        {"state_dict": best_state, "control": control, "seed": seed, "best_epoch": best_epoch},
    )
    result = {
        "control": control,
        "seed": seed,
        "best_epoch": best_epoch,
        "epochs_ran": len(history),
        "best_validation_loss": best_loss,
        "total_parameters": count_parameters(model),
        "trainable_parameters": count_parameters(model, trainable_only=True),
        "device": str(device),
    }
    write_json(output_dir / "result.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPOSITORY_ROOT / "configs/action_effect/structured_factual_only.yaml",
    )
    parser.add_argument("--candidate-cache", type=Path)
    parser.add_argument("--consequence-cache", type=Path)
    parser.add_argument("--scene-feature-cache", type=Path)
    parser.add_argument("--structured-future-cache", type=Path)
    parser.add_argument("--factual-probe-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--controls", nargs="+", choices=tuple(CONTROL_TO_MODE))
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--max-epochs", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = _load_yaml(config_path)
    data = config["data"]
    candidate_cache = _cache(args.candidate_cache, str(data["candidate_cache"]))
    consequence_cache = _cache(args.consequence_cache, str(data["consequence_cache"]))
    scene_feature_cache = _cache(args.scene_feature_cache, str(data["scene_feature_cache"]))
    structured_cache = _cache(args.structured_future_cache, str(data["structured_future_cache"]))
    factual_probe_dir = (
        args.factual_probe_dir.resolve()
        if args.factual_probe_dir is not None
        else _root("ACTION_EFFECT_OUTPUT_ROOT") / str(data["factual_probe_dir"])
    )
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else _root("ACTION_EFFECT_OUTPUT_ROOT") / "structured_factual_only" / "pilot_tiny"
    )
    with (factual_probe_dir / "split.json").open("r", encoding="utf-8") as stream:
        split = json.load(stream)
    fit_scenes, heldout_scenes = split["fit"], split["heldout"]
    arrays, scene_features, _, _, _, _ = load_probe_arrays(
        candidate_cache=candidate_cache,
        consequence_cache=consequence_cache,
        scene_feature_cache=scene_feature_cache,
        fit_scene_ids=fit_scenes,
        assumption="log_replay",
    )
    target, structured_valid = load_structured_targets(structured_cache, arrays)
    train_indices = np.flatnonzero(
        arrays.anchor & arrays.accepted & structured_valid & np.isin(arrays.scene_ids, fit_scenes)
    )
    validation_indices = np.flatnonzero(
        arrays.anchor & arrays.accepted & structured_valid & np.isin(arrays.scene_ids, heldout_scenes)
    )
    binary_mean = target[train_indices, :, :4].astype(np.float64).mean(axis=(0, 1, 3, 4))
    cap = float(config["training"]["dynamic_occupancy_positive_weight_cap"])
    positive_weight = np.clip((1.0 - binary_mean) / np.clip(binary_mean, 1.0e-6, None), 1.0, cap).astype(np.float32)
    controls = args.controls or list(config["controls"])
    seeds = args.seeds or [int(seed) for seed in config["training"]["seeds"]]
    source_paths = {
        "candidate": candidate_cache,
        "consequence": consequence_cache,
        "scene_feature": scene_feature_cache,
        "structured_future": structured_cache,
    }
    source_identity = {
        name: read_manifest(path).compatibility_identity() for name, path in source_paths.items()
    }
    code_files = [
        Path(__file__),
        REPOSITORY_ROOT / "research/action_effect/world_probe.py",
        REPOSITORY_ROOT / "research/action_effect/losses.py",
        REPOSITORY_ROOT / "research/action_effect/probe_data.py",
    ]
    identity = {
        "cache_version": config["experiment"]["cache_version"],
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPOSITORY_ROOT, text=True
        ).strip(),
        "config_hash": content_hash(config),
        "code_tree_hash": content_hash(
            {str(path.relative_to(REPOSITORY_ROOT)): file_sha256(path) for path in code_files}
        ),
        "source_manifests": source_identity,
        "split_sha256": content_hash(split),
        "controls": controls,
        "seeds": seeds,
        "max_epochs_override": args.max_epochs,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    identity_path = output_dir / "build_identity.json"
    if identity_path.is_file():
        with identity_path.open("r", encoding="utf-8") as stream:
            if json.load(stream) != identity:
                raise CacheConflictError(f"structured-probe output identity differs: {output_dir}")
    else:
        write_json(identity_path, identity)
    write_json(
        output_dir / "data_summary.json",
        {
            "fit_scene_count": len(fit_scenes),
            "heldout_scene_count": len(heldout_scenes),
            "factual_fit_count": len(train_indices),
            "factual_heldout_count": len(validation_indices),
            "structured_valid_count": int(structured_valid.sum()),
            "target_shape": list(target.shape),
            "binary_positive_weight": positive_weight.tolist(),
            "provenance": "log_replay",
        },
    )
    results: list[dict[str, Any]] = []
    for control in controls:
        for seed in seeds:
            run_dir = output_dir / control / f"seed_{seed}"
            if (run_dir / "result.json").is_file() and (run_dir / "predictions.npz").is_file():
                with (run_dir / "result.json").open("r", encoding="utf-8") as stream:
                    results.append(json.load(stream))
                continue
            run_dir.mkdir(parents=True, exist_ok=True)
            result = _train_one(
                config=config,
                control=control,
                seed=seed,
                arrays=arrays,
                scene_features=scene_features,
                target=target,
                train_indices=train_indices,
                validation_indices=validation_indices,
                positive_weight=positive_weight,
                output_dir=run_dir,
                max_epochs=args.max_epochs,
            )
            results.append(result)
            print(json.dumps(result, sort_keys=True))

    # Per-cell factual-fit prior is a stronger structured baseline than one
    # global scalar and uses no held-out statistics.
    constant = target[train_indices].astype(np.float32).mean(axis=0)
    probability = np.clip(constant[:, :4], 1.0e-4, 1.0 - 1.0e-4)
    constant[:, :4] = np.log(probability / (1.0 - probability))
    shape = tuple(int(value) for value in config["probe"]["structured_future_shape"])
    for seed in seeds:
        run_dir = output_dir / "constant_mean_control" / f"seed_{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        write_npz(
            run_dir / "predictions.npz",
            structured_future_prediction=np.broadcast_to(constant.astype(np.float16), (len(arrays.scene_ids), *shape)).copy(),
        )
        result = {
            "control": "constant_mean_control",
            "seed": seed,
            "best_epoch": None,
            "epochs_ran": 0,
            "best_validation_loss": None,
            "total_parameters": 0,
            "trainable_parameters": 0,
            "device": "not_applicable",
        }
        write_json(run_dir / "result.json", result)
        results.append(result)
    write_jsonl(output_dir / "training_results.jsonl", results)
    write_json(output_dir / "complete.json", {"run_count": len(results), "identity": identity})
    print(json.dumps({"output_dir": str(output_dir), "run_count": len(results)}, indent=2))


if __name__ == "__main__":
    main()

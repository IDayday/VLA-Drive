#!/usr/bin/env python3
"""Train frozen-feature factual consequence probes and required action controls."""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn
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
from research.action_effect.losses import ConsequencePredictionLoss  # noqa: E402
from research.action_effect.probe_data import (  # noqa: E402
    HARD_TARGET_FIELDS,
    SOFT_TARGET_FIELDS,
    deterministic_scene_split,
    iter_jsonl,
    load_probe_arrays,
    scales_to_json,
)
from research.action_effect.world_probe import (  # noqa: E402
    ActionEffectWorldProbe,
    count_parameters,
)


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


def _environment_root(variable: str) -> Path:
    value = os.environ.get(variable, "").strip()
    if not value:
        raise ValueError(f"source load_env.sh or set {variable}")
    return Path(value).resolve()


def _resolve_cache(explicit: Path | None, relative: str) -> Path:
    return explicit.resolve() if explicit is not None else _environment_root("ACTION_EFFECT_CACHE_ROOT") / relative


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _atomic_torch_save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(handle)
    try:
        torch.save(value, name)
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def _model(config: dict[str, Any], *, consequence_dim: int, mode: str) -> ActionEffectWorldProbe:
    probe = config["probe"]
    shape = probe.get("structured_future_shape")
    return ActionEffectWorldProbe(
        scene_input_dim=int(probe["scene_input_dim"]),
        action_hidden_dim=None,
        consequence_dim=consequence_dim,
        latent_dim=int(probe["latent_dim"]),
        trajectory_input_dim=int(probe["trajectory_input_dim"]),
        trajectory_token_dim=int(probe["trajectory_token_dim"]),
        dropout=float(probe["dropout"]),
        structured_future_shape=tuple(shape) if shape is not None else None,
        input_mode=mode,
    )


@torch.no_grad()
def _predict_all(
    model: nn.Module,
    *,
    scene_features: np.ndarray,
    scene_indices: np.ndarray,
    trajectories: np.ndarray,
    device: torch.device,
    batch_size: int = 512,
) -> np.ndarray:
    model.eval()
    predictions: list[np.ndarray] = []
    for start in range(0, len(trajectories), batch_size):
        stop = min(start + batch_size, len(trajectories))
        scene = torch.from_numpy(scene_features[scene_indices[start:stop]]).to(device)
        trajectory = torch.from_numpy(trajectories[start:stop]).to(device)
        output = model(scene, trajectory)["consequence_prediction"]
        assert isinstance(output, torch.Tensor)
        predictions.append(output.float().cpu().numpy())
    return np.concatenate(predictions, axis=0)


def _loss_on_indices(
    model: nn.Module,
    loss_fn: ConsequencePredictionLoss,
    indices: np.ndarray,
    *,
    arrays,
    scene_features: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> float:
    model.eval()
    values: list[float] = []
    weights: list[int] = []
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            batch = indices[start : start + batch_size]
            scene = torch.from_numpy(scene_features[arrays.scene_feature_indices[batch]]).to(device)
            trajectory = torch.from_numpy(arrays.trajectories[batch]).to(device)
            target = torch.from_numpy(arrays.targets[batch]).to(device)
            prediction = model(scene, trajectory)["consequence_prediction"]
            assert isinstance(prediction, torch.Tensor)
            value = loss_fn(prediction, target)["total"]
            values.append(float(value.item()))
            weights.append(len(batch))
    return float(np.average(values, weights=weights))


def _train_one(
    *,
    config: dict[str, Any],
    control: str,
    seed: int,
    arrays,
    scene_features: np.ndarray,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    output_dir: Path,
    max_epochs: int | None,
) -> dict[str, Any]:
    training = config["training"]
    requested_device = str(training["device"])
    device = torch.device(requested_device if torch.cuda.is_available() else "cpu")
    _seed_everything(seed)
    model = _model(
        config,
        consequence_dim=len(HARD_TARGET_FIELDS) + len(SOFT_TARGET_FIELDS),
        mode=CONTROL_TO_MODE[control],
    ).to(device)
    loss_fn = ConsequencePredictionLoss(len(HARD_TARGET_FIELDS))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    epochs = int(max_epochs if max_epochs is not None else training["epochs"])
    patience = int(training["patience"])
    batch_size = int(training["batch_size"])
    gradient_clip = float(training["gradient_clip_norm"])
    rng = np.random.default_rng(seed)
    best_loss = float("inf")
    best_epoch = -1
    best_state: dict[str, torch.Tensor] | None = None
    stale = 0
    history: list[dict[str, Any]] = []
    for epoch in range(epochs):
        model.train()
        order = rng.permutation(train_indices)
        train_total = 0.0
        train_hard = 0.0
        train_soft = 0.0
        seen = 0
        for start in range(0, len(order), batch_size):
            batch = order[start : start + batch_size]
            scene = torch.from_numpy(scene_features[arrays.scene_feature_indices[batch]]).to(device)
            trajectory = torch.from_numpy(arrays.trajectories[batch]).to(device)
            if control == "shuffled_action_probe" and len(batch) > 1:
                trajectory = torch.roll(trajectory, shifts=1, dims=0)
            target = torch.from_numpy(arrays.targets[batch]).to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(scene, trajectory)["consequence_prediction"]
            assert isinstance(prediction, torch.Tensor)
            losses = loss_fn(prediction, target)
            losses["total"].backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError(f"non-finite gradient norm at epoch {epoch}")
            optimizer.step()
            count = len(batch)
            train_total += float(losses["total"].item()) * count
            train_hard += float(losses["hard"].item()) * count
            train_soft += float(losses["soft"].item()) * count
            seen += count
        validation_loss = _loss_on_indices(
            model,
            loss_fn,
            validation_indices,
            arrays=arrays,
            scene_features=scene_features,
            device=device,
            batch_size=batch_size,
        )
        row = {
            "epoch": epoch,
            "train_loss": train_total / seen,
            "train_hard_loss": train_hard / seen,
            "train_soft_loss": train_soft / seen,
            "validation_loss": validation_loss,
        }
        history.append(row)
        if validation_loss < best_loss - 1.0e-6:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    if best_state is None:
        raise RuntimeError("probe training produced no valid checkpoint")
    model.load_state_dict(best_state)
    prediction = _predict_all(
        model,
        scene_features=scene_features,
        scene_indices=arrays.scene_feature_indices,
        trajectories=arrays.trajectories,
        device=device,
    )
    write_jsonl(output_dir / "history.jsonl", history)
    write_npz(output_dir / "predictions.npz", consequence_prediction=prediction.astype(np.float32))
    _atomic_torch_save(
        output_dir / "probe.pt",
        {
            "state_dict": best_state,
            "control": control,
            "seed": seed,
            "best_epoch": best_epoch,
            "best_validation_loss": best_loss,
        },
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
        "--config", type=Path, default=REPOSITORY_ROOT / "configs/action_effect/factual_only.yaml"
    )
    parser.add_argument("--candidate-cache", type=Path)
    parser.add_argument("--consequence-cache", type=Path)
    parser.add_argument("--scene-feature-cache", type=Path)
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
    candidate_cache = _resolve_cache(args.candidate_cache, str(data["candidate_cache"]))
    consequence_cache = _resolve_cache(args.consequence_cache, str(data["consequence_cache"]))
    scene_feature_cache = _resolve_cache(args.scene_feature_cache, str(data["scene_feature_cache"]))
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else _environment_root("ACTION_EFFECT_OUTPUT_ROOT") / "factual_only" / "pilot_tiny"
    )
    for path, label in (
        (candidate_cache, "candidate cache"),
        (consequence_cache, "consequence cache"),
        (scene_feature_cache, "scene-feature cache"),
    ):
        if not path.is_dir() or read_manifest(path) is None:
            raise FileNotFoundError(f"published {label} is missing: {path}")
    with (candidate_cache / "scene_index.json").open("r", encoding="utf-8") as stream:
        candidate_scene_ids = list(json.load(stream))
    assumption = str(data["target_assumption"])
    factual_rows = [
        row
        for row in iter_jsonl(consequence_cache / "consequences.jsonl")
        if row.get("perturbation_type") == "anchor"
        and row.get("candidate_accepted")
        and row[assumption].get("available")
    ]
    scene_ids = [str(row["scene_id"]) for row in factual_rows]
    if len(scene_ids) != len(set(scene_ids)):
        raise RuntimeError("eligible factual scene IDs are not unique")
    split_seed = int(config["experiment"]["seed"])
    fit_scenes, heldout_scenes = deterministic_scene_split(
        scene_ids,
        fraction=float(config["experiment"]["split_fraction"]),
        seed=split_seed,
    )
    arrays, scene_features, _, scales, trajectory_stats, _ = load_probe_arrays(
        candidate_cache=candidate_cache,
        consequence_cache=consequence_cache,
        scene_feature_cache=scene_feature_cache,
        fit_scene_ids=fit_scenes,
        assumption=assumption,
    )
    fit_set, heldout_set = set(fit_scenes), set(heldout_scenes)
    train_indices = np.flatnonzero(
        arrays.accepted & arrays.anchor & np.isin(arrays.scene_ids, list(fit_set))
    )
    validation_indices = np.flatnonzero(
        arrays.accepted & arrays.anchor & np.isin(arrays.scene_ids, list(heldout_set))
    )
    if len(train_indices) != len(fit_scenes) or len(validation_indices) != len(heldout_scenes):
        raise RuntimeError("each split scene must have exactly one accepted factual anchor")

    controls = args.controls or list(config["controls"])
    seeds = args.seeds or [int(value) for value in config["training"]["seeds"]]
    source_manifests = {
        label: read_manifest(path).compatibility_identity()
        for label, path in (
            ("candidate", candidate_cache),
            ("consequence", consequence_cache),
            ("scene_feature", scene_feature_cache),
        )
    }
    code_files = [
        Path(__file__),
        REPOSITORY_ROOT / "research/action_effect/world_probe.py",
        REPOSITORY_ROOT / "research/action_effect/losses.py",
        REPOSITORY_ROOT / "research/action_effect/probe_data.py",
    ]
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPOSITORY_ROOT, text=True
    ).strip()
    identity = {
        "cache_version": config["experiment"]["cache_version"],
        "git_commit": revision,
        "code_tree_hash": content_hash(
            {str(path.relative_to(REPOSITORY_ROOT)): file_sha256(path) for path in code_files}
        ),
        "config_hash": content_hash(config),
        "source_manifests": source_manifests,
        "fit_scenes_sha256": content_hash(fit_scenes),
        "heldout_scenes_sha256": content_hash(heldout_scenes),
        "controls": controls,
        "seeds": seeds,
        "max_epochs_override": args.max_epochs,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    identity_path = output_dir / "build_identity.json"
    if identity_path.is_file():
        with identity_path.open("r", encoding="utf-8") as stream:
            if json.load(stream) != identity:
                raise CacheConflictError(f"probe output identity differs: {output_dir}")
    else:
        write_json(identity_path, identity)
    write_json(output_dir / "split.json", {"fit": fit_scenes, "heldout": heldout_scenes})
    write_json(output_dir / "target_scales.json", scales_to_json(scales))
    write_json(output_dir / "trajectory_normalization.json", trajectory_stats)
    write_json(
        output_dir / "data_summary.json",
        {
            "fit_scene_count": len(fit_scenes),
            "heldout_scene_count": len(heldout_scenes),
            "factual_fit_count": len(train_indices),
            "factual_heldout_count": len(validation_indices),
            "accepted_candidate_count": int(arrays.accepted.sum()),
            "target_assumption": assumption,
            "candidate_scene_count": len(candidate_scene_ids),
            "excluded_scene_count": len(candidate_scene_ids) - len(scene_ids),
            "target_fields": [*HARD_TARGET_FIELDS, *SOFT_TARGET_FIELDS],
            "normalization_fit_on": "factual anchors in fit scenes only",
        },
    )

    results: list[dict[str, Any]] = []
    for control in controls:
        for seed in seeds:
            run_dir = output_dir / control / f"seed_{seed}"
            result_path = run_dir / "result.json"
            prediction_path = run_dir / "predictions.npz"
            if result_path.is_file() and prediction_path.is_file():
                with result_path.open("r", encoding="utf-8") as stream:
                    results.append(json.load(stream))
                continue
            run_dir.mkdir(parents=True, exist_ok=True)
            result = _train_one(
                config=config,
                control=control,
                seed=seed,
                arrays=arrays,
                scene_features=scene_features,
                train_indices=train_indices,
                validation_indices=validation_indices,
                output_dir=run_dir,
                max_epochs=args.max_epochs,
            )
            results.append(result)
            print(json.dumps(result, sort_keys=True))

    # Two low-information references make "learnable" an explicit comparison.
    hard_prevalence = np.clip(
        arrays.targets[train_indices, : len(HARD_TARGET_FIELDS)].mean(axis=0), 1.0e-4, 1.0 - 1.0e-4
    )
    constant = np.concatenate(
        (
            np.log(hard_prevalence / (1.0 - hard_prevalence)),
            arrays.targets[train_indices, len(HARD_TARGET_FIELDS) :].mean(axis=0),
        )
    ).astype(np.float32)
    for seed in seeds:
        constant_dir = output_dir / "constant_mean_control" / f"seed_{seed}"
        constant_dir.mkdir(parents=True, exist_ok=True)
        write_npz(
            constant_dir / "predictions.npz",
            consequence_prediction=np.broadcast_to(constant, arrays.targets.shape).copy(),
        )
        constant_result = {
            "control": "constant_mean_control",
            "seed": seed,
            "best_epoch": None,
            "epochs_ran": 0,
            "best_validation_loss": None,
            "total_parameters": 0,
            "trainable_parameters": 0,
            "device": "not_applicable",
        }
        write_json(constant_dir / "result.json", constant_result)
        results.append(constant_result)

        random_dir = output_dir / "random_untrained_probe" / f"seed_{seed}"
        random_dir.mkdir(parents=True, exist_ok=True)
        if not (random_dir / "predictions.npz").is_file():
            _seed_everything(seed)
            requested = str(config["training"]["device"])
            device = torch.device(requested if torch.cuda.is_available() else "cpu")
            random_model = _model(
                config,
                consequence_dim=len(HARD_TARGET_FIELDS) + len(SOFT_TARGET_FIELDS),
                mode="scene_action",
            ).to(device)
            random_prediction = _predict_all(
                random_model,
                scene_features=scene_features,
                scene_indices=arrays.scene_feature_indices,
                trajectories=arrays.trajectories,
                device=device,
            )
            write_npz(
                random_dir / "predictions.npz",
                consequence_prediction=random_prediction.astype(np.float32),
            )
            random_result = {
                "control": "random_untrained_probe",
                "seed": seed,
                "best_epoch": None,
                "epochs_ran": 0,
                "best_validation_loss": None,
                "total_parameters": count_parameters(random_model),
                "trainable_parameters": count_parameters(random_model, trainable_only=True),
                "device": str(device),
            }
            write_json(random_dir / "result.json", random_result)
        else:
            with (random_dir / "result.json").open("r", encoding="utf-8") as stream:
                random_result = json.load(stream)
        results.append(random_result)
    write_jsonl(output_dir / "training_results.jsonl", results)
    write_json(output_dir / "complete.json", {"run_count": len(results), "identity": identity})
    print(json.dumps({"output_dir": str(output_dir), "run_count": len(results)}, indent=2))


if __name__ == "__main__":
    main()

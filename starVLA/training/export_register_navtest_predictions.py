#!/usr/bin/env python3
"""Export integrated Register planner trajectories for official NAVSIM scoring."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from accelerate import Accelerator
from accelerate.utils import DataLoaderConfiguration, broadcast_object_list

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from starVLA.dataloader import build_dataloader
from starVLA.model.framework import build_framework
from starVLA.training.config_loader import load_training_config
from starVLA.training.register_stage_utils import atomic_json
from starVLA.model.modules.register_planner.checkpoint import stable_config_hash


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--datalist", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=3)
    parser.add_argument("--generator-checkpoint")
    parser.add_argument("--drivor-checkpoint")
    parser.add_argument("--suprim-checkpoint")
    parser.add_argument(
        "--export-candidates",
        action="store_true",
        help=(
            "Also persist every Register proposal plus the selector index under "
            "OUTPUT_DIR/candidates. The ordinary official trajectory export is "
            "kept unchanged."
        ),
    )
    parser.add_argument(
        "--feature-cache-root",
        help="Optional split-specific test cache. Training caches are never reused implicitly.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true")
    mode.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _load_tokens(path: Path) -> list[str]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"NAVSIM datalist must be a non-empty JSON list: {path}")
    if any(not isinstance(token, str) or not token for token in raw):
        raise TypeError("NAVSIM datalist entries must be non-empty token strings")
    if len(raw) != len(set(raw)):
        raise ValueError("NAVSIM datalist contains duplicate tokens")
    return raw


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repository_commit(root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()


def _valid_prediction(path: Path) -> bool:
    try:
        value = np.load(path, allow_pickle=False)
    except Exception:
        return False
    return value.shape == (8, 3) and np.isfinite(value).all()


def _valid_candidate(path: Path, proposal_num: int) -> bool:
    try:
        with np.load(path, allow_pickle=False) as payload:
            proposals = payload["proposals"]
            selected_index = payload["selected_index"]
    except Exception:
        return False
    return (
        proposals.shape == (proposal_num, 8, 3)
        and np.isfinite(proposals).all()
        and selected_index.shape == ()
        and 0 <= int(selected_index) < proposal_num
    )


def _atomic_numpy(path: Path, value: np.ndarray) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as stream:
        np.save(stream, value, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _atomic_candidate(
    path: Path, proposals: np.ndarray, selected_index: int
) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as stream:
        np.savez(
            stream,
            proposals=proposals,
            selected_index=np.asarray(selected_index, dtype=np.int64),
        )
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _apply_overrides(config: Any, args: argparse.Namespace) -> None:
    config.datasets.vla_data.datalist_path = str(Path(args.datalist).resolve())
    config.datasets.vla_data.data_root = str(Path(args.data_root).resolve())
    config.datasets.vla_data.split = str(args.split)
    config.datasets.vla_data.shuffle = False
    config.datasets.vla_data.per_device_batch_size = int(args.batch_size)
    config.datasets.vla_data.num_workers = int(args.num_workers)
    config.datasets.reward_data.load_reward_data = 0
    config.datasets.vla_data.w_neg_traj = None
    config.framework.inference.return_all_proposals = bool(args.export_candidates)
    if args.generator_checkpoint:
        config.framework.inference.generator_checkpoint = str(
            Path(args.generator_checkpoint).resolve()
        )
    if args.drivor_checkpoint:
        config.framework.inference.drivor_checkpoint = str(
            Path(args.drivor_checkpoint).resolve()
        )
    if args.suprim_checkpoint:
        config.framework.inference.suprim_checkpoint = str(
            Path(args.suprim_checkpoint).resolve()
        )


def _checkpoint_paths(config: Any) -> dict[str, Path]:
    inference = config.framework.inference
    paths = {
        "generator": Path(str(inference.generator_checkpoint)).expanduser().resolve(),
        "drivor": Path(str(inference.drivor_checkpoint)).expanduser().resolve(),
    }
    if inference.get("suprim_checkpoint"):
        paths["suprim"] = Path(str(inference.suprim_checkpoint)).expanduser().resolve()
    minimum_age = float(os.environ.get("CHECKPOINT_MIN_AGE_SECONDS", "0"))
    now = time.time()
    for name, path in paths.items():
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(f"{name} checkpoint is missing or empty: {path}")
        age = now - path.stat().st_mtime
        if age < minimum_age:
            raise RuntimeError(
                f"{name} checkpoint age {age:.1f}s is below "
                f"CHECKPOINT_MIN_AGE_SECONDS={minimum_age:.1f}"
            )
    return paths


def _prediction_identity(
    *,
    config: Any,
    args: argparse.Namespace,
    project_root: Path,
    datalist_path: Path,
    data_root: Path,
    tokens: list[str],
    checkpoints: dict[str, Path],
) -> dict[str, Any]:
    identity = {
        "schema_version": 1,
        "stage": "register_navtest_prediction",
        "repository_commit": _repository_commit(project_root),
        "inference_config": str(Path(args.config).expanduser().resolve()),
        "inference_config_hash": stable_config_hash(config.framework),
        "selector_type": str(config.framework.inference.selector_type),
        "split": str(args.split),
        "data_root": str(data_root),
        "datalist": str(datalist_path),
        "datalist_sha256": _sha256(datalist_path),
        "num_predictions": len(tokens),
        "trajectory_shape": [8, 3],
        "feature_cache_root": (
            str(Path(args.feature_cache_root).expanduser().resolve())
            if args.feature_cache_root
            else None
        ),
        "checkpoints": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in checkpoints.items()
        },
    }
    if args.export_candidates:
        proposal_num = int(config.framework.register_generator.proposal_num)
        identity["candidate_export"] = {
            "relative_directory": "candidates",
            "proposal_shape": [proposal_num, 8, 3],
            "selected_index_shape": [],
            "archive_fields": ["proposals", "selected_index"],
        }
    return identity


def _read_identity(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"prediction identity must be a JSON object: {path}")
    return payload


def _write_rank_manifest(
    output_dir: Path,
    accelerator: Accelerator,
    *,
    identity_hash: str,
    written: int,
    resumed: int,
    wall_time_seconds: float,
) -> None:
    atomic_json(
        output_dir / f"inference_manifest.rank{accelerator.process_index}.json",
        {
            "schema_version": 1,
            "rank": accelerator.process_index,
            "world_size": accelerator.num_processes,
            "identity_hash": identity_hash,
            "written": int(written),
            "resumed": int(resumed),
            "wall_time_seconds": float(wall_time_seconds),
        },
    )


def _finalize_manifest(
    output_dir: Path,
    *,
    identity: dict[str, Any],
    accelerator: Accelerator,
    written: int,
    resumed: int,
    wall_time_seconds: float,
    batch_size_per_rank: int,
    workers_per_rank: int,
) -> dict[str, Any]:
    identity_hash = stable_config_hash(identity)
    rank_manifests = []
    for rank in range(accelerator.num_processes):
        path = output_dir / f"inference_manifest.rank{rank}.json"
        payload = _read_identity(path)
        if (
            payload.get("rank") != rank
            or payload.get("world_size") != accelerator.num_processes
            or payload.get("identity_hash") != identity_hash
        ):
            raise RuntimeError(f"invalid per-rank inference manifest: {path}")
        rank_manifests.append(payload)
    manifest = {
        **identity,
        "identity_hash": identity_hash,
        "distributed": {
            "world_size": accelerator.num_processes,
            "batch_size_per_rank": int(batch_size_per_rank),
            "workers_per_rank": int(workers_per_rank),
        },
        "written": int(written),
        "resumed": int(resumed),
        "wall_time_seconds": float(wall_time_seconds),
        "rank_manifests": rank_manifests,
    }
    atomic_json(output_dir / "prediction_manifest.json", manifest)
    return manifest


def main() -> None:
    args = _parse_args()
    if args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("batch size must be positive and num workers non-negative")
    if args.split != "test":
        raise ValueError("official navtest export requires --split test")

    project_root = Path(__file__).resolve().parents[2]
    datalist_path = Path(args.datalist).expanduser().resolve()
    data_root = Path(args.data_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    tokens = _load_tokens(datalist_path)
    token_set = set(tokens)
    if not (data_root / "meta" / args.split).is_dir():
        raise FileNotFoundError(data_root / "meta" / args.split)

    # A training feature cache may contain only navtrain. Evaluation therefore
    # defaults to raw navtest inputs and accepts a cache only via an explicit
    # split-specific CLI path.
    if args.feature_cache_root:
        feature_cache_root = Path(args.feature_cache_root).expanduser().resolve()
        if not feature_cache_root.is_dir():
            raise FileNotFoundError(feature_cache_root)
        os.environ["NAVSIM_FEATURE_CACHE_ROOT"] = str(feature_cache_root)
    else:
        os.environ.pop("NAVSIM_FEATURE_CACHE_ROOT", None)
        os.environ["NAVSIM_USE_FEATURE_CACHE"] = "0"

    accelerator = Accelerator(
        mixed_precision="bf16",
        dataloader_config=DataLoaderConfiguration(even_batches=False),
    )
    config = load_training_config(args.config)
    _apply_overrides(config, args)
    proposal_num = int(config.framework.register_generator.proposal_num)
    if args.export_candidates and proposal_num <= 1:
        raise ValueError("candidate export requires proposal_num > 1")
    candidate_dir = output_dir / "candidates"
    config.output_dir = str(output_dir)
    checkpoints = _checkpoint_paths(config)
    identity_box = [
        _prediction_identity(
            config=config,
            args=args,
            project_root=project_root,
            datalist_path=datalist_path,
            data_root=data_root,
            tokens=tokens,
            checkpoints=checkpoints,
        )
        if accelerator.is_main_process
        else None
    ]
    broadcast_object_list(identity_box)
    identity = identity_box[0]
    if not isinstance(identity, dict):
        raise RuntimeError("main process did not broadcast a prediction identity")
    identity["trajectory_coordinate_system"] = "ego_relative_x_y_heading"
    identity_hash = stable_config_hash(identity)
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        identity_path = output_dir / "prediction_identity.json"
        existing_entries = list(output_dir.iterdir())
        if args.overwrite:
            # ``--overwrite`` is the only mode allowed to replace exact
            # prediction artifacts. No directory or unrelated file is removed.
            for path in output_dir.glob("*.npy"):
                path.unlink()
            for path in output_dir.glob("inference_manifest.rank*.json"):
                path.unlink()
            if args.export_candidates and candidate_dir.is_dir():
                for path in candidate_dir.glob("*.npz"):
                    path.unlink()
                try:
                    candidate_dir.rmdir()
                except OSError:
                    # Unknown files are intentionally preserved and make the
                    # overwrite contract fail closed below.
                    pass
            for path in (
                identity_path,
                output_dir / "prediction_manifest.json",
            ):
                if path.is_file():
                    path.unlink()
            existing_entries = list(output_dir.iterdir())
        if existing_entries and not args.resume:
            raise FileExistsError(
                f"Refusing to overwrite prediction artifacts under {output_dir}; "
                "use --resume for the same identity or --overwrite explicitly"
            )
        if args.resume and existing_entries:
            if not identity_path.is_file():
                raise RuntimeError(
                    "Refusing to resume prediction artifacts without "
                    "prediction_identity.json"
                )
            if _read_identity(identity_path) != identity:
                raise RuntimeError(
                    "Refusing to mix prediction artifacts from a different "
                    "checkpoint/config/datalist identity"
                )
        if args.export_candidates:
            candidate_dir.mkdir(parents=True, exist_ok=True)
        atomic_json(identity_path, identity)
    accelerator.wait_for_everyone()

    already_complete = torch.zeros((), device=accelerator.device, dtype=torch.int32)
    if accelerator.is_main_process and args.resume:
        actual = {path.stem for path in output_dir.glob("*.npy")}
        candidates_complete = (not args.export_candidates) or (
            {path.stem for path in candidate_dir.glob("*.npz")} == token_set
            and all(
                _valid_candidate(candidate_dir / f"{token}.npz", proposal_num)
                for token in token_set
            )
        )
        if (
            actual == token_set
            and all(
                _valid_prediction(output_dir / f"{token}.npy")
                for token in token_set
            )
            and candidates_complete
        ):
            already_complete.fill_(1)
    already_complete = accelerator.reduce(already_complete, reduction="max")
    if bool(already_complete.item()):
        _write_rank_manifest(
            output_dir,
            accelerator,
            identity_hash=identity_hash,
            written=0,
            resumed=len(tokens) if accelerator.is_main_process else 0,
            wall_time_seconds=0.0,
        )
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            manifest = _finalize_manifest(
                output_dir,
                identity=identity,
                accelerator=accelerator,
                written=0,
                resumed=len(tokens),
                wall_time_seconds=0.0,
                batch_size_per_rank=args.batch_size,
                workers_per_rank=args.num_workers,
            )
            print(json.dumps(manifest, indent=2, sort_keys=True))
        return
    loader = build_dataloader(
        cfg=config, dataset_py=config.datasets.vla_data.dataset_py
    )
    model = build_framework(config)
    if model.__class__.__name__ != "QwenRegisterPlanner":
        raise RuntimeError("navtest export requires QwenRegisterPlanner")
    model = accelerator.prepare_model(model, evaluation_mode=True)
    loader = accelerator.prepare_data_loader(loader)
    model.eval()

    local_written = 0
    local_skipped = 0
    start = time.perf_counter()
    with torch.inference_mode():
        for examples in loader:
            pending = []
            for example in examples:
                path = output_dir / f"{example['token']}.npy"
                candidate_path = candidate_dir / f"{example['token']}.npz"
                artifacts_valid = _valid_prediction(path) and (
                    not args.export_candidates
                    or _valid_candidate(candidate_path, proposal_num)
                )
                if args.resume and artifacts_valid:
                    local_skipped += 1
                else:
                    pending.append(example)
            if not pending:
                continue
            output = model(pending)
            trajectories = output["trajectory_navsim_8"].detach().float().cpu().numpy()
            if trajectories.shape != (len(pending), 8, 3):
                raise RuntimeError(
                    f"integrated planner returned {trajectories.shape}, expected {(len(pending), 8, 3)}"
                )
            if not np.isfinite(trajectories).all():
                raise FloatingPointError("integrated planner produced NaN or Inf")
            proposals = None
            selected_indices = None
            if args.export_candidates:
                all_proposals = output.get("all_proposals")
                if all_proposals is None:
                    raise RuntimeError(
                        "candidate export requested but planner returned no proposals"
                    )
                proposals = all_proposals.detach().float().cpu().numpy()
                selected_indices = (
                    output["selected_index"].detach().to(torch.int64).cpu().numpy()
                )
                expected = (len(pending), proposal_num, 8, 3)
                if proposals.shape != expected:
                    raise RuntimeError(
                        f"integrated planner returned proposals {proposals.shape}, "
                        f"expected {expected}"
                    )
                if selected_indices.shape != (len(pending),):
                    raise RuntimeError(
                        "integrated planner returned invalid selected_index shape: "
                        f"{selected_indices.shape}"
                    )
                if not np.isfinite(proposals).all():
                    raise FloatingPointError("integrated planner proposals contain NaN or Inf")
                if (
                    (selected_indices < 0).any()
                    or (selected_indices >= proposal_num).any()
                ):
                    raise IndexError("integrated planner selected_index is out of range")
                rows = np.arange(len(pending), dtype=np.int64)
                selected_proposals = proposals[rows, selected_indices]
                if not np.allclose(
                    selected_proposals, trajectories, rtol=1.0e-5, atol=1.0e-5
                ):
                    raise RuntimeError(
                        "selected trajectory does not match proposals[selected_index]"
                    )
            for index, (example, trajectory) in enumerate(zip(pending, trajectories)):
                _atomic_numpy(output_dir / f"{example['token']}.npy", trajectory)
                if args.export_candidates:
                    assert proposals is not None and selected_indices is not None
                    _atomic_candidate(
                        candidate_dir / f"{example['token']}.npz",
                        proposals[index],
                        int(selected_indices[index]),
                    )
                local_written += 1

    counts = torch.tensor(
        [local_written, local_skipped],
        device=accelerator.device,
        dtype=torch.int64,
    )
    counts = accelerator.reduce(counts, reduction="sum")
    wall_time_seconds = time.perf_counter() - start
    _write_rank_manifest(
        output_dir,
        accelerator,
        identity_hash=identity_hash,
        written=local_written,
        resumed=local_skipped,
        wall_time_seconds=wall_time_seconds,
    )
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        actual = {path.stem for path in output_dir.glob("*.npy")}
        missing = sorted(token_set - actual)
        extra = sorted(actual - token_set)
        invalid = sorted(
            token
            for token in token_set & actual
            if not _valid_prediction(output_dir / f"{token}.npy")
        )
        candidate_missing: list[str] = []
        candidate_extra: list[str] = []
        candidate_invalid: list[str] = []
        if args.export_candidates:
            candidate_actual = {path.stem for path in candidate_dir.glob("*.npz")}
            candidate_missing = sorted(token_set - candidate_actual)
            candidate_extra = sorted(candidate_actual - token_set)
            candidate_invalid = sorted(
                token
                for token in token_set & candidate_actual
                if not _valid_candidate(
                    candidate_dir / f"{token}.npz", proposal_num
                )
            )
        if (
            missing
            or extra
            or invalid
            or candidate_missing
            or candidate_extra
            or candidate_invalid
        ):
            raise RuntimeError(
                "prediction set validation failed: "
                f"missing={len(missing)} extra={len(extra)} invalid={len(invalid)} "
                f"candidate_missing={len(candidate_missing)} "
                f"candidate_extra={len(candidate_extra)} "
                f"candidate_invalid={len(candidate_invalid)}"
            )
        manifest = _finalize_manifest(
            output_dir,
            identity=identity,
            accelerator=accelerator,
            written=int(counts[0].item()),
            resumed=int(counts[1].item()),
            wall_time_seconds=wall_time_seconds,
            batch_size_per_rank=args.batch_size,
            workers_per_rank=args.num_workers,
        )
        print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

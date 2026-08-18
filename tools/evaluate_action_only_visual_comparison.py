#!/usr/bin/env python3
"""Build and summarize reproducible frozen-vs-trainable visual evaluations."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import pickle
import random
import subprocess
from pathlib import Path
from typing import Any, Iterable

import numpy as np


PDMS_COLUMNS = (
    "pdms",
    "no_at_fault_collisions",
    "drivable_area_compliance",
    "ego_progress",
    "time_to_collision_within_bound",
    "comfort",
    "driving_direction_compliance",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _wrap_to_pi(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2 * np.pi) - np.pi


def make_train_subset(args: argparse.Namespace) -> None:
    datalist_path = Path(args.datalist).resolve()
    data_root = Path(args.data_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    tokens = json.loads(datalist_path.read_text(encoding="utf-8"))
    if not 0 < args.size <= len(tokens):
        raise ValueError(f"subset size must be in [1, {len(tokens)}], got {args.size}")

    rng = random.Random(args.seed)
    indices = sorted(rng.sample(range(len(tokens)), args.size))
    subset = [tokens[index] for index in indices]
    trajectories = []
    for token in subset:
        metadata_path = data_root / "meta" / "train" / f"{token}.pkl"
        if not metadata_path.is_file():
            raise FileNotFoundError(f"training metadata is missing: {metadata_path}")
        with metadata_path.open("rb") as stream:
            raw = pickle.load(stream)
        poses = np.asarray(raw["glo_status"]["global_poses"][:12], dtype=np.float64)
        if poses.shape != (12, 3):
            raise ValueError(f"{token}: expected global poses [12,3], got {poses.shape}")
        reference = poses[3]
        delta = poses[4:12, :2] - reference[None, :2]
        cosine, sine = math.cos(reference[2]), math.sin(reference[2])
        relative_xy = np.stack(
            [
                cosine * delta[:, 0] + sine * delta[:, 1],
                -sine * delta[:, 0] + cosine * delta[:, 1],
            ],
            axis=-1,
        )
        relative_heading = _wrap_to_pi(poses[4:12, 2] - reference[2])
        trajectories.append(np.concatenate([relative_xy, relative_heading[:, None]], axis=-1))

    trajectory = np.asarray(trajectories, dtype=np.float32)
    if trajectory.shape != (args.size, 8, 3) or not np.isfinite(trajectory).all():
        raise ValueError(f"invalid ground-truth trajectory tensor: {trajectory.shape}")

    subset_path = output_dir / "train_subset.json"
    ground_truth_path = output_dir / "train_subset_ground_truth.npz"
    manifest_path = output_dir / "train_subset_manifest.json"
    subset_payload = json.dumps(subset, indent=2) + "\n"
    _atomic_text(subset_path, subset_payload)
    _atomic_npz(
        ground_truth_path,
        tokens=np.asarray(subset),
        trajectory=trajectory,
    )
    manifest = {
        "schema_version": 1,
        "source_datalist": str(datalist_path),
        "source_datalist_sha256": _sha256(datalist_path),
        "source_count": len(tokens),
        "selection": "random_sample_without_replacement_sorted_by_source_index",
        "seed": args.seed,
        "count": args.size,
        "subset_sha256": hashlib.sha256(subset_payload.encode("utf-8")).hexdigest(),
        "ground_truth_sha256": _sha256(ground_truth_path),
        "ground_truth_shape": list(trajectory.shape),
        "coordinate_frame": "ego_at_history_index_3",
        "future_pose_indices": list(range(4, 12)),
    }
    _atomic_text(manifest_path, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(subset_path)
    print(ground_truth_path)


def score_train(args: argparse.Namespace) -> None:
    ground_truth_path = Path(args.ground_truth).resolve()
    prediction_dir = Path(args.prediction_dir).resolve()
    with np.load(ground_truth_path) as payload:
        tokens = payload["tokens"].astype(str).tolist()
        target = np.asarray(payload["trajectory"], dtype=np.float64)
    if target.shape != (len(tokens), 8, 3):
        raise ValueError(f"expected ground truth [N,8,3], got {target.shape}")

    predictions = []
    for token in tokens:
        prediction_path = prediction_dir / f"{token}.npy"
        if not prediction_path.is_file():
            raise FileNotFoundError(f"training-subset prediction is missing: {prediction_path}")
        prediction = np.asarray(np.load(prediction_path), dtype=np.float64)
        if prediction.shape != (8, 3) or not np.isfinite(prediction).all():
            raise ValueError(f"{prediction_path}: expected finite [8,3], got {prediction.shape}")
        predictions.append(prediction)
    prediction = np.stack(predictions)

    displacement = np.linalg.norm(prediction[..., :2] - target[..., :2], axis=-1)
    heading = np.abs(_wrap_to_pi(prediction[..., 2] - target[..., 2]))
    sample_ade = displacement.mean(axis=1)
    sample_fde = displacement[:, -1]
    metrics = {
        "schema_version": 1,
        "arm": args.arm,
        "step": args.step,
        "sample_count": len(tokens),
        "ade": float(sample_ade.mean()),
        "ade_p50": float(np.quantile(sample_ade, 0.5)),
        "ade_p90": float(np.quantile(sample_ade, 0.9)),
        "fde": float(sample_fde.mean()),
        "fde_p50": float(np.quantile(sample_fde, 0.5)),
        "fde_p90": float(np.quantile(sample_fde, 0.9)),
        "heading_mae_rad": float(heading.mean()),
        "prediction_dir": str(prediction_dir),
        "ground_truth": str(ground_truth_path),
    }
    _atomic_text(Path(args.output), json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metrics, sort_keys=True))


def build_manifest(args: argparse.Namespace) -> None:
    repository = Path(__file__).resolve().parents[1]
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repository, text=True
        ).strip()
        branch = subprocess.check_output(
            ["git", "branch", "--show-current"], cwd=repository, text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit, branch = "UNKNOWN", "UNKNOWN"

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "repository": str(repository),
        "commit": commit,
        "branch": branch,
        "inference_seed": args.seed,
        "world_size": args.world_size,
        "batch_size_per_rank": args.batch_size,
        "steps": args.steps,
        "arms": {},
    }
    for arm, run_value in (("frozen", args.frozen_run), ("visual", args.visual_run)):
        run_dir = Path(run_value).resolve()
        config_path = run_dir / "config.yaml"
        if not config_path.is_file():
            raise FileNotFoundError(f"{arm} config is missing: {config_path}")
        arm_manifest: dict[str, Any] = {
            "run_dir": str(run_dir),
            "config_sha256": _sha256(config_path),
            "checkpoints": {},
        }
        for step in args.steps:
            checkpoint = run_dir / "checkpoints" / f"steps_{step}_pytorch_model.pt"
            if not checkpoint.is_file():
                raise FileNotFoundError(f"{arm} checkpoint is missing: {checkpoint}")
            arm_manifest["checkpoints"][str(step)] = {
                "path": str(checkpoint),
                "size": checkpoint.stat().st_size,
                "sha256": _sha256(checkpoint),
            }
        manifest["arms"][arm] = arm_manifest
    _atomic_text(Path(args.output), json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(args.output)


def _read_pdms(summary_path: Path) -> dict[str, float]:
    with summary_path.open(newline="", encoding="utf-8") as stream:
        row = next(csv.DictReader(stream))
    return {column: float(row[column]) for column in PDMS_COLUMNS}


def _fmt(value: float | None, digits: int = 6) -> str:
    return "MISSING" if value is None else f"{value:.{digits}f}"


def summarize(args: argparse.Namespace) -> None:
    root = Path(args.root).resolve()
    rows = []
    for step in args.steps:
        arm_pdms = {}
        arm_train: dict[str, dict[str, float] | None] = {}
        for arm in ("frozen", "visual"):
            summary_path = root / "pdms" / arm / f"step{step}" / "summary.csv"
            if not summary_path.is_file():
                raise FileNotFoundError(f"PDMS summary is missing: {summary_path}")
            arm_pdms[arm] = _read_pdms(summary_path)
            train_path = root / "train_metrics" / arm / f"step{step}.json"
            arm_train[arm] = (
                json.loads(train_path.read_text(encoding="utf-8"))
                if train_path.is_file()
                else None
            )

        row: dict[str, Any] = {"step": step}
        for metric in PDMS_COLUMNS:
            frozen_value = arm_pdms["frozen"][metric]
            visual_value = arm_pdms["visual"][metric]
            row[f"frozen_{metric}"] = frozen_value
            row[f"visual_{metric}"] = visual_value
            row[f"delta_{metric}"] = visual_value - frozen_value
        for metric in ("ade", "fde", "heading_mae_rad"):
            frozen_value = arm_train["frozen"].get(metric) if arm_train["frozen"] else None
            visual_value = arm_train["visual"].get(metric) if arm_train["visual"] else None
            row[f"frozen_train_{metric}"] = frozen_value
            row[f"visual_train_{metric}"] = visual_value
            row[f"delta_train_{metric}"] = (
                visual_value - frozen_value
                if frozen_value is not None and visual_value is not None
                else None
            )
        rows.append(row)

    columns = list(rows[0])
    from io import StringIO

    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    _atomic_text(root / "paired_summary.csv", buffer.getvalue())

    markdown = [
        "# Action-only visual unfreeze comparison",
        "",
        "Positive ΔPDMS means the trainable-visual arm is better. Negative ΔADE/FDE means lower train-subset trajectory error.",
        "",
        "| Step | Frozen PDMS | Visual PDMS | ΔPDMS | ΔNC | ΔDAC | ΔEP | ΔTTC | Frozen ADE | Visual ADE | ΔADE |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        markdown.append(
            "| {step} | {frozen} | {visual} | {delta} | {nc} | {dac} | {ep} | {ttc} | {fade} | {vade} | {dade} |".format(
                step=row["step"],
                frozen=_fmt(row["frozen_pdms"]),
                visual=_fmt(row["visual_pdms"]),
                delta=_fmt(row["delta_pdms"]),
                nc=_fmt(row["delta_no_at_fault_collisions"]),
                dac=_fmt(row["delta_drivable_area_compliance"]),
                ep=_fmt(row["delta_ego_progress"]),
                ttc=_fmt(row["delta_time_to_collision_within_bound"]),
                fade=_fmt(row["frozen_train_ade"]),
                vade=_fmt(row["visual_train_ade"]),
                dade=_fmt(row["delta_train_ade"]),
            )
        )
    best = max(rows, key=lambda row: row["visual_pdms"])
    markdown.extend(
        [
            "",
            f"Best trainable-visual checkpoint: step {best['step']} with PDMS {_fmt(best['visual_pdms'])}.",
            "",
            "This is a single fixed inference seed comparison; use paired scene-level statistics before claiming significance.",
            "",
        ]
    )
    _atomic_text(root / "REPORT.md", "\n".join(markdown))
    print(root / "paired_summary.csv")
    print(root / "REPORT.md")


def _steps(values: Iterable[str]) -> list[int]:
    steps = [int(value) for value in values]
    if not steps or any(step <= 0 for step in steps) or len(set(steps)) != len(steps):
        raise argparse.ArgumentTypeError("steps must be unique positive integers")
    return steps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    subset = subparsers.add_parser("make-train-subset")
    subset.add_argument("--datalist", required=True)
    subset.add_argument("--data-root", required=True)
    subset.add_argument("--output-dir", required=True)
    subset.add_argument("--size", type=int, required=True)
    subset.add_argument("--seed", type=int, required=True)
    subset.set_defaults(function=make_train_subset)

    score = subparsers.add_parser("score-train")
    score.add_argument("--ground-truth", required=True)
    score.add_argument("--prediction-dir", required=True)
    score.add_argument("--arm", choices=("frozen", "visual"), required=True)
    score.add_argument("--step", type=int, required=True)
    score.add_argument("--output", required=True)
    score.set_defaults(function=score_train)

    manifest = subparsers.add_parser("build-manifest")
    manifest.add_argument("--frozen-run", required=True)
    manifest.add_argument("--visual-run", required=True)
    manifest.add_argument("--steps", nargs="+", type=int, required=True)
    manifest.add_argument("--seed", type=int, required=True)
    manifest.add_argument("--world-size", type=int, required=True)
    manifest.add_argument("--batch-size", type=int, required=True)
    manifest.add_argument("--output", required=True)
    manifest.set_defaults(function=build_manifest)

    report = subparsers.add_parser("summarize")
    report.add_argument("--root", required=True)
    report.add_argument("--steps", nargs="+", type=int, required=True)
    report.set_defaults(function=summarize)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    arguments.function(arguments)

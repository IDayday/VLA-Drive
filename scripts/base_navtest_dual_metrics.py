#!/usr/bin/env python3
"""Contracts and artifact utilities for full-navtest PDMS + EPDMS evaluation.

This module deliberately keeps model inference in the existing DriveVLA/NAVSIM
entrypoint.  It validates the immutable inputs, converts the exact inferred
trajectories for the NAVSIM v2 HumanAgent, and refuses partial metric outputs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import pickle
import shutil
import subprocess
import sys
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


EXPECTED_NAVTEST_SCENES = 12_146
EXPECTED_TRAJECTORY_SHAPE = (8, 3)
EXPECTED_CHECKPOINT_BYTES = 4_271_779_662
EXPECTED_CHECKPOINT_SHA256 = (
    "7f7ce61b9492936d9af15c7bb68dc57d53c6629fccd23ac5ae530058286a807d"
)
EXPECTED_DINO_SHA256 = (
    "dca70548ecd7b03ffba6172c4db403014511b5ee6073f9fca72ba9e6e602a25d"
)
EXPECTED_FLASH_ATTN_VERSION = "2.8.2+v0.1.0.ppu2.1.0.oe"
EXPECTED_NAVSIM_V2_COMMIT = "1482f1da87e31907b549f09836a38f99fd18f200"
EXPECTED_NAVSIM_V2_EVALUATOR_SHA256 = (
    "da0e441b036b0bf07d1360aa46070e22be08d95a2ec2b442ab4c810af011f7ee"
)
EXPECTED_NAVSIM_V2_HUMAN_AGENT_SHA256 = (
    "1acf75c19b1abdac2df16e6e86a9f324ee2153d4cc1a18572fbc0310adc344c1"
)
AGGREGATE_TOKENS = frozenset({"average", "average_all_frames"})


class ContractError(RuntimeError):
    """Raised when an evaluation artifact violates the registered protocol."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_tokens(path: Path, expected_count: int = EXPECTED_NAVTEST_SCENES) -> list[str]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ContractError(
            f"Cannot read navtest token list {path}: {error}"
        ) from error
    if not isinstance(payload, list) or not all(
        isinstance(token, str) and token for token in payload
    ):
        raise ContractError(f"Token list must be a non-empty JSON string list: {path}")
    if len(payload) != expected_count:
        raise ContractError(
            f"Expected {expected_count:,} navtest tokens, found {len(payload):,}: {path}"
        )
    if len(set(payload)) != len(payload):
        raise ContractError(f"Duplicate navtest tokens in {path}")
    return payload


def metadata_files(cache_root: Path) -> list[Path]:
    files = sorted((cache_root / "metadata").glob("*_metadata_node_*.csv"))
    if not files:
        raise ContractError(f"Metric-cache metadata is missing below {cache_root}")
    return files


def resolve_cache_entry(cache_root: Path, recorded: str) -> Path:
    recorded_path = Path(recorded)
    if recorded_path.is_file():
        return recorded_path
    if len(recorded_path.parts) < 4:
        raise ContractError(f"Malformed metric-cache path: {recorded}")
    relocated = cache_root.joinpath(*recorded_path.parts[-4:])
    if not relocated.is_file():
        raise ContractError(
            "Metric-cache entry is unreachable through both its recorded path and "
            f"the supplied cache root: {recorded} -> {relocated}"
        )
    return relocated


def inspect_cache(
    cache_root: Path,
    expected_tokens: set[str],
    *,
    resolve_files: bool,
) -> dict[str, Any]:
    rows: list[tuple[str, str]] = []
    metadata_hash = hashlib.sha256()
    for metadata_path in metadata_files(cache_root):
        metadata_hash.update(metadata_path.name.encode("utf-8"))
        metadata_hash.update(metadata_path.read_bytes())
        with metadata_path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            if not reader.fieldnames or "file_name" not in reader.fieldnames:
                raise ContractError(f"Missing file_name column in {metadata_path}")
            for row in reader:
                recorded = row.get("file_name", "")
                if not recorded:
                    raise ContractError(f"Empty file_name in {metadata_path}")
                token = Path(recorded).parent.name
                resolved = (
                    resolve_cache_entry(cache_root, recorded)
                    if resolve_files
                    else Path(recorded)
                )
                rows.append((token, str(resolved)))
    tokens = [token for token, _ in rows]
    if len(tokens) != len(expected_tokens):
        raise ContractError(
            f"Metric cache {cache_root} has {len(tokens):,} rows; "
            f"expected {len(expected_tokens):,}"
        )
    if len(set(tokens)) != len(tokens):
        raise ContractError(f"Duplicate token rows in metric cache {cache_root}")
    if set(tokens) != expected_tokens:
        missing = sorted(expected_tokens - set(tokens))[:5]
        extra = sorted(set(tokens) - expected_tokens)[:5]
        raise ContractError(
            f"Metric-cache token mismatch for {cache_root}: missing={missing}, extra={extra}"
        )
    logical = hashlib.sha256()
    for token, resolved in sorted(rows):
        logical.update(token.encode("utf-8"))
        logical.update(b"\0")
        logical.update(resolved.encode("utf-8"))
        logical.update(b"\n")
    return {
        "root": str(cache_root.resolve()),
        "rows": len(rows),
        "metadata_sha256": metadata_hash.hexdigest(),
        "logical_sha256": logical.hexdigest(),
    }


def prepare_cache_view(source: Path, target: Path, expected_tokens: set[str]) -> None:
    source_info = inspect_cache(source, expected_tokens, resolve_files=True)
    if target.exists():
        manifest_path = target / "cache_view_manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ContractError(
                f"Existing cache view has no valid manifest: {target}"
            ) from error
        if manifest.get("source_metadata_sha256") != source_info[
            "metadata_sha256"
        ] or manifest.get("tokens") != len(expected_tokens):
            raise ContractError(f"Existing cache view identity mismatch: {target}")
        target_info = inspect_cache(target, expected_tokens, resolve_files=True)
        if target_info["rows"] != source_info["rows"]:
            raise ContractError(f"Existing cache view is invalid: {target}")
        print(target)
        return

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
    if temporary.exists():
        raise ContractError(f"Temporary cache-view path already exists: {temporary}")
    (temporary / "metadata").mkdir(parents=True)
    try:
        for source_metadata in metadata_files(source):
            destination = temporary / "metadata" / source_metadata.name
            with source_metadata.open(newline="", encoding="utf-8") as input_stream:
                reader = csv.DictReader(input_stream)
                assert reader.fieldnames is not None
                rows = []
                for row in reader:
                    row = dict(row)
                    row["file_name"] = str(
                        resolve_cache_entry(source, row["file_name"]).resolve()
                    )
                    rows.append(row)
            with destination.open("w", newline="", encoding="utf-8") as output_stream:
                writer = csv.DictWriter(
                    output_stream, fieldnames=reader.fieldnames, lineterminator="\n"
                )
                writer.writeheader()
                writer.writerows(rows)
        atomic_json(
            temporary / "cache_view_manifest.json",
            {
                "schema_version": "drivevla_metric_cache_view.v1",
                "source_root": str(source.resolve()),
                "source_metadata_sha256": source_info["metadata_sha256"],
                "tokens": len(expected_tokens),
                "policy": "metadata-only relocation; cache payloads remain read-only",
            },
        )
        os.replace(temporary, target)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    inspect_cache(target, expected_tokens, resolve_files=True)
    print(target)


def git_value(root: Path, *arguments: str) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), *arguments], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "UNAVAILABLE"


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    files = {
        "checkpoint": args.checkpoint,
        "dino_weights": args.dino_weights,
        "datalist": args.datalist,
        "v1_evaluator": args.repo_root
        / "navsim/planning/script/run_pdm_score_multi_gpu.py",
        "v2_evaluator": args.navsim_v2_root
        / "navsim/planning/script/run_pdm_score_one_stage.py",
        "v2_human_agent": args.navsim_v2_root / "navsim/agents/human_agent.py",
    }
    for label, path in files.items():
        if not path.is_file():
            raise ContractError(f"Missing {label}: {path}")
    dirs = {
        "vlm_config": args.vlm_config,
        "data_root": args.data_root,
        "maps_root": args.maps_root,
        "pdms_cache": args.pdms_cache,
        "epdms_cache": args.epdms_cache,
        "navsim_v2_root": args.navsim_v2_root,
        "test_logs": args.data_root / "meta_datas/test",
        "test_sensors": args.data_root / "sensor_blobs/test",
    }
    for label, path in dirs.items():
        if not path.is_dir():
            raise ContractError(f"Missing {label}: {path}")

    checkpoint_stat = args.checkpoint.stat()
    if checkpoint_stat.st_size != EXPECTED_CHECKPOINT_BYTES:
        raise ContractError(
            f"Checkpoint size mismatch: {checkpoint_stat.st_size} != "
            f"{EXPECTED_CHECKPOINT_BYTES}"
        )
    checkpoint_age = time.time() - checkpoint_stat.st_mtime
    if checkpoint_age < args.checkpoint_min_age:
        raise ContractError(
            f"Checkpoint is only {checkpoint_age:.1f}s old; "
            f"CHECKPOINT_MIN_AGE={args.checkpoint_min_age}s"
        )

    tokens = load_tokens(args.datalist, args.expected_count)
    expected_tokens = set(tokens)
    checkpoint_sha = sha256_file(args.checkpoint)
    dino_sha = sha256_file(args.dino_weights)
    if checkpoint_sha != EXPECTED_CHECKPOINT_SHA256:
        raise ContractError(f"Checkpoint SHA-256 mismatch: {checkpoint_sha}")
    if dino_sha != EXPECTED_DINO_SHA256:
        raise ContractError(f"DINO SHA-256 mismatch: {dino_sha}")

    try:
        flash_version = version("flash-attn")
    except PackageNotFoundError as error:
        raise ContractError("flash-attn is not installed") from error
    if flash_version != EXPECTED_FLASH_ATTN_VERSION:
        raise ContractError(
            "Installed flash-attn must remain unchanged: "
            f"{flash_version} != {EXPECTED_FLASH_ATTN_VERSION}"
        )

    v2_commit = git_value(args.navsim_v2_root, "rev-parse", "HEAD")
    if v2_commit != EXPECTED_NAVSIM_V2_COMMIT:
        raise ContractError(
            f"NAVSIM v2 commit mismatch: {v2_commit} != {EXPECTED_NAVSIM_V2_COMMIT}"
        )
    v2_evaluator_sha = sha256_file(files["v2_evaluator"])
    v2_human_sha = sha256_file(files["v2_human_agent"])
    if v2_evaluator_sha != EXPECTED_NAVSIM_V2_EVALUATOR_SHA256:
        raise ContractError(f"NAVSIM v2 evaluator SHA-256 mismatch: {v2_evaluator_sha}")
    if v2_human_sha != EXPECTED_NAVSIM_V2_HUMAN_AGENT_SHA256:
        raise ContractError(f"NAVSIM v2 HumanAgent SHA-256 mismatch: {v2_human_sha}")

    source_paths = {
        "dual_metric_helper": Path(__file__).resolve(),
        "dual_metric_entrypoint": args.repo_root
        / "scripts/run_base_navtest_dual_metrics_dlc.sh",
        "base_launcher": args.repo_root / "scripts/run_base_pdms.sh",
        "runtime_contract": args.repo_root / "scripts/verify_runtime_versions.py",
        "ppu_runtime_check": args.repo_root / "scripts/check_ppu_runtime.py",
    }
    for label, path in source_paths.items():
        if not path.is_file():
            raise ContractError(f"Missing launcher source {label}: {path}")
    source_hashes = {label: sha256_file(path) for label, path in source_paths.items()}
    pdms_cache_info = inspect_cache(
        args.pdms_cache, expected_tokens, resolve_files=True
    )
    epdms_cache_info = inspect_cache(
        args.epdms_cache, expected_tokens, resolve_files=True
    )
    datalist_sha = sha256_file(args.datalist)
    v1_evaluator_sha = sha256_file(files["v1_evaluator"])
    identity = {
        "schema_version": "drivevla_full_navtest_pdms_epdms.identity.v1",
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_step": args.checkpoint_step,
        "dino_sha256": dino_sha,
        "datalist_sha256": datalist_sha,
        "pdms_cache_metadata_sha256": pdms_cache_info["metadata_sha256"],
        "epdms_cache_metadata_sha256": epdms_cache_info["metadata_sha256"],
        "v1_evaluator_sha256": v1_evaluator_sha,
        "v2_devkit_commit": v2_commit,
        "v2_evaluator_sha256": v2_evaluator_sha,
        "v2_human_agent_sha256": v2_human_sha,
        "flash_attn": flash_version,
        "repository_commit": git_value(args.repo_root, "rev-parse", "HEAD"),
        "source_hashes": source_hashes,
        "topology": {"RANK": 0, "WORLD_SIZE": 1, "visible_ppus": 1},
    }
    payload: dict[str, Any] = {
        "schema_version": "drivevla_full_navtest_pdms_epdms.v1",
        "identity_sha256": canonical_sha256(identity),
        "identity": identity,
        "created_unix_seconds": int(time.time()),
        "split": "navtest",
        "expected_scenes": args.expected_count,
        "trajectory_sampling": {
            "num_poses": 8,
            "interval_seconds": 0.5,
            "horizon_seconds": 4.0,
        },
        "model": {
            "name": "DriveVLA-M0 Base/no-memory",
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_bytes": checkpoint_stat.st_size,
            "checkpoint_sha256": checkpoint_sha,
            "checkpoint_step": args.checkpoint_step,
            "dino_sha256": dino_sha,
            "vlm_config": str(args.vlm_config.resolve()),
            "flash_attn_installed": flash_version,
            "flash_attn_mutation": "forbidden",
            "use_flash_attn": False,
        },
        "data": {
            "datalist": str(args.datalist.resolve()),
            "datalist_sha256": datalist_sha,
            "data_root": str(args.data_root.resolve()),
            "maps_root": str(args.maps_root.resolve()),
        },
        "metrics": {
            "PDMS": {
                "protocol": "official NAVSIM v1.1",
                "aggregate_token": "average",
                "cache": pdms_cache_info,
            },
            "EPDMS": {
                "protocol": "official NAVSIM v2 one-stage",
                "aggregate_token": "average_all_frames",
                "cache": epdms_cache_info,
                "devkit_commit": v2_commit,
                "evaluator_sha256": v2_evaluator_sha,
                "human_agent_sha256": v2_human_sha,
            },
        },
        "topology": {"RANK": 0, "WORLD_SIZE": 1, "visible_ppus": 1},
        "repository": {
            "root": str(args.repo_root.resolve()),
            "commit": git_value(args.repo_root, "rev-parse", "HEAD"),
            "branch": git_value(args.repo_root, "branch", "--show-current"),
            "status": git_value(args.repo_root, "status", "--short"),
            "v1_evaluator_sha256": v1_evaluator_sha,
            "source_hashes": source_hashes,
        },
        "cache_generation": "disabled",
    }
    if args.json_output:
        if args.json_output.exists():
            try:
                existing = json.loads(args.json_output.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as error:
                raise ContractError(
                    f"Existing protocol manifest is unreadable: {args.json_output}"
                ) from error
            if existing.get("identity_sha256") != payload["identity_sha256"]:
                raise ContractError(
                    "Existing run protocol identity differs from the requested inputs: "
                    f"{args.json_output}"
                )
        else:
            atomic_json(args.json_output, payload)
    return payload


def extract_prediction_mapping(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ContractError("Prediction pickle root must be a dictionary")
    if "predictions" in payload:
        predictions = payload["predictions"]
        if not isinstance(predictions, list) or len(predictions) != 1:
            raise ContractError(
                "Submission pickle must contain exactly one prediction seed"
            )
        mapping = predictions[0]
    else:
        mapping = payload
    if not isinstance(mapping, dict):
        raise ContractError("Prediction payload must map scene token to trajectory")
    return mapping


def trajectory_from_value(value: Any) -> Any:
    if isinstance(value, dict):
        if set(value) != {"trajectory"}:
            raise ContractError(f"Unexpected prediction fields: {sorted(value)}")
        value = value["trajectory"]
    if not hasattr(value, "poses"):
        raise ContractError(
            f"Prediction value has no trajectory poses: {type(value)!r}"
        )
    return value


def validate_mapping(mapping: Mapping[str, Any], tokens: Sequence[str]) -> str:
    expected = set(tokens)
    actual = set(mapping)
    if actual != expected:
        raise ContractError(
            f"Prediction token mismatch: expected={len(expected)}, actual={len(actual)}, "
            f"missing={sorted(expected - actual)[:5]}, extra={sorted(actual - expected)[:5]}"
        )
    logical = hashlib.sha256()
    for token in tokens:
        trajectory = trajectory_from_value(mapping[token])
        poses = np.asarray(trajectory.poses)
        if poses.shape != EXPECTED_TRAJECTORY_SHAPE:
            raise ContractError(
                f"Trajectory shape mismatch for {token}: {poses.shape} != "
                f"{EXPECTED_TRAJECTORY_SHAPE}"
            )
        if poses.dtype != np.float32:
            raise ContractError(
                f"Trajectory dtype mismatch for {token}: {poses.dtype} != float32"
            )
        if not np.isfinite(poses).all():
            raise ContractError(f"Non-finite trajectory for {token}")
        logical.update(token.encode("utf-8"))
        logical.update(b"\0")
        logical.update(np.ascontiguousarray(poses).tobytes())
    return logical.hexdigest()


def load_prediction_pickle(
    path: Path, tokens: Sequence[str]
) -> tuple[dict[str, Any], str]:
    try:
        with path.open("rb") as stream:
            payload = pickle.load(stream)
    except (OSError, pickle.PickleError, AttributeError, EOFError) as error:
        raise ContractError(f"Cannot load prediction pickle {path}: {error}") from error
    mapping = extract_prediction_mapping(payload)
    logical_hash = validate_mapping(mapping, tokens)
    return mapping, logical_hash


def convert_predictions(args: argparse.Namespace) -> None:
    tokens = load_tokens(args.datalist, args.expected_count)
    mapping, logical_hash = load_prediction_pickle(args.pickle, tokens)
    if args.prediction_root.exists():
        manifest = validate_prediction_directory(
            args.prediction_root, tokens, args.checkpoint_sha256
        )
        if manifest["trajectory_logical_sha256"] != logical_hash:
            raise ContractError(
                "Refusing unverifiable prediction artifact; use --overwrite only "
                "at the launcher level to archive the whole run"
            )
        print(args.prediction_root)
        return

    args.prediction_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.prediction_root.with_name(
        f".{args.prediction_root.name}.tmp-{os.getpid()}"
    )
    if temporary.exists():
        raise ContractError(f"Temporary prediction path already exists: {temporary}")
    prediction_dir = temporary / "test"
    prediction_dir.mkdir(parents=True)
    try:
        submission_mapping: dict[str, Any] = {}
        for token in tokens:
            trajectory = trajectory_from_value(mapping[token])
            poses = np.asarray(trajectory.poses)
            np.save(prediction_dir / f"{token}.npy", poses, allow_pickle=False)
            submission_mapping[token] = trajectory
        manifest = {
            "schema_version": "drivevla_navtest_predictions.v1",
            "rank": 0,
            "world_size": 1,
            "split": "test",
            "scenes": len(tokens),
            "trajectory_shape": list(EXPECTED_TRAJECTORY_SHAPE),
            "trajectory_dtype": "float32",
            "trajectory_logical_sha256": logical_hash,
            "source_pickle": str(args.pickle.resolve()),
            "source_pickle_sha256": sha256_file(args.pickle),
            "checkpoint_sha256": args.checkpoint_sha256,
            "checkpoint_step": args.checkpoint_step,
            "datalist_sha256": sha256_file(args.datalist),
        }
        atomic_json(prediction_dir / "inference_manifest.rank0.json", manifest)
        submission_path = temporary / "submission.pkl"
        submission_tmp = submission_path.with_name(
            f".{submission_path.name}.tmp-{os.getpid()}"
        )
        with submission_tmp.open("wb") as stream:
            pickle.dump({"predictions": [submission_mapping]}, stream)
        os.replace(submission_tmp, submission_path)
        os.replace(temporary, args.prediction_root)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    validate_prediction_directory(args.prediction_root, tokens, args.checkpoint_sha256)
    print(args.prediction_root)


def validate_prediction_directory(
    prediction_root: Path, tokens: Sequence[str], checkpoint_sha256: str
) -> dict[str, Any]:
    prediction_dir = prediction_root / "test"
    manifest_path = prediction_dir / "inference_manifest.rank0.json"
    submission_path = prediction_root / "submission.pkl"
    if not prediction_dir.is_dir() or not manifest_path.is_file():
        raise ContractError(f"Incomplete prediction directory: {prediction_root}")
    if not submission_path.is_file():
        raise ContractError(f"Missing v1 submission pickle: {submission_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ContractError(f"Invalid prediction manifest: {manifest_path}") from error
    if (
        manifest.get("schema_version") != "drivevla_navtest_predictions.v1"
        or manifest.get("rank") != 0
        or manifest.get("world_size") != 1
        or manifest.get("scenes") != len(tokens)
        or manifest.get("checkpoint_sha256") != checkpoint_sha256
    ):
        raise ContractError(f"Prediction manifest identity mismatch: {manifest_path}")
    expected = set(tokens)
    actual = {path.stem for path in prediction_dir.glob("*.npy")}
    if actual != expected:
        raise ContractError(
            f"Prediction file mismatch: expected={len(expected)}, actual={len(actual)}"
        )
    logical = hashlib.sha256()
    for token in tokens:
        poses = np.load(prediction_dir / f"{token}.npy", allow_pickle=False)
        if poses.shape != EXPECTED_TRAJECTORY_SHAPE or poses.dtype != np.float32:
            raise ContractError(
                f"Invalid saved trajectory for {token}: shape={poses.shape}, dtype={poses.dtype}"
            )
        if not np.isfinite(poses).all():
            raise ContractError(f"Non-finite saved trajectory for {token}")
        logical.update(token.encode("utf-8"))
        logical.update(b"\0")
        logical.update(np.ascontiguousarray(poses).tobytes())
    if logical.hexdigest() != manifest.get("trajectory_logical_sha256"):
        raise ContractError(f"Prediction logical SHA-256 mismatch: {prediction_root}")
    return manifest


def parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0"}:
        return False
    raise ValueError(value)


def validate_score_csv(
    path: Path, aggregate_token: str, expected_tokens: set[str]
) -> dict[str, Any]:
    try:
        with path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
    except (OSError, csv.Error) as error:
        raise ContractError(f"Cannot read score CSV {path}: {error}") from error
    aggregate_rows = [row for row in rows if row.get("token") == aggregate_token]
    if len(aggregate_rows) != 1:
        raise ContractError(
            f"Expected one {aggregate_token!r} row in {path}, found {len(aggregate_rows)}"
        )
    scenario_rows = [row for row in rows if row.get("token") not in AGGREGATE_TOKENS]
    scenario_tokens = {row.get("token", "") for row in scenario_rows}
    if len(scenario_rows) != len(expected_tokens) or scenario_tokens != expected_tokens:
        raise ContractError(
            f"Score token mismatch in {path}: rows={len(scenario_rows)}, "
            f"unique={len(scenario_tokens)}, expected={len(expected_tokens)}"
        )
    failed = 0
    for row in scenario_rows:
        try:
            valid = parse_bool(row.get("valid", ""))
            score = float(row.get("score", "nan"))
        except ValueError as error:
            raise ContractError(f"Malformed score row in {path}: {row}") from error
        if not valid:
            failed += 1
        if not math.isfinite(score):
            raise ContractError(
                f"Non-finite scenario score in {path}: {row.get('token')}"
            )
    aggregate = aggregate_rows[0]
    try:
        aggregate_valid = parse_bool(aggregate.get("valid", ""))
        aggregate_score = float(aggregate.get("score", "nan"))
    except ValueError as error:
        raise ContractError(f"Malformed aggregate score in {path}") from error
    if failed or not aggregate_valid:
        raise ContractError(
            f"Metric evaluation is incomplete in {path}: failed_scenarios={failed}"
        )
    if not math.isfinite(aggregate_score):
        raise ContractError(f"Non-finite aggregate score in {path}")
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "aggregate_token": aggregate_token,
        "score": aggregate_score,
        "scenarios": len(scenario_rows),
        "failed": failed,
        "aggregate": aggregate,
    }


def find_valid_pickle(root: Path, tokens: Sequence[str]) -> Path:
    candidates = sorted(
        root.rglob("*.pkl"), key=lambda path: path.stat().st_mtime_ns, reverse=True
    )
    errors: list[str] = []
    for path in candidates:
        if path.name == "submission.pkl":
            continue
        try:
            load_prediction_pickle(path, tokens)
            return path
        except ContractError as error:
            errors.append(f"{path}: {error}")
    detail = f"; last_error={errors[0]}" if errors else ""
    raise ContractError(f"No complete prediction pickle below {root}{detail}")


def find_valid_score(
    root: Path, aggregate_token: str, expected_tokens: set[str]
) -> Path:
    candidates = sorted(
        root.rglob("*.csv"), key=lambda path: path.stat().st_mtime_ns, reverse=True
    )
    for path in candidates:
        try:
            validate_score_csv(path, aggregate_token, expected_tokens)
            return path
        except ContractError:
            continue
    raise ContractError(f"No complete {aggregate_token} score CSV below {root}")


def atomic_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    shutil.copyfile(source, temporary)
    os.replace(temporary, destination)


def summarize(args: argparse.Namespace) -> None:
    expected = set(load_tokens(args.datalist, args.expected_count))
    pdms = validate_score_csv(args.pdms_csv, "average", expected)
    epdms = validate_score_csv(args.epdms_csv, "average_all_frames", expected)
    payload = {
        "schema_version": "drivevla_full_navtest_pdms_epdms_summary.v1",
        "run_id": args.run_id,
        "status": "PASS",
        "scenarios": len(expected),
        "PDMS": {
            "score": pdms["score"],
            "failed_scenarios": pdms["failed"],
            "csv": pdms["path"],
            "csv_sha256": pdms["sha256"],
            "aggregate_token": "average",
        },
        "EPDMS": {
            "score": epdms["score"],
            "failed_scenarios": epdms["failed"],
            "csv": epdms["path"],
            "csv_sha256": epdms["sha256"],
            "aggregate_token": "average_all_frames",
        },
    }
    atomic_json(args.output_root / "summary.json", payload)
    summary_path = args.output_root / "summary.csv"
    temporary = summary_path.with_name(f".{summary_path.name}.tmp-{os.getpid()}")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "run_id",
                "scenarios",
                "pdms",
                "epdms",
                "pdms_failed",
                "epdms_failed",
                "pdms_csv",
                "epdms_csv",
            ),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerow(
            {
                "run_id": args.run_id,
                "scenarios": len(expected),
                "pdms": f"{pdms['score']:.12g}",
                "epdms": f"{epdms['score']:.12g}",
                "pdms_failed": pdms["failed"],
                "epdms_failed": epdms["failed"],
                "pdms_csv": pdms["path"],
                "epdms_csv": epdms["path"],
            }
        )
    os.replace(temporary, summary_path)
    print(json.dumps(payload, sort_keys=True))


def add_common_count(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--expected-count", type=int, default=EXPECTED_NAVTEST_SCENES)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    preflight_parser = subparsers.add_parser("preflight")
    preflight_parser.add_argument("--repo-root", type=Path, required=True)
    preflight_parser.add_argument("--checkpoint", type=Path, required=True)
    preflight_parser.add_argument("--checkpoint-step", required=True)
    preflight_parser.add_argument("--checkpoint-min-age", type=int, default=120)
    preflight_parser.add_argument("--vlm-config", type=Path, required=True)
    preflight_parser.add_argument("--dino-weights", type=Path, required=True)
    preflight_parser.add_argument("--datalist", type=Path, required=True)
    preflight_parser.add_argument("--data-root", type=Path, required=True)
    preflight_parser.add_argument("--maps-root", type=Path, required=True)
    preflight_parser.add_argument("--pdms-cache", type=Path, required=True)
    preflight_parser.add_argument("--epdms-cache", type=Path, required=True)
    preflight_parser.add_argument("--navsim-v2-root", type=Path, required=True)
    preflight_parser.add_argument("--json-output", type=Path)
    add_common_count(preflight_parser)

    cache_parser = subparsers.add_parser("prepare-cache-view")
    cache_parser.add_argument("--source", type=Path, required=True)
    cache_parser.add_argument("--target", type=Path, required=True)
    cache_parser.add_argument("--datalist", type=Path, required=True)
    add_common_count(cache_parser)

    convert_parser = subparsers.add_parser("convert")
    convert_parser.add_argument("--pickle", type=Path, required=True)
    convert_parser.add_argument("--prediction-root", type=Path, required=True)
    convert_parser.add_argument("--datalist", type=Path, required=True)
    convert_parser.add_argument("--checkpoint-sha256", required=True)
    convert_parser.add_argument("--checkpoint-step", required=True)
    add_common_count(convert_parser)

    validate_predictions_parser = subparsers.add_parser("validate-predictions")
    validate_predictions_parser.add_argument(
        "--prediction-root", type=Path, required=True
    )
    validate_predictions_parser.add_argument("--datalist", type=Path, required=True)
    validate_predictions_parser.add_argument("--checkpoint-sha256", required=True)
    add_common_count(validate_predictions_parser)

    find_pickle_parser = subparsers.add_parser("find-pickle")
    find_pickle_parser.add_argument("--root", type=Path, required=True)
    find_pickle_parser.add_argument("--datalist", type=Path, required=True)
    add_common_count(find_pickle_parser)

    find_score_parser = subparsers.add_parser("find-score")
    find_score_parser.add_argument("--root", type=Path, required=True)
    find_score_parser.add_argument("--datalist", type=Path, required=True)
    find_score_parser.add_argument(
        "--aggregate-token", choices=sorted(AGGREGATE_TOKENS), required=True
    )
    add_common_count(find_score_parser)

    validate_score_parser = subparsers.add_parser("validate-score")
    validate_score_parser.add_argument("--csv", type=Path, required=True)
    validate_score_parser.add_argument("--datalist", type=Path, required=True)
    validate_score_parser.add_argument(
        "--aggregate-token", choices=sorted(AGGREGATE_TOKENS), required=True
    )
    add_common_count(validate_score_parser)

    copy_parser = subparsers.add_parser("atomic-copy")
    copy_parser.add_argument("--source", type=Path, required=True)
    copy_parser.add_argument("--destination", type=Path, required=True)

    summary_parser = subparsers.add_parser("summarize")
    summary_parser.add_argument("--run-id", required=True)
    summary_parser.add_argument("--datalist", type=Path, required=True)
    summary_parser.add_argument("--pdms-csv", type=Path, required=True)
    summary_parser.add_argument("--epdms-csv", type=Path, required=True)
    summary_parser.add_argument("--output-root", type=Path, required=True)
    add_common_count(summary_parser)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "preflight":
            payload = preflight(args)
            print(json.dumps(payload, sort_keys=True))
        elif args.command == "prepare-cache-view":
            tokens = set(load_tokens(args.datalist, args.expected_count))
            prepare_cache_view(args.source, args.target, tokens)
        elif args.command == "convert":
            convert_predictions(args)
        elif args.command == "validate-predictions":
            tokens = load_tokens(args.datalist, args.expected_count)
            manifest = validate_prediction_directory(
                args.prediction_root, tokens, args.checkpoint_sha256
            )
            print(json.dumps(manifest, sort_keys=True))
        elif args.command == "find-pickle":
            tokens = load_tokens(args.datalist, args.expected_count)
            print(find_valid_pickle(args.root, tokens))
        elif args.command == "find-score":
            tokens = set(load_tokens(args.datalist, args.expected_count))
            print(find_valid_score(args.root, args.aggregate_token, tokens))
        elif args.command == "validate-score":
            tokens = set(load_tokens(args.datalist, args.expected_count))
            payload = validate_score_csv(args.csv, args.aggregate_token, tokens)
            payload.pop("aggregate", None)
            print(json.dumps(payload, sort_keys=True))
        elif args.command == "atomic-copy":
            atomic_copy(args.source, args.destination)
            print(args.destination)
        elif args.command == "summarize":
            summarize(args)
        else:  # pragma: no cover - argparse enforces command choices
            raise AssertionError(args.command)
    except ContractError as error:
        print(f"CONTRACT ERROR: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

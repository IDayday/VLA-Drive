#!/usr/bin/env python3
"""Non-mutating validation for the local DriveVLA-M0 Base/no-memory route."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from verify_runtime_versions import EXPECTED_RUNTIME


EXPECTED_CHECKPOINT_BYTES = 4_271_779_662
EXPECTED_CHECKPOINT_SHA256 = (
    "7f7ce61b9492936d9af15c7bb68dc57d53c6629fccd23ac5ae530058286a807d"
)
EXPECTED_DINO_SHA256 = (
    "dca70548ecd7b03ffba6172c4db403014511b5ee6073f9fca72ba9e6e602a25d"
)
EXPECTED_TOKENIZER_SIZE = 151_682
EXPECTED_CACHE_ROWS = 12_146
EXPECTED_LOG_FILES = 147
EXPECTED_SENSOR_LOGS = 147


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--full-hash",
        action="store_true",
        help="recompute checkpoint and DINO SHA-256 values",
    )
    parser.add_argument(
        "--allow-no-accelerator",
        action="store_true",
        help="validate code/assets even when the PPU device is not mounted",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        help="optional path for a machine-readable preflight manifest",
    )
    args = parser.parse_args()

    failures: list[str] = []
    warnings: list[str] = []
    checks: dict[str, Any] = {}

    def report(label: str, value: Any) -> None:
        checks[label] = value
        print(f"[OK] {label}: {value}")

    def warn(message: str) -> None:
        warnings.append(message)
        print(f"[WARN] {message}")

    def fail(message: str) -> None:
        failures.append(message)
        print(f"[FAIL] {message}")

    required_modules = (
        "torch",
        "torchvision",
        "pytorch_lightning",
        "transformers",
        "peft",
        "accelerate",
        "safetensors",
        "timm",
        "hydra",
        "omegaconf",
        "ray",
        "geopandas",
        "rasterio",
        "rtree",
    )
    missing_modules = [
        name for name in required_modules if importlib.util.find_spec(name) is None
    ]
    if missing_modules:
        fail(f"missing Python modules: {', '.join(missing_modules)}")
    else:
        report("python_modules", len(required_modules))

    from importlib.metadata import PackageNotFoundError, version

    runtime_versions: dict[str, str] = {}
    for package, expected in EXPECTED_RUNTIME.items():
        try:
            actual = version(package)
        except PackageNotFoundError:
            fail(f"runtime package missing: {package} (expected {expected})")
            continue
        runtime_versions[package] = actual
        if actual != expected:
            fail(f"runtime mismatch for {package}: found {actual}, expected {expected}")
    if len(runtime_versions) == len(EXPECTED_RUNTIME) and not any(
        value != EXPECTED_RUNTIME[name] for name, value in runtime_versions.items()
    ):
        report("vendor_runtime", runtime_versions)

    env_paths = {
        "DRIVEVLA_BASE_CHECKPOINT": "file",
        "DRIVEVLA_VLM_CONFIG": "dir",
        "DRIVEVLA_DINO_WEIGHTS": "file",
        "OPENSCENE_DATA_ROOT": "dir",
        "NUPLAN_MAPS_ROOT": "dir",
        "METRIC_CACHE_PATH": "dir",
        "NAVSIM_EXP_ROOT": "output",
    }
    paths: dict[str, Path] = {}
    for variable, kind in env_paths.items():
        raw = os.getenv(variable)
        if not raw:
            fail(f"environment variable is unset: {variable}")
            continue
        path = Path(raw).expanduser()
        paths[variable] = path
        if kind == "file":
            exists = path.is_file()
        elif kind == "dir":
            exists = path.is_dir()
        else:
            exists = path.is_dir() or path.parent.is_dir()
        if exists:
            report(variable.lower(), str(path))
        else:
            fail(f"invalid {kind} path in {variable}: {path}")

    checkpoint = paths.get("DRIVEVLA_BASE_CHECKPOINT")
    if checkpoint and checkpoint.is_file():
        size = checkpoint.stat().st_size
        if size == EXPECTED_CHECKPOINT_BYTES:
            report("checkpoint_bytes", size)
        else:
            fail(
                f"checkpoint size is {size:,}, expected "
                f"{EXPECTED_CHECKPOINT_BYTES:,} bytes"
            )
        if args.full_hash:
            actual_hash = sha256(checkpoint)
            if actual_hash == EXPECTED_CHECKPOINT_SHA256:
                report("checkpoint_sha256", actual_hash)
            else:
                fail(f"checkpoint SHA-256 mismatch: {actual_hash}")
        else:
            report("checkpoint_sha256_expected", EXPECTED_CHECKPOINT_SHA256)

    dino = paths.get("DRIVEVLA_DINO_WEIGHTS")
    if dino and dino.is_file():
        if args.full_hash:
            actual_hash = sha256(dino)
            if actual_hash == EXPECTED_DINO_SHA256:
                report("dino_sha256", actual_hash)
            else:
                fail(f"DINO SHA-256 mismatch: {actual_hash}")
        else:
            report("dino_sha256_expected", EXPECTED_DINO_SHA256)

    vlm_path = paths.get("DRIVEVLA_VLM_CONFIG")
    if vlm_path and vlm_path.is_dir() and not missing_modules:
        required_vlm_files = (
            "config.json",
            "configuration_intern_vit.py",
            "configuration_internvl_chat.py",
            "modeling_intern_vit.py",
            "modeling_internvl_chat.py",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "added_tokens.json",
            "vocab.json",
            "merges.txt",
        )
        missing = [name for name in required_vlm_files if not (vlm_path / name).is_file()]
        if missing:
            fail(f"InternVL config/tokenizer files missing: {', '.join(missing)}")
        else:
            from transformers import AutoConfig, AutoTokenizer

            config = AutoConfig.from_pretrained(vlm_path, trust_remote_code=True)
            tokenizer = AutoTokenizer.from_pretrained(
                vlm_path, trust_remote_code=True, use_fast=False
            )
            tokenizer.add_special_tokens(
                {
                    "additional_special_tokens": [
                        f"<DRIVEVLA_EXTRA_{index}>" for index in range(8)
                    ]
                }
            )
            if len(tokenizer) == EXPECTED_TOKENIZER_SIZE:
                report(
                    "internvl",
                    {
                        "hidden_size": int(config.llm_config.hidden_size),
                        "expanded_vocab": len(tokenizer),
                    },
                )
            else:
                fail(
                    f"expanded tokenizer size is {len(tokenizer)}, "
                    f"expected {EXPECTED_TOKENIZER_SIZE}"
                )

    data_root = paths.get("OPENSCENE_DATA_ROOT")
    if data_root and data_root.is_dir():
        logs = data_root / "meta_datas" / "test"
        sensors = data_root / "sensor_blobs" / "test"
        log_count = (
            sum(1 for item in logs.iterdir() if item.suffix == ".pkl")
            if logs.is_dir()
            else 0
        )
        sensor_count = (
            sum(1 for item in sensors.iterdir() if item.is_dir())
            if sensors.is_dir()
            else 0
        )
        if log_count == EXPECTED_LOG_FILES and sensor_count == EXPECTED_SENSOR_LOGS:
            report(
                "navsim_test_data",
                {"log_files": log_count, "sensor_log_directories": sensor_count},
            )
        else:
            fail(
                "NAVSIM test data mismatch: "
                f"found {log_count} logs/{sensor_count} sensor directories; "
                f"expected {EXPECTED_LOG_FILES}/{EXPECTED_SENSOR_LOGS}"
            )

    map_root = paths.get("NUPLAN_MAPS_ROOT")
    if map_root and map_root.is_dir():
        map_version = os.getenv("NUPLAN_MAP_VERSION", "nuplan-maps-v1.0")
        manifest = map_root / f"{map_version}.json"
        if manifest.is_file():
            report("nuplan_map_manifest", manifest.name)
        else:
            fail(f"nuPlan map manifest missing: {manifest}")

    metric_cache = paths.get("METRIC_CACHE_PATH")
    if metric_cache and metric_cache.is_dir():
        metadata_files = sorted((metric_cache / "metadata").glob("*_metadata_node_*.csv"))
        rows = 0
        for metadata_file in metadata_files:
            with metadata_file.open(newline="") as stream:
                rows += max(sum(1 for _ in csv.reader(stream)) - 1, 0)
        if rows == EXPECTED_CACHE_ROWS:
            report("navtest_metric_cache_scenarios", rows)
        elif rows:
            fail(f"metric cache contains {rows:,} rows, expected {EXPECTED_CACHE_ROWS:,}")
        else:
            fail(f"metric cache metadata missing below: {metric_cache}")

    repo_root = Path(os.getenv("DRIVEVLA_REPO_ROOT", Path(__file__).resolve().parents[1]))
    config_path = repo_root / "configs" / "base_model_navtest.yaml"
    if config_path.is_file() and not missing_modules:
        from omegaconf import OmegaConf

        config = OmegaConf.load(config_path)
        signature = {
            "target": str(config._target_),
            "vlm_type": str(config.vlm_config.vlm_type),
            "target_vocab_size": int(config.vlm_config.target_vocab_size),
            "initialize_from_config": bool(config.vlm_config.initialize_from_config),
            "use_flash_attn": bool(config.vlm_config.use_flash_attn),
            "proposal_num": int(config.action_head_config.proposal_num),
            "scorer_ref_num": int(config.action_head_config.scorer_ref_num),
            "num_poses": int(config.action_head_config.num_poses),
        }
        expected = {
            "target": "navsim.agents.EpisodeDrive.episodedrive_agent.EpisodeDriveAgent",
            "vlm_type": "internvl",
            "target_vocab_size": EXPECTED_TOKENIZER_SIZE,
            "initialize_from_config": True,
            "use_flash_attn": False,
            "proposal_num": 64,
            "scorer_ref_num": 4,
            "num_poses": 8,
        }
        if signature == expected:
            report("base_no_memory_topology", signature)
        else:
            fail(f"unexpected Base topology: {signature}")

    if not missing_modules:
        import torch

        if torch.cuda.is_available():
            properties = torch.cuda.get_device_properties(0)
            report(
                "accelerator",
                {
                    "name": properties.name,
                    "memory_gib": round(properties.total_memory / 1024**3, 1),
                    "device_count": torch.cuda.device_count(),
                },
            )
        elif args.allow_no_accelerator:
            warn(
                "torch.cuda.is_available() is false; code/assets were checked but "
                "the PPU execution path was not validated in this process"
            )
        else:
            fail(
                "torch.cuda.is_available() is false. The PPU device/driver is not "
                "visible; do not replace the vendor PyTorch build."
            )

        if checkpoint and checkpoint.is_file():
            payload = torch.load(
                checkpoint, map_location="cpu", mmap=True, weights_only=False
            )
            state_dict = payload.get("state_dict", {})
            embedding_key = (
                "agent.backbone.base_model.model.model.language_model."
                "model.embed_tokens.weight"
            )
            embedding = state_dict.get(embedding_key)
            forbidden_checkpoint_fragments = (
                "memory_attention",
                "retrieve_model",
                "retrieval",
                "memory_pool",
            )
            memory_keys = [
                key
                for key in state_dict
                if any(
                    fragment in key.lower()
                    for fragment in forbidden_checkpoint_fragments
                )
            ]
            if (
                len(state_dict) == 1323
                and embedding is not None
                and tuple(embedding.shape) == (EXPECTED_TOKENIZER_SIZE, 1536)
                and not memory_keys
            ):
                report(
                    "checkpoint_structure",
                    {
                        "tensor_count": len(state_dict),
                        "embedding_shape": list(embedding.shape),
                        "retrieval_or_memory_key_count": len(memory_keys),
                    },
                )
            else:
                fail(
                    "checkpoint state_dict does not match the merged Base/no-memory "
                    f"model (retrieval/memory keys: {memory_keys[:5]})"
                )

    status = "PASS" if not failures else "FAIL"
    manifest = {
        "status": status,
        "checks": checks,
        "warnings": warnings,
        "failures": failures,
    }
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"[OK] json_output: {args.json_output}")

    if failures:
        print(f"\nPreflight failed with {len(failures)} issue(s).", file=sys.stderr)
        return 1
    print("\nDriveVLA-M0 Base/no-memory preflight: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

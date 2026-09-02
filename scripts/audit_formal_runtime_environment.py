#!/usr/bin/env python3
"""Audit the exact runtime used by formal PlanReg-WM launchers.

The two training nodes have host-local ``/root`` environments.  A path whose
last component is shared can therefore still resolve to different Python ABIs
or package versions on each host.  Formal multi-node runs compare this
fingerprint before constructing the model and fail on any difference.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys
from typing import Any, Dict


CORE_FILES = (
    "navsim/agents/EpisodeDrive/drivevla_backbone.py",
    "navsim/agents/EpisodeDrive/drivevla_base_agent.py",
    "navsim/agents/EpisodeDrive/action_decoder.py",
    "navsim/planning/script/run_training_full.py",
)
RUNTIME_ENVIRONMENT_KEYS = (
    "HF_HOME",
    "TRANSFORMERS_OFFLINE",
    "HF_HUB_OFFLINE",
    "PYTHONNOUSERSITE",
    "PLANREG_FORMAL_VLM_PATH",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _module_record(name: str) -> Dict[str, str]:
    module = importlib.import_module(name)
    path = Path(module.__file__).resolve()
    return {
        "version": str(getattr(module, "__version__", "unknown")),
        "module_path": str(path),
    }


def build_fingerprint(repo_root: Path) -> Dict[str, Any]:
    import peft
    import pytorch_lightning
    import torch
    import transformers

    episode_drive_spec = importlib.util.find_spec(
        "navsim.agents.EpisodeDrive.drivevla_base_agent"
    )
    if episode_drive_spec is None or episode_drive_spec.origin is None:
        raise RuntimeError("Could not resolve the formal EpisodeDrive module")
    episode_drive_path = Path(episode_drive_spec.origin).resolve()
    try:
        episode_drive_path.relative_to(repo_root)
    except ValueError as exc:
        raise RuntimeError(
            "Formal runtime imported EpisodeDrive outside the isolated worktree: "
            f"{episode_drive_path} (expected under {repo_root})"
        ) from exc

    vlm_path_value = os.environ.get("PLANREG_FORMAL_VLM_PATH", "")
    vlm_source_sha256: Dict[str, str] = {}
    if vlm_path_value:
        vlm_path = Path(vlm_path_value).expanduser().resolve()
        if not vlm_path.is_dir():
            raise RuntimeError(
                f"PLANREG_FORMAL_VLM_PATH is not a directory: {vlm_path}"
            )
        vlm_source_sha256 = {
            str(path.relative_to(vlm_path)): _sha256(path)
            for path in sorted(vlm_path.rglob("*.py"))
        }
        if not vlm_source_sha256:
            raise RuntimeError(
                f"Formal VLM has no trust_remote_code Python sources: {vlm_path}"
            )

    return {
        "schema_version": 2,
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "abi": getattr(sys, "abiflags", ""),
            "executable": str(Path(sys.executable).resolve()),
        },
        "packages": {
            "torch": _module_record("torch"),
            "transformers": _module_record("transformers"),
            "peft": _module_record("peft"),
            "pytorch_lightning": _module_record("pytorch_lightning"),
        },
        "cuda": {
            "torch_cuda_version": str(torch.version.cuda),
            "nccl_version": list(torch.cuda.nccl.version()),
        },
        "episode_drive_module": str(episode_drive_path),
        "core_file_sha256": {
            relative: _sha256(repo_root / relative) for relative in CORE_FILES
        },
        "runtime_environment": {
            key: os.environ.get(key, "") for key in RUNTIME_ENVIRONMENT_KEYS
        },
        "vlm_python_source_sha256": vlm_source_sha256,
        "declared_versions": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "peft": peft.__version__,
            "pytorch_lightning": pytorch_lightning.__version__,
        },
    }


def _write(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def compare_fingerprints(left: Path, right: Path) -> Dict[str, Any]:
    left_payload = json.loads(left.read_text(encoding="utf-8"))
    right_payload = json.loads(right.read_text(encoding="utf-8"))
    equal = left_payload == right_payload
    result = {
        "schema_version": 2,
        "equal": equal,
        "left": str(left.resolve()),
        "right": str(right.resolve()),
        "left_sha256": _sha256(left),
        "right_sha256": _sha256(right),
    }
    if not equal:
        differing = sorted(
            key
            for key in set(left_payload) | set(right_payload)
            if left_payload.get(key) != right_payload.get(key)
        )
        result["differing_top_level_fields"] = differing
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compare", nargs=2, type=Path, metavar=("LEFT", "RIGHT"))
    args = parser.parse_args()

    if args.compare:
        payload = compare_fingerprints(*args.compare)
        _write(args.output, payload)
        print(json.dumps(payload, sort_keys=True))
        if not payload["equal"]:
            raise SystemExit(2)
        return

    if args.repo_root is None:
        parser.error("--repo-root is required unless --compare is used")
    payload = build_fingerprint(args.repo_root.expanduser().resolve())
    _write(args.output, payload)
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()

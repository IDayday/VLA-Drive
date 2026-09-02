#!/usr/bin/env python3
"""Export a deployment-only PlanReg student checkpoint with provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Dict, Mapping, Optional

import torch


TRAINING_ONLY_STATE_PREFIXES = (
    "ema_register_target.",
    "future_register_predictor.",
)
TRAINING_ONLY_STATE_NAMES = {
    "_ema_optimizer_step",
    "_world_model_optimizer_step",
    "_world_model_total_optimizer_steps",
}
TRAINING_ONLY_TOP_LEVEL_KEYS = {
    "optimizer_states",
    "lr_schedulers",
    "callbacks",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalized_state_key(key: str) -> str:
    return key[len("agent.") :] if key.startswith("agent.") else key


def is_training_only_state_key(key: str) -> bool:
    normalized = _normalized_state_key(key)
    return normalized in TRAINING_ONLY_STATE_NAMES or normalized.startswith(
        TRAINING_ONLY_STATE_PREFIXES
    )


def verify_student_checkpoint_payload(checkpoint: Mapping[str, Any]) -> None:
    state_dict = checkpoint.get("state_dict", checkpoint)
    forbidden_state = sorted(
        key for key in state_dict if is_training_only_state_key(str(key))
    )
    forbidden_top = sorted(
        key for key in TRAINING_ONLY_TOP_LEVEL_KEYS if key in checkpoint
    )
    if forbidden_state or forbidden_top:
        raise RuntimeError(
            "Checkpoint is not student-only: "
            f"training_state_keys={forbidden_state[:16]}, "
            f"training_top_level={forbidden_top}"
        )


def _to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    try:
        from omegaconf import OmegaConf

        if OmegaConf.is_config(value):
            return OmegaConf.to_container(value, resolve=True)
    except ImportError:
        pass
    return str(value)


def _architecture_config(
    checkpoint: Mapping[str, Any],
    resolved_config: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    if resolved_config is not None:
        candidate = resolved_config.get("agent", resolved_config)
        return _to_jsonable(candidate)
    hyper_parameters = checkpoint.get("hyper_parameters", {})
    for key in ("cfg", "config", "agent"):
        candidate = hyper_parameters.get(key) if isinstance(hyper_parameters, Mapping) else None
        if candidate is not None:
            if isinstance(candidate, Mapping) and "agent" in candidate:
                candidate = candidate["agent"]
            return _to_jsonable(candidate)
    return {"available": False, "reason": "source checkpoint has no resolved config"}


def _current_git_commit(repo_root: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def export_student_checkpoint(
    source_path: Path,
    output_path: Path,
    *,
    resolved_config: Optional[Mapping[str, Any]] = None,
    source_git_commit: Optional[str] = None,
) -> Dict[str, Any]:
    source_path = Path(source_path).resolve()
    output_path = Path(output_path).resolve()
    if source_path == output_path:
        raise ValueError("Student export output must differ from source checkpoint")
    checkpoint = torch.load(source_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise TypeError("Training checkpoint must contain a mapping")
    # Shallow-copy the checkpoint container; retained tensors are immutable
    # during export, so a second multi-gigabyte model copy is unnecessary.
    exported = dict(checkpoint)
    source_state = checkpoint.get("state_dict", checkpoint)
    if not isinstance(source_state, Mapping):
        raise TypeError("checkpoint['state_dict'] must be a mapping")
    retained_state = {
        key: value
        for key, value in source_state.items()
        if not is_training_only_state_key(str(key))
    }
    removed_state_keys = sorted(set(source_state) - set(retained_state))
    if "state_dict" in checkpoint:
        exported["state_dict"] = retained_state
    else:
        exported = retained_state
    removed_top_level = []
    if isinstance(exported, dict):
        for key in TRAINING_ONLY_TOP_LEVEL_KEYS:
            if key in exported:
                removed_top_level.append(key)
                del exported[key]
    verify_student_checkpoint_payload(exported)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    torch.save(exported, temporary_path)
    temporary_path.replace(output_path)

    repo_root = Path(__file__).resolve().parents[1]
    manifest = {
        "format": "planreg_student_checkpoint_v1",
        "source_checkpoint": str(source_path),
        "export_checkpoint": str(output_path),
        "source_checkpoint_sha256": sha256_file(source_path),
        "export_checkpoint_sha256": sha256_file(output_path),
        "removed_key_count": len(removed_state_keys),
        "retained_key_count": len(retained_state),
        "removed_state_keys": removed_state_keys,
        "removed_top_level_keys": sorted(removed_top_level),
        "source_git_commit": source_git_commit or _current_git_commit(repo_root),
        "resolved_architecture_config": _architecture_config(
            checkpoint, resolved_config
        ),
    }
    manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def _load_resolved_config(path: Optional[Path]):
    if path is None:
        return None
    from omegaconf import OmegaConf

    return OmegaConf.to_container(OmegaConf.load(path), resolve=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Training or student checkpoint")
    parser.add_argument("output", nargs="?", type=Path, help="Student output checkpoint")
    parser.add_argument("--resolved-config", type=Path)
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Only verify that SOURCE contains no training-only state",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.verify:
        if args.output is not None:
            raise SystemExit("--verify accepts SOURCE only")
        checkpoint = torch.load(
            args.source, map_location="cpu", weights_only=False
        )
        verify_student_checkpoint_payload(checkpoint)
        print(
            "PLANREG_STUDENT_CHECKPOINT_OK "
            f"path={args.source.resolve()} sha256={sha256_file(args.source)}"
        )
        return
    if args.output is None:
        raise SystemExit("OUTPUT is required unless --verify is used")
    manifest = export_student_checkpoint(
        args.source,
        args.output,
        resolved_config=_load_resolved_config(args.resolved_config),
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

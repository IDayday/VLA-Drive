"""Materialize evaluator-facing trajectories from Field2Plan diagnostics."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Dict

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from starVLA.model.modules.field2plan.trajectory_codec import (
    TrajectoryCodec,
    TrajectoryStats,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_save_npy(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as handle:
        np.save(handle, value)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def materialize_predictions(
    diagnostics_dir: Path,
    output_dir: Path,
    datalist_path: Path,
    key: str = "draft_action",
) -> Dict[str, object]:
    """Decode ``[8,4]`` diagnostics into physical ``[8,3]`` predictions."""

    diagnostics_dir = Path(diagnostics_dir).resolve()
    output_dir = Path(output_dir).resolve()
    datalist_path = Path(datalist_path).resolve()
    if key not in {"draft_action", "final_action"}:
        raise ValueError("key must be draft_action or final_action")
    tokens = json.loads(datalist_path.read_text(encoding="utf-8"))
    if not isinstance(tokens, list) or not all(isinstance(token, str) for token in tokens):
        raise ValueError("datalist must be a JSON list of token strings")
    missing = [
        token
        for token in tokens
        if not (diagnostics_dir / f"{token}.npz").is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"missing {len(missing)} diagnostic entries; first={missing[0]}"
        )

    codec = TrajectoryCodec()
    for token in tokens:
        diagnostic_path = diagnostics_dir / f"{token}.npz"
        with np.load(diagnostic_path, allow_pickle=False) as archive:
            if key not in archive.files:
                raise KeyError(f"{diagnostic_path} lacks {key}")
            normalized = np.asarray(archive[key], dtype=np.float32)
        if normalized.shape != (8, 4):
            raise ValueError(
                f"{diagnostic_path}:{key} must have shape [8,4], "
                f"got {normalized.shape}"
            )
        physical = np.asarray(codec.decode_action(normalized), dtype=np.float32)
        if physical.shape != (8, 3) or not np.isfinite(physical).all():
            raise ValueError(f"non-finite or invalid trajectory for token={token}")
        _atomic_save_npy(output_dir / f"{token}.npy", physical)

    inference_manifests = []
    for manifest in sorted(diagnostics_dir.parent.glob("inference_manifest.rank*.json")):
        inference_manifests.append(
            {"path": os.fspath(manifest.resolve()), "sha256": _sha256(manifest)}
        )
    stats = TrajectoryStats()
    summary: Dict[str, object] = {
        "schema_version": 1,
        "diagnostic_key": key,
        "diagnostics_dir": os.fspath(diagnostics_dir),
        "output_dir": os.fspath(output_dir),
        "datalist_path": os.fspath(datalist_path),
        "datalist_sha256": _sha256(datalist_path),
        "entry_count": len(tokens),
        "trajectory_shape": [8, 3],
        "normalization": {
            "x_mean": stats.x_mean,
            "x_std": stats.x_std,
            "y_mean": stats.y_mean,
            "y_std": stats.y_std,
            "heading": "atan2(sin,cos)",
        },
        "inference_manifests": inference_manifests,
    }
    _atomic_write_json(output_dir / "materialization_manifest.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnostics-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--datalist", type=Path, required=True)
    parser.add_argument(
        "--key", choices=("draft_action", "final_action"), default="draft_action"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = materialize_predictions(
        diagnostics_dir=args.diagnostics_dir,
        output_dir=args.output_dir,
        datalist_path=args.datalist,
        key=args.key,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

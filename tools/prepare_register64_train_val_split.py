#!/usr/bin/env python3
"""Create and strictly validate deterministic NAVSIM holdout splits.

The production CLOVER route groups tokens by ``log_name`` and emits separate
calibration (``val``) and untouched checkpoint-selection holdouts.  The legacy
token-only two-way mode remains available for older launchers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 2


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--validation-size", type=int, default=4096)
    parser.add_argument("--selection-size", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    grouping = parser.add_mutually_exclusive_group()
    grouping.add_argument(
        "--token-log-map",
        help="JSON mapping token->log_name (or log_name->[tokens]).",
    )
    grouping.add_argument(
        "--metadata-root",
        help="Processed NAVSIM metadata directory containing <token>.pkl.",
    )
    parser.add_argument("--require-log-disjoint", action="store_true")
    parser.add_argument("--metadata-workers", type=int, default=16)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_tokens(path: Path, *, allow_empty: bool = False) -> list[str]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or (not value and not allow_empty):
        raise ValueError(f"datalist must be a JSON list: {path}")
    if any(not isinstance(token, str) or not token for token in value):
        raise TypeError("datalist entries must be non-empty strings")
    if len(value) != len(set(value)):
        raise ValueError("source datalist contains duplicate tokens")
    return value


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _paths(root: Path) -> tuple[Path, Path, Path, Path]:
    return (
        root / "train.json",
        root / "val.json",
        root / "selection.json",
        root / "manifest.json",
    )


def _legacy_identity(source: Path, validation_size: int, seed: int) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "source": str(source),
        "source_sha256": _sha256_file(source),
        "validation_size": validation_size,
        "seed": seed,
        "selection": "lowest_sha256(seed:token)",
    }


def _mapping_hash(mapping: dict[str, str]) -> str:
    payload = "".join(f"{token}\0{mapping[token]}\n" for token in sorted(mapping))
    return _sha256_bytes(payload.encode("utf-8"))


def _log_name_from_metadata(path: Path) -> str:
    with path.open("rb") as stream:
        metadata = pickle.load(stream)
    direct = metadata.get("log_name") if isinstance(metadata, dict) else None
    if isinstance(direct, str) and direct:
        return direct
    try:
        cameras = metadata["glo_images"]
        camera = cameras.get("cam_f0") or next(iter(cameras.values()))
        image_path = str(camera["image_paths"][0])
    except (KeyError, IndexError, StopIteration, TypeError) as error:
        raise ValueError(f"cannot recover log_name from metadata: {path}") from error
    for marker in ("/trainval/", "/train/"):
        _, separator, suffix = image_path.partition(marker)
        if separator and suffix:
            log_name = suffix.split("/", 1)[0]
            if log_name:
                return log_name
    raise ValueError(f"cannot parse NAVSIM log from image path in {path}: {image_path}")


def _load_token_log_mapping(
    tokens: list[str],
    *,
    token_log_map: str | None,
    metadata_root: str | None,
    metadata_workers: int = 16,
) -> tuple[dict[str, str] | None, str | None]:
    if token_log_map is None and metadata_root is None:
        return None, None
    token_set = set(tokens)
    if token_log_map is not None:
        path = Path(token_log_map).expanduser().resolve()
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or not raw:
            raise ValueError("token-log map must be a non-empty JSON object")
        if all(isinstance(value, str) for value in raw.values()):
            mapping = {str(token): str(log) for token, log in raw.items()}
        elif all(isinstance(value, list) for value in raw.values()):
            pairs = [
                (str(token), str(log))
                for log, log_tokens in raw.items()
                for token in log_tokens
            ]
            mapping = dict(pairs)
            if len(mapping) != len(pairs):
                raise ValueError("one token is assigned to multiple logs")
        else:
            raise TypeError("token-log map values must be all strings or all lists")
        source = str(path)
    else:
        root = Path(str(metadata_root)).expanduser().resolve()
        if not root.is_dir():
            raise NotADirectoryError(root)
        if metadata_workers <= 0:
            raise ValueError("metadata_workers must be positive")

        def read_one(token: str) -> tuple[str, str]:
            path = root / f"{token}.pkl"
            if not path.is_file():
                raise FileNotFoundError(path)
            return token, _log_name_from_metadata(path)

        with ThreadPoolExecutor(max_workers=metadata_workers) as pool:
            mapping = dict(pool.map(read_one, tokens))
        source = str(root)
    if set(mapping) != token_set:
        missing = sorted(token_set.difference(mapping))[:5]
        extra = sorted(set(mapping).difference(token_set))[:5]
        raise RuntimeError(
            f"token-log map differs from source tokens; missing={missing} extra={extra}"
        )
    if any(not log_name for log_name in mapping.values()):
        raise ValueError("token-log map contains an empty log_name")
    return mapping, source


def _log_identity(
    source: Path,
    validation_size: int,
    selection_size: int,
    seed: int,
    mapping: dict[str, str],
    mapping_source: str,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "source": str(source),
        "source_sha256": _sha256_file(source),
        "validation_size_target": validation_size,
        "selection_size_target": selection_size,
        "seed": seed,
        "selection": "lowest_sha256(seed:log_name),whole_logs",
        "grouping": "log_name",
        "token_log_mapping_source": mapping_source,
        "token_log_mapping_sha256": _mapping_hash(mapping),
    }


def _validate(
    root: Path,
    identity: dict[str, Any],
    mapping: dict[str, str] | None = None,
) -> dict[str, Any]:
    train_path, val_path, selection_path, manifest_path = _paths(root)
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for key, value in identity.items():
        if manifest.get(key) != value:
            raise RuntimeError(
                f"split identity mismatch for {key}: {manifest.get(key)!r} != {value!r}"
            )
    train_tokens = _load_tokens(train_path)
    val_tokens = _load_tokens(val_path)
    selection_tokens = (
        _load_tokens(selection_path, allow_empty=True)
        if int(identity.get("schema_version", 1)) >= 2
        else []
    )
    token_sets = (set(train_tokens), set(val_tokens), set(selection_tokens))
    if any(token_sets[left] & token_sets[right] for left in range(3) for right in range(left + 1, 3)):
        raise RuntimeError("train, validation, and selection token sets overlap")
    source_tokens = _load_tokens(Path(identity["source"]))
    if set().union(*token_sets) != set(source_tokens):
        raise RuntimeError("split union differs from source datalist")
    checks = {
        "train_count": len(train_tokens),
        "val_count": len(val_tokens),
        "train_sha256": _sha256_file(train_path),
        "val_sha256": _sha256_file(val_path),
    }
    if int(identity.get("schema_version", 1)) >= 2:
        checks.update(
            selection_count=len(selection_tokens),
            selection_sha256=_sha256_file(selection_path),
        )
        if mapping is None:
            raise RuntimeError("log-disjoint split validation requires token-log map")
        log_sets = [
            {mapping[token] for token in split_tokens} for split_tokens in token_sets
        ]
        if any(log_sets[left] & log_sets[right] for left in range(3) for right in range(left + 1, 3)):
            raise RuntimeError("train, validation, and selection logs overlap")
        expected_logs = {
            "train_logs": sorted(log_sets[0]),
            "val_logs": sorted(log_sets[1]),
            "selection_logs": sorted(log_sets[2]),
        }
        checks.update(expected_logs)
    for key, value in checks.items():
        if manifest.get(key) != value:
            raise RuntimeError(f"split manifest mismatch for {key}")
    return manifest


def _take_complete_logs(
    ranked_logs: list[str],
    grouped_tokens: dict[str, list[str]],
    *,
    start: int,
    target_tokens: int,
) -> tuple[set[str], int]:
    selected: set[str] = set()
    count = 0
    index = start
    while index < len(ranked_logs) and count < target_tokens:
        log_name = ranked_logs[index]
        selected.add(log_name)
        count += len(grouped_tokens[log_name])
        index += 1
    if count < target_tokens:
        raise ValueError("not enough complete logs for requested holdout sizes")
    return selected, index


def main() -> None:
    args = _parse_args()
    source = Path(args.source).expanduser().resolve()
    root = Path(args.output_dir).expanduser().resolve()
    tokens = _load_tokens(source)
    if not 1 <= args.validation_size < len(tokens):
        raise ValueError("validation size must be between 1 and source_size - 1")
    if args.selection_size < 0:
        raise ValueError("selection size must be non-negative")
    mapping, mapping_source = _load_token_log_mapping(
        tokens,
        token_log_map=args.token_log_map,
        metadata_root=args.metadata_root,
        metadata_workers=args.metadata_workers,
    )
    if args.require_log_disjoint and mapping is None:
        raise ValueError("--require-log-disjoint needs --token-log-map or --metadata-root")
    if args.selection_size and mapping is None:
        raise ValueError("a separate selection holdout requires log grouping")
    identity = (
        _legacy_identity(source, args.validation_size, args.seed)
        if mapping is None
        else _log_identity(
            source,
            args.validation_size,
            args.selection_size,
            args.seed,
            mapping,
            str(mapping_source),
        )
    )
    train_path, val_path, selection_path, manifest_path = _paths(root)

    if args.validate_only:
        print(json.dumps(_validate(root, identity, mapping), indent=2, sort_keys=True))
        return
    owned_paths = [train_path, val_path, manifest_path]
    if mapping is not None:
        owned_paths.append(selection_path)
    if (
        any(path.exists() for path in owned_paths)
        and not args.overwrite
    ):
        print(json.dumps(_validate(root, identity, mapping), indent=2, sort_keys=True))
        return

    if mapping is None:
        ranked = sorted(
            tokens,
            key=lambda token: _sha256_bytes(f"{args.seed}:{token}".encode("utf-8")),
        )
        validation_logs: set[str] = set()
        selection_logs: set[str] = set()
        validation = set(ranked[: args.validation_size])
        selection = set()
    else:
        grouped_tokens: dict[str, list[str]] = {}
        for token in tokens:
            grouped_tokens.setdefault(mapping[token], []).append(token)
        ranked_logs = sorted(
            grouped_tokens,
            key=lambda log_name: _sha256_bytes(
                f"{args.seed}:{log_name}".encode("utf-8")
            ),
        )
        validation_logs, next_index = _take_complete_logs(
            ranked_logs,
            grouped_tokens,
            start=0,
            target_tokens=args.validation_size,
        )
        selection_logs, next_index = _take_complete_logs(
            ranked_logs,
            grouped_tokens,
            start=next_index,
            target_tokens=args.selection_size,
        ) if args.selection_size else (set(), next_index)
        if next_index >= len(ranked_logs):
            raise ValueError("holdouts consume every log; no training log remains")
        validation = {token for token in tokens if mapping[token] in validation_logs}
        selection = {token for token in tokens if mapping[token] in selection_logs}
    train_tokens = [
        token for token in tokens if token not in validation and token not in selection
    ]
    val_tokens = [token for token in tokens if token in validation]
    selection_tokens = [token for token in tokens if token in selection]
    _atomic_json(train_path, train_tokens)
    _atomic_json(val_path, val_tokens)
    if mapping is not None:
        _atomic_json(selection_path, selection_tokens)
    manifest = {
        **identity,
        "source_count": len(tokens),
        "train_count": len(train_tokens),
        "val_count": len(val_tokens),
        "train_sha256": _sha256_file(train_path),
        "val_sha256": _sha256_file(val_path),
    }
    if mapping is not None:
        train_logs = sorted(set(mapping.values()) - validation_logs - selection_logs)
        manifest.update(
            selection_count=len(selection_tokens),
            selection_sha256=_sha256_file(selection_path),
            train_logs=train_logs,
            val_logs=sorted(validation_logs),
            selection_logs=sorted(selection_logs),
        )
    _atomic_json(manifest_path, manifest)
    print(json.dumps(_validate(root, identity, mapping), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

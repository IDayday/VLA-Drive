#!/usr/bin/env python3
"""Consolidate raw NAVSIM caches into the formal PlanReg input-only schema."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Iterable, Tuple

from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from navsim.agents.EpisodeDrive.formal_initialization import (  # noqa: E402
    canonical_sha256,
)
from navsim.planning.training.dataset import (  # noqa: E402
    dump_feature_target_to_pickle,
    load_feature_target_from_pickle,
)
from navsim.planning.training.input_only_cache import (  # noqa: E402
    DYNAMIC_FEATURE_CACHE_KEYS,
    INPUT_ONLY_CACHE_NAME,
    INPUT_ONLY_CACHE_SCHEMA_VERSION,
    build_input_only_cache_record,
)


_WORKER_TOKENIZER = None
_WORKER_FEATURE_NAME = None
_WORKER_TARGET_NAME = None
_WORKER_OVERWRITE = False


def _initialize_worker(
    tokenizer_path: str,
    feature_name: str,
    target_name: str,
    overwrite: bool,
) -> None:
    global _WORKER_TOKENIZER, _WORKER_FEATURE_NAME, _WORKER_TARGET_NAME, _WORKER_OVERWRITE
    from transformers import AutoTokenizer

    _WORKER_TOKENIZER = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=True,
        use_fast=False,
        local_files_only=True,
    )
    _WORKER_FEATURE_NAME = feature_name
    _WORKER_TARGET_NAME = target_name
    _WORKER_OVERWRITE = overwrite


def _build_one(token_path_text: str) -> Tuple[str, str]:
    token_path = Path(token_path_text)
    output = token_path / f"{INPUT_ONLY_CACHE_NAME}.gz"
    if output.is_file() and not _WORKER_OVERWRITE:
        return token_path_text, "existing"
    feature_path = token_path / f"{_WORKER_FEATURE_NAME}.gz"
    target_path = token_path / f"{_WORKER_TARGET_NAME}.gz"
    if not feature_path.is_file() or not target_path.is_file():
        missing = [
            str(path)
            for path in (feature_path, target_path)
            if not path.is_file()
        ]
        raise FileNotFoundError(f"Missing raw input cache files: {missing}")
    record = build_input_only_cache_record(
        load_feature_target_from_pickle(feature_path),
        load_feature_target_from_pickle(target_path),
        tokenizer=_WORKER_TOKENIZER,
    )
    dump_feature_target_to_pickle(output, record)
    return token_path_text, "written"


def _sha(values: Iterable[str]) -> str:
    return hashlib.sha256(
        "\n".join(sorted(values)).encode("utf-8")
    ).hexdigest()


def discover_eligible_token_paths(
    root: Path,
    *,
    allowed_logs,
    feature_name: str,
    target_name: str,
):
    discovered = sorted(
        token_path
        for log_path in root.iterdir()
        if log_path.is_dir()
        and (allowed_logs is None or log_path.name in allowed_logs)
        for token_path in log_path.iterdir()
        if token_path.is_dir()
    )
    eligible = [
        path
        for path in discovered
        if (path / f"{feature_name}.gz").is_file()
        and (path / f"{target_name}.gz").is_file()
    ]
    return discovered, eligible


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--feature-name", default="internvl_feature")
    parser.add_argument(
        "--target-name", default="trajectory_target_planreg_wm_v1"
    )
    parser.add_argument("--jobs", type=int, default=16)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--logs", help="Comma-separated log allow-list")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = Path(args.cache_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    allowed_logs = set(args.logs.split(",")) if args.logs else None
    discovered_token_paths, token_paths = discover_eligible_token_paths(
        root,
        allowed_logs=allowed_logs,
        feature_name=args.feature_name,
        target_name=args.target_name,
    )
    # Cache roots can contain historical token directories produced with a
    # different scene filter.  A formal record is eligible only when both raw
    # input sources are present; never count an empty/stale directory toward
    # the 103k protocol.
    skipped_incomplete_count = len(discovered_token_paths) - len(token_paths)
    if args.limit is not None:
        token_paths = token_paths[: args.limit]
    if not token_paths:
        raise RuntimeError("No token directories matched the input-only cache build")
    if args.dry_run:
        print(
            json.dumps(
                {
                    "discovered_token_directory_count": len(
                        discovered_token_paths
                    ),
                    "eligible_record_count": len(token_paths),
                    "skipped_incomplete_count": skipped_incomplete_count,
                    "dry_run": True,
                },
                sort_keys=True,
            )
        )
        return

    statuses = []
    with ProcessPoolExecutor(
        max_workers=args.jobs,
        initializer=_initialize_worker,
        initargs=(
            str(Path(args.tokenizer).expanduser().resolve()),
            args.feature_name,
            args.target_name,
            args.overwrite,
        ),
    ) as pool:
        for _path, status in tqdm(
            pool.map(_build_one, (str(path) for path in token_paths), chunksize=8),
            total=len(token_paths),
            desc="Building PlanReg input-only cache",
        ):
            statuses.append(status)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        trust_remote_code=True,
        use_fast=False,
        local_files_only=True,
    )
    vocab = {token: int(index) for token, index in tokenizer.get_vocab().items()}
    relative_tokens = [str(path.relative_to(root)) for path in token_paths]
    logs = sorted({path.parent.name for path in token_paths})
    manifest = {
        "prompt_version": os.getenv("PLANREG_PROMPT_VERSION", "legacy"),
        "schema_version": INPUT_ONLY_CACHE_SCHEMA_VERSION,
        "cache_mode": "input_only",
        "cache_name": INPUT_ONLY_CACHE_NAME,
        "cache_root": str(root),
        "record_count": len(token_paths),
        "discovered_token_directory_count": len(discovered_token_paths),
        "skipped_incomplete_count": skipped_incomplete_count,
        "required_source_files_complete": True,
        "source_feature_name": args.feature_name,
        "source_target_name": args.target_name,
        "written_count": statuses.count("written"),
        "existing_count": statuses.count("existing"),
        "log_count": len(logs),
        "logs_sha256": _sha(logs),
        "token_paths_sha256": _sha(relative_tokens),
        "tokenizer_vocab_sha256": canonical_sha256(vocab),
        "cached_fields": [
            "current_image_path",
            "future_image_paths",
            "future_valid_mask",
            "gt_trajectory",
            "long_trajectory",
            "ego_status",
            "navigation_command",
            "tokenized_input_ids",
            "attention_mask",
            "image_original_sizes",
            "tile_layout_metadata",
        ],
        "forbidden_dynamic_fields": sorted(DYNAMIC_FEATURE_CACHE_KEYS),
        "front_camera_only": True,
        "sensor_camera_count": 1,
    }
    manifest_path = root / "planreg_input_only_manifest.json"
    temporary = root / ".planreg_input_only_manifest.json.tmp"
    temporary.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(manifest_path)
    print(json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()

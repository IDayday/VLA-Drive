#!/usr/bin/env python3
"""Cache leakage-safe current-scene features from a frozen Qwen+DiT baseline."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
from omegaconf import OmegaConf
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from infer import VLAAgent  # noqa: E402
from research.action_effect.cache_io import (  # noqa: E402
    CacheConflictError,
    CacheManifest,
    content_hash,
    file_sha256,
    finalize_manifest,
    read_manifest,
    write_json,
    write_npz,
)
from research.action_effect.scene_features import extract_qwen_scene_features  # noqa: E402
from starVLA.dataloader.navsim_dataset import NavSimDataset, collate_fn  # noqa: E402


def _code_revision(paths: list[Path]) -> str:
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPOSITORY_ROOT, text=True
    ).strip()
    tree = content_hash(
        {str(path.relative_to(REPOSITORY_ROOT)): file_sha256(path) for path in paths}
    )
    return f"{commit}+tree.{tree[:12]}"


def _environment_path(explicit: Path | None, variable: str, suffix: str = "") -> Path:
    if explicit is not None:
        return explicit.resolve()
    value = os.environ.get(variable, "").strip()
    if not value:
        raise ValueError(f"pass the path explicitly or source load_env.sh to set {variable}")
    return (Path(value) / suffix).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-cache", type=Path)
    parser.add_argument("--checkpoint-run", type=Path)
    parser.add_argument("--model-iter", type=int)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--split", default="train")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-scenes", type=int)
    parser.add_argument(
        "--qwen-forward-mode", choices=("auto", "optimized", "legacy"), default="auto"
    )
    return parser.parse_args()


def _validate_or_prepare_build(cache_dir: Path, manifest: CacheManifest) -> bool:
    existing = read_manifest(cache_dir)
    if existing is not None:
        if existing.compatibility_identity() != manifest.compatibility_identity():
            raise CacheConflictError(f"existing scene-feature cache has a different identity: {cache_dir}")
        required = ("features.npz", "scene_index.json", "selected_scenes.json", "summary.json")
        missing = [name for name in required if not (cache_dir / name).is_file()]
        if missing:
            raise CacheConflictError(f"published scene-feature cache is incomplete ({missing})")
        return True
    cache_dir.mkdir(parents=True, exist_ok=True)
    identity_path = cache_dir / "build_identity.json"
    identity = {"compatibility_identity": manifest.compatibility_identity(), "manifest": asdict(manifest)}
    if identity_path.is_file():
        with identity_path.open("r", encoding="utf-8") as stream:
            if json.load(stream) != identity:
                raise CacheConflictError(
                    f"unfinished scene-feature cache has a different identity: {cache_dir}"
                )
    else:
        write_json(identity_path, identity)
    return False


def main() -> None:
    args = parse_args()
    if args.batch_size < 1 or args.num_workers < 0:
        raise ValueError("batch size must be positive and workers non-negative")
    candidate_cache = _environment_path(
        args.candidate_cache, "ACTION_EFFECT_CACHE_ROOT", "candidates/pilot_tiny/expert"
    )
    checkpoint_run = _environment_path(args.checkpoint_run, "ACTION_EFFECT_BASELINE_RUN")
    data_root = _environment_path(args.data_root, "DATA_ROOT")
    cache_dir = _environment_path(
        args.cache_dir, "ACTION_EFFECT_CACHE_ROOT", "scene_features/pilot_tiny/qwen_dit_100k"
    )
    model_iter = args.model_iter
    if model_iter is None:
        value = os.environ.get("ACTION_EFFECT_BASELINE_STEP", "").strip()
        model_iter = int(value) if value else None
    checkpoint = (
        checkpoint_run / "checkpoints" / f"steps_{model_iter}_pytorch_model.pt"
        if model_iter is not None
        else checkpoint_run / "final_model" / "pytorch_model.pt"
    )
    for path, label in (
        (candidate_cache, "candidate cache"),
        (checkpoint_run, "checkpoint run"),
        (checkpoint, "checkpoint"),
        (data_root, "processed data root"),
    ):
        if not path.exists():
            raise FileNotFoundError(f"{label} is missing: {path}")

    candidate_manifest = read_manifest(candidate_cache)
    if candidate_manifest is None:
        raise FileNotFoundError(f"candidate manifest is missing: {candidate_cache}")
    with (candidate_cache / "scene_index.json").open("r", encoding="utf-8") as stream:
        candidate_scene_index: dict[str, Any] = json.load(stream)
    scene_ids = list(candidate_scene_index)
    if args.max_scenes is not None:
        if args.max_scenes < 1:
            raise ValueError("--max-scenes must be positive")
        scene_ids = scene_ids[: args.max_scenes]

    code_revision = _code_revision(
        [
            Path(__file__),
            REPOSITORY_ROOT / "research/action_effect/scene_features.py",
            REPOSITORY_ROOT / "research/action_effect/data_contract.py",
            REPOSITORY_ROOT / "starVLA/model/framework/QwenOFT.py",
            REPOSITORY_ROOT / "infer.py",
        ]
    )
    manifest = CacheManifest(
        cache_kind="frozen_qwen_scene_feature",
        cache_version="action_effect_scene_features_v1",
        dataset_version=candidate_manifest.dataset_version,
        code_commit=code_revision,
        config_hash=content_hash(
            {
                "qwen_forward_mode_requested": args.qwen_forward_mode,
                "feature_fields": ["scene_tokens", "action_hidden"],
                "input_whitelist": ["image", "lang", "state", "token"],
            }
        ),
        evaluator_hash="not_applicable",
        split=args.split,
        seed=candidate_manifest.seed,
        inputs={
            "candidate_manifest": candidate_manifest.compatibility_identity(),
            "selected_scenes_sha256": content_hash(scene_ids),
            "checkpoint_sha256": file_sha256(checkpoint),
            "checkpoint_relative_to_run": str(checkpoint.relative_to(checkpoint_run)),
        },
    )
    if _validate_or_prepare_build(cache_dir, manifest):
        print(f"[action-effect] reusable scene-feature cache: {cache_dir}")
        return
    write_json(cache_dir / "selected_scenes.json", scene_ids)

    # Optional training caches must not become implicit dependencies of this
    # current-observation-only extraction pass.
    os.environ["NAVSIM_FEATURE_CACHE_ROOT"] = ""
    os.environ["NAVSIM_AGENT_DINO_CACHE_ROOT"] = ""
    os.environ["NAVSIM_VGGT_CACHE_ROOT"] = ""
    agent = VLAAgent(
        str(checkpoint_run),
        model_iter=model_iter,
        device=args.device,
        qwen_forward_mode=args.qwen_forward_mode,
    )
    for parameter in agent.model.parameters():
        parameter.requires_grad_(False)
    agent.model.eval()

    cfg = agent.model_config
    data_cfg = copy.deepcopy(cfg)
    OmegaConf.update(data_cfg, "datasets.video_data.load_2d_data", 0, force_add=True)
    OmegaConf.update(data_cfg, "datasets.video_data.load_3d_data", 0, force_add=True)
    OmegaConf.update(data_cfg, "datasets.reward_data.load_reward_data", False, force_add=True)
    OmegaConf.update(data_cfg, "datasets.vla_data.w_neg_traj", None, force_add=True)
    OmegaConf.update(data_cfg, "w_depth", 0, force_add=True)
    OmegaConf.update(data_cfg, "enable_image_aug", 0, force_add=True)
    OmegaConf.update(data_cfg, "framework.vggt.cache.enabled", False, force_add=True)
    dataset = NavSimDataset(
        datalist_path=str(cache_dir / "selected_scenes.json"),
        split=args.split,
        video_data_cfg=data_cfg.datasets.video_data,
        gs_data_cfg=data_cfg.datasets.gs_data,
        reward_data_cfg=data_cfg.datasets.reward_data,
        ver_1225=OmegaConf.select(cfg, "ver_1225", default=False),
        dataset_cfg=data_cfg.datasets.vla_data,
        all_cfg=data_cfg,
        data_root=str(data_root),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=False,
    )
    scene_dir = cache_dir / "scenes"
    scene_dir.mkdir(parents=True, exist_ok=True)
    resolved_mode = str(agent.qwen_forward_mode)
    for batch in tqdm(loader, desc="Frozen Qwen scene features"):
        expected = [str(sample["token"]) for sample in batch]
        pending = [
            sample for sample in batch if not (scene_dir / f"{sample['token']}.npz").is_file()
        ]
        if not pending:
            continue
        # A partially completed multi-sample batch is intentionally recomputed
        # as a smaller batch; ordering and per-token validation remain exact.
        outputs = extract_qwen_scene_features(
            agent.model, pending, qwen_forward_mode=resolved_mode
        )
        if outputs["scene_tokens"].shape[0] != len(pending):
            raise RuntimeError("feature batch size mismatch")
        for index, sample in enumerate(pending):
            token = str(sample["token"])
            if token not in expected or token not in candidate_scene_index:
                raise RuntimeError(f"dataset returned an unexpected token: {token}")
            write_npz(
                scene_dir / f"{token}.npz",
                scene_tokens=outputs["scene_tokens"][index].cpu().numpy().astype(np.float16),
                action_hidden=outputs["action_hidden"][index].cpu().numpy().astype(np.float16),
            )

    scene_tokens: list[np.ndarray] = []
    action_hidden: list[np.ndarray] = []
    scene_index: dict[str, int] = {}
    for index, token in enumerate(scene_ids):
        path = scene_dir / f"{token}.npz"
        if not path.is_file():
            raise RuntimeError(f"scene cache is incomplete: {path}")
        with np.load(path) as payload:
            scene_tokens.append(np.asarray(payload["scene_tokens"], dtype=np.float16))
            action_hidden.append(np.asarray(payload["action_hidden"], dtype=np.float16))
        scene_index[token] = index
    scene_array = np.stack(scene_tokens)
    action_array = np.stack(action_hidden)
    write_npz(
        cache_dir / "features.npz", scene_tokens=scene_array, action_hidden=action_array
    )
    write_json(cache_dir / "scene_index.json", scene_index)
    summary = {
        "scene_count": len(scene_ids),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": manifest.inputs["checkpoint_sha256"],
        "qwen_forward_mode": resolved_mode,
        "input_fields": ["image", "lang", "state", "token"],
        "scene_tokens_shape": list(scene_array.shape),
        "action_hidden_shape": list(action_array.shape),
        "dtype": str(scene_array.dtype),
        "expert_action_passed_to_qwen": False,
    }
    write_json(cache_dir / "summary.json", summary)
    finalize_manifest(cache_dir, manifest)
    print(json.dumps({"cache_dir": str(cache_dir), **summary}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

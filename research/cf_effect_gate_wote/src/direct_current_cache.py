"""Export frozen WoTE current-only features before any latent world transition."""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import numpy.typing as npt

from .cache_wote_features import (
    RELEASE_RELATIVE_PATHS,
    WOTE_COMMIT,
    validate_asset_manifest,
)
from .direct_rehab_contracts import (
    AccessAuditLog,
    AccessPolicy,
    assert_no_effect_input_stores,
)
from .feature_store import (
    DIRECT_CURRENT_ONLY_SCHEMA_VERSION,
    CacheIdentity,
    FeatureShardReader,
    FeatureShardWriter,
    SceneCacheRecord,
    atomic_write_json,
    sha256_file,
    stable_array_hash,
)


DIRECT_MODEL_INPUT_KEYS = frozenset(
    {
        "current_bev_tokens",
        "current_bev_pool",
        "ego_status_feature",
        "trajectory",
        "trajectory_anchor_feature",
        "candidate_current_feature",
    }
)
SELECTOR_REFERENCE_KEYS = frozenset(
    {"selected_index", "final_rewards", "im_rewards", "sim_rewards"}
)
ALLOWED_CACHE_KEYS = DIRECT_MODEL_INPUT_KEYS | SELECTOR_REFERENCE_KEYS
FORBIDDEN_CACHE_KEYS = frozenset(
    {
        "future_ego_features_by_step",
        "future_bev_tokens_by_step",
        "future_bev_pool_by_step",
        "reward_feature",
        "environment_only_future",
        "shared_environment_future",
        "effect",
        "effect_tensors",
        "factor_labels",
        "oracle_index",
    }
)


class DirectCurrentCacheError(RuntimeError):
    """The current-only cache or its pre-transition provenance is invalid."""


def _activate_wote_navsim(wote_root: Path) -> Path:
    """Make the pinned WoTE NAVSIM tree authoritative before its first import."""

    root = wote_root.resolve()
    expected_package = (root / "navsim" / "__init__.py").resolve()
    if not expected_package.is_file():
        raise DirectCurrentCacheError(
            f"pinned WoTE NAVSIM package is missing: {expected_package}"
        )
    loaded = sys.modules.get("navsim")
    if loaded is not None:
        loaded_file = Path(getattr(loaded, "__file__", "<unknown>")).resolve()
        if loaded_file != expected_package:
            raise DirectCurrentCacheError(
                "navsim was imported before the pinned WoTE tree was activated: "
                f"loaded={loaded_file}, expected={expected_package}"
            )
    root_text = str(root)
    sys.path[:] = [root_text] + [
        entry
        for entry in sys.path
        if not entry or Path(entry).resolve() != root
    ]
    importlib.invalidate_caches()
    spec = importlib.util.find_spec("navsim")
    if spec is None or spec.origin is None or Path(spec.origin).resolve() != expected_package:
        raise DirectCurrentCacheError(
            "pinned WoTE NAVSIM did not win import resolution: "
            f"expected={expected_package}, resolved={getattr(spec, 'origin', None)}"
        )
    return expected_package


def selector_index_is_maximal(
    final_rewards: npt.ArrayLike,
    selected_index: int,
) -> bool:
    """Accept a frozen selector index when cache quantization creates argmax ties.

    WoTE selects from the live reward tensor and the feature writer may later
    store rewards in float16.  Quantization can turn the unique live maximum
    into several equal stored maxima, so recomputing ``np.argmax`` is not an
    identity-preserving check.  The released index must instead point to one of
    the exact stored maxima; a genuinely lower-reward index is still rejected.
    """

    rewards = np.asarray(final_rewards)
    selected = int(selected_index)
    if rewards.ndim != 1 or not len(rewards):
        return False
    if selected < 0 or selected >= len(rewards):
        return False
    if not np.issubdtype(rewards.dtype, np.number) or not np.isfinite(rewards).all():
        return False
    return bool(rewards[selected] == np.max(rewards))


def _tensor_to_numpy(value: Any) -> npt.NDArray[Any]:
    return value.detach().cpu().numpy()


def _source_location(callable_object: Any, wote_root: Path) -> Mapping[str, Any]:
    source_file = Path(inspect.getsourcefile(callable_object) or "<unknown>")
    _, first_line = inspect.getsourcelines(callable_object)
    try:
        displayed = source_file.resolve().relative_to(wote_root.resolve()).as_posix()
    except ValueError:
        displayed = str(source_file.resolve())
    return {"file": displayed, "first_line": int(first_line)}


def validate_direct_scene_arrays(
    arrays: Mapping[str, npt.NDArray[Any]],
    *,
    require_selector_reference: bool,
) -> None:
    keys = set(arrays)
    missing = set(DIRECT_MODEL_INPUT_KEYS) - keys
    unexpected = keys - set(ALLOWED_CACHE_KEYS)
    forbidden = keys & set(FORBIDDEN_CACHE_KEYS)
    forbidden_fragments = sorted(
        key for key in keys if "future" in key.lower() or "effect" in key.lower()
    )
    if require_selector_reference:
        missing.update(set(SELECTOR_REFERENCE_KEYS) - keys)
    if missing or unexpected or forbidden or forbidden_fragments:
        raise DirectCurrentCacheError(
            "current-only array contract failed: "
            f"missing={sorted(missing)}, unexpected={sorted(unexpected)}, "
            f"forbidden={sorted(forbidden)}, fragments={forbidden_fragments}"
        )

    expected_shapes = {
        "current_bev_tokens": (64, 256),
        "current_bev_pool": (256,),
        "ego_status_feature": (8,),
        "trajectory": (256, 8, 3),
        "trajectory_anchor_feature": (256, 256),
        "candidate_current_feature": (256, 256),
    }
    if require_selector_reference:
        expected_shapes.update(
            {
                "selected_index": (),
                "final_rewards": (256,),
                "im_rewards": (256,),
                "sim_rewards": (256, 5),
            }
        )
    for key, expected in expected_shapes.items():
        actual = tuple(np.asarray(arrays[key]).shape)
        if actual != expected:
            raise DirectCurrentCacheError(
                f"{key} expected shape {expected}, got {actual}"
            )
    for key, value in arrays.items():
        array = np.asarray(value)
        if array.dtype.hasobject or (
            np.issubdtype(array.dtype, np.number) and not np.isfinite(array).all()
        ):
            raise DirectCurrentCacheError(f"invalid values in current-only field {key}")
    if "selected_index" in arrays:
        selected = int(np.asarray(arrays["selected_index"]).reshape(-1)[0])
        if not selector_index_is_maximal(arrays["final_rewards"], selected):
            raise DirectCurrentCacheError(
                "selected_index does not reference a maximal stored final_reward"
            )


def validate_direct_current_cache(
    root: Path,
    *,
    require_selector_reference: bool,
) -> Mapping[str, Any]:
    reader = FeatureShardReader(root)
    identity = reader.manifest.get("identity", {})
    if identity.get("feature_schema_version") != DIRECT_CURRENT_ONLY_SCHEMA_VERSION:
        raise DirectCurrentCacheError("cache is not wote_direct_current_only.v1")
    if identity.get("label_source") != "none":
        raise DirectCurrentCacheError("current-only cache must be label-free")
    count = 0
    observed_keys: set[str] | None = None
    for sidecar, arrays in reader.iter_shards():
        keys = set(arrays)
        if observed_keys is None:
            observed_keys = keys
        elif keys != observed_keys:
            raise DirectCurrentCacheError("current-only shards have inconsistent fields")
        records = sidecar["records"]
        for index, record in enumerate(records):
            scene = {key: np.asarray(value[index]) for key, value in arrays.items()}
            validate_direct_scene_arrays(
                scene,
                require_selector_reference=require_selector_reference,
            )
            if stable_array_hash(np.asarray(scene["trajectory"], dtype=np.float32)) != record[
                "trajectory_hash"
            ]:
                raise DirectCurrentCacheError(
                    f"trajectory identity mismatch: {record['scene_token']}"
                )
            count += 1
    if count != int(reader.manifest["scene_count"]):
        raise DirectCurrentCacheError(
            f"read {count} scenes; manifest claims {reader.manifest['scene_count']}"
        )
    return {
        "schema_version": DIRECT_CURRENT_ONLY_SCHEMA_VERSION,
        "scene_count": count,
        "keys": sorted(observed_keys or ()),
        "logical_content_sha256": reader.manifest["logical_content_sha256"],
        "selector_reference_present": bool(
            observed_keys and SELECTOR_REFERENCE_KEYS <= observed_keys
        ),
        "future_fields": [],
        "effect_fields": [],
        "status": "PASS",
    }


def _initialize_agent(args: argparse.Namespace) -> Any:
    _activate_wote_navsim(args.wote_root)
    from navsim.agents.WoTE.WoTE_agent import WoTEAgent
    from navsim.agents.WoTE.configs.default import WoTEConfig

    config = WoTEConfig()
    config.resnet34_path = str(args.release_root / RELEASE_RELATIVE_PATHS["resnet34"])
    config.cluster_file_path = str(
        args.release_root / RELEASE_RELATIVE_PATHS["trajectory_anchors"]
    )
    config.sim_reward_dict_path = None
    config.return_debug_features = True
    config.debug_force_base_anchors = True
    agent = WoTEAgent(
        config=config,
        trajectory_sampling=config.trajectory_sampling,
        lr=0.0,
        checkpoint_path=str(args.release_root / RELEASE_RELATIVE_PATHS["checkpoint"]),
        slice_indices=[3],
    )
    agent.initialize()
    agent.is_eval = True
    agent.eval().to(args.device)
    # ``extract_trajectory_feature`` is called directly below, bypassing the
    # agent wrapper.  The pinned WoTE implementation reads these flags from the
    # inner model, not from WoTEAgent.
    agent.WoTE_model.is_eval = True
    agent.WoTE_model.return_debug_features = True
    agent.WoTE_model.debug_force_base_anchors = True
    return agent


def _selector_reference(
    model: Any,
    trajectory_outputs: Mapping[str, Any],
) -> Mapping[str, Any]:
    import torch

    encoder_results = model.extract_reward_feature(trajectory_outputs, targets=None)
    reward_feature = encoder_results["reward_feature"]
    im_logits = model.reward_head(reward_feature).squeeze(-1)
    im_rewards = torch.softmax(im_logits, dim=-1)
    sim_rewards = [head(reward_feature).sigmoid() for head in model.sim_reward_heads]
    final_rewards = model.weighted_reward_calculation(im_rewards, sim_rewards)
    return {
        "im_rewards": im_rewards,
        "sim_rewards": torch.cat(sim_rewards, dim=-1),
        "final_rewards": final_rewards,
        "selected_index": torch.argmax(final_rewards, dim=-1),
    }


def cache_current_features(args: argparse.Namespace) -> Mapping[str, Any]:
    if args.output.exists():
        raise FileExistsError(f"refusing existing current-only cache: {args.output}")
    assert_no_effect_input_stores(
        ("navsim_current_observation", "fixed_256_candidate_bank", "frozen_wote")
    )
    policy = AccessPolicy.load(args.access_policy)
    tokens = policy.read_token_file(args.tokens, args.access_phase)
    if args.limit is not None:
        tokens = tokens[: args.limit]
    if not tokens or len(tokens) != len(set(tokens)):
        raise DirectCurrentCacheError("token set must be non-empty and unique")
    # This must precede *every* navsim import in this function.  The surrounding
    # DriveVLA repository also contains a different NAVSIM source tree, so merely
    # inserting WoTE's path later in _initialize_agent is too late.
    resolved_navsim_package = _activate_wote_navsim(args.wote_root)
    audit = AccessAuditLog(args.access_log, policy, args.access_phase)
    for token in tokens:
        audit.record(token, "current_only_feature_generation")

    assets = validate_asset_manifest(
        args.wote_root,
        args.release_root,
        args.data_root,
        compute_hashes=True,
        label_source="none",
    )
    if not assets["all_required_present"]:
        raise DirectCurrentCacheError("current-only feature assets are incomplete")

    import torch
    from navsim.common.dataloader import SceneLoader
    from navsim.common.dataclasses import SceneFilter

    seed = 20260827
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=False)

    agent = _initialize_agent(args)
    model = agent.WoTE_model
    loader = SceneLoader(
        data_path=args.data_root / "navsim_logs/trainval",
        sensor_blobs_path=args.data_root / "sensor_blobs/trainval",
        scene_filter=SceneFilter(
            num_history_frames=4,
            num_future_frames=10,
            frame_interval=1,
            has_route=True,
            tokens=list(tokens),
        ),
        sensor_config=agent.get_sensor_config(),
    )
    if set(loader.tokens) != set(tokens):
        missing = sorted(set(tokens) - set(loader.tokens))
        raise DirectCurrentCacheError(f"SceneLoader missing fixed tokens: {missing[:5]}")

    checkpoint_sha = next(
        entry["sha256"] for entry in assets["assets"] if entry["name"] == "checkpoint"
    )
    anchors = np.asarray(
        np.load(
            args.release_root / RELEASE_RELATIVE_PATHS["trajectory_anchors"],
            allow_pickle=False,
        ),
        dtype=np.float32,
    )
    candidate_bank_hash = stable_array_hash(anchors)
    writer = FeatureShardWriter(
        args.output,
        CacheIdentity(
            run_id=args.run_id,
            split=args.split,
            checkpoint_sha256=checkpoint_sha,
            wote_commit_sha=WOTE_COMMIT,
            feature_schema_version=DIRECT_CURRENT_ONLY_SCHEMA_VERSION,
            candidate_count=256,
            horizon=8,
            label_source="none",
            candidate_bank_hash=candidate_bank_hash,
        ),
        float32_keys=("trajectory",),
    )

    pending_arrays: dict[str, list[npt.NDArray[Any]]] = {}
    pending_records: list[SceneCacheRecord] = []
    shard_index = 0
    latent_calls = 0

    def count_latent_call(_module: Any, _inputs: Any) -> None:
        nonlocal latent_calls
        latent_calls += 1

    latent_hook = model.latent_world_model.register_forward_pre_hook(count_latent_call)

    def flush() -> None:
        nonlocal shard_index
        if not pending_records:
            return
        writer.write_shard(
            shard_index,
            {key: np.stack(values, axis=0) for key, values in pending_arrays.items()},
            tuple(pending_records),
        )
        shard_index += 1
        pending_arrays.clear()
        pending_records.clear()

    max_pre_export_latent_calls = 0
    selector_latent_calls = 0
    try:
        for scene_number, token in enumerate(tokens, start=1):
            agent_input = loader.get_agent_input_from_token(token)
            features = {
                key: value.unsqueeze(0).to(args.device)
                for key, value in agent.get_feature_builders()[0]
                .compute_features(agent_input)
                .items()
            }
            before = latent_calls
            with torch.inference_mode():
                trajectory_outputs = model.extract_trajectory_feature(features)
            pre_export_calls = latent_calls - before
            max_pre_export_latent_calls = max(
                max_pre_export_latent_calls, pre_export_calls
            )
            if pre_export_calls != 0:
                raise DirectCurrentCacheError(
                    f"{token}: latent_world_model ran before current feature export"
                )
            result = trajectory_outputs["results"]
            scene_arrays: dict[str, npt.NDArray[Any]] = {
                "current_bev_tokens": _tensor_to_numpy(result["current_bev_tokens"])[0],
                "current_bev_pool": _tensor_to_numpy(result["current_bev_pool"])[0],
                "ego_status_feature": _tensor_to_numpy(result["ego_status_feature"])[0],
                "trajectory": anchors,
                "trajectory_anchor_feature": _tensor_to_numpy(
                    result["trajectory_anchor_feature"]
                )[0],
                "candidate_current_feature": _tensor_to_numpy(
                    result["candidate_current_feature"]
                )[0],
            }
            if not np.array_equal(
                _tensor_to_numpy(result["trajectory_anchor_raw"])[0].astype(np.float32),
                anchors,
            ):
                raise DirectCurrentCacheError(
                    f"{token}: partial export did not use exact base anchors"
                )

            if args.include_selector_reference:
                selector_before = latent_calls
                with torch.inference_mode():
                    selector = _selector_reference(model, trajectory_outputs)
                calls = latent_calls - selector_before
                selector_latent_calls += calls
                if calls <= 0:
                    raise DirectCurrentCacheError(
                        f"{token}: WoTE selector reference did not execute its latent model"
                    )
                scene_arrays.update(
                    {
                        "selected_index": np.asarray(
                            _tensor_to_numpy(selector["selected_index"])[0],
                            dtype=np.int64,
                        ),
                        "final_rewards": _tensor_to_numpy(selector["final_rewards"])[0],
                        "im_rewards": _tensor_to_numpy(selector["im_rewards"])[0],
                        "sim_rewards": _tensor_to_numpy(selector["sim_rewards"])[0],
                    }
                )

            validate_direct_scene_arrays(
                scene_arrays,
                require_selector_reference=args.include_selector_reference,
            )
            record = SceneCacheRecord(
                scene_token=token,
                candidate_indices=tuple(range(256)),
                trajectory_hash=stable_array_hash(anchors),
                label_hash=None,
                candidate_bank_hash=candidate_bank_hash,
            )
            for key, value in scene_arrays.items():
                pending_arrays.setdefault(key, []).append(value)
            pending_records.append(record)
            if len(pending_records) == args.shard_scenes:
                flush()
            print(
                f"[direct-current-cache] {scene_number}/{len(tokens)} {token}",
                flush=True,
            )
        flush()
        manifest_path = writer.finalize()
    finally:
        latent_hook.remove()

    validation = validate_direct_current_cache(
        args.output,
        require_selector_reference=args.include_selector_reference,
    )
    provenance = {
        "schema_version": "direct_feature_provenance.v1",
        "feature_schema_version": DIRECT_CURRENT_ONLY_SCHEMA_VERSION,
        "wote_commit": WOTE_COMMIT,
        "checkpoint_sha256": checkpoint_sha,
        "candidate_bank_file_sha256": sha256_file(
            args.release_root / RELEASE_RELATIVE_PATHS["trajectory_anchors"]
        ),
        "direct_model_input_fields": sorted(DIRECT_MODEL_INPUT_KEYS),
        "selector_reference_fields": sorted(SELECTOR_REFERENCE_KEYS)
        if args.include_selector_reference
        else [],
        "forbidden_fields_absent": True,
        "all_direct_input_exports_before_latent_world_model": True,
        "max_latent_calls_before_direct_export_per_scene": max_pre_export_latent_calls,
        "selector_reference_uses_latent_world_model": bool(
            args.include_selector_reference
        ),
        "selector_reference_total_latent_calls": selector_latent_calls,
        "selector_reference_is_never_exposed_to_direct_models": True,
        "source_locations": {
            "navsim_package": str(resolved_navsim_package),
            "current_export": _source_location(model.extract_trajectory_feature, args.wote_root),
            "latent_transition": _source_location(
                model._latent_world_model_processing, args.wote_root
            ),
            "current_backbone": _source_location(
                model._process_backbone_features, args.wote_root
            ),
            "candidate_encoding": _source_location(
                model._concatenate_ego_and_traj_features, args.wote_root
            ),
        },
        "dependency_graph": {
            "current_bev_tokens": [
                "current camera/lidar",
                "frozen backbone",
                "frozen BEV downscale",
                "fixed current-BEV positional embedding",
            ],
            "current_bev_pool": ["current_bev_tokens", "mean over 64 tokens"],
            "ego_status_feature": ["current ego status input"],
            "trajectory": ["fixed released 256-anchor bank"],
            "trajectory_anchor_feature": [
                "trajectory",
                "frozen mlp_planning_vb",
                "frozen cluster_encoder",
            ],
            "candidate_current_feature": [
                "current ego status",
                "trajectory_anchor_feature",
                "frozen encode_ego_feat_mlp",
            ],
            "selector_reference": [
                "direct inputs above",
                "latent_world_model",
                "frozen reward heads",
            ],
        },
        "cache_validation": validation,
    }
    if args.provenance_output:
        atomic_write_json(args.provenance_output, provenance)
    return {
        "manifest": str(manifest_path),
        "cache": validation,
        "provenance": provenance,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    cache = commands.add_parser("cache")
    cache.add_argument("--wote-root", type=Path, required=True)
    cache.add_argument("--release-root", type=Path, required=True)
    cache.add_argument("--data-root", type=Path, required=True)
    cache.add_argument("--tokens", type=Path, required=True)
    cache.add_argument("--output", type=Path, required=True)
    cache.add_argument("--run-id", required=True)
    cache.add_argument(
        "--split", choices=("train", "val", "dev", "holdout"), required=True
    )
    cache.add_argument("--device", default="cuda")
    cache.add_argument("--shard-scenes", type=int, default=16)
    cache.add_argument("--limit", type=int)
    cache.add_argument("--include-selector-reference", action="store_true")
    cache.add_argument("--provenance-output", type=Path)
    cache.add_argument("--access-policy", type=Path, required=True)
    cache.add_argument("--access-log", type=Path, required=True)
    cache.add_argument("--access-phase", required=True)
    validate = commands.add_parser("validate")
    validate.add_argument("--cache", type=Path, required=True)
    validate.add_argument("--require-selector-reference", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "validate":
        result = validate_direct_current_cache(
            args.cache,
            require_selector_reference=args.require_selector_reference,
        )
    else:
        if args.shard_scenes <= 0:
            raise ValueError("--shard-scenes must be positive")
        if args.limit is not None and args.limit <= 0:
            raise ValueError("--limit must be positive")
        result = cache_current_features(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Asset preflight and frozen WoTE debug-feature caching."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import numpy.typing as npt

from .feature_store import (
    BASE_ANCHOR_FEATURE_SCHEMA_VERSION,
    CacheIdentity,
    FeatureShardReader,
    FeatureShardWriter,
    SceneCacheRecord,
    atomic_write_json,
    fixed_random_projection,
    sha256_file,
    stable_array_hash,
)


WOTE_COMMIT = "298957c128a91d41a1c6075bd0bb6e7e845e093f"
RELEASE_RELATIVE_PATHS = {
    "checkpoint": Path("epoch=29-step=19950.ckpt"),
    "resnet34": Path("resnet34.pth"),
    "trajectory_anchors": Path("extra_data/planning_vb/trajectory_anchors_256.npy"),
    "candidate_scores": Path("extra_data/planning_vb/formatted_pdm_score_256.npy"),
}
DEBUG_KEYS = (
    "current_bev_tokens",
    "current_bev_pool",
    "ego_status_feature",
    "trajectory_anchor_raw",
    "trajectory_anchor_feature",
    "candidate_current_feature",
    "future_ego_features_by_step",
    "future_bev_tokens_by_step",
    "future_bev_pool_by_step",
    "reward_feature",
    "all_trajectory",
    "base_trajectory_anchors",
    "trajectory_offsets",
    "im_rewards",
    "sim_rewards",
    "final_rewards",
    "selected_index",
)


class AssetPreflightError(RuntimeError):
    """Required data or checkpoint assets are absent or inconsistent."""


@dataclass(frozen=True)
class AssetEntry:
    name: str
    expected_name: str
    path: str
    exists: bool
    size_bytes: int | None
    sha256: str | None
    checksum_status: str


def _git_head(repo: Path) -> str:
    process = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return process.stdout.strip()


def _asset_entry(
    name: str,
    expected_name: str,
    path: Path,
    compute_hash: bool,
    manifest_path: str,
) -> AssetEntry:
    exists = path.is_file() and path.stat().st_size > 0
    size = path.stat().st_size if exists else None
    digest = sha256_file(path) if exists and compute_hash else None
    status = "computed" if digest else ("not_computed" if exists else "missing")
    return AssetEntry(
        name=name,
        expected_name=expected_name,
        path=manifest_path,
        exists=exists,
        size_bytes=size,
        sha256=digest,
        checksum_status=status,
    )


def validate_asset_manifest(
    wote_root: Path,
    release_root: Path,
    data_root: Path | None,
    compute_hashes: bool,
    label_source: str = "published",
) -> dict[str, Any]:
    if label_source not in {"published", "none"}:
        raise ValueError(f"unsupported label source: {label_source!r}")
    if not (wote_root / ".git").exists():
        raise AssetPreflightError(f"WoTE checkout is not a Git repository: {wote_root}")
    actual_commit = _git_head(wote_root)
    if actual_commit != WOTE_COMMIT:
        raise AssetPreflightError(
            f"WoTE commit mismatch: expected {WOTE_COMMIT}, got {actual_commit}"
        )

    release_assets = [
        (name, relative)
        for name, relative in RELEASE_RELATIVE_PATHS.items()
        if label_source == "published" or name != "candidate_scores"
    ]
    entries = [
        _asset_entry(
            name,
            relative.name,
            release_root / relative,
            compute_hashes,
            f"$WOTE_RELEASE_ROOT/{relative.as_posix()}",
        )
        for name, relative in release_assets
    ]
    dataset_entries: list[AssetEntry] = []
    if data_root is not None:
        for name, relative in (
            ("nuplan_maps", Path("maps")),
            ("navtrain_logs", Path("navsim_logs/trainval")),
            ("navtrain_sensors", Path("sensor_blobs/trainval")),
        ):
            path = data_root / relative
            exists = path.is_dir()
            dataset_entries.append(
                AssetEntry(
                    name=name,
                    expected_name=str(relative),
                    path=f"$NAVSIM_DATA_ROOT/{relative.as_posix()}",
                    exists=exists,
                    size_bytes=None,
                    sha256=None,
                    checksum_status="directory_present" if exists else "missing",
                )
            )

    manifest = {
        "schema_version": "cf_effect_gate_assets.v1",
        "wote_commit_sha": actual_commit,
        "release_root": "$WOTE_RELEASE_ROOT",
        "data_root": "$NAVSIM_DATA_ROOT" if data_root is not None else None,
        "label_source": label_source,
        "assets": [asdict(entry) for entry in entries + dataset_entries],
        "all_required_present": all(entry.exists for entry in entries + dataset_entries),
    }
    return manifest


def _atomic_write_text(path: Path, text: str) -> None:
    if path.exists():
        raise AssetPreflightError(f"refusing to overwrite report: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.partial")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_asset_reports(manifest: Mapping[str, Any], report_dir: Path) -> None:
    atomic_write_json(report_dir / "ASSET_MANIFEST.json", manifest)
    missing = [entry for entry in manifest["assets"] if not entry["exists"]]
    if not missing:
        return
    lines = [
        "# Missing assets",
        "",
        "WoTE full training is intentionally not a fallback. The following",
        "published or local data assets were not available during preflight:",
        "",
        "| Asset | Official expected name | Target path | Checksum status |",
        "| --- | --- | --- | --- |",
    ]
    for entry in missing:
        lines.append(
            f"| {entry['name']} | `{entry['expected_name']}` | "
            f"`{entry['path']}` | {entry['checksum_status']} |"
        )
    lines.extend(
        [
            "",
            "## Searched paths",
            "",
            f"- WoTE release root: `{manifest['release_root']}`",
            f"- NAVSIM data root: `{manifest.get('data_root') or 'not provided'}`",
            "",
            "## Work that can continue without the checkpoint",
            "",
            "- deterministic split generation from available score/log tokens;",
            "- candidate-label source audit;",
            "- replay-effect geometry unit and leakage tests;",
            "- dry-run/preflight validation for every launcher.",
            "",
            "G0 model reproduction and all checkpoint-dependent Gates remain `NOT_RUN`.",
        ]
    )
    _atomic_write_text(report_dir / "MISSING_ASSETS.md", "\n".join(lines) + "\n")


def validate_debug_shapes(
    output: Mapping[str, Any], expected_candidates: int = 256, expected_horizon: int = 8
) -> None:
    missing = [key for key in DEBUG_KEYS if key not in output]
    if missing:
        raise ValueError(f"WoTE debug output is missing keys: {missing}")

    def shape(key: str) -> tuple[int, ...]:
        return tuple(output[key].shape)

    batch = shape("current_bev_tokens")[0]
    if shape("current_bev_tokens")[1:] != (64, 256):
        raise ValueError(
            f"current_bev_tokens expected [B,64,256], got {shape('current_bev_tokens')}"
        )
    expected_prefix = (batch, expected_candidates)
    for key in (
        "trajectory_anchor_raw",
        "trajectory_anchor_feature",
        "candidate_current_feature",
        "future_ego_features_by_step",
        "future_bev_tokens_by_step",
        "future_bev_pool_by_step",
        "reward_feature",
        "im_rewards",
        "sim_rewards",
        "final_rewards",
    ):
        if shape(key)[:2] != expected_prefix:
            raise ValueError(
                f"{key} candidate alignment mismatch: expected prefix "
                f"{expected_prefix}, got {shape(key)}"
            )
    if shape("trajectory_anchor_raw")[2:] != (expected_horizon, 3):
        raise ValueError(
            f"trajectory_anchor_raw expected horizon {expected_horizon}, "
            f"got {shape('trajectory_anchor_raw')}"
        )
    if shape("sim_rewards")[-1] != 5:
        raise ValueError(f"sim_rewards must expose five factors, got {shape('sim_rewards')}")
    for key in DEBUG_KEYS:
        value = output[key]
        if hasattr(value, "is_floating_point") and value.is_floating_point():
            import torch

            if not torch.isfinite(value).all():
                raise ValueError(f"NaN/Inf detected in WoTE debug tensor {key}")


def assert_official_equivalence(
    reference: Mapping[str, Any], instrumented: Mapping[str, Any]
) -> None:
    import torch

    for key in ("trajectory", "all_trajectory", "final_rewards"):
        if key not in reference or key not in instrumented:
            raise ValueError(f"equivalence comparison missing key: {key}")
        torch.testing.assert_close(
            reference[key], instrumented[key], atol=1e-6, rtol=1e-6
        )


def environment_only_future(
    current_bev_tokens: npt.NDArray[np.floating[Any]],
    future_bev_tokens: npt.NDArray[np.floating[Any]],
    trajectories: npt.NDArray[np.floating[Any]],
    projection: npt.NDArray[np.floating[Any]],
) -> npt.NDArray[np.float32]:
    """Mask candidate ego cells, pool environment tokens, and project to 64D."""

    current = np.asarray(current_bev_tokens, dtype=np.float32)
    future = np.asarray(future_bev_tokens, dtype=np.float32)
    trajectory = np.asarray(trajectories, dtype=np.float32)
    if current.ndim != 3 or current.shape[1] != 64:
        raise ValueError(f"current BEV must be [B,64,C], got {current.shape}")
    if future.ndim != 5 or future.shape[3] != 64:
        raise ValueError(f"future BEV must be [B,K,T,64,C], got {future.shape}")
    batch, candidates, steps, cells, channels = future.shape
    if trajectory.shape[:3] != (batch, candidates, 8) or trajectory.shape[-1] < 2:
        raise ValueError(f"trajectory shape mismatch: {trajectory.shape}")
    if projection.ndim != 2 or projection.shape[0] != channels:
        raise ValueError(f"invalid projection shape: {projection.shape}")

    current_pool = current.mean(axis=1)
    output = np.empty((batch, candidates, steps, projection.shape[1]), dtype=np.float32)
    interval = 8 // steps
    if interval * steps != 8:
        raise ValueError(f"rollout steps must divide 8, got {steps}")
    for batch_index in range(batch):
        for candidate_index in range(candidates):
            for step_index in range(steps):
                pose_index = (step_index + 1) * interval - 1
                x, y = trajectory[batch_index, candidate_index, pose_index, :2]
                row = int(np.floor(x * 8.0 / 32.0))
                column = int(np.floor(y * 8.0 / 64.0 + 4.0))
                valid = np.ones(cells, dtype=bool)
                for row_delta in (-1, 0, 1):
                    for column_delta in (-1, 0, 1):
                        rr, cc = row + row_delta, column + column_delta
                        if 0 <= rr < 8 and 0 <= cc < 8:
                            valid[rr * 8 + cc] = False
                if not valid.any():
                    raise ValueError("ego-cell mask removed every BEV token")
                pooled = future[
                    batch_index, candidate_index, step_index, valid
                ].mean(axis=0)
                output[batch_index, candidate_index, step_index] = (
                    pooled - current_pool[batch_index]
                ) @ projection
    if not np.isfinite(output).all():
        raise ValueError("environment-only feature contains NaN/Inf")
    return output


def _load_score_dictionary(score_path: Path) -> Mapping[str, Any]:
    score_dictionary = np.load(score_path, allow_pickle=True).item()
    if not isinstance(score_dictionary, dict):
        raise ValueError(f"candidate score file is not a dictionary: {score_path}")
    return score_dictionary


def _load_score_factors(
    score_dictionary: Mapping[str, Any], token: str
) -> npt.NDArray[np.float32]:
    if token not in score_dictionary:
        raise KeyError(f"scene token absent from candidate score table: {token}")
    table = score_dictionary[token]["trajectory_scores"][0]
    keys = (
        "no_at_fault_collisions",
        "drivable_area_compliance",
        "ego_progress",
        "time_to_collision_within_bound",
        "comfort",
    )
    factors = np.stack([np.asarray(table[key], dtype=np.float32) for key in keys], axis=-1)
    if factors.shape != (256, 5):
        raise ValueError(f"factor label shape mismatch for {token}: {factors.shape}")
    if not np.isfinite(factors).all():
        raise ValueError(f"NaN/Inf factor label for {token}")
    return factors


def _score_dictionary_for_label_source(
    release_root: Path, label_source: str
) -> Mapping[str, Any] | None:
    """Load published labels only when explicitly selected."""

    if label_source == "none":
        return None
    if label_source != "published":
        raise ValueError(f"unsupported label source: {label_source!r}")
    return _load_score_dictionary(
        release_root / RELEASE_RELATIVE_PATHS["candidate_scores"]
    )


def assert_base_anchor_contract(
    output: Mapping[str, npt.NDArray[Any]],
    released_anchors: npt.NDArray[np.float32],
) -> None:
    """Require every candidate tensor to equal the released base bank bit-for-bit."""

    anchors = np.asarray(released_anchors, dtype=np.float32)
    if anchors.shape != (256, 8, 3):
        raise ValueError(f"released base anchors expected [256,8,3], got {anchors.shape}")
    base = np.asarray(output["base_trajectory_anchors"], dtype=np.float32)
    raw = np.asarray(output["trajectory_anchor_raw"], dtype=np.float32)
    all_trajectory = np.asarray(output["all_trajectory"], dtype=np.float32)
    if base.shape != anchors.shape or not np.array_equal(base, anchors):
        raise ValueError("model base_trajectory_anchors differ from released anchor bank")
    expected = anchors[None]
    if raw.shape != expected.shape or not np.array_equal(raw, expected):
        raise ValueError("trajectory_anchor_raw is not exactly the base anchor bank")
    if all_trajectory.shape != anchors.shape or not np.array_equal(
        all_trajectory, anchors
    ):
        raise ValueError("all_trajectory is not exactly the base anchor bank")


def _tensor_to_numpy(value: Any) -> npt.NDArray[Any]:
    return value.detach().cpu().numpy()


def run_cache(args: argparse.Namespace) -> None:
    if args.output.exists():
        raise FileExistsError(f"refusing existing cache output: {args.output}")
    manifest = validate_asset_manifest(
        args.wote_root,
        args.release_root,
        args.data_root,
        compute_hashes=True,
        label_source=args.label_source,
    )
    if not manifest["all_required_present"]:
        raise AssetPreflightError("required assets are missing")

    token_lines = [
        line.strip()
        for line in args.tokens.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not token_lines or len(token_lines) != len(set(token_lines)):
        raise ValueError("token file must be non-empty and contain unique tokens")
    if args.limit is not None:
        token_lines = token_lines[: args.limit]

    seed = 20260827
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    sys.path.insert(0, str(args.wote_root))
    import torch
    from navsim.agents.WoTE.WoTE_agent import WoTEAgent
    from navsim.agents.WoTE.configs.default import WoTEConfig
    from navsim.common.dataloader import SceneLoader
    from navsim.common.dataclasses import SceneFilter

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=False)

    config = WoTEConfig()
    config.resnet34_path = str(args.release_root / RELEASE_RELATIVE_PATHS["resnet34"])
    config.cluster_file_path = str(
        args.release_root / RELEASE_RELATIVE_PATHS["trajectory_anchors"]
    )
    config.sim_reward_dict_path = (
        str(args.release_root / RELEASE_RELATIVE_PATHS["candidate_scores"])
        if args.label_source == "published"
        else None
    )
    config.return_debug_features = False
    config.debug_force_base_anchors = False
    checkpoint_path = args.release_root / RELEASE_RELATIVE_PATHS["checkpoint"]
    agent = WoTEAgent(
        config=config,
        trajectory_sampling=config.trajectory_sampling,
        lr=0.0,
        checkpoint_path=str(checkpoint_path),
        slice_indices=[3],
    )
    agent.initialize()
    agent.is_eval = True
    agent.eval().to(args.device)

    scene_filter = SceneFilter(
        num_history_frames=4,
        num_future_frames=10,
        frame_interval=1,
        has_route=True,
        tokens=token_lines,
    )
    loader = SceneLoader(
        data_path=args.data_root / "navsim_logs/trainval",
        sensor_blobs_path=args.data_root / "sensor_blobs/trainval",
        scene_filter=scene_filter,
        sensor_config=agent.get_sensor_config(),
    )
    if set(loader.tokens) != set(token_lines):
        missing = sorted(set(token_lines) - set(loader.tokens))
        extra = sorted(set(loader.tokens) - set(token_lines))
        raise ValueError(
            f"scene token mismatch: missing={missing[:5]} extra={extra[:5]}"
        )

    checkpoint_sha = next(
        entry["sha256"] for entry in manifest["assets"] if entry["name"] == "checkpoint"
    )
    released_anchors = np.asarray(
        np.load(
            args.release_root / RELEASE_RELATIVE_PATHS["trajectory_anchors"],
            allow_pickle=False,
        ),
        dtype=np.float32,
    )
    candidate_bank_hash = stable_array_hash(released_anchors)
    identity = CacheIdentity(
        run_id=args.run_id,
        split=args.split,
        checkpoint_sha256=checkpoint_sha,
        wote_commit_sha=WOTE_COMMIT,
        feature_schema_version=(
            BASE_ANCHOR_FEATURE_SCHEMA_VERSION
            if args.label_source == "none"
            else "wote_debug.v1"
        ),
        label_source=args.label_source,
        candidate_bank_hash=(
            candidate_bank_hash if args.label_source == "none" else None
        ),
    )
    # Candidate identity is contractual for Gate2O joins.  Preserve anchors in
    # float32 while continuing to compress the much larger frozen activations.
    writer = FeatureShardWriter(args.output, identity, float32_keys=("trajectory",))
    projection = fixed_random_projection(256, 64, seed=20260827)
    score_dictionary = _score_dictionary_for_label_source(
        args.release_root, args.label_source
    )
    pending_arrays: dict[str, list[npt.NDArray[Any]]] = {}
    pending_records: list[SceneCacheRecord] = []
    shard_index = 0

    def flush() -> None:
        nonlocal shard_index
        if not pending_records:
            return
        arrays = {key: np.stack(values, axis=0) for key, values in pending_arrays.items()}
        writer.write_shard(shard_index, arrays, tuple(pending_records))
        shard_index += 1
        pending_arrays.clear()
        pending_records.clear()

    for token in token_lines:
        agent_input = loader.get_agent_input_from_token(token)
        features = {
            key: value.unsqueeze(0).to(args.device)
            for key, value in agent.get_feature_builders()[0].compute_features(agent_input).items()
        }
        with torch.inference_mode():
            agent.WoTE_model.return_debug_features = False
            agent.WoTE_model.debug_force_base_anchors = False
            official = agent(features)
            agent.WoTE_model.return_debug_features = True
            agent.WoTE_model.debug_force_base_anchors = False
            debug_official = agent(features)
            assert_official_equivalence(official, debug_official)
            validate_debug_shapes(debug_official)
            agent.WoTE_model.debug_force_base_anchors = True
            gate_output = agent(features)
            validate_debug_shapes(gate_output)

        output = {key: _tensor_to_numpy(value) for key, value in gate_output.items()}
        assert_base_anchor_contract(output, released_anchors)
        selected_from_output = int(np.asarray(output["selected_index"]).reshape(-1)[0])
        selected_from_rewards = int(np.argmax(output["final_rewards"][0]))
        if selected_from_output != selected_from_rewards:
            raise ValueError(
                f"{token}: selected_index {selected_from_output} is not "
                f"argmax(final_rewards) {selected_from_rewards}"
            )
        base_anchors = np.broadcast_to(
            output["base_trajectory_anchors"][None], (1, 256, 8, 3)
        ).copy()
        env_future = environment_only_future(
            output["current_bev_tokens"],
            output["future_bev_tokens_by_step"],
            base_anchors,
            projection,
        )
        scene_arrays = {
            "current_bev_tokens": output["current_bev_tokens"][0],
            "current_bev_pool": output["current_bev_pool"][0],
            "ego_status_feature": output["ego_status_feature"][0],
            "trajectory": base_anchors[0],
            "candidate_current_feature": output["candidate_current_feature"][0],
            "reward_feature": output["reward_feature"][0],
            "future_ego_features_by_step": output["future_ego_features_by_step"][0],
            "future_bev_pool_by_step": output["future_bev_pool_by_step"][0],
            # Gate2O v2 requires the frozen candidate-specific future spatial
            # tokens themselves.  They remain label-free WoTE activations and
            # are never copied into the replay-grounded effect cache.
            "future_bev_tokens_by_step": output["future_bev_tokens_by_step"][0],
            "environment_only_future": env_future[0],
            "shared_environment_future": np.broadcast_to(
                env_future[0].mean(axis=0, keepdims=True), env_future[0].shape
            ).copy(),
            "im_rewards": output["im_rewards"][0],
            "sim_rewards": output["sim_rewards"][0],
            "final_rewards": output["final_rewards"][0],
            "selected_index": np.asarray(output["selected_index"][0], dtype=np.int64),
        }
        factors = None
        if score_dictionary is not None:
            factors = _load_score_factors(score_dictionary, token)
            scene_arrays["factor_labels"] = factors
        trajectory_hash = stable_array_hash(base_anchors[0])
        record = SceneCacheRecord(
            scene_token=token,
            candidate_indices=tuple(range(256)),
            trajectory_hash=trajectory_hash,
            label_hash=(stable_array_hash(factors) if factors is not None else None),
            candidate_bank_hash=(
                candidate_bank_hash if args.label_source == "none" else None
            ),
        )
        for key, value in scene_arrays.items():
            pending_arrays.setdefault(key, []).append(value)
        pending_records.append(record)
        if len(pending_records) == args.shard_scenes:
            flush()
    flush()
    writer.finalize()


def summarize_g0(
    cache_first: Path,
    cache_second: Path,
    alignment_summary_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Validate two complete smoke caches and write the four G0 artifacts."""

    first = FeatureShardReader(cache_first)
    second = FeatureShardReader(cache_second)
    first_manifest = first.manifest
    second_manifest = second.manifest
    if first_manifest["identity"] != second_manifest["identity"]:
        raise ValueError("G0 cache identities differ")
    first_hash = first_manifest["logical_content_sha256"]
    second_hash = second_manifest["logical_content_sha256"]
    cache_reproducible = first_hash == second_hash
    alignment = json.loads(alignment_summary_path.read_text(encoding="utf-8"))
    first_shard_sidecar, _ = next(first.iter_shards())
    tensor_shapes = {
        name: details for name, details in first_shard_sidecar["arrays"].items()
    }
    scene_count = int(first_manifest["scene_count"])
    summary = {
        "scene_count": scene_count,
        "scene_failures": 0,
        "official_debug_equivalence": True,
        "candidate_alignment_pass": bool(alignment.get("pass", False)),
        "alignment_domain": alignment.get("alignment_domain"),
        "alignment_audited_scenes": int(alignment.get("audited_scenes", 0)),
        "alignment_audited_candidates_per_scene": int(
            alignment.get("audited_candidates_per_scene", 0)
        ),
        "alignment_audited_factor_values": int(
            alignment.get("audited_factor_values", 0)
        ),
        "alignment_maximum_absolute_error": float(
            alignment.get("maximum_absolute_error", float("nan"))
        ),
        "alignment_mean_absolute_error": float(
            alignment.get("mean_absolute_error", float("nan"))
        ),
        "alignment_mismatched_candidate_fraction": float(
            alignment.get("mismatched_candidate_fraction", float("nan"))
        ),
        "alignment_tolerance": float(alignment.get("tolerance", float("nan"))),
        "alignment_recompute_proposal_num_poses": alignment.get(
            "recompute_proposal_num_poses"
        ),
        "alignment_published_score_generator_proposal_num_poses": alignment.get(
            "published_score_generator_proposal_num_poses"
        ),
        "alignment_default_metric_cache_proposal_num_poses": alignment.get(
            "default_metric_cache_proposal_num_poses"
        ),
        "alignment_default_metric_cache_future_num_poses": alignment.get(
            "default_metric_cache_future_num_poses"
        ),
        "alignment_published_generator_default_cache_conflict": bool(
            alignment.get("published_generator_default_cache_conflict", False)
        ),
        "alignment_upstream_horizon_issue": alignment.get("upstream_horizon_issue"),
        "cache_first_logical_sha256": first_hash,
        "cache_second_logical_sha256": second_hash,
        "cache_reproducible": cache_reproducible,
        "gate_g0_pass": bool(
            scene_count == 200
            and alignment.get("pass", False)
            and cache_reproducible
        ),
    }
    checkpoint_manifest = {
        "checkpoint_sha256": first_manifest["identity"]["checkpoint_sha256"],
        "wote_commit_sha": first_manifest["identity"]["wote_commit_sha"],
        "feature_schema_version": first_manifest["identity"]["feature_schema_version"],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_dir / "g0_smoke_summary.json", summary)
    atomic_write_json(output_dir / "g0_tensor_shapes.json", tensor_shapes)
    atomic_write_json(output_dir / "g0_checkpoint_manifest.json", checkpoint_manifest)
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    preflight = subparsers.add_parser("preflight", help="validate assets without model load")
    preflight.add_argument("--wote-root", type=Path, required=True)
    preflight.add_argument("--release-root", type=Path, required=True)
    preflight.add_argument("--data-root", type=Path)
    preflight.add_argument("--report-dir", type=Path)
    preflight.add_argument("--write-reports", action="store_true")
    preflight.add_argument(
        "--label-source", choices=("published", "none"), default="published"
    )

    cache = subparsers.add_parser("cache", help="cache frozen WoTE features")
    cache.add_argument("--wote-root", type=Path, required=True)
    cache.add_argument("--release-root", type=Path, required=True)
    cache.add_argument("--data-root", type=Path, required=True)
    cache.add_argument("--tokens", type=Path, required=True)
    cache.add_argument("--output", type=Path, required=True)
    cache.add_argument("--run-id", required=True)
    cache.add_argument(
        "--split",
        choices=("train", "val", "test", "smoke", "headroom"),
        required=True,
    )
    cache.add_argument("--device", default="cuda")
    cache.add_argument("--shard-scenes", type=int, default=16)
    cache.add_argument("--limit", type=int)
    cache.add_argument("--dry-run", action="store_true")
    cache.add_argument("--preflight-only", action="store_true")
    cache.add_argument(
        "--label-source", choices=("published", "none"), default="published"
    )

    summary = subparsers.add_parser("summarize-g0", help="validate repeated G0 caches")
    summary.add_argument("--cache-first", type=Path, required=True)
    summary.add_argument("--cache-second", type=Path, required=True)
    summary.add_argument("--alignment-summary", type=Path, required=True)
    summary.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "summarize-g0":
        result = summarize_g0(
            args.cache_first,
            args.cache_second,
            args.alignment_summary,
            args.output_dir,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["gate_g0_pass"] else 4
    if args.command == "preflight":
        manifest = validate_asset_manifest(
            args.wote_root,
            args.release_root,
            args.data_root,
            compute_hashes=args.write_reports,
            label_source=args.label_source,
        )
        print(json.dumps(manifest, indent=2, sort_keys=True))
        if args.write_reports:
            if args.report_dir is None:
                raise ValueError("--report-dir is required with --write-reports")
            write_asset_reports(manifest, args.report_dir)
        if not manifest["all_required_present"]:
            raise AssetPreflightError("one or more required assets are missing")
        return 0

    if args.shard_scenes <= 0:
        raise ValueError("--shard-scenes must be positive")
    resolved = {
        "wote_root": str(args.wote_root),
        "release_root": str(args.release_root),
        "data_root": str(args.data_root),
        "tokens": str(args.tokens),
        "output": str(args.output),
        "run_id": args.run_id,
        "split": args.split,
        "device": args.device,
        "shard_scenes": args.shard_scenes,
        "limit": args.limit,
        "label_source": args.label_source,
    }
    print(json.dumps(resolved, indent=2, sort_keys=True))
    if args.dry_run:
        return 0
    manifest = validate_asset_manifest(
        args.wote_root,
        args.release_root,
        args.data_root,
        compute_hashes=False,
        label_source=args.label_source,
    )
    if not manifest["all_required_present"]:
        raise AssetPreflightError("one or more required assets are missing")
    if args.preflight_only:
        return 0
    run_cache(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

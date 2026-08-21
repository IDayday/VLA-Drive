#!/usr/bin/env python3
"""Generate deterministic DDP proposals and offline NAVSIM supervision.

The ``proposals`` pass is GPU-distributed and invokes the unchanged Qwen+DiT
DDP once per token.  The ``score`` pass is CPU-sharded and uses NAVSIM's
official metric cache, simulator, traffic policy, and PDM scorer.  Nothing in
this tool is imported by planner inference.
"""

from __future__ import annotations

import argparse
import copy
import json
import multiprocessing as mp
import os
import pickle
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

_REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

import numpy as np
import torch
from omegaconf import OmegaConf

from starVLA.model.modules.action_model.multi_trajectory.cache_schema import (
    CandidateCacheManifest,
    ProposalCacheRecord,
    TRAINING_METRIC_SCHEMA,
    TrainingCacheRecord,
    load_proposal_record,
    load_training_record,
    mark_training_cache_complete,
    read_manifest,
    save_proposal_record,
    save_training_record,
    sha256_file,
    stable_config_hash,
    write_manifest,
)
from starVLA.model.modules.action_model.multi_trajectory.trajectory_codec import (
    ACTION_Q01,
    ACTION_Q99,
    normalized_deltas_to_poses,
)
from starVLA.model.modules.action_model.multi_trajectory.trajectory_resampler import (
    trajectory_8_to_40,
)


_PRECOMPUTED_STATIC_SCORES: Mapping[str, Any] | None = None


def _load_tokens(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as stream:
        values = json.load(stream)
    if not isinstance(values, list) or not all(
        isinstance(value, str) and value for value in values
    ):
        raise TypeError("datalist must be a JSON list of non-empty token strings")
    if len(set(values)) != len(values):
        raise ValueError("datalist contains duplicate tokens")
    return values


def _git_commit(repository_root: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repository_root, text=True
    ).strip()


def _generator_contract(args: argparse.Namespace) -> dict[str, Any]:
    repository_root = Path(__file__).resolve().parent.parent
    paths = {
        "training_config_sha": sha256_file(args.config_yaml),
        "static_vocab_sha": sha256_file(args.vocab_path),
        "generator_tool_sha": sha256_file(Path(__file__).resolve()),
        "trajectory_codec_sha": sha256_file(
            repository_root
            / "starVLA/model/modules/action_model/multi_trajectory/trajectory_codec.py"
        ),
        "trajectory_resampler_sha": sha256_file(
            repository_root
            / "starVLA/model/modules/action_model/multi_trajectory/trajectory_resampler.py"
        ),
        "datalist_sha": sha256_file(args.datalist_path),
    }
    static_score_path = getattr(args, "static_score_path", None)
    if static_score_path is not None:
        paths["precomputed_static_score_sha"] = sha256_file(static_score_path)
    else:
        paths["precomputed_static_score_sha"] = None
    return {
        "baseline": "DDP-DRS-scene2048",
        "num_dynamic_candidates": int(args.num_candidates),
        "seed": int(args.seed),
        "split": str(args.split),
        "ddp_checkpoint_sha": str(args.ddp_checkpoint_sha),
        "base_vlm": str(Path(args.base_vlm).resolve()),
        "action_q01": list(ACTION_Q01),
        "action_q99": list(ACTION_Q99),
        "trajectory_8_times": [0.5 * index for index in range(1, 9)],
        "trajectory_40_interval": 0.1,
        **paths,
    }


def command_init(args: argparse.Namespace) -> None:
    tokens = _load_tokens(args.datalist_path)
    vocabulary = np.load(args.vocab_path, allow_pickle=False, mmap_mode="r")
    if vocabulary.shape != (8192, 40, 3):
        raise ValueError(
            f"static vocabulary must have shape [8192,40,3], got {vocabulary.shape}"
        )
    if not np.isfinite(vocabulary).all():
        raise ValueError("static vocabulary contains NaN or Inf")
    actual_sha = sha256_file(args.base_ddp_checkpoint)
    if actual_sha != args.ddp_checkpoint_sha:
        raise ValueError(
            "provided DDP checkpoint SHA does not match the checkpoint file"
        )
    contract = _generator_contract(args)
    generator_hash = stable_config_hash(contract)
    repository_root = Path(__file__).resolve().parent.parent
    manifest = CandidateCacheManifest(
        split=args.split,
        ddp_checkpoint_sha=args.ddp_checkpoint_sha,
        repository_commit_sha=_git_commit(repository_root),
        generator_config_hash=generator_hash,
        seed=args.seed,
        metric_schema=tuple(TRAINING_METRIC_SCHEMA),
        label_source_split=args.split,
    )
    root = Path(args.cache_root)
    existing = root / "manifest.json"
    if existing.is_file():
        previous = read_manifest(root)
        if previous != manifest:
            raise RuntimeError(
                "cache root already contains a different manifest; choose a new root"
            )
    else:
        write_manifest(root, manifest)
    metadata_path = root / "generator_contract.json"
    temporary = root / f".generator_contract.tmp-{os.getpid()}.json"
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(
            {**contract, "generator_config_hash": generator_hash, "tokens": len(tokens)},
            stream,
            indent=2,
            sort_keys=True,
        )
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, metadata_path)
    print(generator_hash)


def _build_cache_config(args: argparse.Namespace):
    cfg = OmegaConf.load(args.config_yaml)
    overrides = OmegaConf.from_dotlist(
        [
            f"framework.qwenvl.base_vlm={args.base_vlm}",
            f"framework.qwenvl.attn_implementation={args.attn_implementation}",
            "multi_trajectory.enabled=true",
            "multi_trajectory.training_stage=cache_candidates",
            f"multi_trajectory.num_dynamic_candidates={args.num_candidates}",
            f"multi_trajectory.deterministic_seed={args.seed}",
            f"datasets.vla_data.datalist_path={args.datalist_path}",
            f"datasets.vla_data.split={args.split}",
            f"datasets.vla_data.data_root={args.processed_data_root}",
            "datasets.vla_data.per_device_batch_size=1",
            "datasets.vla_data.w_neg_traj=null",
            "datasets.vla_data.act_norm=false",
            "datasets.video_data.load_2d_data=0",
            "datasets.gs_data.load_3d_data=0",
            "datasets.reward_data.load_reward_data=0",
            "enable_image_aug=0",
            "w_depth=0",
            "doing_s2=0",
            "vit_pre=0",
            "pretrain_model_2d=null",
            "trainer.pretrained_checkpoint=null",
            "trainer.resume_ckpt=none",
        ]
    )
    return OmegaConf.merge(cfg, overrides)


def _load_tensor_state(path: Path) -> Mapping[str, torch.Tensor]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if isinstance(payload, Mapping) and isinstance(payload.get("state_dict"), Mapping):
        payload = payload["state_dict"]
    if not isinstance(payload, Mapping) or not all(
        isinstance(key, str) and torch.is_tensor(value)
        for key, value in payload.items()
    ):
        raise TypeError("base DDP checkpoint must contain a tensor state_dict")
    return payload


def command_proposals(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("DDP proposal generation requires a CUDA-compatible device")
    manifest = read_manifest(args.cache_root)
    if manifest.ddp_checkpoint_sha != args.ddp_checkpoint_sha:
        raise ValueError("proposal checkpoint SHA does not match cache manifest")
    rank = int(os.environ.get("RANK", args.rank))
    world_size = int(os.environ.get("WORLD_SIZE", args.world_size))
    local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
    if world_size <= 0 or rank < 0 or rank >= world_size:
        raise ValueError("invalid proposal rank/world-size")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)

    # Full Qwen hidden states are required. Frozen Qwen feature caches from
    # other baselines must not replace the image/Qwen path in this pass.
    os.environ.pop("NAVSIM_FEATURE_CACHE_ROOT", None)
    from starVLA.dataloader.navsim_dataset import NavSimDataset
    from starVLA.model.framework import build_framework
    from starVLA.model.modules.action_model.multi_trajectory.checkpointing import (
        load_base_checkpoint_strict,
    )

    cfg = _build_cache_config(args)
    dataset = NavSimDataset(
        datalist_path=args.datalist_path,
        split=args.split,
        video_data_cfg=cfg.datasets.video_data,
        gs_data_cfg=cfg.datasets.gs_data,
        reward_data_cfg=cfg.datasets.reward_data,
        ver_1225=cfg.ver_1225,
        dataset_cfg=cfg.datasets.vla_data,
        all_cfg=cfg,
    )
    model = build_framework(cfg)
    load_base_checkpoint_strict(
        model, _load_tensor_state(Path(args.base_ddp_checkpoint))
    )
    model.to(device=device, dtype=torch.bfloat16)
    model.eval()
    owned_indices = list(range(rank, len(dataset), world_size))
    written = skipped = 0
    with torch.inference_mode():
        for local_index, dataset_index in enumerate(owned_indices):
            token = dataset.raw_list[dataset_index]
            try:
                previous = load_proposal_record(args.cache_root, token, manifest)
                if previous.trajectory_8.shape[0] != args.num_candidates:
                    raise ValueError("existing proposal candidate count is incompatible")
                skipped += 1
                continue
            except FileNotFoundError:
                pass
            sample = dataset[dataset_index]
            output = model([sample])
            trajectories_8 = (
                output["candidate_trajectories"]
                .detach()
                .to(device="cpu", dtype=torch.float32)[0]
            )
            trajectories_40 = trajectory_8_to_40(trajectories_8)
            target = normalized_deltas_to_poses(
                torch.as_tensor(sample["action"], dtype=torch.float32)
            )
            record = ProposalCacheRecord(
                token=token,
                split=manifest.split,
                ddp_checkpoint_sha=manifest.ddp_checkpoint_sha,
                repository_commit_sha=manifest.repository_commit_sha,
                generator_config_hash=manifest.generator_config_hash,
                seed=manifest.seed,
                candidate_ids=np.arange(args.num_candidates, dtype=np.int64),
                trajectory_8=trajectories_8.numpy(),
                trajectory_40=trajectories_40.numpy(),
                target_trajectory_8=target.numpy(),
            )
            save_proposal_record(args.cache_root, record, manifest)
            written += 1
            if (local_index + 1) % args.log_interval == 0:
                print(
                    f"[proposals rank={rank}] processed={local_index + 1}/"
                    f"{len(owned_indices)} written={written} skipped={skipped}",
                    flush=True,
                )
    print(
        f"[proposals rank={rank}] complete owned={len(owned_indices)} "
        f"written={written} skipped={skipped}",
        flush=True,
    )


# DriveSuprim donor mapping:
# William-Yao-2000/DriveSuprim
#   navsim/evaluate/pdm_score.py::pdm_score_full_v2
# Compatibility changes: target NAVSIM scorer returns DataFrames instead of
# ``if_return_pdms``; this implementation reads the same scorer metric arrays
# after the official score_proposals call. Formulas and default sampling are
# unchanged, and static/dynamic pools are scored independently so the donor
# 8192-static supervision remains unchanged.
def _score_trajectory_pool(
    metric_cache,
    trajectories: np.ndarray,
    sampling,
    simulator,
    scorer,
    traffic_policy,
) -> dict[str, np.ndarray]:
    from navsim.common.dataclasses import Trajectory
    from navsim.common.enums import SceneFrameType
    from navsim.evaluate.pdm_score import (
        get_trajectory_as_array,
        transform_trajectory,
    )
    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import (
        MultiMetricIndex,
        WeightedMetricIndex,
    )
    from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_comfort_metrics import (
        ego_is_comfortable,
    )

    if trajectories.ndim != 3 or trajectories.shape[1:] != (40, 3):
        raise ValueError("PDM trajectory pool must have shape [N,40,3]")
    initial_ego_state = metric_cache.ego_state
    all_states = [
        get_trajectory_as_array(
            metric_cache.trajectory, sampling, initial_ego_state.time_point
        )
    ]
    trajectory_sampling = type(sampling)(time_horizon=4.0, interval_length=0.1)
    for poses in trajectories:
        transformed = transform_trajectory(
            Trajectory(poses, trajectory_sampling), initial_ego_state
        )
        all_states.append(
            get_trajectory_as_array(
                transformed, sampling, initial_ego_state.time_point
            )
        )
    states = np.stack(all_states, axis=0)
    simulated_states = simulator.simulate_proposals(states, initial_ego_state)
    traffic_tracks = traffic_policy.simulate_environment(
        simulated_states[1], metric_cache
    )
    if len(traffic_tracks) != states.shape[1]:
        raise RuntimeError("traffic policy returned an invalid trajectory length")
    results = scorer.score_proposals(
        simulated_states,
        metric_cache.observation,
        metric_cache.centerline,
        metric_cache.route_lane_ids,
        metric_cache.drivable_area_map,
        metric_cache.map_parameters,
        traffic_tracks,
        metric_cache.past_human_trajectory,
    )
    scores = {
        "no_at_fault_collisions": scorer._multi_metrics[
            MultiMetricIndex.NO_COLLISION
        ][1:],
        "drivable_area_compliance": scorer._multi_metrics[
            MultiMetricIndex.DRIVABLE_AREA
        ][1:],
        "driving_direction_compliance": scorer._multi_metrics[
            MultiMetricIndex.DRIVING_DIRECTION
        ][1:],
        "traffic_light_compliance": scorer._multi_metrics[
            MultiMetricIndex.TRAFFIC_LIGHT_COMPLIANCE
        ][1:],
        "ego_progress": scorer._weighted_metrics[WeightedMetricIndex.PROGRESS][1:],
        "time_to_collision_within_bound": scorer._weighted_metrics[
            WeightedMetricIndex.TTC
        ][1:],
        "lane_keeping": scorer._weighted_metrics[WeightedMetricIndex.LANE_KEEPING][
            1:
        ],
        "history_comfort": scorer._weighted_metrics[
            WeightedMetricIndex.HISTORY_COMFORT
        ][1:],
        "pdm_score": np.asarray(
            [float(frame["pdm_score"].iloc[0]) for frame in results[1:]],
            dtype=np.float64,
        ),
    }
    time_points = np.arange(
        0, sampling.num_poses + 1, dtype=np.float64
    ) * sampling.interval_length
    scores["comfort"] = ego_is_comfortable(
        simulated_states, time_points
    ).all(axis=-1)[1:]
    if scorer._config.human_penalty_filter and (
        metric_cache.scene_type == SceneFrameType.ORIGINAL
    ):
        human = transform_trajectory(
            metric_cache.human_trajectory, initial_ego_state
        )
        human_states = get_trajectory_as_array(
            human, sampling, initial_ego_state.time_point
        )
        human_simulated = simulator.simulate_proposals(
            human_states[None], initial_ego_state
        )
        human_tracks = traffic_policy.simulate_environment(
            human_simulated[0], metric_cache
        )
        human_result = scorer.score_proposals(
            human_simulated,
            metric_cache.observation,
            metric_cache.centerline,
            metric_cache.route_lane_ids,
            metric_cache.drivable_area_map,
            metric_cache.map_parameters,
            human_tracks,
        )[0]
        for name in tuple(scores):
            if name not in {"pdm_score", "comfort"} and float(
                human_result[name].iloc[0]
            ) == 0.0:
                scores[name] = np.ones_like(scores[name])
        human_comfort = ego_is_comfortable(
            human_simulated, time_points
        ).all(axis=-1)[0]
        if not human_comfort:
            scores["comfort"] = np.ones_like(scores["comfort"])
    expected_count = trajectories.shape[0]
    for name, values in scores.items():
        if values.shape != (expected_count,):
            raise RuntimeError(
                f"PDM score {name} has shape {values.shape}, expected [{expected_count}]"
            )
        if not np.isfinite(values).all():
            raise ValueError(f"PDM score {name} contains NaN or Inf")
    return {name: values.astype(np.float16) for name, values in scores.items()}


def _load_precomputed_static_scores(path: Path) -> Mapping[str, Any]:
    with path.open("rb") as stream:
        payload = pickle.load(stream)
    if not isinstance(payload, Mapping):
        raise TypeError("DriveSuprim static score cache must contain a token mapping")
    return payload


def _normalize_precomputed_static_scores(
    token: str,
    payload: Mapping[str, Any],
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Normalize the official DriveSuprim navtrain.pkl token payload.

    DriveSuprim stores its eight selector metrics plus ``pdm_score``.  The
    ninth ``comfort`` field belongs to the DrivoR cache contract and is never
    consumed for a static candidate; for a complete uniform record it aliases
    the donor's ``history_comfort`` value without changing selector training.
    """

    if token not in payload:
        raise KeyError(f"official DriveSuprim static cache has no token {token!r}")
    token_payload = payload[token]
    if not isinstance(token_payload, Mapping):
        raise TypeError(f"DriveSuprim static payload for {token!r} is not a mapping")
    metrics: dict[str, np.ndarray] = {}
    for name in TRAINING_METRIC_SCHEMA:
        donor_name = "history_comfort" if name == "comfort" else name
        if donor_name not in token_payload:
            raise KeyError(
                f"DriveSuprim static payload for {token!r} is missing {donor_name!r}"
            )
        values = np.asarray(token_payload[donor_name], dtype=np.float16)
        if values.shape != (8192,):
            raise ValueError(
                f"DriveSuprim static metric {donor_name!r} for {token!r} "
                f"has shape {values.shape}, expected [8192]"
            )
        if not np.isfinite(values).all():
            raise ValueError(
                f"DriveSuprim static metric {donor_name!r} contains NaN or Inf"
            )
        metrics[name] = values
    if "pdm_score" not in token_payload:
        raise KeyError(
            f"DriveSuprim static payload for {token!r} is missing 'pdm_score'"
        )
    final_score = np.asarray(token_payload["pdm_score"], dtype=np.float16)
    if final_score.shape != (8192,):
        raise ValueError(
            f"DriveSuprim static pdm_score for {token!r} has shape "
            f"{final_score.shape}, expected [8192]"
        )
    if not np.isfinite(final_score).all():
        raise ValueError("DriveSuprim static pdm_score contains NaN or Inf")
    return metrics, final_score


def command_score(args: argparse.Namespace) -> None:
    from nuplan.planning.simulation.trajectory.trajectory_sampling import (
        TrajectorySampling,
    )
    from navsim.common.dataloader import MetricCacheLoader
    from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import (
        PDMScorer,
    )
    from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import (
        PDMSimulator,
    )
    from navsim.traffic_agents_policies.log_replay_traffic_agents import (
        LogReplayTrafficAgents,
    )

    if args.num_workers <= 0 or args.worker_index not in range(args.num_workers):
        raise ValueError("invalid score worker index/count")
    manifest = read_manifest(args.cache_root)
    tokens = _load_tokens(args.datalist_path)
    static_vocab = np.asarray(
        np.load(args.vocab_path, allow_pickle=False, mmap_mode="r"),
        dtype=np.float32,
    )
    if static_vocab.shape != (8192, 40, 3):
        raise ValueError("static vocabulary must have shape [8192,40,3]")
    cache_loader = MetricCacheLoader(Path(args.metric_cache_root))
    sampling = TrajectorySampling(num_poses=40, interval_length=0.1)
    simulator = PDMSimulator(sampling)
    scorer = PDMScorer(sampling)
    traffic_policy = LogReplayTrafficAgents(sampling)
    static_score_payload = _PRECOMPUTED_STATIC_SCORES
    if args.static_score_path is not None and static_score_payload is None:
        static_score_payload = _load_precomputed_static_scores(
            Path(args.static_score_path)
        )
    owned_tokens = tokens[args.worker_index :: args.num_workers]
    written = skipped = 0
    for index, token in enumerate(owned_tokens):
        try:
            load_training_record(
                args.cache_root,
                token,
                expected_split=args.split,
                expected_ddp_checkpoint_sha=manifest.ddp_checkpoint_sha,
                expected_generator_config_hash=manifest.generator_config_hash,
                require_complete=False,
                manifest=manifest,
            )
            skipped += 1
            continue
        except FileNotFoundError:
            pass
        proposal = load_proposal_record(args.cache_root, token, manifest)
        if token not in cache_loader.metric_cache_paths:
            raise FileNotFoundError(
                f"NAVSIM metric cache has no entry for token {token!r}"
            )
        metric_cache = cache_loader.get_from_token(token)
        if static_score_payload is None:
            static_scores = _score_trajectory_pool(
                metric_cache,
                static_vocab,
                sampling,
                simulator,
                scorer,
                traffic_policy,
            )
            static_final_score = static_scores["pdm_score"]
        else:
            static_scores, static_final_score = (
                _normalize_precomputed_static_scores(token, static_score_payload)
            )
        dynamic_scores = _score_trajectory_pool(
            metric_cache,
            proposal.trajectory_40.astype(np.float32, copy=False),
            sampling,
            simulator,
            scorer,
            traffic_policy,
        )
        record = TrainingCacheRecord(
            proposal=proposal,
            dynamic_metrics={
                name: dynamic_scores[name] for name in TRAINING_METRIC_SCHEMA
            },
            static_metrics={
                name: static_scores[name] for name in TRAINING_METRIC_SCHEMA
            },
            dynamic_final_score=dynamic_scores["pdm_score"],
            static_final_score=static_final_score,
        )
        save_training_record(args.cache_root, record, manifest)
        written += 1
        if (index + 1) % args.log_interval == 0:
            print(
                f"[score worker={args.worker_index}] processed={index + 1}/"
                f"{len(owned_tokens)} written={written} skipped={skipped}",
                flush=True,
            )
    print(
        f"[score worker={args.worker_index}] complete owned={len(owned_tokens)} "
        f"written={written} skipped={skipped}",
        flush=True,
    )


def _shared_static_score_worker(args: argparse.Namespace, worker_index: int) -> None:
    worker_args = copy.copy(args)
    worker_args.worker_index = worker_index
    command_score(worker_args)


def command_score_all(args: argparse.Namespace) -> None:
    """Fork CPU score workers after loading the 15-GB donor cache once."""

    if args.static_score_path is None:
        raise ValueError("score-all requires --static-score-path")
    if args.num_workers <= 0:
        raise ValueError("score-all num-workers must be positive")
    global _PRECOMPUTED_STATIC_SCORES
    _PRECOMPUTED_STATIC_SCORES = _load_precomputed_static_scores(
        Path(args.static_score_path)
    )
    context = mp.get_context("fork")
    processes = [
        context.Process(
            target=_shared_static_score_worker,
            args=(args, worker_index),
            name=f"ddp-drs-score-{worker_index}",
        )
        for worker_index in range(args.num_workers)
    ]
    for process in processes:
        process.start()
    failed: list[tuple[str, int | None]] = []
    for process in processes:
        process.join()
        if process.exitcode != 0:
            failed.append((process.name, process.exitcode))
    if failed:
        raise RuntimeError(f"DDP-DRS score workers failed: {failed}")


def command_finalize(args: argparse.Namespace) -> None:
    tokens = _load_tokens(args.datalist_path)
    path = mark_training_cache_complete(args.cache_root, tokens=tokens)
    print(path)


def _add_shared_cache_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--datalist-path", type=Path, required=True)
    parser.add_argument("--split", required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init")
    _add_shared_cache_arguments(init_parser)
    init_parser.add_argument("--config-yaml", type=Path, required=True)
    init_parser.add_argument("--base-vlm", type=Path, required=True)
    init_parser.add_argument("--base-ddp-checkpoint", type=Path, required=True)
    init_parser.add_argument("--ddp-checkpoint-sha", required=True)
    init_parser.add_argument("--vocab-path", type=Path, required=True)
    init_parser.add_argument("--static-score-path", type=Path)
    init_parser.add_argument("--seed", type=int, default=3047)
    init_parser.add_argument("--num-candidates", type=int, default=64)
    init_parser.set_defaults(handler=command_init)

    proposal_parser = subparsers.add_parser("proposals")
    _add_shared_cache_arguments(proposal_parser)
    proposal_parser.add_argument("--config-yaml", type=Path, required=True)
    proposal_parser.add_argument("--base-vlm", type=Path, required=True)
    proposal_parser.add_argument("--base-ddp-checkpoint", type=Path, required=True)
    proposal_parser.add_argument("--processed-data-root", type=Path, required=True)
    proposal_parser.add_argument("--ddp-checkpoint-sha", required=True)
    proposal_parser.add_argument("--seed", type=int, default=3047)
    proposal_parser.add_argument("--num-candidates", type=int, default=64)
    proposal_parser.add_argument("--attn-implementation", default="flash_attention_2")
    proposal_parser.add_argument("--rank", type=int, default=0)
    proposal_parser.add_argument("--world-size", type=int, default=1)
    proposal_parser.add_argument("--local-rank", type=int, default=0)
    proposal_parser.add_argument("--log-interval", type=int, default=10)
    proposal_parser.set_defaults(handler=command_proposals)

    score_parser = subparsers.add_parser("score")
    _add_shared_cache_arguments(score_parser)
    score_parser.add_argument("--vocab-path", type=Path, required=True)
    score_parser.add_argument("--metric-cache-root", type=Path, required=True)
    score_parser.add_argument("--static-score-path", type=Path)
    score_parser.add_argument("--worker-index", type=int, default=0)
    score_parser.add_argument("--num-workers", type=int, default=1)
    score_parser.add_argument("--log-interval", type=int, default=1)
    score_parser.set_defaults(handler=command_score)

    score_all_parser = subparsers.add_parser("score-all")
    _add_shared_cache_arguments(score_all_parser)
    score_all_parser.add_argument("--vocab-path", type=Path, required=True)
    score_all_parser.add_argument("--metric-cache-root", type=Path, required=True)
    score_all_parser.add_argument("--static-score-path", type=Path, required=True)
    score_all_parser.add_argument("--num-workers", type=int, default=1)
    score_all_parser.add_argument("--worker-index", type=int, default=0)
    score_all_parser.add_argument("--log-interval", type=int, default=1)
    score_all_parser.set_defaults(handler=command_score_all)

    finalize_parser = subparsers.add_parser("finalize")
    _add_shared_cache_arguments(finalize_parser)
    finalize_parser.set_defaults(handler=command_finalize)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.handler(args)


if __name__ == "__main__":
    main()

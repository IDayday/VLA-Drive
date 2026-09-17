"""Official one-stage NAVSIM v2 per-scene reward; isolated spawn workers.

Every candidate invokes pdm_score with [official PDM reference, one candidate].
We never concatenate the G candidates into the scorer proposal pool.
"""
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import lzma
import multiprocessing as mp
import os
import pickle
import time
import numpy as np
from .contracts import digest

VERSION = "navsim_v2_one_stage_single_scene"


@dataclass
class ScoreResult:
    token: str
    candidate: int
    score: float
    metrics: dict
    status: str = "valid"


class OfficialEvaluator:
    def __init__(self, devkit, cache_index, cache_capacity=4):
        from hydra.utils import instantiate
        from omegaconf import OmegaConf
        from nuplan.planning.simulation.trajectory.trajectory_sampling import (
            TrajectorySampling,
        )
        from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import (
            PDMSimulator,
        )
        from navsim.traffic_agents_policies.log_replay_traffic_agents import (
            LogReplayTrafficAgents,
        )

        self.cache_index = cache_index
        self.cache = OrderedDict()
        self.capacity = cache_capacity
        self.sampling = TrajectorySampling(num_poses=40, interval_length=0.1)
        cfg = OmegaConf.load(
            Path(devkit)
            / "navsim/planning/script/config/pdm_scoring/scorer/pdm_scorer.yaml"
        )
        cfg.proposal_sampling = OmegaConf.create(
            {
                "_target_": "nuplan.planning.simulation.trajectory.trajectory_sampling.TrajectorySampling",
                "num_poses": 40,
                "interval_length": 0.1,
            }
        )
        self.scorer = instantiate(cfg)
        self.simulator = PDMSimulator(self.sampling)
        self.traffic = LogReplayTrafficAgents(self.sampling)

    def load_cache(self, token):
        if token not in self.cache_index:
            raise FileNotFoundError(
                f"metric cache missing for {token}; run scripts/flow_grpo/cache.sh"
            )
        if token not in self.cache:
            with lzma.open(self.cache_index[token], "rb") as stream:
                cache = pickle.load(stream)
            required = {
                "human_trajectory",
                "past_human_trajectory",
                "map_parameters",
                "future_tracked_objects",
            }
            if not required <= vars(cache).keys():
                raise ValueError("cache is not NAVSIM v2")
            self.cache[token] = cache
            while len(self.cache) > self.capacity:
                self.cache.popitem(last=False)
        self.cache.move_to_end(token)
        return self.cache[token]

    def score(self, token, trajectory, candidate=0):
        from navsim.common.dataclasses import Trajectory
        from navsim.evaluate.pdm_score import pdm_score
        from navsim.planning.script.run_pdm_score_one_stage import compute_final_scores
        from nuplan.planning.simulation.trajectory.trajectory_sampling import (
            TrajectorySampling,
        )

        trajectory = np.asarray(trajectory, dtype=np.float64)
        if trajectory.shape != (8, 3) or not np.isfinite(trajectory).all():
            raise ValueError("trajectory must be finite ego-frame [8,3] at 0.5s")
        cache = self.load_cache(token)
        row, _ = pdm_score(
            cache,
            Trajectory(
                trajectory, TrajectorySampling(num_poses=8, interval_length=0.5)
            ),
            self.sampling,
            self.simulator,
            self.scorer,
            self.traffic,
        )
        # Single logged observation has no adjacent predicted frame. The official
        # finalizer removes that unavailable metric's weight, exactly as evaluator.
        row["two_frame_extended_comfort"] = np.nan
        final = compute_final_scores(row)
        score = float(final.iloc[0]["score"])
        if not np.isfinite(score):
            raise FloatingPointError("official evaluator returned nonfinite score")
        metrics = {}
        for key, value in final.iloc[0].items():
            if np.isscalar(value):
                metrics[key] = (
                    None
                    if isinstance(value, (float, np.floating)) and np.isnan(value)
                    else float(value)
                )
        return ScoreResult(token, candidate, score, metrics)


_WORKER = None


def worker_init(devkit, index, threads):
    global _WORKER
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[key] = str(threads)
    import torch

    torch.set_num_threads(threads)
    _WORKER = OfficialEvaluator(devkit, index)


def worker_score(job):
    token, trajectories = job
    return [
        _WORKER.score(token, trajectory, i) for i, trajectory in enumerate(trajectories)
    ]


def build_cache_index(cache_root):
    root = Path(cache_root)
    if not root.is_dir():
        raise FileNotFoundError(
            f"missing cache directory {root}; run scripts/flow_grpo/cache.sh"
        )
    index = {p.parent.name: str(p) for p in root.glob("*/*/*/metric_cache.pkl")}
    if not index:
        raise FileNotFoundError(f"no official metric caches in {root}")
    return index


class RewardService:
    def __init__(
        self,
        devkit,
        cache_root,
        split,
        allowed_tokens,
        workers=2,
        timeout=180,
        cache_dir=None,
        threads=1,
    ):
        if split not in ("train", "validation"):
            raise ValueError(
                "RL reward accepts only train/validation; navtest is forbidden"
            )
        if workers < 0 or workers > 32:
            raise ValueError("reward workers must be bounded in [0,32]")
        self.index = build_cache_index(cache_root)
        self.allowed = set(allowed_tokens)
        self.timeout = timeout
        missing = self.allowed - self.index.keys()
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} scenes lack metric cache, e.g. {next(iter(missing))}; run cache.sh"
            )
        self.metadata = dict(
            version=VERSION,
            split=split,
            trajectory="rear_axle_ego_x_y_heading",
            horizon=8,
            interval=0.5,
            protocol_hash=hashlib.sha256(
                (Path(devkit) / "navsim/evaluate/pdm_score.py").read_bytes()
            ).hexdigest(),
        )
        protocol_files = [
            "navsim/evaluate/pdm_score.py",
            "navsim/planning/script/run_pdm_score_one_stage.py",
            "navsim/planning/simulation/planner/pdm_planner/scoring/pdm_scorer.py",
            "navsim/planning/simulation/planner/pdm_planner/simulation/pdm_simulator.py",
            "navsim/planning/script/config/pdm_scoring/scorer/pdm_scorer.yaml",
            "navsim/traffic_agents_policies/log_replay_traffic_agents.py",
        ]
        self.metadata["evaluator_files"] = {
            name: hashlib.sha256((Path(devkit) / name).read_bytes()).hexdigest()
            for name in protocol_files
        }
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.pool = (
            ProcessPoolExecutor(
                max_workers=workers,
                mp_context=mp.get_context("spawn"),
                initializer=worker_init,
                initargs=(str(devkit), self.index, threads),
            )
            if workers
            else None
        )
        self.serial = OfficialEvaluator(devkit, self.index) if not workers else None
        self.errors = 0
        self.calls = 0

    def score(self, tokens, trajectories):
        trajectories = np.asarray(trajectories)
        if any(t not in self.allowed for t in tokens):
            raise ValueError("scene outside declared reward split")
        if trajectories.shape[:1] != (len(tokens),):
            raise ValueError("token/trajectory batch mismatch")
        jobs = []
        cached = {}
        futures = {}
        for i, (token, group) in enumerate(zip(tokens, trajectories)):
            # Cache metric file identity and exact physical bytes; never reward=0 as miss.
            stat = Path(self.index[token]).stat()
            key = digest(
                dict(
                    self.metadata,
                    token=token,
                    metric_file=[stat.st_size, stat.st_mtime_ns],
                    trajectory_sha=hashlib.sha256(group.tobytes()).hexdigest(),
                )
            )
            path = self.cache_dir / f"{key}.json" if self.cache_dir else None
            jobs.append((i, token, group, path))
            if path and path.exists():
                cached[i] = [
                    ScoreResult(**x) for x in json.loads(path.read_text())["results"]
                ]
            elif self.pool:
                futures[i] = self.pool.submit(worker_score, (token, group))
        results = []
        try:
            deadline = time.monotonic() + self.timeout
            for i, token, group, path in jobs:
                rows = cached.get(i)
                if rows is None:
                    rows = (
                        futures[i].result(
                            timeout=max(0.01, deadline - time.monotonic())
                        )
                        if self.pool
                        else [
                            self.serial.score(token, x, j) for j, x in enumerate(group)
                        ]
                    )
                    if path:
                        tmp = path.with_suffix(f".{os.getpid()}.tmp")
                        tmp.write_text(
                            json.dumps(
                                {
                                    "metadata": dict(self.metadata, token=token),
                                    "results": [vars(r) for r in rows],
                                },
                                allow_nan=False,
                            )
                        )
                        os.replace(tmp, path)
                results.append(rows)
            self.calls += sum(len(r) for r in results)
            return results
        except BaseException:
            self.errors += 1
            self.close(abort=True)
            raise

    def close(self, abort=False):
        if self.pool:
            if abort:
                # Only our own bounded scoring children; never external jobs.
                processes = list((self.pool._processes or {}).values())
                for process in processes:
                    process.terminate()
                deadline = time.monotonic() + 2
                for process in processes:
                    process.join(timeout=max(0, deadline - time.monotonic()))
                    if process.is_alive():
                        process.kill()
            self.pool.shutdown(wait=not abort, cancel_futures=True)
            self.pool = None

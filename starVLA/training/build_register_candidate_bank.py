#!/usr/bin/env python3
"""Stage B: generate Register64 once, score each full pool once, and write LMDB."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import subprocess
import sys
import time
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from accelerate import Accelerator
from accelerate.utils import DataLoaderConfiguration, set_seed
from omegaconf import OmegaConf

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from starVLA.candidate_bank import (
    CandidateBankBuildIdentity,
    CandidateBankReader,
    CandidateBankWriter,
    build_identity_hash,
    finalize_candidate_bank,
    prepare_candidate_bank_root,
    read_candidate_bank_build_identity,
)
from starVLA.candidate_bank.schema import CANDIDATE_METRICS
from starVLA.model.modules.planning.trajectory_codec import TrajectoryCodec
from starVLA.model.modules.register_planner.checkpoint import (
    load_register_generator_checkpoint,
    sha256_file,
)
from starVLA.training.config_loader import load_training_config
from starVLA.training.register_stage_utils import atomic_json


def _commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        cwd=Path(__file__).resolve().parents[2],
    ).stdout.strip()


class CandidateBankReport:
    def __init__(self, proposal_num: int) -> None:
        self.proposal_num = int(proposal_num)
        self.oracle_scores: list[float] = []
        self.single_scores: list[float] = []
        self.random_scores: list[float] = []
        self.feasible = 0
        self.candidates = 0
        self.winner_histogram = torch.zeros(proposal_num, dtype=torch.long)
        self.pairwise_ade_sum = 0.0
        self.pairwise_fde_sum = 0.0
        self.finite = {name: 0 for name in CANDIDATE_METRICS}
        self.metric_values = {name: 0 for name in CANDIDATE_METRICS}
        self.wall_time_seconds = 0.0

    @classmethod
    def from_payload(
        cls, payload: Mapping[str, Any], proposal_num: int
    ) -> "CandidateBankReport":
        report = cls(proposal_num)
        report.oracle_scores = list(payload["oracle_scores"])
        report.single_scores = list(payload["single_scores"])
        report.random_scores = list(payload.get("random_scores", []))
        report.feasible = int(payload["feasible"])
        report.candidates = int(payload["candidates"])
        report.winner_histogram = torch.tensor(
            payload["winner_histogram"], dtype=torch.long
        )
        report.pairwise_ade_sum = float(payload["pairwise_ade_sum"])
        report.pairwise_fde_sum = float(payload["pairwise_fde_sum"])
        report.finite = {name: int(payload["finite"][name]) for name in CANDIDATE_METRICS}
        report.metric_values = {
            name: int(payload["metric_values"][name])
            for name in CANDIDATE_METRICS
        }
        report.wall_time_seconds = float(payload.get("wall_time", 0.0))
        return report

    @torch.no_grad()
    def update(
        self,
        proposals: torch.Tensor,
        gt_trajectory: torch.Tensor,
        metrics: Mapping[str, torch.Tensor],
        tokens: Sequence[str] | None = None,
    ) -> None:
        proposals = proposals.float().cpu()
        gt_trajectory = gt_trajectory.float().cpu()
        metrics = {name: value.float().cpu() for name, value in metrics.items()}
        aggregate = metrics["aggregate_score"]
        self.oracle_scores.extend(aggregate.max(dim=1).values.tolist())
        self.single_scores.extend(aggregate[:, 0].tolist())
        if tokens is None:
            random_index = torch.arange(aggregate.shape[0]) % self.proposal_num
        else:
            random_index = torch.tensor(
                [
                    int.from_bytes(
                        hashlib.sha256(str(token).encode("utf-8")).digest()[:8],
                        byteorder="little",
                    )
                    % self.proposal_num
                    for token in tokens
                ],
                dtype=torch.long,
            )
        self.random_scores.extend(
            aggregate[torch.arange(aggregate.shape[0]), random_index].tolist()
        )
        feasible = (
            (metrics["no_at_fault_collisions"] > 0.5)
            & (metrics["drivable_area_compliance"] > 0.5)
        )
        self.feasible += int(feasible.sum())
        self.candidates += int(feasible.numel())
        l1 = torch.linalg.vector_norm(
            proposals - gt_trajectory[:, None], ord=1, dim=-1
        ).mean(dim=-1)
        winner = l1.argmin(dim=1)
        self.winner_histogram += torch.bincount(
            winner, minlength=self.proposal_num
        )
        if self.proposal_num > 1:
            pairwise = torch.linalg.vector_norm(
                proposals[:, :, None, :, :2]
                - proposals[:, None, :, :, :2],
                ord=2,
                dim=-1,
            )
            rows, cols = torch.triu_indices(
                self.proposal_num, self.proposal_num, offset=1
            )
            self.pairwise_ade_sum += float(
                pairwise[:, rows, cols].mean(dim=(-1, -2)).sum()
            )
            self.pairwise_fde_sum += float(
                pairwise[:, rows, cols, -1].mean(dim=-1).sum()
            )
        for name in CANDIDATE_METRICS:
            self.finite[name] += int(torch.isfinite(metrics[name]).sum())
            self.metric_values[name] += int(metrics[name].numel())

    def payload(self, *, wall_time: float) -> dict[str, Any]:
        return {
            "oracle_scores": self.oracle_scores,
            "single_scores": self.single_scores,
            "random_scores": self.random_scores,
            "feasible": self.feasible,
            "candidates": self.candidates,
            "winner_histogram": self.winner_histogram.tolist(),
            "pairwise_ade_sum": self.pairwise_ade_sum,
            "pairwise_fde_sum": self.pairwise_fde_sum,
            "finite": self.finite,
            "metric_values": self.metric_values,
            "wall_time": self.wall_time_seconds + float(wall_time),
        }


def _records_from_batch(
    examples: Sequence[Mapping[str, Any]],
    scene,
    proposals: torch.Tensor,
    ego_state: torch.Tensor,
    metrics: Mapping[str, torch.Tensor],
    *,
    include_dense_memory: bool,
    storage_dtype: torch.dtype,
) -> list[dict[str, Any]]:
    codec = TrajectoryCodec()
    actions = torch.as_tensor(
        np.asarray([example["action"] for example in examples]),
        device=proposals.device,
        dtype=torch.float32,
    )
    gt = codec.flow_to_navsim(actions)
    records = []
    for index, example in enumerate(examples):
        record: dict[str, Any] = {
            "token": str(example["token"]),
            "ego_state": ego_state[index].detach().float().cpu(),
            "scene_global_tokens": scene.global_tokens[index]
            .detach()
            .to(device="cpu", dtype=storage_dtype),
            "proposals": proposals[index]
            .detach()
            .to(device="cpu", dtype=storage_dtype),
            "gt_trajectory": gt[index].detach().float().cpu(),
            "metrics": {
                name: metrics[name][index].detach().float().cpu()
                for name in CANDIDATE_METRICS
            },
        }
        if include_dense_memory:
            record["scene_dense_memory"] = scene.dense_memory[index]
            record["scene_dense_memory"] = record["scene_dense_memory"].detach().to(
                device="cpu", dtype=storage_dtype
            )
            padding = scene.memory_key_padding_mask
            record["attention_mask"] = (
                torch.ones(
                    scene.dense_memory.shape[1], dtype=torch.bool
                )
                if padding is None
                else (~padding[index]).detach().cpu()
            )
        records.append(record)
    return records


@torch.no_grad()
def score_and_write_candidate_batch(
    *,
    examples,
    scene,
    generator_output,
    ego_state,
    metric_supervisor,
    writer: CandidateBankWriter,
    report: CandidateBankReport,
    include_dense_memory: bool = False,
    storage_dtype: torch.dtype = torch.float16,
) -> None:
    """Testable synchronous path; the supervisor sees one complete [B,64] pool."""

    tokens = [str(example["token"]) for example in examples]
    metrics = metric_supervisor.score(tokens, generator_output.proposals.float())
    gt = TrajectoryCodec().flow_to_navsim(
        torch.as_tensor(
            np.asarray([example["action"] for example in examples]),
            device=generator_output.proposals.device,
            dtype=torch.float32,
        )
    )
    report.update(generator_output.proposals, gt, metrics, tokens=tokens)
    for record in _records_from_batch(
        examples,
        scene,
        generator_output.proposals,
        ego_state,
        metrics,
        include_dense_memory=include_dense_memory,
        storage_dtype=storage_dtype,
    ):
        writer.put(record)


def _rank_report_path(root: Path, rank: int) -> Path:
    return root / f"rank_{rank:05d}.report.json"


def _aggregate_reports(root: Path, world_size: int, bank_size: int) -> dict[str, Any]:
    payloads = []
    for rank in range(world_size):
        path = _rank_report_path(root, rank)
        if not path.is_file():
            raise FileNotFoundError(f"candidate bank rank report is missing: {path}")
        with path.open("r", encoding="utf-8") as stream:
            payloads.append(json.load(stream))
    oracle = [value for payload in payloads for value in payload["oracle_scores"]]
    single = [value for payload in payloads for value in payload["single_scores"]]
    random_scores = [
        value for payload in payloads for value in payload.get("random_scores", [])
    ]
    histogram = torch.tensor(
        [payload["winner_histogram"] for payload in payloads], dtype=torch.long
    ).sum(dim=0)
    scenes = len(oracle)
    active = float((histogram > 0).float().mean()) if histogram.numel() else 0.0
    finite = {}
    for name in CANDIDATE_METRICS:
        numerator = sum(payload["finite"][name] for payload in payloads)
        denominator = sum(payload["metric_values"][name] for payload in payloads)
        finite[name] = numerator / max(denominator, 1)
    wall_time = max(float(payload["wall_time"]) for payload in payloads)
    oracle_mean = statistics.fmean(oracle) if oracle else float("nan")
    single_mean = statistics.fmean(single) if single else float("nan")
    return {
        "number_of_scenes": scenes,
        "mean_oracle_at_64": oracle_mean,
        "median_oracle_at_64": statistics.median(oracle) if oracle else float("nan"),
        "mean_single_register_score": single_mean,
        "mean_random_register_score": (
            statistics.fmean(random_scores) if random_scores else float("nan")
        ),
        "oracle_gain": oracle_mean - single_mean,
        "feasible_candidate_rate": sum(p["feasible"] for p in payloads)
        / max(sum(p["candidates"] for p in payloads), 1),
        "register_winner_histogram": histogram.tolist(),
        "active_register_ratio": active,
        "pairwise_ade": sum(p["pairwise_ade_sum"] for p in payloads)
        / max(scenes, 1),
        "pairwise_fde": sum(p["pairwise_fde_sum"] for p in payloads)
        / max(scenes, 1),
        "metric_finite_statistics": finite,
        "bank_size_bytes": int(bank_size),
        "scenes_per_second": scenes / max(wall_time, 1e-9),
        "total_wall_time_seconds": wall_time,
    }


def _validate_bank(root: Path) -> None:
    reader = CandidateBankReader(root, strict=True)
    try:
        for token in reader.tokens():
            reader.get(token)
    finally:
        reader.close()
    print(f"validated candidate bank: {root} ({len(reader)} scenes)")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", choices=("train", "val"), required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--include-dense-memory", action="store_true")
    parser.add_argument("--workers-per-rank", type=int)
    parser.add_argument("--backend", choices=("local", "thread", "process"))
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.resume and args.overwrite:
        raise ValueError("--resume and --overwrite are mutually exclusive")
    config = load_training_config(args.config)
    bank_config = config.candidate_bank
    root = Path(str(bank_config.output_root)) / args.split
    if args.validate_only:
        _validate_bank(root)
        return
    checkpoint_path = Path(str(bank_config.generator_checkpoint))
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"generator component checkpoint is missing: {checkpoint_path}")
    include_dense = bool(bank_config.get("include_dense_memory", False)) or bool(
        args.include_dense_memory
    )
    label_protocol = str(
        bank_config.get("label_protocol", "navsim_v2_epdms")
    )
    split_config = bank_config.get("splits", {}).get(args.split, {})
    config.datasets.vla_data.split = str(
        split_config.get("dataset_split", args.split)
    )
    if split_config.get("datalist_path"):
        config.datasets.vla_data.datalist_path = split_config.datalist_path
    # Bank ownership and resume identity require a stable sample order.
    config.datasets.vla_data.shuffle = False
    config.output_dir = str(root)
    accelerator = Accelerator(
        mixed_precision=str(config.get("precision", "bf16")),
        dataloader_config=DataLoaderConfiguration(even_batches=False),
    )
    set_seed(int(config.get("seed", 42)), device_specific=True)
    from starVLA.dataloader import build_dataloader
    from starVLA.model.framework import build_framework
    from starVLA.training.navsim_metric_supervisor import DynamicMetricSupervisor

    dataloader = build_dataloader(
        cfg=config, dataset_py=config.datasets.vla_data.dataset_py
    )
    model = build_framework(config)
    if model.__class__.__name__ != "QwenRegisterGenerator":
        raise RuntimeError("candidate-bank construction requires QwenRegisterGenerator")
    metadata = load_register_generator_checkpoint(
        checkpoint_path,
        qwen_vl_interface=model.qwen_vl_interface,
        action_input_model=model.action_input_model,
        scene_encoder=model.scene_encoder,
        register_generator=model.register_generator,
        expected_metadata={
            "qwen_base_model": str(config.framework.qwenvl.base_vlm),
            "proposal_num": 64,
            "num_poses": 8,
            "state_dim": 3,
            "scene_queries": 16,
            "scene_dim": 256,
            "decoder_layers": 4,
            "decoder_heads": 1,
            "proposal_head_style": "donor_mlp_v1",
            "stage_loss_mode": str(model.register_generator.stage_loss_mode),
            "proposal_head_count": int(
                model.register_generator.proposal_head_count
            ),
        },
    )
    # Bank ranks may receive different final batch counts. Avoid DDP wrapping
    # for this inference-only pass, and do not pad the dataloader with duplicate
    # scenes merely to equalize ranks.
    model = accelerator.prepare_model(model, evaluation_mode=True)
    dataloader = accelerator.prepare_data_loader(dataloader)
    model.eval()
    supervisor_config = OmegaConf.create(
        OmegaConf.to_container(bank_config.metric_supervisor, resolve=True)
    )
    configured_supervisor_protocol = str(
        supervisor_config.get("protocol", label_protocol)
    )
    if configured_supervisor_protocol != label_protocol:
        raise ValueError(
            "candidate-bank label_protocol differs from metric supervisor protocol"
        )
    supervisor_config.protocol = label_protocol
    if args.workers_per_rank is not None:
        supervisor_config.workers_per_rank = args.workers_per_rank
    if args.backend is not None:
        supervisor_config.backend = args.backend
    storage_dtype_name = str(bank_config.get("storage_dtype", "float16"))
    storage_dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[storage_dtype_name]
    checkpoint_sha256 = sha256_file(checkpoint_path)
    repository_commit = _commit()
    build_identity = CandidateBankBuildIdentity(
        split=args.split,
        world_size=accelerator.num_processes,
        proposal_num=64,
        generator_checkpoint_sha256=checkpoint_sha256,
        generator_config_hash=str(metadata["config_hash"]),
        repository_commit=repository_commit,
        metric_cache_root=str(supervisor_config.metric_cache_root),
        datalist_path=str(config.datasets.vla_data.datalist_path),
        scene_dim=256,
        scene_queries=16,
        include_dense_memory=include_dense,
        storage_dtype=storage_dtype_name,
        label_protocol=label_protocol,
    )
    expected_identity_hash = build_identity_hash(build_identity)
    if accelerator.is_main_process:
        prepared_hash = prepare_candidate_bank_root(
            root,
            identity=build_identity,
            resume=args.resume,
            overwrite=args.overwrite,
        )
        if prepared_hash != expected_identity_hash:
            raise AssertionError("candidate-bank identity hash is not deterministic")
    accelerator.wait_for_everyone()
    stored_identity = read_candidate_bank_build_identity(root)
    if stored_identity != build_identity:
        raise RuntimeError("candidate-bank identity differs across distributed ranks")
    supervisor = DynamicMetricSupervisor(
        supervisor_config,
        rank=accelerator.process_index,
        world_size=accelerator.num_processes,
    )
    prior_report_path = _rank_report_path(root, accelerator.process_index)
    can_skip_existing = bool(args.resume and prior_report_path.is_file())
    if can_skip_existing:
        with prior_report_path.open("r", encoding="utf-8") as stream:
            prior_payload = json.load(stream)
        can_skip_existing = "random_scores" in prior_payload
        report = (
            CandidateBankReport.from_payload(prior_payload, 64)
            if can_skip_existing
            else CandidateBankReport(proposal_num=64)
        )
    else:
        report = CandidateBankReport(proposal_num=64)
    start = time.perf_counter()
    pending = deque()
    max_inflight = max(1, int(bank_config.get("max_inflight_batches", 2)))
    writer = CandidateBankWriter(
        root,
        rank=accelerator.process_index,
        proposal_num=64,
        scene_queries=16,
        scene_dim=256,
        include_dense_memory=include_dense,
        map_size_bytes=int(bank_config.get("map_size_bytes", 1 << 40)),
        commit_interval=int(bank_config.get("commit_interval", 16)),
        resume=args.resume,
        overwrite=args.overwrite,
        expected_build_identity_hash=expected_identity_hash,
    )

    def flush_one() -> None:
        examples, scene, proposals, ego_state, future = pending.popleft()
        metrics = future.result()
        gt = TrajectoryCodec().flow_to_navsim(
            torch.as_tensor(
                np.asarray([example["action"] for example in examples]),
                device=proposals.device,
                dtype=torch.float32,
            )
        )
        report.update(
            proposals,
            gt,
            metrics,
            tokens=[str(example["token"]) for example in examples],
        )
        for record in _records_from_batch(
            examples,
            scene,
            proposals,
            ego_state,
            metrics,
            include_dense_memory=include_dense,
            storage_dtype=storage_dtype,
        ):
            writer.put(record)

    try:
        with torch.inference_mode():
            for examples in dataloader:
                if can_skip_existing:
                    examples = [
                        example
                        for example in examples
                        if not writer.contains(str(example["token"]))
                    ]
                    if not examples:
                        continue
                generated_batch = model(examples, generate_only=True)
                scene = generated_batch["scene_context"]
                generated = generated_batch["generator_output"]
                ego_state = generated_batch["ego_state"]
                tokens = [str(example["token"]) for example in examples]
                stored_proposals = generated.proposals.detach().to(
                    device="cpu", dtype=torch.float32
                )
                # Exactly one supervisor submission contains the complete K=64 pool.
                # Submit the same CPU snapshot that will be written to LMDB so
                # proposals cross the accelerator/host boundary only once and
                # metric results remain on CPU.
                future = supervisor.score_async(tokens, stored_proposals)
                record_examples = [
                    {"token": example["token"], "action": example["action"]}
                    for example in examples
                ]
                stored_scene = SimpleNamespace(
                    global_tokens=scene.global_tokens.detach().cpu(),
                    dense_memory=(
                        scene.dense_memory.detach().cpu()
                        if include_dense
                        else None
                    ),
                    memory_key_padding_mask=(
                        None
                        if scene.memory_key_padding_mask is None
                        else scene.memory_key_padding_mask.detach().cpu()
                    ),
                )
                pending.append(
                    (
                        record_examples,
                        stored_scene,
                        stored_proposals,
                        ego_state.detach().cpu(),
                        future,
                    )
                )
                if len(pending) >= max_inflight:
                    flush_one()
            while pending:
                flush_one()
        writer.close(complete=True)
    except BaseException:
        writer.__exit__(*__import__("sys").exc_info())
        raise
    finally:
        supervisor.close()
    wall_time = time.perf_counter() - start
    atomic_json(
        _rank_report_path(root, accelerator.process_index),
        report.payload(wall_time=wall_time),
    )
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        manifest = finalize_candidate_bank(
            root,
            manifest_fields={
                "split": args.split,
                "proposal_num": 64,
                "generator_checkpoint_sha256": checkpoint_sha256,
                "generator_config_hash": str(metadata["config_hash"]),
                "repository_commit": repository_commit,
                "metric_cache_root": str(supervisor_config.metric_cache_root),
                "scene_dim": 256,
                "scene_queries": 16,
                "include_dense_memory": include_dense,
                "label_protocol": label_protocol,
            },
            world_size=accelerator.num_processes,
            expected_build_identity_hash=expected_identity_hash,
        )
        size = sum(
            path.stat().st_size for path in root.rglob("*") if path.is_file()
        )
        final_report = _aggregate_reports(
            root, accelerator.num_processes, bank_size=size
        )
        final_report["manifest_num_scenes"] = manifest.num_scenes
        atomic_json(root / "candidate_bank_report.json", final_report)
        print(json.dumps(final_report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

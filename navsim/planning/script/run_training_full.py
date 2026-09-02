import os
import random
import multiprocessing as mp
from typing import Any, Dict, Iterable, Optional, Tuple
from pathlib import Path
import logging
import pickle
import hashlib
import json
import math
from datetime import datetime

import hydra
import numpy as np
from hydra.utils import instantiate
from omegaconf import DictConfig, open_dict
from torch.utils.data import DataLoader, ConcatDataset, Subset
import torch.distributed as dist
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import SceneFilter
from navsim.common.dataloader import SceneLoader
from navsim.planning.training.dataset import (
    CacheOnlyDataset,
    Dataset,
    drivevla_cached_collate,
)
from navsim.planning.training.agent_lightning_module import AgentLightningModule
from navsim.planning.training.formal_throughput import (
    FormalThroughputBenchmarkCallback,
)

logger = logging.getLogger(__name__)

CONFIG_PATH = "config/training"
CONFIG_NAME = "default_training"


def _parse_cpu_list(cpu_list: str):
    cpus = []
    for item in cpu_list.strip().split(","):
        if not item:
            continue
        if "-" in item:
            start, end = (int(value) for value in item.split("-", 1))
            cpus.extend(range(start, end + 1))
        else:
            cpus.append(int(item))
    return cpus


def _bind_rank_to_local_cpus(devices: int) -> None:
    """Bind each local GPU rank and its children to NUMA-local CPU cores."""
    if os.getenv("DRIVEVLA_BIND_RANK_CPUS", "0").lower() not in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return
    if not hasattr(os, "sched_setaffinity") or devices <= 0:
        return

    node_paths = sorted(Path("/sys/devices/system/node").glob("node[0-9]*"))
    if not node_paths:
        return
    local_rank = int(os.getenv("LOCAL_RANK", "0"))

    # Divide ranks over NUMA nodes, then divide physical-core sibling groups
    # within the selected node.  Keeping SMT siblings together avoids assigning
    # the two threads of one core to different GPU ranks.
    rank_counts = [devices // len(node_paths)] * len(node_paths)
    for index in range(devices % len(node_paths)):
        rank_counts[index] += 1
    rank_start = 0
    node_index = 0
    for index, count in enumerate(rank_counts):
        if rank_start <= local_rank < rank_start + count:
            node_index = index
            break
        rank_start += count
    rank_within_node = local_rank - rank_start
    ranks_on_node = rank_counts[node_index]

    node_cpus = set(
        _parse_cpu_list((node_paths[node_index] / "cpulist").read_text())
    )
    sibling_groups = []
    seen = set()
    for cpu in sorted(node_cpus):
        siblings_path = Path(
            f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list"
        )
        siblings = tuple(
            value
            for value in _parse_cpu_list(siblings_path.read_text())
            if value in node_cpus
        )
        if siblings not in seen:
            sibling_groups.append(siblings)
            seen.add(siblings)

    assigned_groups = sibling_groups[rank_within_node::ranks_on_node]
    assigned_cpus = {cpu for group in assigned_groups for cpu in group}
    if assigned_cpus:
        os.sched_setaffinity(0, assigned_cpus)
        logger.info(
            "Bound local rank %d to NUMA node %d CPUs %s",
            local_rank,
            node_index,
            sorted(assigned_cpus),
        )


def _configure_forkserver_preload(cfg: DictConfig) -> None:
    """Prepare one clean, shared CPU forkserver before any CUDA-backed worker."""
    dataloader_context = cfg.dataloader.params.get("multiprocessing_context", None)
    scorer_context = os.getenv("DRIVEVLA_SCORE_START_METHOD", "spawn")
    if dataloader_context != "forkserver" and scorer_context != "forkserver":
        return
    mp.set_forkserver_preload(
        [
            "navsim.planning.training.dataset",
            "navsim.agents.EpisodeDrive.utils.internvl_preprocess",
            "navsim.agents.EpisodeDrive.score_module.compute_navsim_score",
        ]
    )


def _pad_dataset_to_multiple(dataset, multiple: int, name: str):
    """Pad a dataset deterministically so DDP never drops a partial global batch."""
    if multiple <= 0:
        raise ValueError(f"Dataset padding multiple must be positive, got {multiple}")

    remainder = len(dataset) % multiple
    if remainder == 0:
        logger.info("%s dataset already divisible by global batch %d", name, multiple)
        return dataset

    pad_count = multiple - remainder
    if len(dataset) < pad_count:
        raise ValueError(
            f"Cannot pad {name} dataset of length {len(dataset)} with {pad_count} samples"
        )

    logger.info(
        "Padding %s dataset from %d to %d samples for global batch %d",
        name,
        len(dataset),
        len(dataset) + pad_count,
        multiple,
    )
    return ConcatDataset([dataset, Subset(dataset, range(pad_count))])


def _configured_global_batch_size(cfg: DictConfig) -> int:
    devices = cfg.trainer.params.devices
    if not isinstance(devices, int) or devices <= 0:
        raise ValueError(
            "pad_datasets_to_global_batch requires trainer.params.devices to be a positive integer"
        )
    num_nodes = int(cfg.trainer.params.get("num_nodes", 1))
    per_device_batch = int(cfg.dataloader.params.batch_size)
    return devices * num_nodes * per_device_batch

def dist_ready():
    return dist.is_available() and dist.is_initialized()


def preinit_global_zero() -> bool:
    """Identify global rank zero before Lightning initializes torch.distributed."""
    if dist_ready():
        return dist.get_rank() == 0
    return int(os.getenv("RANK", os.getenv("LOCAL_RANK", "0"))) == 0


def _ordered_unique_log_names(log_names: Iterable[str]) -> list[str]:
    """Return stable, duplicate-free log names as plain strings."""
    return list(dict.fromkeys(str(log_name) for log_name in log_names))


def _sha256_log_set(log_names: Iterable[str]) -> str:
    """Hash a log set independently of Hydra list order."""
    payload = "\n".join(sorted(set(str(name) for name in log_names))).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def resolve_data_protocol(
    cfg: DictConfig,
    *,
    scene_filter_log_names: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """Resolve and strictly audit the train/validation log contract."""
    protocol_cfg = cfg.get("data_protocol", {})
    include_val = bool(protocol_cfg.get("include_val_in_train", False))
    require_disjoint = bool(protocol_cfg.get("require_disjoint_train_val", True))

    train_logs = _ordered_unique_log_names(cfg.train_logs)
    val_logs = _ordered_unique_log_names(cfg.val_logs)
    if scene_filter_log_names is not None:
        allowed = set(_ordered_unique_log_names(scene_filter_log_names))
        train_logs = [name for name in train_logs if name in allowed]
        val_logs = [name for name in val_logs if name in allowed]

    overlap = sorted(set(train_logs).intersection(val_logs))
    if require_disjoint and overlap:
        preview = ", ".join(overlap[:10])
        raise RuntimeError(
            "Train/validation log overlap is forbidden by "
            "data_protocol.require_disjoint_train_val=true: "
            f"count={len(overlap)}, first_logs=[{preview}]"
        )

    if include_val:
        if bool(cfg.get("validation_run", False)):
            raise RuntimeError(
                "data_protocol.include_val_in_train=true cannot be combined "
                "with validation_run=true"
            )
        limit_val_batches = cfg.trainer.params.get("limit_val_batches", 1.0)
        if float(limit_val_batches) != 0.0:
            raise RuntimeError(
                "data_protocol.include_val_in_train=true is final-fit mode and "
                "requires trainer.params.limit_val_batches=0"
            )

    effective_train_logs = _ordered_unique_log_names(
        [*train_logs, *(val_logs if include_val else [])]
    )
    audit = {
        "mode": "final_fit" if include_val else "train_with_validation",
        "include_val_in_train": include_val,
        "require_disjoint_train_val": require_disjoint,
        "hyperparameter_selection_allowed": not include_val,
        "best_checkpoint_allowed": not include_val,
        "train_log_count": len(train_logs),
        "val_log_count": len(val_logs),
        "effective_train_log_count": len(effective_train_logs),
        "overlap_count": len(overlap),
        "overlap_logs": overlap,
        "train_logs_sha256": _sha256_log_set(train_logs),
        "val_logs_sha256": _sha256_log_set(val_logs),
        "effective_train_logs_sha256": _sha256_log_set(effective_train_logs),
        "train_logs": train_logs,
        "val_logs": val_logs,
        "effective_train_logs": effective_train_logs,
    }
    logger.info(
        "TRAIN_VAL_PROTOCOL mode=%s train_log_count=%d val_log_count=%d "
        "overlap_count=%d train_sha256=%s val_sha256=%s",
        audit["mode"],
        audit["train_log_count"],
        audit["val_log_count"],
        audit["overlap_count"],
        audit["train_logs_sha256"],
        audit["val_logs_sha256"],
    )
    if include_val:
        logger.warning(
            "FINAL-FIT MODE: validation is included in training; validation, "
            "best-checkpoint selection, and hyperparameter selection are disabled"
        )
    return audit


def write_data_protocol_metadata(cfg: DictConfig, audit: Dict[str, Any]) -> Path:
    """Persist the exact split audit in the run metadata directory."""
    metadata_dir = Path(str(cfg.output_dir)) / "run_metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    output_path = metadata_dir / "train_val_protocol.json"
    temporary_path = metadata_dir / ".train_val_protocol.json.tmp"
    temporary_path.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary_path.replace(output_path)
    return output_path


class FormalEpochCheckpointCallback(pl.Callback):
    """Save predetermined recovery checkpoints without validation selection."""

    MILESTONES = frozenset({5, 10, 15, 20, 25, 27})

    def __init__(self, output_dir: str):
        super().__init__()
        self.output_dir = str(Path(output_dir).expanduser().resolve())

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        completed_epoch = int(trainer.current_epoch) + 1
        if completed_epoch not in self.MILESTONES:
            return
        directory = Path(self.output_dir) / "checkpoints"
        if trainer.is_global_zero:
            directory.mkdir(parents=True, exist_ok=True)
        trainer.strategy.barrier()
        filename = (
            "epoch_27_final.ckpt"
            if completed_epoch == 27
            else f"epoch_{completed_epoch:02d}.ckpt"
        )
        trainer.save_checkpoint(str(directory / filename), weights_only=False)


def configure_callbacks_for_data_protocol(
    callbacks: Iterable[pl.Callback],
    audit: Dict[str, Any],
    *,
    output_dir: Optional[str] = None,
    formal_milestones: bool = False,
) -> list[pl.Callback]:
    """Enforce last-only checkpointing for final-fit runs."""
    callbacks = list(callbacks)
    if not audit["include_val_in_train"]:
        return callbacks
    callbacks = [
        callback for callback in callbacks if not isinstance(callback, ModelCheckpoint)
    ]
    callbacks.append(
        ModelCheckpoint(
            dirpath=(
                str(Path(output_dir).expanduser().resolve() / "checkpoints")
                if output_dir is not None
                else None
            ),
            monitor=None,
            save_top_k=0,
            save_last=True,
            save_on_train_epoch_end=True,
        )
    )
    if formal_milestones:
        if output_dir is None:
            raise ValueError("Formal milestone checkpoints require output_dir")
        callbacks.append(FormalEpochCheckpointCallback(output_dir))
    return callbacks


def configure_formal_throughput_callback(
    cfg: DictConfig, callbacks: Iterable[pl.Callback]
) -> list[pl.Callback]:
    """Add the bounded formal-layout benchmark when explicitly requested."""
    callbacks = list(callbacks)
    benchmark = cfg.get("throughput_benchmark", {})
    if not bool(benchmark.get("enabled", False)):
        return callbacks
    if not bool(cfg.get("formal_training", {}).get("enabled", False)):
        raise RuntimeError("Formal throughput benchmark requires formal_training.enabled")
    output_path = benchmark.get("output_path")
    layout_name = benchmark.get("layout_name")
    if not output_path or not layout_name:
        raise ValueError(
            "throughput_benchmark.output_path and layout_name are required"
        )
    if os.getenv("PLANREG_FORMAL_TIMING", "0").lower() not in {
        "1",
        "true",
        "yes",
        "on",
    }:
        raise RuntimeError(
            "Formal throughput benchmark requires PLANREG_FORMAL_TIMING=1 "
            "before agent construction"
        )
    callbacks.append(
        FormalThroughputBenchmarkCallback(
            str(output_path),
            global_batch_size=_configured_global_batch_size(cfg),
            warmup_steps=int(benchmark.get("warmup_steps", 20)),
            timed_steps=int(benchmark.get("timed_steps", 300)),
            layout_name=str(layout_name),
            scorer_processes_per_rank=int(
                benchmark.get("scorer_processes_per_rank", 0)
            ),
            num_workers=int(cfg.dataloader.params.num_workers),
        )
    )
    return callbacks


def compute_formal_step_budget(
    dataset_length: int,
    global_batch_size: int,
    *,
    dataset_epochs: int = 27,
) -> Dict[str, int]:
    if dataset_length <= 0 or global_batch_size <= 0 or dataset_epochs <= 0:
        raise ValueError("Dataset length, global batch, and epochs must be positive")
    steps_per_epoch = int(math.ceil(dataset_length / global_batch_size))
    padded_samples_per_epoch = steps_per_epoch * global_batch_size
    return {
        "dataset_length": int(dataset_length),
        "global_batch_size": int(global_batch_size),
        "dataset_epochs": int(dataset_epochs),
        "steps_per_epoch": steps_per_epoch,
        "total_steps": steps_per_epoch * int(dataset_epochs),
        "padded_samples_per_epoch": padded_samples_per_epoch,
        "sampler_padding_per_epoch": padded_samples_per_epoch - int(dataset_length),
        "unpadded_sample_exposures": int(dataset_length) * int(dataset_epochs),
        "optimizer_sample_slots": padded_samples_per_epoch * int(dataset_epochs),
    }


def configure_formal_step_budget(
    cfg: DictConfig, dataset_length: int
) -> Optional[Dict[str, Any]]:
    formal = cfg.get("formal_training", {})
    if not bool(formal.get("enabled", False)):
        return None
    if not bool(cfg.data_protocol.include_val_in_train):
        raise RuntimeError("Formal 103k protocol requires include_val_in_train=true")
    if float(cfg.trainer.params.limit_val_batches) != 0.0:
        raise RuntimeError("Formal 103k protocol prohibits internal validation")
    if bool(cfg.validation_run):
        raise RuntimeError("Formal 103k protocol cannot run validation mode")
    if bool(cfg.get("auto_resume", True)):
        raise RuntimeError("Formal runs prohibit automatic cross-experiment resume")
    if cfg.get("train_ckpt_path") is not None:
        raise RuntimeError(
            "Formal resolved config must keep train_ckpt_path=null; resume is "
            "accepted only through the explicitly validated RESUME_CHECKPOINT environment variable"
        )
    epochs = int(formal.get("dataset_epochs", 27))
    if epochs != 27:
        raise RuntimeError(f"Formal PlanReg training requires 27 epochs, got {epochs}")
    expected_size = int(formal.get("expected_dataset_size", 103288))
    if bool(formal.get("require_exact_dataset_size", True)) and dataset_length != expected_size:
        raise RuntimeError(
            "Formal trainval dataset size mismatch: "
            f"expected={expected_size}, actual={dataset_length}"
        )
    budget = compute_formal_step_budget(
        dataset_length,
        _configured_global_batch_size(cfg),
        dataset_epochs=epochs,
    )
    with open_dict(cfg):
        cfg.trainer.params.max_epochs = epochs
        cfg.trainer.params.max_steps = budget["total_steps"]
    if preinit_global_zero():
        metadata_dir = Path(str(cfg.output_dir)) / "run_metadata"
        metadata_dir.mkdir(parents=True, exist_ok=True)
        output = metadata_dir / "formal_step_budget.json"
        temporary = metadata_dir / ".formal_step_budget.json.tmp"
        temporary.write_text(
            json.dumps(budget, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        temporary.replace(output)
    logger.info(
        "FORMAL_STEP_BUDGET dataset=%d global_batch=%d steps_per_epoch=%d "
        "epochs=%d total_steps=%d sampler_padding=%d",
        budget["dataset_length"],
        budget["global_batch_size"],
        budget["steps_per_epoch"],
        budget["dataset_epochs"],
        budget["total_steps"],
        budget["sampler_padding_per_epoch"],
    )
    return budget


def combine_cached_train_data(train_data, val_data, audit: Dict[str, Any]):
    """Include validation cache entries only for explicit final-fit runs."""
    if audit["include_val_in_train"]:
        return ConcatDataset([train_data, val_data])
    return train_data


def build_datasets(
    cfg: DictConfig,
    agent: AbstractAgent,
    data_protocol: Optional[Dict[str, Any]] = None,
) -> Tuple[Dataset, Dataset]:
    """
    Builds training and validation datasets from omega config
    :param cfg: omegaconf dictionary
    :param agent: interface of agents in NAVSIM
    :return: tuple for training and validation dataset
    """
    
    logger.info("Training without cache-only dataset")
    train_scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    configured_filter_logs = train_scene_filter.log_names
    if data_protocol is None:
        data_protocol = resolve_data_protocol(
            cfg, scene_filter_log_names=configured_filter_logs
        )
    train_scene_filter.log_names = list(data_protocol["effective_train_logs"])
    logger.info("Train scene filter log count: %d", len(train_scene_filter.log_names))

    val_scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    val_scene_filter.log_names = list(data_protocol["val_logs"])

    data_path = Path(cfg.navsim_log_path)
    sensor_blobs_path = Path(cfg.sensor_blobs_path)

    train_scene_loader = SceneLoader(
        sensor_blobs_path=sensor_blobs_path,
        data_path=data_path,
        scene_filter=train_scene_filter,
        sensor_config=agent.get_sensor_config(),
        load_image_path=cfg.load_image_path,
    )

    val_scene_loader = SceneLoader(
        sensor_blobs_path=sensor_blobs_path,
        data_path=data_path,
        scene_filter=val_scene_filter,
        sensor_config=agent.get_sensor_config(),
        load_image_path=cfg.load_image_path,
    )

    train_data = Dataset(
        scene_loader=train_scene_loader,
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        cache_path=cfg.cache_path,
        force_cache_computation=cfg.force_cache_computation,
    )

    val_data = Dataset(
        scene_loader=val_scene_loader,
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        cache_path=cfg.cache_path,
        force_cache_computation=cfg.force_cache_computation,
    )

    return train_data, val_data


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main entrypoint for training an agent.
    :param cfg: omegaconf dictionary
    """

    _configure_forkserver_preload(cfg)

    devices = cfg.trainer.params.devices
    if isinstance(devices, int):
        _bind_rank_to_local_cpus(devices)

    pl.seed_everything(cfg.seed, workers=True)
    logger.info(f"Global Seed set to {cfg.seed}")

    logger.info(f"Path where all results are stored: {cfg.output_dir}")

    scene_filter_log_names = None
    if not cfg.use_cache_without_dataset:
        audit_scene_filter: SceneFilter = instantiate(
            cfg.train_test_split.scene_filter
        )
        scene_filter_log_names = audit_scene_filter.log_names
    data_protocol = resolve_data_protocol(
        cfg, scene_filter_log_names=scene_filter_log_names
    )
    if preinit_global_zero():
        metadata_path = write_data_protocol_metadata(cfg, data_protocol)
        logger.info("Wrote train/validation protocol metadata: %s", metadata_path)

    logger.info("Building Agent")
    agent: AbstractAgent = instantiate(cfg.agent)
    agent.initialize()
    if 'mem' in agent.name().lower() and agent._config.memory_type=="error" and agent._config.memory_mode=="waver":
        print("=========initialize=========")
        agent.initialize()
        print("========freeze action head=========")
        agent.freeze_action_head()
    
    elif 'mem' in agent.name().lower() and agent._config.memory_type=="error" and agent._config.memory_mode=="adapter":
        print("=========initialize=========")
        agent.initialize()
        print("========freeze action head=========")
        agent.freeze_non_adapter_head()

    logger.info("Building Lightning Module")
    lightning_module = AgentLightningModule(
        agent=agent,
        diagnostics=cfg.get("diagnostics", {}),
    )

    if cfg.use_cache_without_dataset:
        logger.info("Using cached data without building SceneLoader")
        assert (
            not cfg.force_cache_computation
        ), "force_cache_computation must be False when using cached data without building SceneLoader"
        assert (
            cfg.cache_path is not None
        ), "cache_path must be provided when using cached data without building SceneLoader"
        preprocess_images = bool(cfg.get("preprocess_images_in_workers", False))
        preprocess_image_dtype = str(cfg.get("preprocess_image_dtype", "bfloat16"))
        pretokenize_inputs = bool(cfg.get("pretokenize_inputs_in_workers", False))
        tokenizer = agent.backbone.tokenizer if pretokenize_inputs else None
        formal_input_only = bool(
            getattr(agent, "_dynamic_feature_cache_guard_enabled", False)
        )
        input_only_cache_name = (
            str(cfg.get("input_only_cache_name", "planreg_input_only"))
            if formal_input_only
            else None
        )
        train_data = CacheOnlyDataset(
            cache_path=cfg.cache_path,
            feature_builders=agent.get_feature_builders(),
            target_builders=agent.get_target_builders(),
            log_names=data_protocol["train_logs"],
            preprocess_images=preprocess_images,
            preprocess_image_dtype=preprocess_image_dtype,
            pretokenize_inputs=pretokenize_inputs,
            tokenizer=tokenizer,
            preprocess_future_images=bool(
                cfg.get("preprocess_future_images_in_workers", False)
            ),
            reject_dynamic_feature_keys=formal_input_only,
            input_only_cache_name=input_only_cache_name,
            require_input_only_manifest=formal_input_only,
        )
        val_data = CacheOnlyDataset(
            cache_path=cfg.cache_path,
            feature_builders=agent.get_feature_builders(),
            target_builders=agent.get_target_builders(),
            log_names=data_protocol["val_logs"],
            preprocess_images=preprocess_images,
            preprocess_image_dtype=preprocess_image_dtype,
            pretokenize_inputs=pretokenize_inputs,
            tokenizer=tokenizer,
            preprocess_future_images=bool(
                cfg.get("preprocess_future_images_in_workers", False)
            ),
            reject_dynamic_feature_keys=formal_input_only,
            input_only_cache_name=input_only_cache_name,
            require_input_only_manifest=formal_input_only,
        )


        logger.info("Building Datasets")

        train_data = combine_cached_train_data(train_data, val_data, data_protocol)
    else:
        logger.info("Building SceneLoader")
        train_data, val_data = build_datasets(cfg, agent, data_protocol)

    formal_step_budget = configure_formal_step_budget(cfg, len(train_data))

    if bool(cfg.get("pad_datasets_to_global_batch", False)):
        global_batch_size = _configured_global_batch_size(cfg)
        train_data = _pad_dataset_to_multiple(train_data, global_batch_size, "train")
        val_data = _pad_dataset_to_multiple(val_data, global_batch_size, "validation")

    logger.info("Building Datasets")
    collate_fn = (
        drivevla_cached_collate
        if cfg.use_cache_without_dataset
        and bool(cfg.get("preprocess_images_in_workers", False))
        else None
    )
    train_dataloader = DataLoader(
        train_data,
        **cfg.dataloader.params,
        shuffle=True,
        drop_last=True,
        collate_fn=collate_fn,
    )
    logger.info("Num training samples: %d", len(train_data))
    val_dataloader = DataLoader(
        val_data,
        **cfg.dataloader.params,
        shuffle=False,
        drop_last=True,
        collate_fn=collate_fn,
    )
    logger.info("Num validation samples: %d", len(val_data))

    logger.info("Building Trainer")

    # automatically resume training
    # find latest ckpt
    import glob
    def find_latest_checkpoint(search_pattern):
        # List all files matching the pattern
        list_of_files = glob.glob(search_pattern, recursive=True)
        # Find the file with the latest modification time
        if not list_of_files:
            return None
        latest_file = max(list_of_files, key=os.path.getmtime)
        return latest_file


    if cfg.train_ckpt_path is None and bool(cfg.get("auto_resume", True)):
        # Pattern to match all .ckpt files in the base_path recursively
        search_pattern = "/".join(str(cfg.output_dir).split("/")[:-1]) + "/*/lightning_logs/version_*/checkpoints/" + '*.ckpt'
        print("/".join(str(cfg.output_dir).split("/")[:-1]))
        print("search_pattern ", search_pattern)
        cfg.train_ckpt_path = find_latest_checkpoint(search_pattern)
        print("cfg.train_ckpt_path ", cfg.train_ckpt_path)

    callbacks = configure_callbacks_for_data_protocol(
        agent.get_training_callbacks(),
        data_protocol,
        output_dir=str(cfg.output_dir),
        formal_milestones=formal_step_budget is not None,
    )
    callbacks = configure_formal_throughput_callback(cfg, callbacks)
    trainer = pl.Trainer(**cfg.trainer.params, callbacks=callbacks)

    if cfg.validation_run:
        logger.info("Starting Validation")
        timestamp = datetime.now().strftime("%Y.%m.%d.%H.%M.%S")
        dump_root = os.path.join(os.getenv('SUBSCORE_PATH'), "navsim1_pdm_scores", cfg.experiment_name)
        os.makedirs(dump_root, exist_ok=True)
        dump_path = os.path.join(dump_root, f"{timestamp}.pkl")
        trainer.validate(
            model=lightning_module,
            dataloaders=[val_dataloader],
            ckpt_path=cfg.train_ckpt_path,
            verbose=True
        )
        logger.info("Running predictions to collect trajectories")
        predictions = trainer.predict(
            AgentLightningModule(
                agent=agent,
                for_viz=True,
                diagnostics=cfg.get("diagnostics", {}),
            ),
            val_dataloader,
            return_predictions=True
        )

        if dist_ready():
            dist.barrier()
        
        world_size = dist.get_world_size() if dist_ready() else 1
        all_predictions = [None for _ in range(world_size)]

        if dist_ready():
            dist.all_gather_object(all_predictions, predictions)
        else:
            all_predictions = [predictions]

        rank = dist.get_rank() if dist_ready() else 0
        if rank != 0:
            return None

        merged_predictions = {}
        for proc_prediction in all_predictions:
            for d in proc_prediction:
                merged_predictions.update(d)

        pickle.dump(predictions, open(dump_path, 'wb'))
    else:
        logger.info("Starting Training")
        fit_checkpoint = cfg.train_ckpt_path
        explicit_resume = os.getenv("RESUME_CHECKPOINT")
        if formal_step_budget is not None and explicit_resume:
            fit_checkpoint = str(Path(explicit_resume).expanduser().resolve())
            if not Path(fit_checkpoint).is_file():
                raise FileNotFoundError(
                    f"Explicit formal RESUME_CHECKPOINT does not exist: {fit_checkpoint}"
                )
        trainer.fit(
            model=lightning_module,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
            ckpt_path=fit_checkpoint
        )


if __name__ == "__main__":
    main()

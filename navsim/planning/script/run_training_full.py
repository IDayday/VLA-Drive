import os
import random
import multiprocessing as mp
from typing import Tuple
from pathlib import Path
import logging
import pickle
from datetime import datetime

import hydra
import numpy as np
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader, ConcatDataset, Subset
import torch.distributed as dist
import pytorch_lightning as pl

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import SceneFilter
from navsim.common.dataloader import SceneLoader
from navsim.planning.training.dataset import (
    CacheOnlyDataset,
    Dataset,
    drivevla_cached_collate,
)
from navsim.planning.training.agent_lightning_module import AgentLightningModule

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

def build_datasets(cfg: DictConfig, agent: AbstractAgent) -> Tuple[Dataset, Dataset]:
    """
    Builds training and validation datasets from omega config
    :param cfg: omegaconf dictionary
    :param agent: interface of agents in NAVSIM
    :return: tuple for training and validation dataset
    """
    
    print("Train without caching....")
    train_scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    if train_scene_filter.log_names is not None:
        train_scene_filter.log_names = [
            log_name for log_name in train_scene_filter.log_names if log_name in cfg.train_logs or log_name in cfg.val_logs 
        ]
        print("-----------")
    else:
        train_scene_filter.log_names = cfg.train_logs + cfg.val_logs
        print("===========")
    

    print("len(train_scene_filter.log_names) ", len(train_scene_filter.log_names))

    val_scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    if val_scene_filter.log_names is not None:
        val_scene_filter.log_names = [log_name for log_name in val_scene_filter.log_names if log_name in cfg.val_logs]
    else:
        val_scene_filter.log_names = cfg.val_logs

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
        train_data = CacheOnlyDataset(
            cache_path=cfg.cache_path,
            feature_builders=agent.get_feature_builders(),
            target_builders=agent.get_target_builders(),
            log_names=cfg.train_logs,
            preprocess_images=preprocess_images,
            preprocess_image_dtype=preprocess_image_dtype,
            pretokenize_inputs=pretokenize_inputs,
            tokenizer=tokenizer,
        )
        val_data = CacheOnlyDataset(
            cache_path=cfg.cache_path,
            feature_builders=agent.get_feature_builders(),
            target_builders=agent.get_target_builders(),
            log_names=cfg.val_logs,
            preprocess_images=preprocess_images,
            preprocess_image_dtype=preprocess_image_dtype,
            pretokenize_inputs=pretokenize_inputs,
            tokenizer=tokenizer,
        )


        logger.info("Building Datasets")

        train_data = ConcatDataset([train_data, val_data])
    else:
        logger.info("Building SceneLoader")
        train_data, val_data = build_datasets(cfg, agent)

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

    trainer = pl.Trainer(**cfg.trainer.params, callbacks=agent.get_training_callbacks())

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
            AgentLightningModule(agent=agent, for_viz=True),
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
        trainer.fit(
            model=lightning_module,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
            ckpt_path=cfg.train_ckpt_path
        )


if __name__ == "__main__":
    main()

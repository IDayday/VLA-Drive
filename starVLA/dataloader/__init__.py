import json
import os
from accelerate.logging import get_logger
import numpy as np
from torch.utils.data import DataLoader
import numpy as np
import torch
import torch.distributed as dist
from pathlib import Path
from starVLA.dataloader.vlm_datasets import make_vlm_dataloader

from omegaconf import OmegaConf

logger = get_logger(__name__)


def _navsim_worker_init_fn(_worker_id):
    """Keep each loader worker single-threaded to avoid 96-worker oversubscription."""
    worker_threads = max(1, int(os.environ.get("NAVSIM_WORKER_THREADS", "1")))
    torch.set_num_threads(worker_threads)
    os.environ["OMP_NUM_THREADS"] = str(worker_threads)
    os.environ["MKL_NUM_THREADS"] = str(worker_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(worker_threads)
    try:
        import cv2

        cv2.setNumThreads(0)
    except Exception:
        pass

def save_dataset_statistics(dataset_statistics, run_dir):
    """Saves a `dataset_statistics.json` file."""
    out_path = run_dir / "dataset_statistics.json"
    with open(out_path, "w") as f_json:
        for _, stats in dataset_statistics.items():
            for k in stats["action"].keys():
                if isinstance(stats["action"][k], np.ndarray):
                    stats["action"][k] = stats["action"][k].tolist()
            if "proprio" in stats:
                for k in stats["proprio"].keys():
                    if isinstance(stats["proprio"][k], np.ndarray):
                        stats["proprio"][k] = stats["proprio"][k].tolist()
            if "num_trajectories" in stats:
                if isinstance(stats["num_trajectories"], np.ndarray):
                    stats["num_trajectories"] = stats["num_trajectories"].item()
            if "num_transitions" in stats:
                if isinstance(stats["num_transitions"], np.ndarray):
                    stats["num_transitions"] = stats["num_transitions"].item()
        json.dump(dataset_statistics, f_json, indent=2)
    logger.info(f"Saved dataset statistics file at path {out_path}")



def build_dataloader(cfg, dataset_py="lerobot_datasets_oxe"): # TODO now here only is get dataset, we need mv dataloader to here

    if dataset_py == "lerobot_datasets":
        from starVLA.dataloader.lerobot_datasets import get_vla_dataset, collate_fn
        vla_dataset_cfg = cfg.datasets.vla_data

        vla_dataset = get_vla_dataset(data_cfg=vla_dataset_cfg)
        
        vla_train_dataloader = DataLoader(
            vla_dataset,
            batch_size=cfg.datasets.vla_data.per_device_batch_size,
            collate_fn=collate_fn,
            num_workers=4,
            # shuffle=True
        )        
        if dist.get_rank() == 0: 
            
            output_dir = Path(cfg.output_dir)
            vla_dataset.save_dataset_statistics(output_dir / "dataset_statistics.json")
        return vla_train_dataloader
    elif dataset_py == "vlm_datasets":
        vlm_data_module = make_vlm_dataloader(cfg)
        vlm_train_dataloader = vlm_data_module["train_dataloader"]
        
        return vlm_train_dataloader

    elif dataset_py == "navsim_dataset":
        from starVLA.dataloader.navsim_dataset import NavSimDataset, collate_fn
        navsim_dataset_cfg = cfg.datasets.vla_data

        ver_1225 = OmegaConf.select(cfg, "ver_1225", default=False)

        navsim_dataset = NavSimDataset(
            datalist_path=navsim_dataset_cfg.datalist_path,
            split=navsim_dataset_cfg.split,
            video_data_cfg=cfg.datasets.video_data,
            gs_data_cfg=cfg.datasets.gs_data,
            reward_data_cfg=cfg.datasets.reward_data,
            ver_1225 = ver_1225,
            dataset_cfg = navsim_dataset_cfg,
            all_cfg = cfg,
        )
        
        navsim_num_workers = int(
            os.environ.get(
                "NAVSIM_NUM_WORKERS", str(navsim_dataset_cfg.get("num_workers", 7))
            )
        )
        navsim_prefetch_factor = int(os.environ.get("NAVSIM_PREFETCH_FACTOR", "2"))
        navsim_pin_memory = os.environ.get("NAVSIM_PIN_MEMORY", "1") == "1"
        navsim_shuffle = bool(navsim_dataset_cfg.get("shuffle", False))
        if navsim_num_workers < 0:
            raise ValueError("NAVSIM_NUM_WORKERS must be non-negative")
        if navsim_prefetch_factor < 1:
            raise ValueError("NAVSIM_PREFETCH_FACTOR must be positive")
        dataloader_kwargs = {}
        if navsim_num_workers > 0:
            dataloader_kwargs.update(
                persistent_workers=True,
                prefetch_factor=navsim_prefetch_factor,
                worker_init_fn=_navsim_worker_init_fn,
            )
        if not dist.is_initialized() or dist.get_rank() == 0:
            batch_size = int(cfg.datasets.vla_data.per_device_batch_size)
            logger.info(
                "NAVSIM DataLoader: workers_per_rank=%d prefetch_factor=%d "
                "prefetched_batches_per_rank=%d prefetched_samples_per_rank=%d "
                "pin_memory=%s shuffle=%s",
                navsim_num_workers,
                navsim_prefetch_factor,
                navsim_num_workers * navsim_prefetch_factor,
                navsim_num_workers * navsim_prefetch_factor * batch_size,
                navsim_pin_memory,
                navsim_shuffle,
            )
        navsim_train_dataloader = DataLoader(
            navsim_dataset,
            batch_size=cfg.datasets.vla_data.per_device_batch_size,
            collate_fn=collate_fn,
            num_workers=navsim_num_workers,
            shuffle=navsim_shuffle,
            drop_last=False,
            pin_memory=navsim_pin_memory,
            **dataloader_kwargs,
        )
        return navsim_train_dataloader

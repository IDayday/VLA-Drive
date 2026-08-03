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
from starVLA.cache.navsim_feature_cache import append_world_action_tokens

from omegaconf import OmegaConf

logger = get_logger(__name__)


class NavsimQwenBatchCollator:
    """Build the CPU side of Qwen inputs in DataLoader workers.

    The original path runs ``AutoProcessor.apply_chat_template`` serially in
    the training process immediately before the GPU forward.  For NAVSIM this
    takes hundreds of milliseconds when the rank is intentionally limited to
    one CPU thread.  Collating the exact same batch in loader workers lets the
    81 MiB batch payload be prefetched while the accelerator is busy.
    """

    def __init__(self, model_path, act_tok, w_depth, cot_prompt=None):
        from transformers import AutoProcessor

        self.processor = AutoProcessor.from_pretrained(model_path)
        self.processor.tokenizer.padding_side = "left"
        self.act_tok = int(act_tok)
        self.w_depth = bool(w_depth)
        self.cot_prompt = cot_prompt

    def __call__(self, batch):
        instructions = [
            append_world_action_tokens(sample["lang"], self.act_tok, self.w_depth)
            for sample in batch
        ]
        messages = []
        for sample, instruction in zip(batch, instructions):
            content = [
                {"type": "image", "image": image}
                for image in sample["image"]
            ]
            prompt = (
                self.cot_prompt.replace("{instruction}", instruction)
                if self.cot_prompt is not None
                else instruction
            )
            content.append({"type": "text", "text": prompt})
            messages.append([{"role": "user", "content": content}])

        qwen_inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            padding=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        # The framework consumes this batch-level payload from the first
        # sample.  Keeping the surrounding list-of-dicts contract unchanged
        # avoids perturbing the other three training branches and evaluation.
        batch[0]["_qwen_prefetched_inputs"] = dict(qwen_inputs)
        return batch


def _navsim_worker_init_fn(_worker_id):
    """Apply the per-worker CPU budget without oversubscribing the host."""
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
        
        navsim_num_workers = int(os.environ.get("NAVSIM_NUM_WORKERS", "7"))
        navsim_prefetch_factor = int(os.environ.get("NAVSIM_PREFETCH_FACTOR", "2"))
        navsim_pin_memory = os.environ.get("NAVSIM_PIN_MEMORY", "1") == "1"
        prefetch_qwen = os.environ.get("STARVLA_PREFETCH_QWEN", "0") == "1"
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
        navsim_collate_fn = collate_fn
        if prefetch_qwen:
            if navsim_num_workers == 0:
                raise ValueError(
                    "STARVLA_PREFETCH_QWEN=1 requires NAVSIM_NUM_WORKERS > 0"
                )
            cot_prompt = OmegaConf.select(
                cfg, "datasets.vla_data.CoT_prompt", default=None
            )
            navsim_collate_fn = NavsimQwenBatchCollator(
                model_path=cfg.framework.qwenvl.base_vlm,
                act_tok=OmegaConf.select(cfg, "act_tok", default=8),
                w_depth=OmegaConf.select(cfg, "w_depth", default=False),
                cot_prompt=cot_prompt,
            )
        if not dist.is_initialized() or dist.get_rank() == 0:
            batch_size = int(cfg.datasets.vla_data.per_device_batch_size)
            logger.info(
                "NAVSIM DataLoader: workers_per_rank=%d prefetch_factor=%d "
                "prefetched_batches_per_rank=%d prefetched_samples_per_rank=%d "
                "pin_memory=%s qwen_worker_preprocess=%s",
                navsim_num_workers,
                navsim_prefetch_factor,
                navsim_num_workers * navsim_prefetch_factor,
                navsim_num_workers * navsim_prefetch_factor * batch_size,
                navsim_pin_memory,
                prefetch_qwen,
            )
        navsim_train_dataloader = DataLoader(
            navsim_dataset,
            batch_size=cfg.datasets.vla_data.per_device_batch_size,
            collate_fn=navsim_collate_fn,
            num_workers=navsim_num_workers,
            # shuffle=True,
            drop_last=False,
            pin_memory=navsim_pin_memory,
            **dataloader_kwargs,
        )
        return navsim_train_dataloader

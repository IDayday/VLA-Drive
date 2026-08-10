# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License"); 
# Implemented by [Jinhui YE / HKUST University] in [2025].


"""
StarVLA’s trainer is built directly on native PyTorch + Accelerate + DeepSpeed, keeping the loop explicit and easy to hack.
Conventions:
1. Store runtime state in dicts where possible (simplifies data info, procesing info, config, etc).  
2. Use multiple dataloaders to adapt heterogeneous data types / task mixtures.  
3. Put each training strategy in its own `trainer_*.py` file (avoid large if‑else chains).  
"""

# Standard Library
import argparse
import json
import os
from pathlib import Path
from typing import Tuple
from torch.utils.data import Dataset, DataLoader
import numpy as np
import time

# DeepSpeed's Triton autotune table is updated at interpreter exit. Give each
# local rank its own node-local cache to avoid concurrent pickle updates and
# stale handles on shared filesystems.
if "TRITON_CACHE_DIR" in os.environ and "LOCAL_RANK" in os.environ:
    os.environ["TRITON_CACHE_DIR"] = os.path.join(
        os.environ["TRITON_CACHE_DIR"],
        f"local_rank{os.environ['LOCAL_RANK']}",
    )
    os.makedirs(os.environ["TRITON_CACHE_DIR"], exist_ok=True)

# Third-Party Libraries
import torch
import torch.distributed as dist
import wandb
import yaml
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from omegaconf import OmegaConf
from tqdm import tqdm
from transformers import AutoProcessor, get_scheduler

# Local Modules
from starVLA.training.trainer_utils.trainer_tools import normalize_dotlist_args
from starVLA.model.framework import build_framework
from starVLA.training.trainer_utils.trainer_tools import TrainerUtils
from starVLA.training.trainer_utils.trainer_tools import build_param_lr_groups

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# Initialize Overwatch =>> Wraps `logging.Logger`
from accelerate.logging import get_logger

logger = get_logger(__name__)


def _cfg_select(cfg, path, default=None):
    """Read an OmegaConf/dict/namespace path without mutating configuration."""

    if OmegaConf.is_config(cfg):
        return OmegaConf.select(cfg, path, default=default)
    value = cfg
    for part in path.split("."):
        if isinstance(value, dict):
            if part not in value:
                return default
            value = value[part]
        elif hasattr(value, part):
            value = getattr(value, part)
        else:
            return default
    return value


def aggregate_losses(output_dict, cfg):
    """Aggregate structured Field2Plan or unchanged legacy framework losses.

    Returns ``(total_loss, named_losses)``. Structured ``output_dict['losses']``
    uses ``trainer.loss_weights``. A legacy flat output follows the exact
    historical action/rgb/gs/reward switches and ignores the new weights.
    """

    structured = output_dict.get("losses")
    if structured is not None:
        if not isinstance(structured, dict) or not structured:
            raise ValueError("output_dict['losses'] must be a non-empty dict")
        weights = _cfg_select(cfg, "trainer.loss_weights", {}) or {}
        total = None
        named = {}
        for name, loss in structured.items():
            if not isinstance(loss, torch.Tensor) or loss.numel() != 1:
                raise ValueError(f"structured loss {name!r} must be a scalar tensor")
            weight = float(weights.get(name, 1.0))
            named[name] = loss
            weighted = loss * weight
            total = weighted if total is None else total + weighted
        return total, named

    named = {}
    if int(_cfg_select(cfg, "datasets.video_data.load_2d_data", 0)) == 1:
        named["rgb"] = output_dict["rgb_loss"]
    if int(_cfg_select(cfg, "datasets.gs_data.load_3d_data", 0)) == 1 or bool(
        _cfg_select(cfg, "w_depth", 0)
    ):
        named["gs"] = output_dict["gs_loss"]
    if int(_cfg_select(cfg, "datasets.reward_data.load_reward_data", 0)) == 1:
        named["reward"] = output_dict["reward_loss"]
    if int(_cfg_select(cfg, "datasets.vla_data.load_act_data", 0)) == 1:
        named["action"] = output_dict["action_loss"]
    if not named:
        raise ValueError("no enabled legacy losses")
    total = None
    for loss in named.values():
        total = loss if total is None else total + loss
    return total, named


def json_scalar_metrics(metrics, step: int):
    """Return a JSON-safe scalar record for local checkpoint diagnostics."""

    record = {"step": int(step)}
    for name, value in metrics.items():
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                continue
            value = value.detach().float().item()
        if isinstance(value, (bool, int, float, str)):
            record[str(name)] = value
    return record



def setup_directories(cfg) -> Path:
    """create output directory and save config"""
    cfg.output_dir = os.path.join(cfg.run_root_dir, cfg.run_id)
    output_dir = Path(cfg.output_dir)

    if not dist.is_initialized() or dist.get_rank() == 0:
        # create output directory and checkpoint directory
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(output_dir / "checkpoints", exist_ok=True)

        # save config
        OmegaConf.save(cfg, output_dir / "config.yaml")
        with open(output_dir / "config.yaml", "r") as f_yaml, open(output_dir / "config.json", "w") as f_json:
            yaml_cfg = yaml.safe_load(f_yaml)
            json.dump(yaml_cfg, f_json, indent=2)

    return output_dir


def build_model(cfg) -> torch.nn.Module:
    """build model framework"""
    logger.info(f"Loading Base VLM `{cfg.framework.qwenvl.base_vlm}` from ID/Path")
    model = build_framework(cfg)

    return model


# here changes need to 📦 encapsulate Dataloader
from starVLA.dataloader import build_dataloader


def prepare_data(cfg, accelerator, output_dir) -> DataLoader:
    """prepare training data"""
    # VLA data loader
    # logger.info(f"Creating VLA Dataset with Mixture `{cfg.datasets.vla_data.data_mix}`")
    vla_train_dataloader = build_dataloader(cfg=cfg, dataset_py=cfg.datasets.vla_data.dataset_py)

    accelerator.dataloader_config.dispatch_batches = False
    dist.barrier()

    return vla_train_dataloader


def setup_optimizer_and_scheduler(model, cfg) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    """set optimizer and scheduler"""
    # initialize optimizer
    param_groups = build_param_lr_groups(model=model, cfg=cfg)
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=cfg.trainer.learning_rate.base,
        betas=tuple(cfg.trainer.optimizer.betas),
        weight_decay=cfg.trainer.optimizer.weight_decay,
        eps=cfg.trainer.optimizer.eps,
    )

    # print optimizer group info
    if dist.is_initialized() and dist.get_rank() == 0:
        for i, group in enumerate(optimizer.param_groups):
            logger.info(f"LR Group {group['name']}: lr={group['lr']}, num_params={len(group['params'])}")

    # initialize learning rate scheduler
    lr_scheduler = get_scheduler(
        name=cfg.trainer.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=cfg.trainer.num_warmup_steps,
        num_training_steps=cfg.trainer.max_train_steps,
        scheduler_specific_kwargs=cfg.trainer.scheduler_specific_kwargs,  # minimum learning rate
    )

    return optimizer, lr_scheduler


class VLATrainer(TrainerUtils):
    def __init__(self, cfg, model, vla_train_dataloader, optimizer, lr_scheduler, accelerator):
        self.config = cfg
        self.model = model
        self.vla_train_dataloader = vla_train_dataloader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.accelerator = accelerator

        # training status tracking
        self.completed_steps = 0
        self.total_batch_size = self._calculate_total_batch_size()
        self._timing_window = []

        # --- Grad monitor (NEW) ---
        self._gm_handles = []
        self._gm_names = []
        self._gm_mask = None

    # ====== NEW: 参数梯度监控（DeepSpeed/ZeRO 兼容） ======
    def _setup_grad_monitor(self):
        """
        在 prepare 之后、训练前调用。
        对每个 requires_grad 的参数注册反向 hook。
        """
        raw_model = self.accelerator.unwrap_model(self.model)

        names, params = [], []
        for n, p in raw_model.named_parameters():
            if p.requires_grad:
                names.append(n)
                params.append(p)

        if len(params) == 0:
            return  # 没有可训练参数

        device = params[0].device
        self._gm_names = names
        # 用 int32 做标记，方便 all_reduce(max)
        self._gm_mask = torch.zeros(len(params), dtype=torch.int32, device=device)

        handles = []
        for idx, p in enumerate(params):
            def _make_hook(i):
                def _hook(grad):
                    # 只要本 rank 的这个 shard 收到过梯度，就置 1
                    self._gm_mask[i] = 1
                return _hook
            handles.append(p.register_hook(_make_hook(idx)))

        self._gm_handles = handles
        if self.accelerator.is_main_process:
            logger.info(f"[GradMonitor] Registered hooks for {len(names)} trainable params.")

    def _report_unused_after_backward(self):
        """
        在 accelerator.backward(loss) 之后、optimizer.step() 之前调用。
        合并各 rank 的 touched_mask，并在主进程打印“未用参数”。
        """
        if self._gm_mask is None:
            return  # 未初始化

        self.accelerator.wait_for_everyone()
        if dist.is_initialized():
            dist.all_reduce(self._gm_mask, op=dist.ReduceOp.MAX)

        if self.accelerator.is_main_process:
            mask = self._gm_mask.tolist()
            unused = [n for n, f in zip(self._gm_names, mask) if f == 0]
            used   = [n for n, f in zip(self._gm_names, mask) if f == 1]
            if len(unused) == 0:
                logger.info(f"[GradMonitor] All {len(used)} trainable params received grads this step.")
            else:
                logger.info(f"[GradMonitor] Unused params this step ({len(unused)}):")
                for n in unused:
                    logger.info(f"  - {n}")

        # 清零进入下一步（想跨多步累计就把这行注释掉）
        self._gm_mask.zero_()

    def prepare_training(self):
        rank = dist.get_rank() if dist.is_initialized() else 0
        seed = self.config.seed + rank if hasattr(self.config, "seed") else rank + 3047
        set_seed(seed)

        # load pretrained weights
        if hasattr(self.config.trainer, "pretrained_checkpoint") and self.config.trainer.pretrained_checkpoint:
            pretrained_checkpoint = self.config.trainer.pretrained_checkpoint
            reload_modules = (
                self.config.trainer.reload_modules if hasattr(self.config.trainer, "reload_modules") else None
            )
            self.model = self.load_pretrained_backbones(self.model, pretrained_checkpoint, reload_modules=reload_modules)



        # freeze parameters
        freeze_modules = (
            self.config.trainer.freeze_modules
            if (self.config and hasattr(self.config.trainer, "freeze_modules"))
            else None
        )
        self.model = self.freeze_backbones(self.model, freeze_modules=freeze_modules)

        if self.config.trainer.resume_ckpt != 'none':
            state = torch.load(self.config.trainer.resume_ckpt, map_location="cpu", weights_only=True)
            missing, unexpected = self.model.load_state_dict(state, strict=False)
            print("missing:", missing, "unexpected:", unexpected)
        
        if self.config.pretrain_model_2d is not None:
            state = torch.load(self.config.pretrain_model_2d, map_location="cpu", weights_only=True)
            missing, unexpected = self.model.load_state_dict(state, strict=False)
            print("missing:", missing, "unexpected:", unexpected)

        #  print model trainable parameters:
        self.print_trainable_parameters(self.model)

        # initialize distributed training components
        self.model, self.optimizer, self.vla_train_dataloader = self.setup_distributed_training(
            self.accelerator,  # must be the first param
            self.model,
            self.optimizer,
            self.vla_train_dataloader,
            # self.vlm_train_dataloader
        )
        # self._setup_grad_monitor()

        self._init_wandb()
        self._init_checkpointing()

    def _calculate_total_batch_size(self):
        """calculate global batch size"""
        return (
            self.config.datasets.vla_data.per_device_batch_size
            * self.accelerator.num_processes
            * self.accelerator.gradient_accumulation_steps
        )

    def _init_wandb(self):
        """initialize Weights & Biases"""
        if self.accelerator.is_main_process:
            wandb.init(
                name=self.config.run_id,
                dir=os.path.join(self.config.output_dir, "wandb"),
                project=self.config.wandb_project,
                entity=self.config.wandb_entity,
                group="vla-train",
            )

    def _init_checkpointing(self):
        """initialize checkpoint directory"""
        self.checkpoint_dir = os.path.join(self.config.output_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        pretrained_checkpoint = getattr(self.config.trainer, "pretrained_checkpoint", None)
        is_resume = getattr(self.config.trainer, "is_resume", False)

        # resume training state
        if pretrained_checkpoint and is_resume:
            self._load_checkpoint(self.config.resume_from_checkpoint)

    def _load_checkpoint(self, checkpoint_path):
        """load checkpoint"""
        self.accelerator.load_state(checkpoint_path)
        self.accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")

    def _save_checkpoint(self):
        """save current training state"""

        if self.accelerator.is_main_process:

            checkpoint_path = os.path.join(self.checkpoint_dir, f"steps_{self.completed_steps}")
            # save model state
            state_dict = self.accelerator.get_state_dict(self.model)
            torch.save(state_dict, checkpoint_path + "_pytorch_model.pt")

            # save training metadata
            summary_data = {
                "steps": self.completed_steps,
            }
            with open(os.path.join(self.config.output_dir, "summary.jsonl"), "a") as f:
                f.write(json.dumps(summary_data) + "\n")
            self.accelerator.print(f"✅ Checkpoint saved at {checkpoint_path}")
        self.accelerator.wait_for_everyone()

    def _log_metrics(self, metrics):
        """record training metrics"""
        if self.completed_steps % self.config.trainer.logging_frequency == 0:
            if dist.get_rank() == 0:
                # Scalar extraction synchronizes the accelerator.  Keep it at
                # logging frequency instead of forcing four syncs every step.
                metrics = {
                    key: (
                        value.detach().float().item()
                        if isinstance(value, torch.Tensor) and value.numel() == 1
                        else value
                    )
                    for key, value in metrics.items()
                }
                # add learning rate
                metrics["learning_rate"] = self.lr_scheduler.get_last_lr()[0]

                # add epoch info
                metrics["epoch"] = round(self.completed_steps / len(self.vla_train_dataloader), 2)

                if "rgbs" in metrics:
                    vids = metrics["rgbs"]
                    wandb_videos = []
                    for vid in vids:
                        # 如果只有一张图，扩展成 T=1 的视频
                        if vid.ndim == 3:
                            vid = vid[None]

                        # wandb.Video 需要 (T,H,W,C)->(T,C,H,W)
                        vid_chw = vid.transpose(1, 0, 2, 3)

                        wandb_videos.append(
                            wandb.Video(vid_chw, fps=2, format="mp4")
                        )

                    metrics["generated_videos"] = wandb_videos
                    del metrics["rgbs"]  # 不建议直接 log 巨大的 numpy list

                if "gs" in metrics:
                    vids = metrics["gs"]
                    wandb_gs = []
                    for vid in vids:
                        # 如果只有一张图，扩展成 T=1 的视频
                        if vid.ndim == 3:
                            vid = vid[None]

                        # wandb.Video 需要 (T,H,W,C)->(T,C,H,W)
                        vid_chw = vid.transpose(0, 3, 1, 2)

                        wandb_gs.append(
                            wandb.Video(vid_chw, fps=2, format="mp4")
                        )

                    metrics["generated_gs"] = wandb_gs
                    del metrics["gs"]  # 不建议直接 log 巨大的 numpy list

                local_record = json_scalar_metrics(metrics, self.completed_steps)
                with open(
                    os.path.join(self.config.output_dir, "training_metrics.jsonl"),
                    "a",
                    encoding="utf-8",
                ) as stream:
                    stream.write(json.dumps(local_record, sort_keys=True) + "\n")

                # record to W&B
                wandb.log(metrics, step=self.completed_steps)
                # debug output
                logger.info(f"Step {self.completed_steps}, Loss: {metrics})")

    def _create_data_iterators(self):
        """create data iterators"""
        self.vla_iter = iter(self.vla_train_dataloader)

    def _get_next_batch(self):
        """get next batch (automatically handle data loop)"""
        try:
            batch_vla = next(self.vla_iter)
        except StopIteration:
            self.accelerator.print(f"[R{dist.get_rank()}] HIT StopIteration at step={self.completed_steps}")
            if not hasattr(self, "vla_epoch_count"):
                self.vla_epoch_count = 0
            self.vla_iter, self.vla_epoch_count = TrainerUtils._reset_dataloader(
                self.vla_train_dataloader, self.vla_epoch_count
            )
            batch_vla = next(self.vla_iter)

        return batch_vla

    def train(self):
        """execute training loop"""
        # print training config
        self._log_training_config()

        # prepare data iterators
        self._create_data_iterators()

        # create progress bar
        progress_bar = tqdm(
            range(self.config.trainer.max_train_steps), disable=not self.accelerator.is_local_main_process
        )

        # main training loop
        optimizer_step_start = time.perf_counter()
        optimizer_data_time = 0.0
        optimizer_model_time = 0.0
        while self.completed_steps < self.config.trainer.max_train_steps:
            # get data batch
            t_start_data = time.perf_counter()
            batch_vla = self._get_next_batch()
            t_end_data = time.perf_counter()

            # execute training step
            t_start_model = time.perf_counter()
            step_metrics = self._train_step(batch_vla)
            t_end_model = time.perf_counter()
            optimizer_data_time += t_end_data - t_start_data
            optimizer_model_time += t_end_model - t_start_model

            # update progress
            if self.accelerator.sync_gradients:
                progress_bar.update(1)
                self.completed_steps += 1

                # Evaluation, logging, and checkpointing are optimizer-step
                # operations. Running them on unsynchronized accumulation
                # microsteps repeatedly evaluates "step 0" and breaks the
                # intended accumulation schedule.
                if self.completed_steps % self.config.trainer.eval_interval == 0:
                    step_metrics = self.eval_action_model(step_metrics)

                step_metrics["data_time"] = optimizer_data_time
                step_metrics["model_time"] = optimizer_model_time
                self._log_metrics(step_metrics)

                if self.completed_steps % self.config.trainer.save_interval == 0:
                    self._save_checkpoint()

                optimizer_step_end = time.perf_counter()
                wall_time = optimizer_step_end - optimizer_step_start
                overhead_time = max(
                    0.0,
                    wall_time - optimizer_data_time - optimizer_model_time,
                )
                self._timing_window.append(
                    (optimizer_data_time, optimizer_model_time, overhead_time, wall_time)
                )

                if self.completed_steps % self.config.trainer.logging_frequency == 0:
                    if self.accelerator.is_local_main_process:
                        timing = np.asarray(self._timing_window, dtype=np.float64)
                        means = timing.mean(axis=0)
                        p95 = np.percentile(timing, 95, axis=0)
                        progress_bar.set_postfix(
                            {
                                "data_avg": f"{means[0]:.3f}",
                                "data_p95": f"{p95[0]:.3f}",
                                "model_avg": f"{means[1]:.3f}",
                                "wall_avg": f"{means[3]:.3f}",
                            },
                            refresh=False,
                        )
                        logger.info(
                            "Timing[%d steps]: data avg/p95=%.3f/%.3fs, "
                            "model avg/p95=%.3f/%.3fs, overhead avg/p95=%.3f/%.3fs, "
                            "wall avg/p95=%.3f/%.3fs",
                            len(self._timing_window),
                            means[0], p95[0],
                            means[1], p95[1],
                            means[2], p95[2],
                            means[3], p95[3],
                        )
                    # Every rank owns a timing window. Clear it everywhere so
                    # non-main ranks do not retain one tuple per training step.
                    self._timing_window.clear()

                optimizer_step_start = time.perf_counter()
                optimizer_data_time = 0.0
                optimizer_model_time = 0.0

                if self.completed_steps >= self.config.trainer.max_train_steps:
                    break

        # training end processing
        self._finalize_training()

        # execute evaluation step

    def eval_action_model(self, step_metrics: dict = None) -> float:
        """
        Evaluate the model on the given dataset using the specified metric function.

        :param eval_dataset: List of evaluation samples, each containing 'image', 'instruction', and 'action'.
        :param metric_fn: Function to compute the distance between predicted and ground truth actions.
        :return: Average metric score across the evaluation dataset.
        """

        if self.accelerator.is_main_process:

            examples = self._get_next_batch()

            score = 0.0
            num_samples = len(examples)

            batch_images = [example["image"] for example in examples]
            instructions = [example["lang"] for example in examples]  # [B, str]
            try:
                actions = [example["action"] for example in examples]  # label
            except:
                actions = None

            # Predict actions using the model
            output_dict = self.model.predict_action(
                # batch_images=batch_images, instructions=instructions, use_ddim=True, num_ddim_steps=20
                examples,
            )

            normalized_actions = output_dict["normalized_actions"]  # B, T, D

            if normalized_actions is not None:

                actions = np.array(actions)  # convert actions to numpy.ndarray
                # B, Chunk, dim = actions.shape
                num_pots = np.prod(actions.shape)
                # Compute the metric score
                # actions = np.cumsum(actions, axis=1)
                # normalized_actions = np.cumsum(normalized_actions, axis=1)
                score = TrainerUtils.euclidean_distance(normalized_actions, actions)
                average_score = score / num_pots
                step_metrics["mse_score"] = average_score

            if self.config.datasets.video_data.load_2d_data:
                rgbs = output_dict["rgbs"]
                step_metrics["rgbs"] = rgbs
            
            if self.config.datasets.gs_data.load_3d_data:
                gs = output_dict["gs"]
                step_metrics["gs"] = gs
            
            if self.config.datasets.reward_data.load_reward_data:
                pred_reward = output_dict["reward"].detach().cpu().numpy()  # B, T, D
                reward = [example["reward_data"] for example in examples]

                reward = np.array(reward)  # convert actions to numpy.ndarray
                # B, Chunk, dim = actions.shape
                num_pots = np.prod(reward.shape)
                # Compute the metric score
                # actions = np.cumsum(actions, axis=1)
                # normalized_actions = np.cumsum(normalized_actions, axis=1)
                score = TrainerUtils.euclidean_distance(pred_reward, reward)
                average_score = score / num_pots
                step_metrics["reward_mse_score"] = average_score

        pass
        dist.barrier()  # ensure all processes are synchronized
        return step_metrics

    def _log_training_config(self):
        """record training config"""
        if self.accelerator.is_main_process:
            logger.info("***** Training Configuration *****")
            logger.info(f"  Total optimization steps = {self.config.trainer.max_train_steps}")
            logger.info(f"  Per device batch size = {self.config.datasets.vla_data.per_device_batch_size}")
            logger.info(f"  Gradient accumulation steps = {self.config.trainer.gradient_accumulation_steps}")
            logger.info(f"  Total batch size = {self.total_batch_size}")

    def _train_step(self, batch_vla, batch_vlm=None):
        """execute single training step"""
        if self.config.datasets.gs_data.load_3d_data:
            if self.config.framework.gs_model.enable_perceptual_loss and self.completed_steps >= self.config.framework.gs_model.perceptual_loss_start_iter:
                logger.info('starting set_perceptual_loss')
                self.model.gs_model.rgb_and_lpips_loss.set_perceptual_loss(True)
        with self.accelerator.accumulate(self.model):
            # VLA task forward propagation
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output_dict = self.model.forward(batch_vla)
                total_loss, named_losses = aggregate_losses(output_dict, self.config)


            # VLA backward propagation
            self.accelerator.backward(total_loss)
            # for debug
            # self._report_unused_after_backward()

            # gradient clipping
            if self.accelerator.sync_gradients:
                if self.config.trainer.gradient_clipping is not None:
                    self.accelerator.clip_grad_norm_(self.model.parameters(), self.config.trainer.gradient_clipping)

                # optimizer step
                self.optimizer.step()
                self.lr_scheduler.step()
                raw_model = self.accelerator.unwrap_model(self.model)
                update_ema = getattr(raw_model, "update_ema", None)
                if callable(update_ema):
                    update_ema()
                self.optimizer.zero_grad()

        if "losses" in output_dict:
            metrics = {
                f"{name}_loss": loss.detach() for name, loss in named_losses.items()
            }
            metrics.update(
                {
                    name: value.detach() if isinstance(value, torch.Tensor) else value
                    for name, value in output_dict.get("metrics", {}).items()
                }
            )
            return metrics
        return {
            "action_dit_loss": named_losses["action"].detach(),
            "rgb_gen_loss": 0 if "rgb" not in named_losses else named_losses["rgb"].detach(),
            "gs_loss": 0 if "gs" not in named_losses else named_losses["gs"].detach(),
            "reward_loss": 0 if "reward" not in named_losses else named_losses["reward"].detach(),
        }

    def _finalize_training(self):
        """training end processing"""
        # save final model
        skip_final_save = os.environ.get("TRAINING_SKIP_FINAL_SAVE", "0") == "1"
        if self.accelerator.is_main_process and not skip_final_save:
            final_checkpoint = os.path.join(self.config.output_dir, "final_model")
            os.makedirs(final_checkpoint, exist_ok=True)
            state_dict = self.accelerator.get_state_dict(self.model)
            torch.save(state_dict, os.path.join(final_checkpoint, "pytorch_model.pt"))
            logger.info(f"Training complete. Final model saved at {final_checkpoint}")
        elif self.accelerator.is_main_process:
            logger.info("TRAINING_SKIP_FINAL_SAVE=1; skipped final checkpoint (smoke test only)")

        # close W&B
        if self.accelerator.is_main_process:
            wandb.finish()

        self.accelerator.wait_for_everyone()


def main(cfg) -> None:
    deepspeed_plugin = DeepSpeedPlugin()
    accelerator = Accelerator(
        gradient_accumulation_steps=int(
            cfg.trainer.gradient_accumulation_steps
        ),
        deepspeed_plugin=deepspeed_plugin,
    )
    accelerator.print(accelerator.state)
    logger.info("VLA Training :: Warming Up")

    # create output directory and save config
    output_dir = setup_directories(cfg=cfg)

    import shutil
    import os
    from shutil import ignore_patterns
    code_dir = os.path.join(output_dir, 'code/')
    project_root = str(Path(__file__).resolve().parent.parent.parent)
    # Only rank 0 writes the source snapshot.  Sixteen ranks copying the same
    # tree concurrently needlessly hammers shared storage at job startup.
    if accelerator.is_main_process:
        os.makedirs(code_dir, exist_ok=True)
        for fname in ('debug.sh', '8-train.sh', 'training.sh', 'pre_cache.sh'):
            src = os.path.join(project_root, fname)
            if os.path.exists(src):
                shutil.copy2(src, code_dir)
        shutil.copytree(
            os.path.join(project_root, 'starVLA'),
            os.path.join(code_dir, 'starVLA'),
            ignore=ignore_patterns('__pycache__', '*.pyc', '*.egg-info'),
            dirs_exist_ok=True,
        )
    accelerator.wait_for_everyone()




    # build model
    vla = build_framework(cfg, accelerator)
    if accelerator.is_main_process:
        total_parameters = sum(parameter.numel() for parameter in vla.parameters())
        trainable_parameters = sum(
            parameter.numel() for parameter in vla.parameters() if parameter.requires_grad
        )
        logger.info(
            "Model ready: %s total_params=%.3fB trainable_params=%.3fB",
            type(vla).__name__,
            total_parameters / 1e9,
            trainable_parameters / 1e9,
        )
    # prepare data
    vla_train_dataloader = prepare_data(cfg=cfg, accelerator=accelerator, output_dir=output_dir)

    # set optimizer and scheduler
    optimizer, lr_scheduler = setup_optimizer_and_scheduler(model=vla, cfg=cfg)

    # create trainer
    # Run VLA Training
    trainer = VLATrainer(
        cfg=cfg,
        model=vla,
        vla_train_dataloader=vla_train_dataloader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        accelerator=accelerator,
    )

    # execute training preparation
    trainer.prepare_training()
    # execute training
    trainer.train()

    # And... we're done!
    logger.info("... and that's all, folks!")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", type=str, default="starVLA/config/training/starvla_cotrain_oxe.yaml", help="Path to YAML config")
    args, clipargs = parser.parse_known_args()

    # Load YAML config & Convert CLI overrides to dotlist config
    cfg = OmegaConf.load(args.config_yaml)
    dotlist = normalize_dotlist_args(clipargs)  # Normalize CLI args to dotlist format
    cli_cfg = OmegaConf.from_dotlist(dotlist)
    cfg = OmegaConf.merge(cfg, cli_cfg)

    # if cfg.is_debug:
    # if cfg.is_debug and dist.is_initialized() and dist.get_rank() == 0:
    #     import debugpy
    #     debugpy.listen(("0.0.0.0", 10092))
    #     print("🔍 Rank 0 waiting for debugger attach on port 10092...")
    #     debugpy.wait_for_client()

    main(cfg)

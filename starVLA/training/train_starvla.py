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
from starVLA.training.trainer_utils.trainer_tools import aggregate_output_losses
from starVLA.path_config import apply_environment_path_overrides
from starVLA.model.framework import build_framework
from starVLA.training.trainer_utils.trainer_tools import TrainerUtils
from starVLA.training.trainer_utils.trainer_tools import build_param_lr_groups
from starVLA.training.trainer_utils.trainer_tools import collect_learning_rate_metrics
from starVLA.training.trainer_utils.trainer_tools import resolve_training_step_contract

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# Initialize Overwatch =>> Wraps `logging.Logger`
from accelerate.logging import get_logger

logger = get_logger(__name__)



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

    expected_sample_count = OmegaConf.select(
        cfg,
        "datasets.vla_data.expected_sample_count",
        default=None,
    )
    if expected_sample_count is not None:
        expected_sample_count = int(expected_sample_count)
        actual_sample_count = len(vla_train_dataloader.dataset)
        if actual_sample_count != expected_sample_count:
            raise RuntimeError(
                "NAVSIM training dataset size mismatch: "
                f"expected {expected_sample_count}, found {actual_sample_count}"
            )
        if accelerator.is_main_process:
            logger.info(
                "Validated complete NAVSIM training set: %d samples",
                actual_sample_count,
            )

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

    # Weight-only continuation starts a fresh optimizer/scheduler for only the
    # remaining global steps. The launcher keeps the learning rates equal to
    # the source run's endpoint so no LR restart is introduced.
    _, scheduler_training_steps = resolve_training_step_contract(cfg)
    lr_scheduler = get_scheduler(
        name=cfg.trainer.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=cfg.trainer.num_warmup_steps,
        num_training_steps=scheduler_training_steps,
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
        self.initial_step, self.remaining_steps = resolve_training_step_contract(cfg)
        self.completed_steps = self.initial_step
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

        gp_stage_a_checkpoint = str(
            OmegaConf.select(
                self.config, "trainer.gp_stage_a_checkpoint", default=""
            )
            or ""
        ).strip()
        if gp_stage_a_checkpoint:
            if not os.path.isfile(gp_stage_a_checkpoint):
                raise FileNotFoundError(
                    f"Stage-A GP checkpoint is missing: {gp_stage_a_checkpoint}"
                )
            if not hasattr(self.model, "load_gp_modules_state_dict"):
                raise RuntimeError(
                    "trainer.gp_stage_a_checkpoint requires QwenOFT_GPSQ3DMix"
                )
            gp_state = torch.load(
                gp_stage_a_checkpoint, map_location="cpu", weights_only=True
            )
            self.model.load_gp_modules_state_dict(gp_state)



        # freeze parameters
        freeze_modules = (
            self.config.trainer.freeze_modules
            if (self.config and hasattr(self.config.trainer, "freeze_modules"))
            else None
        )
        self.model = self.freeze_backbones(self.model, freeze_modules=freeze_modules)

        if self.config.trainer.resume_ckpt != 'none':
            resume_ckpt = os.fspath(self.config.trainer.resume_ckpt)
            if not os.path.isfile(resume_ckpt):
                raise FileNotFoundError(f"Resume model checkpoint is missing: {resume_ckpt}")
            state = torch.load(resume_ckpt, map_location="cpu", weights_only=True)
            resume_strict = bool(
                OmegaConf.select(
                    self.config,
                    "trainer.resume_strict",
                    default=False,
                )
            )
            missing, unexpected = self.model.load_state_dict(
                state,
                strict=resume_strict,
            )
            if self.accelerator.is_main_process:
                logger.info(
                    "Loaded model continuation checkpoint %s "
                    "(strict=%s, missing=%d, unexpected=%d, initial_step=%d)",
                    resume_ckpt,
                    resume_strict,
                    len(missing),
                    len(unexpected),
                    self.initial_step,
                )
        
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
                # Keep the legacy scalar and expose each optimizer group. This
                # is required when Qwen visual uses a lower LR than language
                # and Action DiT parameters.
                metrics.update(collect_learning_rate_metrics(self.lr_scheduler))

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

                # record to W&B
                wandb.log(metrics, step=self.completed_steps)
                if str(getattr(self.config.framework, "name", "")) in (
                    "QwenOFT_VGGT",
                    "QwenOFT_VGGT_Bottleneck",
                    "QwenOFT_GPSQ3DMix",
                ):
                    diagnostic_record = {
                        "step": self.completed_steps,
                        **{
                            key: value
                            for key, value in metrics.items()
                            if key.startswith("vggt")
                            or key.startswith("gp_sq3dmix")
                            or key
                            in {
                                "source_token_count_mean",
                                "source_feature_norm",
                                "task_geometry_norm",
                                "horizon_readout_norm",
                                "planning_delta_norm",
                                "planning_delta_ratio",
                                "slot_pairwise_cosine",
                            }
                            or key.startswith("weighted_loss/")
                            or key == "action_dit_loss"
                        },
                    }
                    with open(
                        os.path.join(
                            self.config.output_dir,
                            "gp_sq3dmix_diagnostics.jsonl"
                            if str(getattr(self.config.framework, "name", ""))
                            == "QwenOFT_GPSQ3DMix"
                            else "vggt_diagnostics.jsonl",
                        ),
                        "a",
                        encoding="utf-8",
                    ) as diagnostic_stream:
                        diagnostic_stream.write(json.dumps(diagnostic_record) + "\n")
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

    def _run_vggt_intervention_diagnostics(self, batch_vla):
        """Compare real/zero/shuffled/template memory with identical RNG."""

        framework_name = str(
            OmegaConf.select(self.config, "framework.name", default="")
        )
        is_dense_bottleneck = framework_name == "QwenOFT_VGGT_Bottleneck"
        diagnostic_path = (
            "framework.vggt_bottleneck.diagnostics.intervention_interval"
            if is_dense_bottleneck
            else "framework.vggt.diagnostics.intervention_interval"
        )
        interval = int(OmegaConf.select(self.config, diagnostic_path, default=0))
        if interval <= 0 or (self.completed_steps + 1) % interval != 0:
            return {}
        raw_model = self.accelerator.unwrap_model(self.model)
        if not hasattr(raw_model, "set_vggt_intervention"):
            return {}
        modes = (
            ("real", "zero", "shuffled")
            if is_dense_bottleneck
            else ("real", "zero", "shuffled", "slot_mean")
        )
        losses = {}
        predictions = {}
        collect_trajectory = bool(
            OmegaConf.select(
                self.config,
                "framework.vggt.diagnostics.intervention_trajectory",
                default=True,
            )
        )
        seed_path = (
            "framework.vggt_bottleneck.diagnostics.intervention_seed"
            if is_dense_bottleneck
            else "framework.vggt.diagnostics.intervention_seed"
        )
        diagnostic_seed = int(
            OmegaConf.select(self.config, seed_path, default=20260811)
        ) + self.completed_steps
        rng_devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
        try:
            for mode in modes:
                raw_model.set_vggt_intervention(mode)
                with torch.random.fork_rng(devices=rng_devices):
                    torch.manual_seed(diagnostic_seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(diagnostic_seed)
                    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                        diagnostic_output = self.model.forward(batch_vla)
                losses[mode] = diagnostic_output["action_loss"].detach()
                if collect_trajectory:
                    with torch.random.fork_rng(devices=rng_devices):
                        torch.manual_seed(diagnostic_seed)
                        if torch.cuda.is_available():
                            torch.cuda.manual_seed_all(diagnostic_seed)
                        with torch.no_grad():
                            prediction_output = raw_model.predict_action(batch_vla)
                    predictions[mode] = torch.as_tensor(
                        prediction_output["normalized_actions"], dtype=torch.float32
                    )
        finally:
            raw_model.set_vggt_intervention("real")
        real = losses["real"]
        metrics = {
            f"intervention_flow_loss_{mode}": value for mode, value in losses.items()
        }
        metrics.update(
            {
                f"intervention_{mode}_minus_real": value - real
                for mode, value in losses.items()
                if mode != "real"
            }
        )
        if predictions:
            real_prediction = predictions["real"]
            target = torch.as_tensor(
                np.asarray([example["action"] for example in batch_vla]),
                dtype=torch.float32,
            )
            assert target.shape == real_prediction.shape
            for mode, prediction in predictions.items():
                xy_error = (prediction[..., :2] - target[..., :2]).norm(dim=-1)
                heading_cosine = torch.nn.functional.cosine_similarity(
                    prediction[..., 2:4], target[..., 2:4], dim=-1
                )
                metrics[f"intervention_{mode}_ade"] = xy_error.mean()
                metrics[f"intervention_{mode}_fde"] = xy_error[:, -1].mean()
                metrics[f"intervention_{mode}_heading_error"] = (
                    1.0 - heading_cosine
                ).mean()
                if mode != "real":
                    trajectory_change = (
                        prediction[..., :2] - real_prediction[..., :2]
                    ).norm(dim=-1)
                    metrics[f"intervention_{mode}_trajectory_l2"] = (
                        trajectory_change.mean()
                    )
                    metrics[f"intervention_{mode}_final_trajectory_l2"] = (
                        trajectory_change[:, -1].mean()
                    )
        return metrics

    def train(self):
        """execute training loop"""
        # print training config
        self._log_training_config()

        # prepare data iterators
        self._create_data_iterators()

        # create progress bar
        progress_bar = tqdm(
            range(self.initial_step, self.config.trainer.max_train_steps),
            disable=not self.accelerator.is_local_main_process,
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
            except KeyError:
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
            logger.info(f"  Initial global step = {self.initial_step}")
            logger.info(f"  Remaining optimization steps = {self.remaining_steps}")
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

                action_loss = output_dict["action_loss"]
                # total_loss = action_loss

                if self.config.datasets.video_data.load_2d_data == 1:
                    rgb_loss = output_dict['rgb_loss']
                
                if self.config.datasets.gs_data.load_3d_data == 1 or self.config.w_depth:
                    gs_loss = output_dict['gs_loss']
                
                if self.config.datasets.reward_data.load_reward_data == 1:
                    reward_loss = output_dict['reward_loss']
                agent_dino_loss = output_dict.get('agent_dino_loss', torch.tensor(0.0, device=action_loss.device))

                if "losses" in output_dict:
                    total_loss, weighted_named_losses = aggregate_output_losses(
                        output_dict,
                        self.config,
                        optimizer_step=self.completed_steps,
                    )
                else:
                    weighted_named_losses = {}
                    total_loss = 0
                    if self.config.datasets.video_data.load_2d_data == 1:
                        total_loss += rgb_loss
                    if self.config.datasets.gs_data.load_3d_data == 1 or self.config.w_depth:
                        total_loss += gs_loss
                    if self.config.datasets.reward_data.load_reward_data == 1:
                        total_loss += reward_loss
                    if self.config.datasets.vla_data.load_act_data == 1:
                        total_loss += action_loss
                    if getattr(self.config.framework, "action_prompt_mode", "full") == "minimal_agent":
                        total_loss += agent_dino_loss


            # VLA backward propagation
            self.accelerator.backward(total_loss)
            raw_model = self.accelerator.unwrap_model(self.model)
            if hasattr(raw_model, "get_planning_usage_metrics"):
                output_dict.setdefault("backward_metrics", {}).update(
                    raw_model.get_planning_usage_metrics()
                )
            if hasattr(raw_model, "get_backbone_training_metrics"):
                output_dict.setdefault("backward_metrics", {}).update(
                    raw_model.get_backbone_training_metrics()
                )
            # for debug
            # self._report_unused_after_backward()

            # gradient clipping
            if self.accelerator.sync_gradients:
                if self.config.trainer.gradient_clipping is not None:
                    self.accelerator.clip_grad_norm_(self.model.parameters(), self.config.trainer.gradient_clipping)

                # optimizer step
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()
                output_dict.setdefault("metrics", {}).update(
                    self._run_vggt_intervention_diagnostics(batch_vla)
                )

        step_metrics = {
            "action_dit_loss": action_loss.detach(),
            "rgb_gen_loss": 0 if self.config.datasets.video_data.load_2d_data == 0 else rgb_loss.detach(),
            "gs_loss": 0 if self.config.datasets.gs_data.load_3d_data == 0 and self.config.w_depth==0 else gs_loss.detach(),
            "reward_loss": 0 if self.config.datasets.reward_data.load_reward_data == 0 else reward_loss.detach(),
            "agent_dino_loss": 0 if getattr(self.config.framework, "action_prompt_mode", "full") != "minimal_agent" else agent_dino_loss.detach()
        }
        for name, value in output_dict.get("metrics", {}).items():
            if torch.is_tensor(value) and value.numel() == 1:
                step_metrics[f"vggt/{name}"] = value.detach()
        for name, value in output_dict.get("backward_metrics", {}).items():
            if torch.is_tensor(value) and value.numel() == 1:
                step_metrics[name] = value.detach()
        for name, value in weighted_named_losses.items():
            step_metrics[f"weighted_loss/{name}"] = value.detach()
        return step_metrics

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
        for fname in (
            'debug.sh',
            '8-train.sh',
            '8-train_action-only-qwen-visual.sh',
            '8-continue_action-only-qwen-visual-200k.sh',
            'training.sh',
            'pre_cache.sh',
        ):
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
    parser.add_argument(
        "--config_overlay",
        action="append",
        default=[],
        help="Optional YAML overlay; may be repeated. CLI values still have highest precedence.",
    )
    args, clipargs = parser.parse_known_args()

    # Load YAML config & Convert CLI overrides to dotlist config
    cfg = OmegaConf.load(args.config_yaml)
    for overlay_path in args.config_overlay:
        cfg = OmegaConf.merge(cfg, OmegaConf.load(overlay_path))
    apply_environment_path_overrides(cfg)
    dotlist = normalize_dotlist_args(clipargs)  # Normalize CLI args to dotlist format
    cli_cfg = OmegaConf.from_dotlist(dotlist)
    cfg = OmegaConf.merge(cfg, cli_cfg)
    if bool(
        OmegaConf.select(
            cfg,
            "framework.qwenvl.visual_action_only_experiment",
            default=False,
        )
    ):
        from starVLA.model.modules.vlm.visual_training import (
            validate_visual_action_only_config,
        )

        validate_visual_action_only_config(cfg)

    # if cfg.is_debug:
    # if cfg.is_debug and dist.is_initialized() and dist.get_rank() == 0:
    #     import debugpy
    #     debugpy.listen(("0.0.0.0", 10092))
    #     print("🔍 Rank 0 waiting for debugger attach on port 10092...")
    #     debugpy.wait_for_client()

    main(cfg)

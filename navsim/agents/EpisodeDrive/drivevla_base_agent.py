from typing import Any, List, Dict, Optional, Union
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import os
import torch
from torch.optim import Optimizer
import torch.optim as optim
from torch.optim.lr_scheduler import LRScheduler
from omegaconf import DictConfig, OmegaConf
from transformers.feature_extraction_utils import BatchFeature
import math
import sys
import pickle
import time
from pathlib import Path
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from navsim.planning.training.dataset import load_feature_target_from_pickle
from navsim.planning.training.input_only_cache import (
    reject_dynamic_feature_cache,
    validate_input_only_cache_policy,
)
from pytorch_lightning.callbacks import Callback, ModelCheckpoint, ProgressBar, LearningRateMonitor
from navsim.common.dataloader import MetricCacheLoader

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import AgentInput, SensorConfig, Trajectory
from navsim.planning.training.abstract_feature_target_builder import AbstractFeatureBuilder, AbstractTargetBuilder
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from pytorch_lightning.callbacks import ModelCheckpoint, ProgressBar, LearningRateMonitor

from .utils.internvl_preprocess import load_image
from .utils.lr_scheduler import WarmupCosLR
from .utils.utils import build_drivevla_questions, build_from_configs
from .drivevla_features import DriveVLAFeatureBuilder ,TrajectoryTargetBuilder
from .drivevla_backbone import (
    DriveVLABackbone,
    load_legacy_checkpoint_with_planreg_audit,
)
from .formal_initialization import (
    FORMAL_INITIALIZATION_MODE,
    sha256_file,
    validate_formal_initialization_config,
)
from .shared_planreg_initialization import load_shared_trainable_initialization
from .action_decoder import ActionDecoder
from .layers.planning_registers import freeze_vision_except_qv_lora
from .layers.planning_registers.register_diagnostics import (
    compute_register_diagnostics,
)
from .layers.world_model import (
    EMARegisterTarget,
    EMARegisterTargetCallback,
    FutureRegisterPredictor,
    cosine_ema_momentum,
    decode_path_tensor,
)

from peft import LoraConfig, get_peft_model


def planreg_warmup_cosine_multiplier(
    optimizer_step: int,
    *,
    total_optimizer_steps: int,
    warmup_ratio: float,
    start_lr_ratio: float,
    min_lr_ratio: float,
) -> float:
    """Resume-safe step-indexed warmup/cosine multiplier."""
    if total_optimizer_steps <= 0:
        raise ValueError("total_optimizer_steps must be positive")
    if not 0.0 <= warmup_ratio < 1.0:
        raise ValueError("warmup_ratio must be in [0,1)")
    if not 0.0 <= start_lr_ratio <= 1.0:
        raise ValueError("start_lr_ratio must be in [0,1]")
    if not 0.0 <= min_lr_ratio <= 1.0:
        raise ValueError("min_lr_ratio must be in [0,1]")
    last_step = max(1, int(total_optimizer_steps) - 1)
    warmup_steps = min(
        last_step, max(1, int(round(total_optimizer_steps * warmup_ratio)))
    )
    step = min(last_step, max(0, int(optimizer_step)))
    if step <= warmup_steps:
        progress = step / warmup_steps
        return float(start_lr_ratio + (1.0 - start_lr_ratio) * progress)
    decay_steps = max(1, last_step - warmup_steps)
    progress = (step - warmup_steps) / decay_steps
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return float(min_lr_ratio + (1.0 - min_lr_ratio) * cosine)

class LitProgressBar(ProgressBar):

    def __init__(self):
        super().__init__()  # don't forget this :)
        self.enable = True

    def disable(self):
        self.enable = False

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)
        if batch_idx%100 == 0:
            print(f"Epoch {trainer.current_epoch} - train {batch_idx} / {self.total_train_batches} - {self.get_metrics(trainer, pl_module)}")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx)
        if batch_idx%100 == 0:
            print(f"Epoch {trainer.current_epoch} - val {batch_idx} / {self.total_train_batches} - {self.get_metrics(trainer, pl_module)}")

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        super().on_train_epoch_end(self, pl_module)
        metrics = self.get_metrics(trainer, pl_module)
        train_metrics = dict()
        val_metrics = dict()
        other_metrics = dict()
        for k,v in metrics.items():
            if "train/" in k:
                train_metrics[k]=v
            elif "val/" in k:
                val_metrics[k]=v
            else:
                other_metrics[k]=v
        print(f"\n###########  Epoch {trainer.current_epoch} ##########")
        for k,v in train_metrics.items():
            print(f"{k},{v:.3f}")
        for k,v in val_metrics.items():
            print(f"{k},{v:.3f}")
        for k,v in other_metrics.items():
            print(f"{k},{v:.3f}")
        print(f"###########\n")


class TrainingThroughputCallback(Callback):
    """Report end-to-end DDP throughput over stable multi-step windows."""

    def __init__(self, interval: int, warmup: int = 5):
        super().__init__()
        self.interval = interval
        self.warmup = warmup
        self._start_step = None
        self._start_time = None

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        completed = batch_idx + 1
        if completed == self.warmup:
            self._start_step = completed
            self._start_time = time.perf_counter()
            return
        if (
            self._start_time is not None
            and completed - self._start_step >= self.interval
        ):
            now = time.perf_counter()
            steps = completed - self._start_step
            if trainer.is_global_zero:
                print(
                    "TRAIN_THROUGHPUT "
                    f"epoch={trainer.current_epoch} step={trainer.global_step} "
                    f"seconds_per_step={(now - self._start_time) / steps:.6f} "
                    f"steps_per_second={steps / (now - self._start_time):.6f}",
                    flush=True,
                )
            self._start_step = completed
            self._start_time = now


class EfficientBestAndLastCheckpoint(ModelCheckpoint):
    """Keep exact best/latest states while writing only once per epoch.

    Lightning's ``save_last="link"`` links to the most recently *saved* top-k
    checkpoint.  When an epoch is not a new best, that target is stale and is
    therefore not a valid latest resume point.  A new best is written once and
    linked as ``last.ckpt``; otherwise ``last.ckpt`` is overwritten once while
    the prior best remains untouched.
    """

    def _save_topk_checkpoint(self, trainer, monitor_candidates) -> None:
        super()._save_topk_checkpoint(trainer, monitor_candidates)
        # Always refresh latest. ModelCheckpoint does not restore its private
        # _last_checkpoint_saved field when resuming a run.
        if self.save_last:
            self._save_last_checkpoint(trainer, monitor_candidates)

    def _save_last_checkpoint(self, trainer, monitor_candidates) -> None:
        filepath = self.format_checkpoint_name(
            monitor_candidates, self.CHECKPOINT_NAME_LAST
        )
        if self._enable_version_counter:
            version_cnt = self.STARTING_VERSION
            while self.file_exists(filepath, trainer) and filepath != self.last_model_path:
                filepath = self.format_checkpoint_name(
                    monitor_candidates,
                    self.CHECKPOINT_NAME_LAST,
                    ver=version_cnt,
                )
                version_cnt += 1

        # The base on_validation_end hook invokes _save_last_checkpoint again
        # after _save_topk_checkpoint.  A non-best epoch was already written
        # directly to this path by our first invocation.
        if (
            self._last_global_step_saved == trainer.global_step
            and self._last_checkpoint_saved
            and os.path.abspath(self._last_checkpoint_saved)
            == os.path.abspath(filepath)
        ):
            return

        previous, self.last_model_path = self.last_model_path, filepath
        current_epoch_was_saved_as_best = (
            self._last_global_step_saved == trainer.global_step
            and bool(self._last_checkpoint_saved)
            and os.path.abspath(self._last_checkpoint_saved)
            != os.path.abspath(filepath)
        )
        if current_epoch_was_saved_as_best:
            self._link_checkpoint(trainer, self._last_checkpoint_saved, filepath)
        else:
            # Do not follow a prior symlink and overwrite the retained best.
            if trainer.is_global_zero and os.path.islink(filepath):
                os.remove(filepath)
            trainer.strategy.barrier()
            self._save_checkpoint(trainer, filepath)

        if previous and self._should_remove_checkpoint(
            trainer, previous, filepath
        ):
            self._remove_checkpoint(trainer, previous)


class DriveVLABaseAgent(AbstractAgent):
    def __init__(
        self,
        vlm_config,
        lora_config,
        action_head_config,
        vision_adaptation=None,
        planning_registers=None,
        scene_fusion=None,
        world_model=None,
        ema=None,
        initialization=None,
        cache_policy=None,
        semantic_path=None,
        lr_args=None,
        loss=None,
        progress_bar=True,
        scheduler_args: dict=None,
        batch_size: int=64,
        num_gpus: int=1,
        trajectory_sampling=None,
        checkpoint_path:str = None,
        stage1_checkpoint_path: str = None,
        cache_data: bool = False,
    ):
        super().__init__()
        self.action_head_config=action_head_config
        self.vlm_config=vlm_config
        self.lora_config=lora_config
        self.vision_adaptation = vision_adaptation
        self.planning_registers_config = planning_registers
        self.scene_fusion = scene_fusion
        self.world_model_config = world_model
        self.ema_config = ema
        self.initialization_config = initialization
        self.cache_policy = cache_policy
        self.semantic_path_config = semantic_path
        self.world_model_enabled = bool(
            world_model is not None and getattr(world_model, "enabled", False)
        )
        self.future_mode = (
            str(getattr(world_model, "future_mode", "correct"))
            if self.world_model_enabled
            else "disabled"
        )
        if self.future_mode not in {
            "disabled",
            "correct",
            "no_action_condition",
            "shuffled_batch",
            "repeated_current",
        }:
            raise ValueError(f"Unsupported world_model.future_mode={self.future_mode!r}")

        self._lr_args=lr_args
        self.progress_bar=progress_bar
        self.scheduler_args=scheduler_args
        self.batch_size=batch_size
        self.num_gpus=num_gpus
        self.checkpoint_path=checkpoint_path
        self.stage1_checkpoint_path = stage1_checkpoint_path
        self._formal_initialization = (
            initialization is not None
            and str(getattr(initialization, "mode", ""))
            == FORMAL_INITIALIZATION_MODE
        )
        self._formal_initialization_audit = None
        self._agent_checkpoint_loaded = False
        self._shared_trainable_initialization_metadata = None
        if self._formal_initialization:
            validate_input_only_cache_policy(cache_policy)
            self._formal_initialization_audit = validate_formal_initialization_config(
                initialization,
                checkpoint_path=checkpoint_path,
                stage1_checkpoint_path=stage1_checkpoint_path,
                vlm_config=vlm_config,
            )
        self._dynamic_feature_cache_guard_enabled = bool(
            self._formal_initialization
            and cache_policy is not None
            and str(getattr(cache_policy, "mode", "")) == "input_only"
        )
        self.cache_data = cache_data
        self._initialized = False
        self._latest_registers_for_diagnostics = None

        self.future_register_predictor = None
        self.ema_register_target = None
        if self.world_model_enabled:
            if self.vlm_config.cache_hidden_state or self.vlm_config.cache_mode:
                raise ValueError(
                    "PlanReg-WM-V1 requires cache_hidden_state=false and cache_mode=false"
                )
            if not bool(getattr(self.vlm_config, "planning_registers_enabled", False)):
                raise ValueError("World-model training requires planning_registers_enabled=true")
            if self.ema_config is None or not bool(
                getattr(self.ema_config, "enabled", False)
            ):
                raise ValueError("World-model training requires ema.enabled=true")
            self.future_register_predictor = FutureRegisterPredictor(
                hidden_dim=int(getattr(world_model, "hidden_dim", 256)),
                predictor_layers=int(getattr(world_model, "predictor_layers", 2)),
                horizons_sec=tuple(
                    float(value)
                    for value in getattr(
                        world_model, "horizons_sec", (0.5, 1.5, 3.0)
                    )
                ),
                normalize_state_space=bool(
                    getattr(world_model, "normalize_state_space", True)
                ),
                x_scale=float(getattr(world_model, "x_scale", 30.0)),
                y_scale=float(getattr(world_model, "y_scale", 10.0)),
                speed_scale=float(getattr(world_model, "speed_scale", 15.0)),
                acceleration_scale=float(
                    getattr(world_model, "acceleration_scale", 8.0)
                ),
            )
            self.register_buffer(
                "_ema_optimizer_step",
                torch.zeros((), dtype=torch.long),
                persistent=True,
            )
            self.register_buffer(
                "_world_model_optimizer_step",
                torch.zeros((), dtype=torch.long),
                persistent=True,
            )
            self.register_buffer(
                "_world_model_total_optimizer_steps",
                torch.zeros((), dtype=torch.long),
                persistent=True,
            )

        if self.checkpoint_path and self.stage1_checkpoint_path:
            raise ValueError(
                "checkpoint_path (full-agent restore) and stage1_checkpoint_path "
                "(VLM-only warm start) are mutually exclusive."
            )

        if not self.cache_data:
            self.action_head = ActionDecoder(
                action_head_config,
                scene_fusion_config=self.scene_fusion,
                # Lightning supplies its authoritative stepping-batch estimate
                # after dataloaders and gradient accumulation are configured.
                total_optimizer_steps=None,
            )

        if not self.cache_data and self.action_head_config.checkpoint_path=="":
            self.bce_logit_loss=nn.BCEWithLogitsLoss

            # Training-time oracle scoring can use Ray, but starting a local Ray
            # cluster in every Lightning DDP inference rank is unnecessary and
            # can make the released multi-GPU evaluator fail.  Keep the public
            # behavior by default and allow deployment jobs to opt out.
            self.ray = os.getenv("DRIVEVLA_SCORE_RAY", "1").lower() not in {
                "0",
                "false",
                "no",
                "off",
            }
            self.score_process_count = int(os.getenv("DRIVEVLA_SCORE_PROCESSES", "0"))
            self.score_partition_count = int(os.getenv("DRIVEVLA_SCORE_PARTITIONS", "1"))
            self.score_start_method = os.getenv(
                "DRIVEVLA_SCORE_START_METHOD", "spawn"
            )
            if self.score_process_count < 0:
                raise ValueError("DRIVEVLA_SCORE_PROCESSES must be non-negative")
            if self.score_partition_count < 1:
                raise ValueError("DRIVEVLA_SCORE_PARTITIONS must be positive")
            if self.score_start_method not in {"spawn", "forkserver"}:
                raise ValueError(
                    "DRIVEVLA_SCORE_START_METHOD must be spawn or forkserver"
                )
            if self.ray and self.score_process_count:
                raise ValueError(
                    "DRIVEVLA_SCORE_RAY and DRIVEVLA_SCORE_PROCESSES cannot both be enabled"
                )
            self._score_process_pool = None

            if self.ray:
                from navsim.planning.utils.multithreading.worker_ray_no_torch import RayDistributedNoTorch
                from nuplan.planning.utils.multithreading.worker_utils import worker_map
                self.worker = RayDistributedNoTorch(threads_per_node=8)
                self.worker_map=worker_map

            from .score_module.compute_navsim_score import get_scores, get_sub_score

            self.score_metric_cache_path = Path(
                os.getenv(
                    "NAVSIM_TRAIN_METRIC_CACHE",
                    str(Path(os.getenv("NAVSIM_EXP_ROOT", "outputs")) / "train_metric_cache_Haswell"),
                )
            )
            self.train_metric_cache_paths = {}
            self.test_metric_cache_paths = {}
            if self.score_metric_cache_path.exists():
                metric_cache = MetricCacheLoader(self.score_metric_cache_path)
                self.train_metric_cache_paths = metric_cache.metric_cache_paths
                self.test_metric_cache_paths = metric_cache.metric_cache_paths
            else:
                print(
                    "Score metric cache not found at "
                    f"{self.score_metric_cache_path}. "
                    "Set NAVSIM_TRAIN_METRIC_CACHE when training with score loss."
                )

            self.get_scores = get_scores
            self.get_sub_score = get_sub_score

            self.loss = loss

        self._trajectory_sampling = trajectory_sampling
        self.backbone = None
        
        local_rank = int(os.getenv("LOCAL_RANK", "0"))
        device = f"cuda:{local_rank}"
        self.device = device
        if (
            not self.cache_data
            and not self.vlm_config.cache_hidden_state
            and not self.vlm_config.cache_mode
        ):
            print("Agent running in 'no-cache' mode. Initializing internal backbone.")
            if not self.vlm_config.vlm_path or not self.vlm_config.vlm_type:
                raise ValueError("In 'no-cache' mode, vlm_path and vlm_type are required.")
            vision_mode = (
                getattr(self.vision_adaptation, "mode", "none")
                if self.vision_adaptation is not None
                else "none"
            )
            vision_qv_lora_enabled = bool(
                getattr(self.vlm_config, "vision_qv_lora_enabled", False)
                or vision_mode == "qv_lora"
            )
            if vision_qv_lora_enabled and self.lora_config.use_lora:
                raise ValueError(
                    "PlanReg-WM-V1 Q/V LoRA cannot be combined with the legacy "
                    "whole-VLM PEFT target_modules path. Set lora_config.use_lora=false."
                )
            self.backbone = DriveVLABackbone(
                model_type=self.vlm_config.vlm_type,
                checkpoint_path=self.vlm_config.vlm_path,
                device=device,
                extra_token_count=int(
                    getattr(self.vlm_config, "extra_token_count", 0)
                ),
                target_vocab_size=getattr(
                    self.vlm_config, "target_vocab_size", None
                ),
                use_flash_attn=bool(
                    getattr(self.vlm_config, "use_flash_attn", True)
                ),
                initialize_from_config=bool(
                    getattr(self.vlm_config, "initialize_from_config", False)
                ),
                skip_lm_head=bool(
                    getattr(self.vlm_config, "skip_lm_head", False)
                ),
                gradient_checkpointing=bool(
                    getattr(self.vlm_config, "gradient_checkpointing", False)
                ),
                planning_registers_enabled=bool(
                    getattr(self.vlm_config, "planning_registers_enabled", False)
                ),
                num_planning_registers=int(
                    getattr(self.vlm_config, "num_planning_registers", 16)
                ),
                planning_register_dim=int(
                    getattr(self.vlm_config, "planning_register_dim", 256)
                ),
                tile_register_aggregation=getattr(
                    self.vlm_config, "tile_register_aggregation", "mean"
                ),
                planning_register_attention_mode=str(
                    getattr(
                        self.planning_registers_config,
                        "attention_mode",
                        "bidirectional",
                    )
                    if self.planning_registers_config is not None
                    else "bidirectional"
                ),
                vision_qv_lora_enabled=vision_qv_lora_enabled,
                vision_qv_lora_rank=int(
                    getattr(
                        self.vision_adaptation,
                        "rank",
                        getattr(self.vlm_config, "vision_qv_lora_rank", 32),
                    )
                    if self.vision_adaptation is not None
                    else getattr(self.vlm_config, "vision_qv_lora_rank", 32)
                ),
                vision_qv_lora_dropout=float(
                    getattr(self.vision_adaptation, "dropout", 0.0)
                    if self.vision_adaptation is not None
                    else 0.0
                ),
                strict_vocab_alignment=bool(
                    getattr(self.vlm_config, "strict_vocab_alignment", False)
                ),
                semantic_frozen_llm_no_grad=bool(
                    getattr(
                        self.semantic_path_config,
                        "frozen_llm_no_grad",
                        False,
                    )
                    if self.semantic_path_config is not None
                    else False
                ),
                semantic_backprop_to_vision=bool(
                    getattr(
                        self.semantic_path_config,
                        "backprop_to_vision",
                        True,
                    )
                    if self.semantic_path_config is not None
                    else True
                ),
            )
            
            if self.lora_config.use_lora:
                self.backbone = self._apply_lora_to_backbone(self.backbone)
                self._freeze_backbone_for_lora()

        self.num_inference_samples = 1
        self.inference_selection_mode = "median"

    def name(self) -> str:
        return self.__class__.__name__

    def set_memory_attention(
        self, memory_attention: Optional[nn.Module]
    ) -> None:
        """Attach Attention Memory after the legacy checkpoint is initialized."""
        self.action_head.set_memory_attention(memory_attention)

    def set_optimizer_step(self, optimizer_step: int) -> None:
        """Synchronize step-dependent modules with Lightning's resume-safe step."""
        if not self.cache_data:
            self.action_head.set_optimizer_step(optimizer_step)
        if self.world_model_enabled:
            self._world_model_optimizer_step.fill_(int(optimizer_step))

    def configure_total_optimizer_steps(self, total_optimizer_steps: int) -> None:
        if not self.cache_data:
            self.action_head.configure_total_optimizer_steps(
                total_optimizer_steps
            )
        if self.world_model_enabled:
            self._world_model_total_optimizer_steps.fill_(
                int(total_optimizer_steps)
            )

    def current_world_model_weight(self) -> float:
        """Return the resume-safe optimizer-step WM coefficient."""
        if not self.world_model_enabled:
            return 0.0
        if not hasattr(self.world_model_config, "max_weight"):
            # Old PlanReg checkpoints/configs used one fixed coefficient.
            return float(getattr(self.world_model_config, "weight", 0.25))
        total_steps = int(self._world_model_total_optimizer_steps.item())
        if total_steps <= 0:
            raise RuntimeError(
                "World-model weight schedule requires total optimizer steps"
            )
        step = int(self._world_model_optimizer_step.item())
        start_fraction = float(
            getattr(self.world_model_config, "start_fraction", 0.05)
        )
        ramp_fraction = float(
            getattr(self.world_model_config, "ramp_fraction", 0.10)
        )
        max_weight = float(self.world_model_config.max_weight)
        if not 0.0 <= start_fraction <= 1.0:
            raise ValueError("world_model.start_fraction must be in [0,1]")
        if not 0.0 < ramp_fraction <= 1.0:
            raise ValueError("world_model.ramp_fraction must be in (0,1]")
        start_step = start_fraction * total_steps
        ramp_steps = max(1.0, ramp_fraction * total_steps)
        progress = min(1.0, max(0.0, (step - start_step) / ramp_steps))
        return max_weight * progress

    def _initialize_ema_register_target(self) -> None:
        if not self.world_model_enabled:
            return
        if self.ema_register_target is not None:
            return
        if self.backbone is None:
            raise RuntimeError("Cannot initialize EMA target without a student backbone")
        # This is intentionally called only after legacy/student checkpoint
        # restoration and custom Q/V LoRA construction.
        self.ema_register_target = EMARegisterTarget(self.backbone)
        self.ema_register_target.eval()
        print(
            "✅ Initialized training-only EMA target with InternViT, Q/V LoRA, "
            "planning registers and neck (no LLM/Q-Former/action/scorer modules)."
        )

    @torch.no_grad()
    def update_ema_after_optimizer_step(
        self,
        optimizer_step: int,
        total_optimizer_steps: int,
    ) -> None:
        if not self.world_model_enabled or self.ema_register_target is None:
            return
        completed_step = int(optimizer_step)
        if completed_step <= int(self._ema_optimizer_step.item()):
            return
        start = float(getattr(self.ema_config, "start_momentum", 0.996))
        end = float(getattr(self.ema_config, "end_momentum", 0.9999))
        momentum = cosine_ema_momentum(
            completed_step,
            total_optimizer_steps,
            start=start,
            end=end,
        )
        self.ema_register_target.update(self.backbone, momentum)
        self._ema_optimizer_step.fill_(completed_step)

    def remove_training_only_world_model(self) -> None:
        """Strip predictor/EMA modules from a deployment-only process."""
        self.ema_register_target = None
        self.future_register_predictor = None
    
    def _apply_lora_to_backbone(self, backbone):
        """Apply LoRA to the backbone."""
        lora_config = LoraConfig(
            r=self.lora_config.lora_rank,
            lora_alpha=2*self.lora_config.lora_rank,
            target_modules=self.lora_config.lora_target_modules,
            lora_dropout=self.lora_config.lora_dropout,
            bias="none",
        )
        lora_backbone = get_peft_model(backbone, lora_config)
        lora_module_count = sum(
            1 for name, _ in lora_backbone.named_modules() if "lora" in name
        )
        print(f"LoRA applied to backbone ({lora_module_count} LoRA modules).")
        
        return lora_backbone
    
    def _freeze_backbone_for_lora(self,freeze_vision=True):
        if self.backbone is None:
            return
        
        if self.lora_config.use_lora:
            # LoRA mode: freeze all parameters except LoRA adapter
            for name, param in self.backbone.named_parameters():
                if "lora" not in name:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
            
            self.backbone.eval()
            
            # Print trainable parameter statistics
            print("Trainable parameters in LoRA backbone:")
            trainable_params = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in self.backbone.parameters())
            print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({trainable_params/total_params:.2%})")
        else:
            self._freeze_backbone_selective()
            
            
    def _freeze_backbone(self):
        """冻结backbone所有参数"""
        if self.backbone is None:
            return
        
        # 设置所有参数不更新梯度
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # 设置为评估模式（禁用dropout、batchnorm更新）
        self.backbone.eval()
        
        # 可选：打印冻结信息
        frozen_params = sum(p.numel() for p in self.backbone.parameters())
        print(f"✅ Backbone冻结完成：{frozen_params:,} 个参数已冻结")

    def _freeze_backbone_for_planreg(self) -> None:
        """Freeze the VLM and enable only planning neck/registers and Q/V LoRA."""
        if self.backbone is None:
            return
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

        adapter = self.backbone.planning_register_adapter
        if adapter is None:
            raise RuntimeError(
                "planning_registers_enabled=true but no planning adapter exists"
            )
        for parameter in adapter.parameters():
            parameter.requires_grad = True

        if self.backbone.vision_qv_lora_enabled:
            freeze_vision_except_qv_lora(self.backbone.model.vision_model)

        leaked_language = [
            name
            for name, parameter in self.backbone.named_parameters()
            if "language_model" in name and parameter.requires_grad
        ]
        if leaked_language:
            raise RuntimeError(
                "PlanReg-WM-V1 must freeze the LLM; trainable language keys: "
                f"{leaked_language[:8]}"
            )
        trainable = sum(
            parameter.numel()
            for parameter in self.backbone.parameters()
            if parameter.requires_grad
        )
        print(
            "✅ PlanReg backbone frozen except planning registers/neck and "
            f"vision Q/V LoRA: {trainable:,} trainable parameters"
        )

    def _freeze_lm_head(self) -> None:
        """Freeze the language decoder head while keeping the VLM trainable."""
        if self.backbone is None:
            return
        output_embeddings = self.backbone.model.language_model.get_output_embeddings()
        if output_embeddings is None:
            raise RuntimeError("VLM has no output embeddings to freeze as lm_head")
        for parameter in output_embeddings.parameters():
            parameter.requires_grad = False

        frozen = sum(parameter.numel() for parameter in output_embeddings.parameters())
        leaked = [
            name
            for name, parameter in self.backbone.named_parameters()
            if "lm_head" in name and parameter.requires_grad
        ]
        if leaked:
            raise RuntimeError(f"lm_head parameters remain trainable: {leaked[:5]}")
        print(f"✅ Frozen VLM lm_head: {frozen:,} parameters")

    def _report_backbone_trainability(self) -> None:
        if self.backbone is None:
            return
        totals = {
            "vision": [0, 0],
            "projector": [0, 0],
            "language": [0, 0],
            "lm_head": [0, 0],
            "other": [0, 0],
        }
        for name, parameter in self.backbone.named_parameters():
            if "lm_head" in name:
                group = "lm_head"
            elif "vision_model" in name:
                group = "vision"
            elif ".mlp1." in name:
                group = "projector"
            elif "language_model" in name:
                group = "language"
            else:
                group = "other"
            totals[group][0] += parameter.numel()
            if parameter.requires_grad:
                totals[group][1] += parameter.numel()
        print("VLM parameter trainability:")
        for group, (total, trainable) in totals.items():
            if total:
                print(f"  {group}: {trainable:,} / {total:,} trainable")

    def train(self, mode: bool = True):
        """Keep a paper-style frozen Stage-1 VLM in inference mode."""
        super().train(mode)
        if (
            self.backbone is not None
            and bool(getattr(self.vlm_config, "freeze_backbone", False))
        ):
            self.backbone.eval()
            if mode and bool(
                getattr(self.vlm_config, "gradient_checkpointing", False)
            ):
                self.backbone.activate_gradient_checkpointing_train_mode()
        if self.ema_register_target is not None:
            self.ema_register_target.eval()
        return self
        
    def _freeze_backbone_selective(self):
        """选择性冻结backbone参数"""
        if self.backbone is None:
            return
        
        # 默认冻结所有参数
        for name, param in self.backbone.named_parameters():
            param.requires_grad = False
        
        # 解冻指定的层
        for layer_name in self.vlm_config.trainable_layers:
            for name, param in self.backbone.named_parameters():
                if layer_name in name:
                    param.requires_grad = True
                    print(f"🔓 解冻层: {name}")
        
        # 统计信息
        total_params = sum(p.numel() for p in self.backbone.parameters())
        trainable_params = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        print(f"📊 Backbone参数统计: {trainable_params:,}/{total_params:,} 可训练")

    def _load_stage1_backbone(self, checkpoint_path: str) -> None:
        """Strictly restore only the VQA-pretrained VLM from a merged checkpoint.

        The Compress/Q-Former, trajectory decoder, and scorer all live under
        ``action_head`` and are intentionally left at their seeded random
        initialization for Stage 2.
        """
        if self.backbone is None:
            raise RuntimeError("Cannot warm-start Stage 1 because no backbone is initialized.")

        payload = torch.load(checkpoint_path, map_location="cpu")
        source_state = payload.get("state_dict", payload)
        target_state = self.backbone.state_dict()
        backbone_state = {}
        shape_errors = []

        for key, value in source_state.items():
            if key.startswith("agent.backbone."):
                normalized_key = key[len("agent.backbone."):]
            elif key.startswith("backbone."):
                normalized_key = key[len("backbone."):]
            else:
                normalized_key = key

            if normalized_key not in target_state:
                continue
            if value.shape != target_state[normalized_key].shape:
                shape_errors.append(
                    f"{normalized_key}: checkpoint={tuple(value.shape)}, "
                    f"model={tuple(target_state[normalized_key].shape)}"
                )
                continue
            backbone_state[normalized_key] = value

        missing = sorted(set(target_state) - set(backbone_state))
        if shape_errors or missing:
            details = []
            if shape_errors:
                details.append("shape mismatches: " + "; ".join(shape_errors[:10]))
            if missing:
                details.append(
                    f"missing {len(missing)} backbone tensors: " + ", ".join(missing[:10])
                )
            raise RuntimeError("Invalid Stage-1 VLM checkpoint; " + " | ".join(details))

        self.backbone.load_state_dict(backbone_state, strict=True)
        parameter_count = sum(
            target_state[key].numel() for key in backbone_state
        )
        print(
            "✅ Stage-1 VLM-only warm start loaded "
            f"{len(backbone_state):,} tensors / {parameter_count:,} values from: "
            f"{checkpoint_path}"
        )
        print(
            "✅ Stage-2 action_head was not restored; Compress/Q-Former, "
            "trajectory head, and scorer retain seeded random initialization."
        )

    def initialize(self) -> None:
        if self._initialized:
            return

        if self.checkpoint_path:
            ckpt = torch.load(self.checkpoint_path, map_location="cpu")["state_dict"]
            if self.world_model_enabled and any(
                _key.startswith("agent.ema_register_target.")
                or _key.startswith("ema_register_target.")
                for _key in ckpt
            ):
                # A PlanReg training checkpoint carries an EMA state. Build the
                # exact training-only topology so strict restoration can load
                # it; legacy/base checkpoints initialize EMA after student load.
                self._initialize_ema_register_target()
            load_legacy_checkpoint_with_planreg_audit(
                self,
                ckpt,
                legacy_lora_scale=2.0,
            )
            self._agent_checkpoint_loaded = True
            print(f"✅ Agent loaded from checkpoint: {self.checkpoint_path}")
        elif self.stage1_checkpoint_path:
            self._load_stage1_backbone(self.stage1_checkpoint_path)
            
        if bool(getattr(self.vlm_config, "planning_registers_enabled", False)):
            self._freeze_backbone_for_planreg()
        elif self.vlm_config.freeze_backbone:
            self._freeze_backbone()
        elif bool(getattr(self.vlm_config, "freeze_lm_head", False)):
            self._freeze_lm_head()
            # ``from_pretrained`` initializes InternVL in eval mode. Full
            # fine-tuning must restore training mode before Lightning inspects
            # module state (the frozen linear lm_head has no mode-dependent
            # behavior).
            self.backbone.train()
        if self._formal_initialization:
            shared_path = getattr(
                self.initialization_config,
                "shared_trainable_init_path",
                None,
            )
            if not shared_path:
                raise RuntimeError(
                    "Formal PlanReg training requires initialization."
                    "shared_trainable_init_path. Generate it once and use the same "
                    "artifact for BaseInit and VQAInit."
                )
            self._shared_trainable_initialization_metadata = (
                load_shared_trainable_initialization(self, str(shared_path))
            )
            self._shared_trainable_initialization_metadata["artifact_path"] = str(
                Path(str(shared_path)).expanduser().resolve()
            )
            self._shared_trainable_initialization_metadata["artifact_sha256"] = (
                sha256_file(Path(str(shared_path)).expanduser().resolve())
            )
            print(
                "✅ Restored shared formal PlanReg initialization: "
                f"{self._shared_trainable_initialization_metadata['trainable_state_key_count']} "
                "trainable tensors, SHA-256="
                f"{self._shared_trainable_initialization_metadata['artifact_sha256']}"
            )
        self._report_backbone_trainability()
        self._initialize_ema_register_target()
        self._initialized = True

    def get_sensor_config(self) -> SensorConfig:
        def _history(name: str) -> List[int]:
            values = getattr(self.action_head_config, name, [])
            return list(values) if values else []

        return SensorConfig(
            # Future-register targets use current+0.5/1.5/3.0 s front views.
            # Loading all front-view frame paths avoids assuming a fixed
            # history length in SensorConfig's absolute iteration indices.
            cam_f0=True if self.world_model_enabled else _history("cam_f0"),
            cam_l0=_history("cam_l0"),
            cam_l1=_history("cam_l1"),
            cam_l2=_history("cam_l2"),
            cam_r0=_history("cam_r0"),
            cam_r1=_history("cam_r1"),
            cam_r2=_history("cam_r2"),
            cam_b0=_history("cam_b0"),
            lidar_pc=_history("lidar_pc"),
        )

    def get_target_builders(self) -> List[AbstractTargetBuilder]:
        return [
            TrajectoryTargetBuilder(
                config=self.action_head_config,
                world_model_config=self.world_model_config,
            )
        ]

    def get_feature_builders(self) -> List[AbstractFeatureBuilder]:
        feature_builders = DriveVLAFeatureBuilder(
            cache_hidden_state=self.vlm_config.cache_hidden_state,
            model_type=self.vlm_config.vlm_type,
            checkpoint_path=self.vlm_config.vlm_path,
            device=self.device,
            cache_mode=self.vlm_config.cache_mode,
        )
        if feature_builders.backbone:
            feature_builders.backbone = self._apply_lora_to_backbone(feature_builders.backbone)
            if self.lora_config.checkpoint_path:
                adapter_ckpt = torch.load(self.lora_config.checkpoint_path, map_location=self.device)['state_dict']
                filtered_ckpt = {}
                for k, v in adapter_ckpt.items():
                    full_name = k.split('agent.backbone.')[-1]
                    filtered_ckpt[full_name] = v
                # feature_builders.backbone.load_state_dict(filtered_ckpt, strict=False)
                
                missing_keys, unexpected_keys = feature_builders.backbone.load_state_dict(filtered_ckpt, strict=False)
                for name, param in feature_builders.backbone.named_parameters():
                    param.requires_grad = False
                feature_builders.backbone.eval()
                print(f"✅ Feature Builder loaded from checkpoint: {self.lora_config.checkpoint_path}")
                print("LoRA adapter loaded successfully")
                # print(f" - Missing keys: {missing_keys}")
                # print(f" - Unexpected keys: {unexpected_keys}")
        return [feature_builders]

    def forward(self, features: Dict[str, torch.Tensor], targets=None, tokens_list=None) -> Dict[str, torch.Tensor]:
        reject_dynamic_feature_cache(
            features,
            enabled=self._dynamic_feature_cache_guard_enabled,
            source="DriveVLABaseAgent.forward inputs",
        )
        # Lightning/model-summary mode transitions can recursively reset child
        # ``training`` flags after ``DriveVLABaseAgent.train`` returns. Restore
        # the selective checkpoint wrappers at the actual training boundary;
        # this is idempotent and leaves frozen dropout/drop-path children in
        # eval mode.
        if (
            self.training
            and self.backbone is not None
            and bool(getattr(self.vlm_config, "gradient_checkpointing", False))
        ):
            self.backbone.eval()
            self.backbone.activate_gradient_checkpointing_train_mode()
        # Work on a shallow local copy. The original feature dictionary remains
        # available to compute_loss (notably for future image path tensors).
        runtime_features = dict(features)
        pixel_values_batch = runtime_features.get("pixel_values")
        questions = runtime_features.get("questions")
        image_path_tensor = runtime_features.get("image_path_tensor")
        tile_metadata_batch = runtime_features.get("tile_metadata")
        input_ids = runtime_features.get("input_ids")
        attention_mask = runtime_features.get("attention_mask")
        pretokenized_inputs = None
        if input_ids is not None or attention_mask is not None:
            if input_ids is None or attention_mask is None:
                raise ValueError("input_ids and attention_mask must be provided together")
            pretokenized_inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }

        if (
            questions is None
            and pretokenized_inputs is None
            and not self.vlm_config.cache_hidden_state
        ):
            prompt_history = runtime_features["history_trajectory"]
            prompt_command = runtime_features["high_command_one_hot"]
            if prompt_history.is_cuda:
                prompt_history = prompt_history.detach().cpu()
            if prompt_command.is_cuda:
                prompt_command = prompt_command.detach().cpu()
            questions = build_drivevla_questions(prompt_history, prompt_command)
        elif isinstance(questions, str):
            questions = [questions]

        host_only_keys = {
            "pixel_values",
            "questions",
            "image_path_tensor",
            "input_ids",
            "attention_mask",
            "tile_metadata",
            "future_image_paths",
            "future_image_path_lengths",
            "future_valid_mask",
            "future_pixel_values",
            "future_tile_metadata",
            "image_path_length",
            "input_decode_time",
            "input_transform_time",
        }
        for key, tensor in list(runtime_features.items()):
            if key in host_only_keys:
                continue
            if isinstance(tensor, torch.Tensor):
                runtime_features[key] = tensor.cuda(non_blocking=True)

        features = runtime_features

        history_trajectory = features["history_trajectory"]
        high_command_one_hot = features["high_command_one_hot"]
        
        
        if history_trajectory.ndim == 2: history_trajectory = history_trajectory.unsqueeze(0)
        if high_command_one_hot.ndim == 1: high_command_one_hot = high_command_one_hot.unsqueeze(0)

        planning_registers = features.get("planning_registers")
        if self.vlm_config.cache_hidden_state:
            last_hidden_state = features["last_hidden_state"]
        else:
            if self.backbone is None:
                raise RuntimeError("Agent is in 'no-cache' mode, but backbone is not initialized.")
            image_paths = None
            if image_path_tensor is not None:
                if image_path_tensor.is_cuda:
                    image_path_tensor = image_path_tensor.detach().cpu()
                if image_path_tensor.ndim == 1:
                    image_path_tensor = image_path_tensor.unsqueeze(0)
                image_paths = self._decode_paths_from_tensor(image_path_tensor)
            
            if self.vlm_config.vlm_type == "internvl":
                tile_metadata = None
                needs_tile_metadata = bool(
                    self.backbone.planning_register_adapter is not None
                    and self.backbone.planning_register_adapter.tile_aggregation
                    != "mean"
                )
                if pixel_values_batch is None:
                    if image_paths is None:
                        raise RuntimeError("InternVL requires image paths or pixel_values")
                    if needs_tile_metadata:
                        loaded_images = [
                            load_image(path, return_tile_metadata=True)
                            for path in image_paths
                        ]
                        pixel_values_list = [item[0] for item in loaded_images]
                        tile_metadata = torch.cat(
                            [item[1] for item in loaded_images], dim=0
                        )
                    else:
                        pixel_values_list = [load_image(path) for path in image_paths]
                    num_patches_list = [value.shape[0] for value in pixel_values_list]
                    pixel_values_cat = torch.cat(pixel_values_list, dim=0).cuda(
                        non_blocking=True
                    )
                elif isinstance(pixel_values_batch, torch.Tensor):
                    pixel_values_batch = pixel_values_batch.cuda(non_blocking=True)
                    if pixel_values_batch.ndim == 5:
                        num_patches_list = [pixel_values_batch.shape[1]] * pixel_values_batch.shape[0]
                        pixel_values_cat = pixel_values_batch.flatten(0, 1)
                    elif pixel_values_batch.ndim == 4:
                        num_patches_list = [pixel_values_batch.shape[0]]
                        pixel_values_cat = pixel_values_batch
                    else:
                        raise ValueError(
                            f"Unexpected pixel_values shape: {pixel_values_batch.shape}"
                        )
                    if tile_metadata_batch is not None:
                        if not isinstance(tile_metadata_batch, torch.Tensor):
                            raise TypeError(
                                "Stacked pixel_values require tensor tile_metadata"
                            )
                        if tile_metadata_batch.ndim == 3:
                            tile_metadata = tile_metadata_batch.flatten(0, 1)
                        elif tile_metadata_batch.ndim == 2:
                            tile_metadata = tile_metadata_batch
                        else:
                            raise ValueError(
                                "tile_metadata must be [B,T,5] or [T,5], got "
                                f"{tuple(tile_metadata_batch.shape)}"
                            )
                else:
                    pixel_values_list = [
                        value.cuda(non_blocking=True) for value in pixel_values_batch
                    ]
                    num_patches_list = [value.shape[0] for value in pixel_values_list]
                    pixel_values_cat = torch.cat(pixel_values_list, dim=0)
                    if tile_metadata_batch is not None:
                        if not isinstance(tile_metadata_batch, (list, tuple)):
                            raise TypeError(
                                "List pixel_values require list tile_metadata"
                            )
                        tile_metadata = torch.cat(
                            [torch.as_tensor(value) for value in tile_metadata_batch],
                            dim=0,
                        )

                if needs_tile_metadata and tile_metadata is None:
                    raise KeyError(
                        "Spatial tile aggregation requires tile_metadata; use "
                        "load_image(..., return_tile_metadata=True)"
                    )

                outputs = self.backbone(
                    pixel_values_cat,
                    questions,
                    num_patches_list=num_patches_list,
                    model_inputs=pretokenized_inputs,
                    tile_metadata=tile_metadata,
                )
                if isinstance(outputs, dict):
                    last_hidden_state = outputs["last_hidden_state"]
                    planning_registers = outputs.get("planning_registers")
                else:
                    last_hidden_state = outputs.hidden_states[-1]
            
            elif self.vlm_config.vlm_type == "qwen3vl":
                if image_paths is None:
                    raise RuntimeError("Qwen3VL requires image paths")
                pixel_values_list = image_paths

                outputs, visual_feature_idx = self.backbone(pixel_values_list, questions)
                last_hidden_state = outputs.hidden_states[-1]
                
                # Get the alignment feature (index the visual token)
                start_index = visual_feature_idx[0]
                end_index = visual_feature_idx[-1]
                alignment_feature = outputs.hidden_states[-7][:, start_index:end_index+1, :]  # align the 3/4 layer (21/28) with the geometry feature

        status_feature = features["status_feature"]
        if status_feature.ndim == 1: status_feature = status_feature.unsqueeze(0)
        if last_hidden_state.ndim == 2: last_hidden_state = last_hidden_state.unsqueeze(0)

        history_trajectory_reshaped = history_trajectory.view(history_trajectory.size(0), -1)
        input_state = torch.cat([status_feature, history_trajectory_reshaped], dim=1)
        
        if not self.training:
            action_inputs={}
            action_inputs = {
                "last_hidden_state":last_hidden_state.float(),
                "status_feature":status_feature
            }
        else:
            action_inputs={}
            action_inputs = {
                "last_hidden_state":last_hidden_state,
                "status_feature":status_feature
            }

        if bool(getattr(self.vlm_config, "planning_registers_enabled", False)):
            if planning_registers is None:
                raise KeyError(
                    "planning_registers_enabled=true but forward produced no "
                    "planning_registers"
                )
            action_inputs["planning_registers"] = (
                planning_registers.float()
                if not self.training
                else planning_registers
            )

        for key in (
            "memory_map_query_key",
            "memory_agent_query_key",
        ):
            if key in features:
                query_key = features[key]
                if query_key.ndim == 1:
                    query_key = query_key.unsqueeze(0)
                action_inputs[key] = query_key
        for key in (
            "memory_profile_latency",
            "memory_return_attention_weights",
            "memory_excluded_index",
        ):
            if key in features:
                action_inputs[key] = features[key]

        predictions = self.action_head(action_inputs)
        diagnostic_registers = predictions.get("planning_registers")
        self._latest_registers_for_diagnostics = (
            diagnostic_registers.detach()
            if isinstance(diagnostic_registers, torch.Tensor)
            else None
        )
        return predictions

    def compute_trajectory(self, agent_input: AgentInput) -> Trajectory:
        self.eval()
        if self.world_model_enabled:
            self.remove_training_only_world_model()

        features: Dict[str, torch.Tensor] = {}
        # build features
        # if not self.evaluation:
        if self.evaluation:
            for builder in self.feature_builders:
                features.update(builder.compute_features(agent_input))
        
            # add batch dimension
            features = {k: v.unsqueeze(0) for k, v in features.items()}
        else:
            features = agent_input

        with torch.no_grad():
            predictions = self.forward(features)
            poses = predictions["trajectory"].float().cpu().squeeze(0)

        return Trajectory(poses)

    def compute_trajectory_vis(self, agent_input: AgentInput) -> Trajectory:
        self.eval()
        if self.world_model_enabled:
            self.remove_training_only_world_model()

        features: Dict[str, torch.Tensor] = {}
        # build features
        for builder in self.get_feature_builders():
            features.update(builder.compute_features(agent_input))

        # add batch dimension
        features = {k: v.unsqueeze(0) for k, v in features.items()}

        with torch.no_grad():
            predictions = self.forward(features)
            poses = predictions["trajectory"].float().cpu().squeeze(0)
        return Trajectory(poses)


    def _current_image_tiles(
        self,
        features: Dict[str, torch.Tensor],
        batch_size: int,
    ):
        pixel_values = features.get("pixel_values")
        tile_metadata = features.get("tile_metadata")
        if isinstance(pixel_values, torch.Tensor):
            if pixel_values.ndim == 5:
                if pixel_values.shape[0] != batch_size:
                    raise ValueError("Current pixel batch does not match register batch")
                pixel_groups = [pixel_values[index] for index in range(batch_size)]
                if isinstance(tile_metadata, torch.Tensor) and tile_metadata.ndim == 3:
                    metadata_groups = [
                        tile_metadata[index] for index in range(batch_size)
                    ]
                elif tile_metadata is None:
                    metadata_groups = [None] * batch_size
                else:
                    raise ValueError(
                        "Batched current pixels require tile_metadata [B,T,5]"
                    )
                return list(zip(pixel_groups, metadata_groups))
            if pixel_values.ndim == 4 and batch_size == 1:
                if isinstance(tile_metadata, torch.Tensor) and tile_metadata.ndim == 3:
                    tile_metadata = tile_metadata[0]
                return [(pixel_values, tile_metadata)]
            raise ValueError(
                "World-model current pixel_values must be [B,T,C,H,W] or "
                f"single-sample [T,C,H,W], got {tuple(pixel_values.shape)}"
            )
        if isinstance(pixel_values, (list, tuple)):
            if len(pixel_values) != batch_size:
                raise ValueError("Current pixel list does not match register batch")
            if tile_metadata is None:
                metadata_groups = [None] * batch_size
            elif isinstance(tile_metadata, (list, tuple)):
                if len(tile_metadata) != batch_size:
                    raise ValueError("Current tile metadata list does not match batch")
                metadata_groups = list(tile_metadata)
            else:
                raise ValueError("List current pixels require list tile_metadata")
            return list(zip(pixel_values, metadata_groups))

        path_tensor = features.get("image_path_tensor")
        if path_tensor is None:
            raise KeyError(
                "World-model loss requires current pixel_values or image_path_tensor"
            )
        if path_tensor.ndim == 1:
            path_tensor = path_tensor.unsqueeze(0)
        current_paths = self._decode_paths_from_tensor(path_tensor.detach().cpu())
        if len(current_paths) != batch_size:
            raise ValueError("Current image path batch does not match registers")
        return [
            load_image(path, return_tile_metadata=True) for path in current_paths
        ]

    def _encode_ema_register_targets(
        self,
        features: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        batch_size: int,
    ):
        if self.ema_register_target is None:
            raise RuntimeError("EMA register target has not been initialized")
        if self.future_mode == "shuffled_batch" and batch_size == 1:
            raise ValueError(
                "world_model.future_mode=shuffled_batch requires batch_size > 1"
            )
        required = (
            "future_image_paths",
            "future_image_path_lengths",
            "future_valid_mask",
        )
        missing = [key for key in required if key not in targets]
        if missing:
            raise KeyError(f"World-model targets are missing future fields: {missing}")

        future_paths = targets["future_image_paths"]
        future_lengths = targets["future_image_path_lengths"]
        future_valid = targets["future_valid_mask"]
        if future_paths.ndim == 2:
            future_paths = future_paths.unsqueeze(0)
            future_lengths = future_lengths.unsqueeze(0)
            future_valid = future_valid.unsqueeze(0)
        if future_paths.shape[:2] != (batch_size, 3):
            raise ValueError(
                "future_image_paths must have shape [B,3,1024], got "
                f"{tuple(future_paths.shape)}"
            )
        valid_mask = future_valid.detach().to(dtype=torch.bool, device="cpu")
        current_tiles = self._current_image_tiles(features, batch_size)
        worker_future_pixels = features.get("future_pixel_values")
        worker_future_metadata = features.get("future_tile_metadata")
        if (worker_future_pixels is None) != (worker_future_metadata is None):
            raise ValueError(
                "future_pixel_values and future_tile_metadata must be provided together"
            )
        if worker_future_pixels is not None:
            if len(worker_future_pixels) != batch_size or len(worker_future_metadata) != batch_size:
                raise ValueError("Worker-preprocessed future image batch size mismatch")
            if not all(len(group) == 3 for group in worker_future_pixels):
                raise ValueError("Worker-preprocessed future images require three horizons")
            if not all(len(group) == 3 for group in worker_future_metadata):
                raise ValueError("Worker-preprocessed future metadata require three horizons")

        all_image_tiles: List[torch.Tensor] = []
        all_tile_metadata: List[torch.Tensor] = []
        tile_counts: List[int] = []
        for batch_index in range(batch_size):
            current, current_metadata = current_tiles[batch_index]
            current = current.detach().cpu()
            if current_metadata is not None:
                current_metadata = torch.as_tensor(current_metadata).detach().cpu()
            image_group = [(current, current_metadata)]
            for horizon_index in range(3):
                if self.future_mode == "repeated_current" or not bool(
                    valid_mask[batch_index, horizon_index]
                ):
                    future, future_metadata = current, current_metadata
                elif worker_future_pixels is not None:
                    future = torch.as_tensor(
                        worker_future_pixels[batch_index][horizon_index]
                    ).detach().cpu()
                    future_metadata = torch.as_tensor(
                        worker_future_metadata[batch_index][horizon_index]
                    ).detach().cpu()
                else:
                    path = decode_path_tensor(
                        future_paths[batch_index, horizon_index],
                        future_lengths[batch_index, horizon_index],
                    )
                    future, future_metadata = load_image(
                        path, return_tile_metadata=True
                    )
                image_group.append((future, future_metadata))
            for image_tiles, image_tile_metadata in image_group:
                if image_tiles.ndim != 4:
                    raise ValueError(
                        f"InternVL image preprocessing must return [T,C,H,W], got {image_tiles.shape}"
                    )
                all_image_tiles.append(image_tiles)
                tile_counts.append(int(image_tiles.shape[0]))
                if image_tile_metadata is None:
                    if (
                        getattr(
                            getattr(
                                self.ema_register_target,
                                "planning_register_adapter",
                                None,
                            ),
                            "tile_aggregation",
                            "mean",
                        )
                        != "mean"
                    ):
                        raise KeyError(
                            "EMA spatial tile aggregation requires tile_metadata"
                        )
                else:
                    all_tile_metadata.append(image_tile_metadata)

        teacher_parameter = next(self.ema_register_target.parameters())
        pixel_values = torch.cat(all_image_tiles, dim=0).to(
            device=teacher_parameter.device,
            dtype=teacher_parameter.dtype,
            non_blocking=True,
        )
        tile_metadata_tensor = (
            torch.cat(all_tile_metadata, dim=0) if all_tile_metadata else None
        )
        with torch.no_grad():
            all_targets = self.ema_register_target(
                pixel_values, tile_counts, tile_metadata_tensor
            )
        all_targets = all_targets.reshape(batch_size, 4, *all_targets.shape[1:])
        target_current = all_targets[:, 0].detach()
        target_future = all_targets[:, 1:].detach()

        if self.future_mode == "repeated_current":
            target_future = target_current[:, None].expand_as(target_future)
            valid_mask = torch.ones_like(valid_mask)
        elif self.future_mode == "shuffled_batch":
            target_future = torch.roll(target_future, shifts=1, dims=0)
            valid_mask = torch.roll(valid_mask, shifts=1, dims=0)
        return (
            target_current,
            target_future,
            valid_mask.to(device=target_current.device),
        )

    @staticmethod
    def _masked_horizon_mean(
        values: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        weights = valid_mask.to(device=values.device, dtype=values.dtype)
        return (values * weights).sum() / weights.sum().clamp_min(1.0)

    @staticmethod
    def _weighted_masked_horizon_mean(
        values: torch.Tensor,
        valid_mask: torch.Tensor,
        horizon_weights: torch.Tensor,
    ) -> torch.Tensor:
        weights = valid_mask.to(device=values.device, dtype=values.dtype)
        weights = weights * horizon_weights.to(
            device=values.device, dtype=values.dtype
        )[None]
        return (values * weights).sum() / weights.sum().clamp_min(1.0)

    @staticmethod
    def _require_every_future_horizon_present(
        valid_mask: torch.Tensor,
    ) -> None:
        """Require batch coverage at 0.5/1.5/3.0s, allowing masked samples."""
        if valid_mask.ndim != 2 or valid_mask.shape[1] != 3:
            raise ValueError(
                "future_valid_mask must be [B,3], got "
                f"{tuple(valid_mask.shape)}"
            )
        horizon_present = valid_mask.to(dtype=torch.bool).any(dim=0)
        if not bool(horizon_present.all()):
            missing = (
                (~horizon_present).nonzero(as_tuple=False).flatten().tolist()
            )
            raise RuntimeError(
                "Real-data smoke requires at least one valid future image for "
                "every 0.5/1.5/3.0s horizon; "
                f"missing horizon indices={missing}"
            )

    def _compute_world_model_loss_from_registers(
        self,
        current_registers: torch.Tensor,
        gt_trajectory: torch.Tensor,
        target_current: torch.Tensor,
        target_future: torch.Tensor,
        future_valid_mask: torch.Tensor,
        current_speed: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if self.future_register_predictor is None:
            raise RuntimeError("Future register predictor is unavailable")
        if gt_trajectory.ndim == 2:
            gt_trajectory = gt_trajectory.unsqueeze(0)
        if gt_trajectory.shape[-2:] != (8, 3):
            raise ValueError(
                f"GT trajectory must be [B,8,3], got {tuple(gt_trajectory.shape)}"
            )
        if target_future.shape != (
            current_registers.shape[0],
            3,
            current_registers.shape[1],
            current_registers.shape[2],
        ):
            raise ValueError(
                "Future EMA register target shape mismatch: "
                f"got {tuple(target_future.shape)}"
            )
        if not bool(future_valid_mask.any()):
            raise RuntimeError("World-model batch has no valid future image targets")

        predictor_only = bool(
            getattr(self.world_model_config, "predictor_only", False)
        )
        predictor_current = (
            current_registers.detach() if predictor_only else current_registers
        )
        trajectories = gt_trajectory.to(
            device=current_registers.device,
            dtype=current_registers.dtype,
        )[:, None]
        horizons = tuple(
            float(value)
            for value in getattr(
                self.world_model_config, "horizons_sec", (0.5, 1.5, 3.0)
            )
        )
        pred_future = self.future_register_predictor(
            predictor_current,
            trajectories,
            horizons,
            use_action_condition=self.future_mode != "no_action_condition",
            current_speed=current_speed,
        )
        target_current = target_current.detach().to(
            device=pred_future.device, dtype=pred_future.dtype
        )
        target_future = target_future.detach().to(
            device=pred_future.device, dtype=pred_future.dtype
        )
        valid_mask = future_valid_mask.to(device=pred_future.device, dtype=torch.bool)
        if bool(
            getattr(self.world_model_config, "require_all_horizons_valid", False)
        ):
            self._require_every_future_horizon_present(valid_mask)

        target_current_n = self.future_register_predictor.normalize_register_state(
            target_current
        )
        target_future_n = self.future_register_predictor.normalize_register_state(
            target_future
        )
        current_n = self.future_register_predictor.normalize_register_state(
            predictor_current
        )

        cosine_by_register = 1.0 - F.cosine_similarity(
            pred_future,
            target_future_n[:, None],
            dim=-1,
            eps=1e-8,
        )
        cosine_by_horizon = cosine_by_register.mean(dim=(1, 3))
        delta_error = F.smooth_l1_loss(
            pred_future - current_n[:, None, None],
            target_future_n[:, None] - target_current_n[:, None, None],
            reduction="none",
        )
        delta_by_horizon = delta_error.mean(dim=(1, 3, 4))
        horizon_weights = torch.as_tensor(
            getattr(self.world_model_config, "horizon_weights", (1.0, 1.0, 1.0)),
            device=pred_future.device,
            dtype=pred_future.dtype,
        )
        if horizon_weights.shape != (3,) or bool((horizon_weights < 0).any()):
            raise ValueError(
                "world_model.horizon_weights must be three non-negative values"
            )
        wm_abs_loss = self._weighted_masked_horizon_mean(
            cosine_by_horizon, valid_mask, horizon_weights
        )
        wm_delta_loss = self._weighted_masked_horizon_mean(
            delta_by_horizon, valid_mask, horizon_weights
        )
        abs_weight = float(getattr(self.world_model_config, "abs_weight", 1.0))
        delta_weight = float(getattr(self.world_model_config, "delta_weight", 0.25))
        wm_loss = abs_weight * wm_abs_loss + delta_weight * wm_delta_loss

        loss_dict = {
            "wm_loss": wm_loss,
            "wm_abs_loss": wm_abs_loss,
            "wm_delta_loss": wm_delta_loss,
            "predicted_future_registers": pred_future,
        }
        horizon_names = ("0p5", "1p5", "3p0")
        for index, name in enumerate(horizon_names):
            mask = valid_mask[:, index]
            horizon_abs = self._masked_horizon_mean(
                cosine_by_horizon[:, index], mask
            )
            loss_dict[f"wm_abs_{name}"] = horizon_abs
            # Backward-compatible metric name from the initial implementation.
            loss_dict[f"wm_cos_{name}"] = horizon_abs
            loss_dict[f"wm_delta_{name}"] = self._masked_horizon_mean(
                delta_by_horizon[:, index], mask
            )
        return loss_dict

    def get_planreg_gradient_norms(self) -> Dict[str, torch.Tensor]:
        device = next(self.parameters()).device

        def module_norm(modules) -> torch.Tensor:
            squares = []
            seen = set()
            for module in modules:
                if module is None:
                    continue
                for parameter in module.parameters():
                    if id(parameter) in seen:
                        continue
                    seen.add(id(parameter))
                    if parameter.grad is not None:
                        squares.append(parameter.grad.detach().float().square().sum())
            if not squares:
                return torch.zeros((), device=device)
            return torch.stack(squares).sum().sqrt()

        planning_adapter = None
        vision_lora_modules = []
        if self.backbone is not None:
            planning_adapter = getattr(
                self.backbone, "planning_register_adapter", None
            )
            for module_name, module in self.backbone.named_modules():
                if module_name.endswith(("q_lora_a", "q_lora_b", "v_lora_a", "v_lora_b")):
                    vision_lora_modules.append(module)

        action_modules = []
        scorer_modules = []
        if hasattr(self, "action_head"):
            action_modules = [
                getattr(self.action_head, name, None)
                for name in (
                    "hist_encoding",
                    "init_feature",
                    "trajectory_decoder",
                    "traj_head",
                )
            ]
            scorer_modules = [
                getattr(self.action_head, name, None)
                for name in ("scorer_attention", "pos_embed", "scorer")
            ]

        return {
            "vision_lora_grad_norm": module_norm(vision_lora_modules),
            "register_grad_norm": module_norm([planning_adapter]),
            "future_predictor_grad_norm": module_norm(
                [self.future_register_predictor]
            ),
            "action_head_grad_norm": module_norm(action_modules),
            "scorer_grad_norm": module_norm(scorer_modules),
        }

    def get_planreg_register_diagnostics(self) -> Dict[str, torch.Tensor]:
        """Compute interval-gated diagnostics from a detached forward snapshot."""
        registers = self._latest_registers_for_diagnostics
        self._latest_registers_for_diagnostics = None
        if registers is None:
            return {}
        return compute_register_diagnostics(registers)


    def compute_loss(
            self,
            features: Dict[str, torch.Tensor],
            targets: Dict[str, torch.Tensor],
            pred: Dict[str, torch.Tensor],
    ) -> Dict:
        base_loss_dict = self.loss(
            targets, pred, self.action_head_config, self.compute_score
        )
        if not self.world_model_enabled:
            return base_loss_dict
        if not self.training:
            return base_loss_dict
        current_registers = pred.get("planning_registers")
        if current_registers is None:
            raise KeyError("World-model loss requires predictions['planning_registers']")
        target_current, target_future, valid_mask = self._encode_ema_register_targets(
            features,
            targets,
            batch_size=current_registers.shape[0],
        )
        status_feature = features.get("status_feature")
        if status_feature is None:
            raise KeyError("World-model training requires features['status_feature']")
        if status_feature.ndim == 1:
            status_feature = status_feature.unsqueeze(0)
        if status_feature.shape[-1] < 6:
            raise ValueError(
                "status_feature must contain command[4] followed by vx,vy"
            )
        current_speed = torch.linalg.vector_norm(
            status_feature[:, 4:6].to(
                device=current_registers.device,
                dtype=current_registers.dtype,
            ),
            dim=-1,
        )
        wm_loss_dict = self._compute_world_model_loss_from_registers(
            current_registers,
            targets["trajectory"],
            target_current,
            target_future,
            valid_mask,
            current_speed=current_speed,
        )
        wm_weight = current_registers.new_tensor(
            self.current_world_model_weight()
        )
        weighted_wm_loss = wm_weight * wm_loss_dict["wm_loss"]
        total = base_loss_dict["loss"] + weighted_wm_loss
        result = dict(base_loss_dict)
        result["loss"] = total
        result["wm_weight_current"] = wm_weight
        result["weighted_wm_loss"] = weighted_wm_loss
        result.update(
            {
                key: value
                for key, value in wm_loss_dict.items()
                if key != "predicted_future_registers"
            }
        )
        return result

    def compute_score(self, targets, proposals, test=True):
        if self.training:
            metric_cache_paths = self.train_metric_cache_paths
        else:
            metric_cache_paths = self.test_metric_cache_paths

        if not metric_cache_paths:
            raise RuntimeError(
                "Metric cache is required for compute_score. "
                "Set NAVSIM_TRAIN_METRIC_CACHE to a valid metric cache directory."
            )

        target_trajectory = targets["trajectory"]
        proposals=proposals.detach()

        
        data_points = [
            {
                "token": metric_cache_paths[token],
                "poses": poses,
                "test": test
            }
            for token, poses in zip(targets["token"], proposals.float().cpu().numpy())
        ]

        if self.ray:
            all_res = self.worker_map(self.worker, self.get_scores, data_points)
        elif self.score_process_count and (
            len(data_points) > 1
            or any(len(point["poses"]) > 1 for point in data_points)
        ):
            if self._score_process_pool is None:
                # CUDA is already initialized in each DDP rank at this point.
                # Spawn gives the CPU-only scorer workers fresh interpreters and
                # avoids inheriting an unsafe CUDA context through fork.
                if self.score_start_method == "forkserver":
                    # The forkserver itself is spawned after CUDA init, so it
                    # has a clean CPU-only address space.  Preloading the scorer
                    # there lets its children share imports safely and avoids
                    # importing torch/navsim eight times per rank.
                    mp.set_forkserver_preload(
                        [
                            "navsim.agents.EpisodeDrive.score_module.compute_navsim_score"
                        ]
                    )
                self._score_process_pool = ProcessPoolExecutor(
                    max_workers=self.score_process_count,
                    mp_context=mp.get_context(self.score_start_method),
                )

            task_tokens = []
            task_poses = []
            task_tests = []
            scene_task_ranges = []
            for point in data_points:
                partition_count = min(self.score_partition_count, len(point["poses"]))
                start = len(task_poses)
                for poses_partition in np.array_split(
                    point["poses"], partition_count, axis=0
                ):
                    task_tokens.append(point["token"])
                    task_poses.append(poses_partition)
                    task_tests.append(point["test"])
                scene_task_ranges.append((start, len(task_poses)))

            task_results = list(
                self._score_process_pool.map(
                    self.get_sub_score,
                    task_tokens,
                    task_poses,
                    task_tests,
                    chunksize=1,
                )
            )
            all_res = [
                tuple(
                    np.concatenate(component_parts, axis=0)
                    for component_parts in zip(*task_results[start:end])
                )
                for start, end in scene_task_ranges
            ]
        else:
            all_res = self.get_scores(data_points)

        target_scores = torch.FloatTensor(np.stack([res[0] for res in all_res])).to(proposals.device)

        final_scores = target_scores[:, :, -1]

        best_scores = torch.amax(final_scores, dim=-1)

        if test:
            l2_2s = torch.linalg.norm(proposals[:, 0] - target_trajectory, dim=-1)[:, :4]

            return final_scores[:, 0].mean(), best_scores.mean(), final_scores, l2_2s.mean(), target_scores[:, 0]
        else:
            key_agent_corners = torch.FloatTensor(np.stack([res[1] for res in all_res])).to(proposals.device)

            key_agent_labels = torch.BoolTensor(np.stack([res[2] for res in all_res])).to(proposals.device)

            all_ego_areas = torch.BoolTensor(np.stack([res[3] for res in all_res])).to(proposals.device)

            return final_scores, best_scores, target_scores, key_agent_corners, key_agent_labels, all_ego_areas


    def _uses_planreg_optimizer_groups(self) -> bool:
        return bool(
            self.scene_fusion is not None
            or getattr(self.vlm_config, "planning_registers_enabled", False)
            or self.world_model_enabled
        )

    def _get_planreg_optimizers(
        self, total_optimizer_steps: Optional[int] = None
    ):
        if self._lr_args["name"] not in {"Adam", "AdamW"}:
            raise NotImplementedError
        if float(self._lr_args.get("language_model_lr", 0.0)) != 0.0:
            raise ValueError("PlanReg-WM-V1 requires language_model_lr=0")

        global_batch_size = self.batch_size * self.num_gpus
        batch_scale = 1.0
        if bool(self._lr_args.get("scale_with_batch_size", False)):
            batch_scale = math.sqrt(
                global_batch_size / self._lr_args["base_batch_size"]
            )
        legacy_new_module_lr = float(
            self._lr_args.get("new_module_lr", 2e-4)
        )
        learning_rates = {
            "planning_adapter": float(
                self._lr_args.get("planning_adapter_lr", legacy_new_module_lr)
            )
            * batch_scale,
            "future_predictor": float(
                self._lr_args.get("future_predictor_lr", legacy_new_module_lr)
            )
            * batch_scale,
            "fusion": float(self._lr_args.get("fusion_lr", 1e-4))
            * batch_scale,
            "action_head": float(self._lr_args.get("action_head_lr", 1e-4))
            * batch_scale,
            "scorer": float(self._lr_args.get("scorer_lr", 1e-4))
            * batch_scale,
            "vision_qv_lora": float(
                self._lr_args.get("vision_qv_lora_lr", 5e-5)
            )
            * batch_scale,
            "semantic_qformer": float(
                self._lr_args.get("semantic_qformer_lr", 1e-5)
            )
            * batch_scale,
        }
        decay_weight_decay = float(
            self._lr_args.get("decay_weight_decay", 0.01)
        )
        no_decay_weight_decay = float(
            self._lr_args.get("no_decay_weight_decay", 0.0)
        )
        grouped = {name: [] for name in learning_rates}
        unclassified = []
        trainable_language = []

        for name, parameter in self.named_parameters():
            if not parameter.requires_grad:
                continue
            if "ema_register_target" in name:
                unclassified.append(name)
            elif "language_model" in name or "lm_head" in name:
                trainable_language.append(name)
            elif name.startswith("future_register_predictor."):
                grouped["future_predictor"].append((name, parameter))
            elif name.startswith("backbone.planning_register_adapter."):
                grouped["planning_adapter"].append((name, parameter))
            elif name.startswith("backbone.") and (
                ".q_lora_" in name or ".v_lora_" in name
            ):
                grouped["vision_qv_lora"].append((name, parameter))
            elif name.startswith("action_head."):
                action_name = name[len("action_head."):]
                if action_name.startswith(
                    ("scorer_attention.", "pos_embed.", "scorer.")
                ):
                    grouped["scorer"].append((name, parameter))
                elif action_name.startswith(
                    ("semantic_gate", "scene_norm.")
                ):
                    grouped["fusion"].append((name, parameter))
                elif action_name.startswith(
                    ("q_former.", "scene_embeds")
                ):
                    grouped["semantic_qformer"].append((name, parameter))
                elif action_name.startswith(
                    (
                        "hist_encoding.",
                        "init_feature.",
                        "trajectory_decoder.",
                        "traj_head.",
                    )
                ):
                    grouped["action_head"].append((name, parameter))
                else:
                    unclassified.append(name)
            else:
                unclassified.append(name)

        if trainable_language:
            raise RuntimeError(
                "PlanReg-WM-V1 LLM must be frozen; trainable parameters: "
                f"{trainable_language[:16]}"
            )
        if unclassified:
            raise RuntimeError(
                "Unclassified requires_grad=True PlanReg parameters: "
                f"{unclassified[:32]}"
            )

        modules = dict(self.named_modules())

        def is_no_decay(parameter_name: str, parameter: nn.Parameter) -> bool:
            local_name = parameter_name.rsplit(".", 1)[-1]
            module_name = parameter_name.rsplit(".", 1)[0]
            owner = modules.get(module_name)
            owner_class = type(owner).__name__.lower() if owner is not None else ""
            if local_name == "bias" or parameter.ndim < 2:
                return True
            if isinstance(owner, nn.Embedding) or "norm" in owner_class:
                return True
            if any(
                marker in parameter_name
                for marker in (
                    ".q_lora_a.",
                    ".q_lora_b.",
                    ".v_lora_a.",
                    ".v_lora_b.",
                    "planning_registers",
                    "scene_embeds",
                    "semantic_gate",
                    "tile_gate",
                )
            ):
                return True
            return False

        optimizer_groups = []
        summaries = []
        for logical_name, named_parameters in grouped.items():
            if not named_parameters:
                continue
            split_parameters = {"decay": [], "no_decay": []}
            for parameter_name, parameter in named_parameters:
                split = (
                    "no_decay"
                    if is_no_decay(parameter_name, parameter)
                    else "decay"
                )
                split_parameters[split].append((parameter_name, parameter))
            for split_name, split_named_parameters in split_parameters.items():
                if not split_named_parameters:
                    continue
                parameters = [parameter for _, parameter in split_named_parameters]
                weight_decay = (
                    decay_weight_decay
                    if split_name == "decay"
                    else no_decay_weight_decay
                )
                group_name = f"{logical_name}_{split_name}"
                parameter_count = sum(
                    parameter.numel() for parameter in parameters
                )
                optimizer_groups.append(
                    {
                        "params": parameters,
                        "lr": learning_rates[logical_name],
                        "weight_decay": weight_decay,
                        "name": group_name,
                        "logical_name": logical_name,
                    }
                )
                summary = {
                    "name": group_name,
                    "logical_name": logical_name,
                    "tensor_count": len(parameters),
                    "parameter_count": parameter_count,
                    "lr": learning_rates[logical_name],
                    "weight_decay": weight_decay,
                }
                summaries.append(summary)
                print(
                    "PLANREG_OPTIMIZER_GROUP "
                    f"name={group_name} logical_name={logical_name} "
                    f"tensors={len(parameters)} parameters={parameter_count} "
                    f"lr={learning_rates[logical_name]:.8g} "
                    f"weight_decay={weight_decay:.8g}"
                )
        if not optimizer_groups:
            raise RuntimeError("No trainable PlanReg-WM-V1 parameters found")
        self._planreg_optimizer_group_summary = summaries
        optimizer_class = (
            torch.optim.AdamW
            if self._lr_args["name"] == "AdamW"
            else torch.optim.Adam
        )
        optimizer = optimizer_class(
            optimizer_groups,
            betas=tuple(self._lr_args.get("betas", (0.9, 0.999))),
            eps=float(self._lr_args.get("eps", 1e-8)),
        )
        if self.scheduler_args is None:
            return [optimizer]
        if total_optimizer_steps is None or int(total_optimizer_steps) <= 0:
            raise ValueError(
                "PlanReg scheduler requires trainer.estimated_stepping_batches"
            )

        scheduler_args = self.scheduler_args
        warmup_ratio = float(scheduler_args.get("warmup_ratio", 0.03))
        start_lr_ratio = float(scheduler_args.get("start_lr_ratio", 0.01))
        default_min_ratios = {
            "planning_adapter": 0.10,
            "future_predictor": 0.10,
            "fusion": 0.10,
            "vision_qv_lora": 0.10,
            "action_head": 0.20,
            "scorer": 0.20,
            "semantic_qformer": 0.20,
        }
        configured_min_ratios = scheduler_args.get("min_lr_ratios", {})
        min_ratios = {
            name: float(configured_min_ratios.get(name, default))
            for name, default in default_min_ratios.items()
        }
        lr_lambdas = []
        for group in optimizer.param_groups:
            logical_name = group["logical_name"]
            min_ratio = min_ratios[logical_name]

            def lr_lambda(step, *, _min_ratio=min_ratio):
                return planreg_warmup_cosine_multiplier(
                    step,
                    total_optimizer_steps=int(total_optimizer_steps),
                    warmup_ratio=warmup_ratio,
                    start_lr_ratio=start_lr_ratio,
                    min_lr_ratio=_min_ratio,
                )

            lr_lambdas.append(lr_lambda)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lr_lambdas
        )
        print(
            "PLANREG_LR_SCHEDULER type=warmup_cosine interval=step "
            f"total_steps={int(total_optimizer_steps)} "
            f"warmup_ratio={warmup_ratio:.6g} "
            f"start_lr_ratio={start_lr_ratio:.6g} "
            f"min_lr_ratios={min_ratios}"
        )
        return [optimizer], [
            {"scheduler": scheduler, "interval": "step", "frequency": 1}
        ]


    def get_optimizers(
        self, total_optimizer_steps: Optional[int] = None
    ) -> Union[Optimizer, Dict[str, LRScheduler]]:
        """
        pack all trainable parameters into optimizer
        """
        if self._uses_planreg_optimizer_groups():
            return self._get_planreg_optimizers(total_optimizer_steps)
        global_batchsize = self.batch_size * self.num_gpus
        if self._lr_args["name"] not in {"Adam", "AdamW"}:
            raise NotImplementedError

        batch_scale = math.sqrt(
            global_batchsize / self._lr_args["base_batch_size"]
        )
        if not bool(self._lr_args.get("scale_with_batch_size", True)):
            batch_scale = 1.0
        base_lr = float(self._lr_args["base_lr"]) * batch_scale

        learning_rates = {
            "action_head": float(
                self._lr_args.get("action_head_lr", base_lr)
            ) * batch_scale,
            "vlm_vision": float(
                self._lr_args.get("vlm_vision_lr", base_lr * 0.1)
            ) * batch_scale,
            "vlm_projector": float(
                self._lr_args.get("vlm_projector_lr", base_lr * 0.1)
            ) * batch_scale,
            "vlm_language": float(
                self._lr_args.get("vlm_language_lr", base_lr * 0.1)
            ) * batch_scale,
            "vlm_lora": float(
                self._lr_args.get("vlm_lora_lr", base_lr)
            ) * batch_scale,
            "vlm_other": float(
                self._lr_args.get("vlm_other_lr", base_lr * 0.1)
            ) * batch_scale,
            "other": float(self._lr_args.get("other_lr", base_lr)) * batch_scale,
        }
        # ``base_lr`` above is already scaled when no explicit module LR was
        # supplied. Avoid applying the factor twice to those fallback values.
        for group_name, config_key in {
            "action_head": "action_head_lr",
            "vlm_vision": "vlm_vision_lr",
            "vlm_projector": "vlm_projector_lr",
            "vlm_language": "vlm_language_lr",
            "vlm_lora": "vlm_lora_lr",
            "vlm_other": "vlm_other_lr",
            "other": "other_lr",
        }.items():
            if config_key not in self._lr_args:
                if group_name in {"vlm_vision", "vlm_projector", "vlm_language", "vlm_other"}:
                    learning_rates[group_name] = base_lr * 0.1
                else:
                    learning_rates[group_name] = base_lr

        default_weight_decay = float(self._lr_args.get("weight_decay", 1e-4))
        weight_decays = {
            "action_head": float(
                self._lr_args.get("action_head_weight_decay", default_weight_decay)
            ),
            "vlm": float(
                self._lr_args.get("vlm_weight_decay", default_weight_decay)
            ),
            "other": default_weight_decay,
        }

        grouped_parameters = {name: [] for name in learning_rates}
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if "action_head" in name:
                group = "action_head"
            elif "backbone" in name:
                lower_name = name.lower()
                if "lm_head" in lower_name:
                    raise RuntimeError(
                        f"Trainable lm_head parameter entered optimizer: {name}"
                    )
                if "lora" in lower_name:
                    group = "vlm_lora"
                elif "vision_model" in name:
                    group = "vlm_vision"
                elif ".mlp1." in name:
                    group = "vlm_projector"
                elif "language_model" in name:
                    group = "vlm_language"
                else:
                    group = "vlm_other"
            else:
                group = "other"
            grouped_parameters[group].append((name, param))

        param_groups = []
        for group_name, named_parameters in grouped_parameters.items():
            if not named_parameters:
                continue
            decay_parameters = []
            no_decay_parameters = []
            for parameter_name, parameter in named_parameters:
                if parameter.ndim < 2 or parameter_name.endswith(".bias"):
                    no_decay_parameters.append(parameter)
                else:
                    decay_parameters.append(parameter)

            group_weight_decay = (
                weight_decays["action_head"]
                if group_name == "action_head"
                else weight_decays["vlm"]
                if group_name.startswith("vlm_")
                else weight_decays["other"]
            )
            for decay_name, parameters, weight_decay in (
                ("decay", decay_parameters, group_weight_decay),
                ("no_decay", no_decay_parameters, 0.0),
            ):
                if not parameters:
                    continue
                param_groups.append(
                    {
                        "params": parameters,
                        "lr": learning_rates[group_name],
                        "weight_decay": weight_decay,
                        "name": f"{group_name}_{decay_name}",
                    }
                )
            parameter_count = sum(
                parameter.numel() for _, parameter in named_parameters
            )
            print(
                f"✅ Optimizer group {group_name}: "
                f"{len(named_parameters)} tensors / {parameter_count:,} values, "
                f"lr={learning_rates[group_name]:.2e}, "
                f"weight_decay={group_weight_decay:.2e}"
            )

        if not param_groups:
            raise RuntimeError("No trainable parameters found.")

        optimizer_class = (
            torch.optim.AdamW
            if self._lr_args["name"] == "AdamW"
            else torch.optim.Adam
        )
        optimizer = optimizer_class(
            param_groups,
            betas=tuple(self._lr_args.get("betas", (0.9, 0.95))),
            eps=float(self._lr_args.get("eps", 1e-8)),
            lr=base_lr,
        )

        if self.scheduler_args is not None:
            total_steps = int(
                math.ceil(self.scheduler_args.dataset_size / global_batchsize)
                * self.scheduler_args.num_epochs
            )
            warmup_ratio = float(self.scheduler_args.get("warmup_ratio", 0.03))
            min_lr_ratio = float(self.scheduler_args.get("min_lr_ratio", 0.0))
            action_head_min_lr_ratio = float(
                self.scheduler_args.get(
                    "action_head_min_lr_ratio", min_lr_ratio
                )
            )
            vlm_min_lr_ratio = float(
                self.scheduler_args.get("vlm_min_lr_ratio", min_lr_ratio)
            )
            start_lr_ratio = float(
                self.scheduler_args.get("start_lr_ratio", 1e-3)
            )
            if not 0.0 <= warmup_ratio < 1.0:
                raise ValueError("scheduler warmup_ratio must be in [0, 1)")
            if not 0.0 <= min_lr_ratio <= 1.0:
                raise ValueError("scheduler min_lr_ratio must be in [0, 1]")
            if not 0.0 <= action_head_min_lr_ratio <= 1.0:
                raise ValueError(
                    "scheduler action_head_min_lr_ratio must be in [0, 1]"
                )
            if not 0.0 <= vlm_min_lr_ratio <= 1.0:
                raise ValueError(
                    "scheduler vlm_min_lr_ratio must be in [0, 1]"
                )
            warmup_steps = max(1, int(total_steps * warmup_ratio))

            def make_lr_multiplier(group_min_lr_ratio: float):
                def lr_multiplier(step: int) -> float:
                    if step < warmup_steps:
                        progress = step / warmup_steps
                        return start_lr_ratio + (1.0 - start_lr_ratio) * progress
                    decay_steps = max(1, total_steps - warmup_steps)
                    progress = min(1.0, (step - warmup_steps) / decay_steps)
                    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
                    return group_min_lr_ratio + (
                        1.0 - group_min_lr_ratio
                    ) * cosine

                return lr_multiplier

            lr_lambdas = []
            for group in optimizer.param_groups:
                group_name = str(group.get("name", ""))
                if group_name.startswith("action_head_"):
                    group_min_lr_ratio = action_head_min_lr_ratio
                elif group_name.startswith("vlm_"):
                    group_min_lr_ratio = vlm_min_lr_ratio
                else:
                    group_min_lr_ratio = min_lr_ratio
                lr_lambdas.append(make_lr_multiplier(group_min_lr_ratio))

            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda=lr_lambdas
            )
            print(
                "✅ LR scheduler: linear warmup + cosine decay, "
                f"total_steps={total_steps:,}, warmup_steps={warmup_steps:,}, "
                f"action_min={action_head_min_lr_ratio:.3f}, "
                f"vlm_min={vlm_min_lr_ratio:.3f}, "
                f"other_min={min_lr_ratio:.3f}"
            )
            return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
        
        else:
            return [optimizer]

    def get_training_callbacks(self):

        checkpoint_cb_best = EfficientBestAndLastCheckpoint(save_top_k=1,
                                        monitor='val/score_epoch',
                                        filename='best-{epoch}-{step}',
                                        mode="max",
                                        # The optimized subclass retains a
                                        # real latest state with one write.
                                        save_last="link",
                                        )

        lr_monitor = LearningRateMonitor(logging_interval="step", 
                                            log_momentum=False,
                                            log_weight_decay=False)
        timing_interval = int(os.getenv("DRIVEVLA_TIMING_INTERVAL", "0"))
        timing_callbacks = (
            [TrainingThroughputCallback(timing_interval)]
            if timing_interval > 0
            else []
        )
        ema_callbacks = (
            [EMARegisterTargetCallback()] if self.world_model_enabled else []
        )
        
        if self.progress_bar:
            return [
                checkpoint_cb_best,
                lr_monitor,
                *timing_callbacks,
                *ema_callbacks,
            ]
        else:
            progress_bar = LitProgressBar()
            return [
                checkpoint_cb_best,
                progress_bar,
                lr_monitor,
                *timing_callbacks,
                *ema_callbacks,
            ]

    def verify_lora_activation(self):
        """
        验证LoRA参数是否确实可训练
        """
        print("=== LoRA配置验证 ===")
        print(f"使用LoRA: {self.lora_config.use_lora}")
        
        if self.backbone is None:
            print("Backbone未初始化")
            return
        
        # 统计参数
        total_params = 0
        trainable_params = 0
        lora_params = 0
        
        for name, param in self.backbone.named_parameters():
            total_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
                if "lora" in name.lower():
                    lora_params += param.numel()
        
        print(f"Backbone总参数: {total_params:,}")
        print(f"可训练参数: {trainable_params:,} ({trainable_params/total_params*100:.4f}%)")
        print(f"其中LoRA参数: {lora_params:,}")
        
        # 列出LoRA模块
        print("\nLoRA模块列表:")
        for name, module in self.backbone.named_modules():
            if hasattr(module, "lora_A") or hasattr(module, "lora_B"):
                print(f"  - {name}")


    @staticmethod
    def _decode_paths_from_tensor(path_tensor: torch.Tensor) -> List[str]:
        """
        Decodes a batch of path tensors back into a list of file path strings.
        
        Args:
            path_tensor (torch.Tensor): A 2D tensor of shape 
                (batch_size, max_path_length) from the collate_fn.
        
        Returns:
            List[str]: A list of decoded file path strings.
        """
        decoded_paths = []
        for single_path_tensor in path_tensor:
            chars = []
            for code in single_path_tensor:
                code_item = code.item()
                if code_item == 0: 
                    break
                chars.append(chr(code_item))
            decoded_paths.append("".join(chars))
        return decoded_paths

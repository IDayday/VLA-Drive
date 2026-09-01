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
from .action_decoder import ActionDecoder
from .layers.planning_registers import freeze_vision_except_qv_lora

from peft import LoraConfig, get_peft_model

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

        self._lr_args=lr_args
        self.progress_bar=progress_bar
        self.scheduler_args=scheduler_args
        self.batch_size=batch_size
        self.num_gpus=num_gpus
        self.checkpoint_path=checkpoint_path
        self.stage1_checkpoint_path = stage1_checkpoint_path
        self.cache_data = cache_data
        self._initialized = False

        if self.checkpoint_path and self.stage1_checkpoint_path:
            raise ValueError(
                "checkpoint_path (full-agent restore) and stage1_checkpoint_path "
                "(VLM-only warm start) are mutually exclusive."
            )

        if not self.cache_data:
            self.action_head=ActionDecoder(action_head_config)

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
            load_legacy_checkpoint_with_planreg_audit(
                self,
                ckpt,
                legacy_lora_scale=2.0,
            )
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
        self._report_backbone_trainability()
        self._initialized = True

    def get_sensor_config(self) -> SensorConfig:
        def _history(name: str) -> List[int]:
            values = getattr(self.action_head_config, name, [])
            return list(values) if values else []

        return SensorConfig(
            cam_f0=_history("cam_f0"),
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
        return [TrajectoryTargetBuilder(config=self.action_head_config)]

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
        # These payloads are consumed only by the frozen VLM.  Pop them before
        # the generic feature transfer so paths and prompt construction never
        # force repeated CUDA-to-host synchronization.
        pixel_values_batch = features.pop("pixel_values", None)
        questions = features.pop("questions", None)
        image_path_tensor = features.pop("image_path_tensor", None)
        input_ids = features.pop("input_ids", None)
        attention_mask = features.pop("attention_mask", None)
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
            prompt_history = features["history_trajectory"]
            prompt_command = features["high_command_one_hot"]
            if prompt_history.is_cuda:
                prompt_history = prompt_history.detach().cpu()
            if prompt_command.is_cuda:
                prompt_command = prompt_command.detach().cpu()
            questions = build_drivevla_questions(prompt_history, prompt_command)
        elif isinstance(questions, str):
            questions = [questions]

        for key, tensor in features.items():
            if isinstance(tensor, torch.Tensor):
                features[key] = tensor.cuda(non_blocking=True)

        history_trajectory = features["history_trajectory"]
        high_command_one_hot = features["high_command_one_hot"]
        
        
        if history_trajectory.ndim == 2: history_trajectory = history_trajectory.unsqueeze(0)
        if high_command_one_hot.ndim == 1: high_command_one_hot = high_command_one_hot.unsqueeze(0)

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
                if pixel_values_batch is None:
                    if image_paths is None:
                        raise RuntimeError("InternVL requires image paths or pixel_values")
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
                else:
                    pixel_values_list = [
                        value.cuda(non_blocking=True) for value in pixel_values_batch
                    ]
                    num_patches_list = [value.shape[0] for value in pixel_values_list]
                    pixel_values_cat = torch.cat(pixel_values_list, dim=0)

                outputs = self.backbone(
                    pixel_values_cat,
                    questions,
                    num_patches_list=num_patches_list,
                    model_inputs=pretokenized_inputs,
                )
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

        return self.action_head(action_inputs)

    def compute_trajectory(self, agent_input: AgentInput) -> Trajectory:
        self.eval()

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
            poses = predictions["pred_traj"].float().cpu().squeeze(0)

        return Trajectory(poses)

    def compute_trajectory_vis(self, agent_input: AgentInput) -> Trajectory:
        self.eval()

        features: Dict[str, torch.Tensor] = {}
        # build features
        for builder in self.get_feature_builders():
            features.update(builder.compute_features(agent_input))

        # add batch dimension
        features = {k: v.unsqueeze(0) for k, v in features.items()}

        with torch.no_grad():
            predictions = self.forward(features)
            poses = predictions["pred_traj"].float().cpu().squeeze(0)
        return Trajectory(poses)


    def compute_loss(
            self,
            features: Dict[str, torch.Tensor],
            targets: Dict[str, torch.Tensor],
            pred: Dict[str, torch.Tensor],
    ) -> Dict:
        return self.loss(targets, pred, self.action_head_config, self.compute_score)

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


    def get_optimizers(self) -> Union[Optimizer, Dict[str, LRScheduler]]:
        """
        pack all trainable parameters into optimizer
        """
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
        
        if self.progress_bar:
            return [checkpoint_cb_best, lr_monitor, *timing_callbacks]
        else:
            progress_bar = LitProgressBar()
            return [checkpoint_cb_best, progress_bar, lr_monitor, *timing_callbacks]

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

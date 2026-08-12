"""NAVSIM training adapter for Retrieve Model V1."""

import json
from pathlib import Path
from typing import Any, Dict, Optional

import pytorch_lightning as pl
import torch
import torch.distributed as dist
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import SensorConfig

from .retrieve_features import RetrieveFeatureBuilder, RetrieveTargetBuilder
from .retrieve_loss import RetrieveLoss
from .retrieve_model import RetrieveModelV1


class ValidationLossEpochLogger(pl.Callback):
    """Log a sample-weighted, DDP-reduced validation loss once per epoch."""

    def __init__(self, metric_name: str = "val_loss_epoch") -> None:
        super().__init__()
        self.metric_name = metric_name
        self._loss_sum: Optional[torch.Tensor] = None
        self._sample_count = 0

    def on_validation_epoch_start(self, trainer, pl_module) -> None:
        del trainer, pl_module
        self._loss_sum = None
        self._sample_count = 0

    def on_validation_batch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx,
        dataloader_idx=0,
    ) -> None:
        del trainer, pl_module, batch_idx, dataloader_idx
        loss = outputs.get("loss") if isinstance(outputs, dict) else outputs
        if not isinstance(loss, torch.Tensor):
            return
        features = batch[0]
        first_feature = next(iter(features.values()))
        batch_size = int(first_feature.shape[0])
        weighted_loss = loss.detach().float() * batch_size
        self._loss_sum = (
            weighted_loss
            if self._loss_sum is None
            else self._loss_sum + weighted_loss
        )
        self._sample_count += batch_size

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        del trainer
        if self._loss_sum is None or self._sample_count == 0:
            return
        state = torch.stack(
            [
                self._loss_sum,
                self._loss_sum.new_tensor(float(self._sample_count)),
            ]
        )
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(state, op=dist.ReduceOp.SUM)
        mean_loss = state[0] / state[1].clamp_min(1.0)
        pl_module.log(
            self.metric_name,
            mean_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )


def _to_plain_dict(config) -> Dict[str, Any]:
    if isinstance(config, DictConfig):
        return OmegaConf.to_container(config, resolve=True)
    return dict(config)


def _extract_model_state_dict(payload: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    state_dict = payload.get(
        "state_dict",
        payload.get("model_state_dict", payload),
    )
    prefixes = ("agent.model.", "model.")
    for prefix in prefixes:
        if any(key.startswith(prefix) for key in state_dict):
            return {
                key[len(prefix) :]: value
                for key, value in state_dict.items()
                if key.startswith(prefix)
            }
    return dict(state_dict)


class RetrieveModelAgent(AbstractAgent):
    """Trainable NAVSIM agent wrapper for the standalone retrieval model."""

    def __init__(
        self,
        model_config,
        feature_config,
        target_config,
        loss_config,
        optimizer_config,
        training_config=None,
        checkpoint_path: str = "",
    ) -> None:
        super().__init__()
        self.model_config = _to_plain_dict(model_config)
        self.feature_config = _to_plain_dict(feature_config)
        self.target_config = _to_plain_dict(target_config)
        self.loss_config = _to_plain_dict(loss_config)
        self.optimizer_config = _to_plain_dict(optimizer_config)
        self.training_config = (
            _to_plain_dict(training_config) if training_config is not None else {}
        )
        self.checkpoint_path = checkpoint_path

        self.model = RetrieveModelV1(**self.model_config)
        resolved_loss_config = self._resolve_loss_config(self.loss_config)
        resolved_loss_config.setdefault(
            "map_num_classes",
            int(self.model_config.get("map_num_classes", 4)),
        )
        resolved_loss_config.setdefault(
            "agent_num_classes",
            int(self.model_config.get("agent_num_classes", 3)),
        )
        self.loss_module = RetrieveLoss(**resolved_loss_config)

    @staticmethod
    def _resolve_loss_config(loss_config: Dict[str, Any]) -> Dict[str, Any]:
        loss_config = dict(loss_config)
        stats_path = loss_config.pop("stats_path", "")
        if stats_path:
            stats = json.loads(Path(stats_path).read_text(encoding="utf-8"))
            loss_config["map_class_weights"] = stats["map_class_weights"]
            loss_config["agent_class_weights"] = stats["agent_class_weights"]
        required = {"map_class_weights", "agent_class_weights"}
        missing = required - set(loss_config)
        if missing:
            raise ValueError(
                f"Missing class weights {sorted(missing)}. Generate target statistics "
                "or set explicit weights."
            )
        return loss_config

    def name(self) -> str:
        return "RetrieveModelAgentV1"

    def initialize(self) -> None:
        if not self.checkpoint_path:
            return
        payload = torch.load(self.checkpoint_path, map_location="cpu")
        self.model.load_state_dict(
            _extract_model_state_dict(payload),
            strict=True,
        )

    def get_sensor_config(self) -> SensorConfig:
        camera_names = set(self.feature_config.get("camera_names", ["cam_f0"]))
        history_iteration = 3
        return SensorConfig(
            cam_f0=[history_iteration] if "cam_f0" in camera_names else [],
            cam_l0=[history_iteration] if "cam_l0" in camera_names else [],
            cam_l1=[],
            cam_l2=[],
            cam_r0=[history_iteration] if "cam_r0" in camera_names else [],
            cam_r1=[],
            cam_r2=[],
            cam_b0=[history_iteration] if "cam_b0" in camera_names else [],
            lidar_pc=[],
        )

    def get_feature_builders(self):
        return [RetrieveFeatureBuilder(**self.feature_config)]

    def get_target_builders(self):
        return [RetrieveTargetBuilder(**self.target_config)]

    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self.model(features)

    def compute_loss(
        self,
        features: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        predictions: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        del features
        return self.loss_module(predictions, targets)

    def get_optimizers(self):
        parameters = [parameter for parameter in self.parameters() if parameter.requires_grad]
        optimizer_name = self.optimizer_config.get("name", "AdamW")
        kwargs = {
            "lr": float(self.optimizer_config.get("lr", 2e-4)),
            "weight_decay": float(self.optimizer_config.get("weight_decay", 1e-4)),
        }
        if optimizer_name == "AdamW":
            optimizer = torch.optim.AdamW(parameters, **kwargs)
        elif optimizer_name == "Adam":
            optimizer = torch.optim.Adam(parameters, **kwargs)
        else:
            raise ValueError(f"Unsupported optimizer {optimizer_name}")

        scheduler_config = dict(self.optimizer_config.get("scheduler", {}))
        if not scheduler_config.get("enabled", False):
            return optimizer
        scheduler_name = scheduler_config.get("name", "ReduceLROnPlateau")
        if scheduler_name != "ReduceLROnPlateau":
            raise ValueError(f"Unsupported scheduler {scheduler_name}")
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=float(scheduler_config.get("factor", 0.5)),
            patience=int(scheduler_config.get("patience", 2)),
            threshold=float(scheduler_config.get("threshold", 1e-3)),
            threshold_mode="rel",
            cooldown=int(scheduler_config.get("cooldown", 0)),
            min_lr=float(scheduler_config.get("min_lr", 1e-6)),
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": scheduler_config.get("monitor", "val_loss_epoch"),
                "interval": "epoch",
                "frequency": 1,
                "strict": True,
            },
        }

    def get_training_callbacks(self) -> list[pl.Callback]:
        monitor = self.training_config.get("monitor", "val_loss_epoch")
        callbacks: list[pl.Callback] = []
        if self.training_config.get("aggregate_validation_loss", False):
            callbacks.append(ValidationLossEpochLogger(metric_name=monitor))

        checkpoint_config = dict(self.training_config.get("checkpoint", {}))
        save_top_k = int(checkpoint_config.get("save_top_k", -1))
        checkpoint_kwargs: Dict[str, Any] = {
            "save_last": bool(checkpoint_config.get("save_last", True)),
            "save_top_k": save_top_k,
            "every_n_epochs": 1,
            "filename": checkpoint_config.get(
                "filename",
                "retrieve-v1-{epoch:02d}-{step}",
            ),
        }
        if save_top_k != -1:
            checkpoint_kwargs.update(monitor=monitor, mode="min")
        callbacks.append(ModelCheckpoint(**checkpoint_kwargs))

        early_stopping_config = dict(
            self.training_config.get("early_stopping", {})
        )
        if early_stopping_config.get("enabled", False):
            callbacks.append(
                EarlyStopping(
                    monitor=monitor,
                    mode="min",
                    patience=int(early_stopping_config.get("patience", 6)),
                    min_delta=float(early_stopping_config.get("min_delta", 1e-3)),
                    check_finite=True,
                    verbose=True,
                )
            )
        callbacks.append(LearningRateMonitor(logging_interval="step"))
        return callbacks

    def encode_keys(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self.model.encode_keys(features)

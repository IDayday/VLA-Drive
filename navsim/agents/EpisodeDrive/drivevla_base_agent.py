from typing import Any, List, Dict, Optional, Union
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
from pathlib import Path
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from navsim.planning.training.dataset import load_feature_target_from_pickle
from pytorch_lightning.callbacks import ModelCheckpoint, ProgressBar, LearningRateMonitor
from navsim.common.dataloader import MetricCacheLoader

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import AgentInput, SensorConfig, Trajectory
from navsim.planning.training.abstract_feature_target_builder import AbstractFeatureBuilder, AbstractTargetBuilder
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from pytorch_lightning.callbacks import ModelCheckpoint, ProgressBar, LearningRateMonitor

from .utils.internvl_preprocess import load_image
from .utils.lr_scheduler import WarmupCosLR
from .utils.utils import format_number, build_from_configs
from .drivevla_features import DriveVLAFeatureBuilder ,TrajectoryTargetBuilder
from .drivevla_backbone import DriveVLABackbone
from .action_decoder import ActionDecoder

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

class DriveVLABaseAgent(AbstractAgent):
    def __init__(
        self,
        vlm_config,
        lora_config,
        action_head_config,
        lr_args=None,
        loss=None,
        progress_bar=True,
        scheduler_args: dict=None,
        batch_size: int=64,
        num_gpus: int=1,
        trajectory_sampling=None,
        checkpoint_path:str = None,
    ):
        super().__init__()
        self.action_head_config=action_head_config
        self.vlm_config=vlm_config
        self.lora_config=lora_config

        self._lr_args=lr_args
        self.progress_bar=progress_bar
        self.scheduler_args=scheduler_args
        self.batch_size=batch_size
        self.num_gpus=num_gpus
        self.checkpoint_path=checkpoint_path

        cache_data = False

        if not cache_data:
            self.action_head=ActionDecoder(action_head_config)

        if not cache_data and self.action_head_config.checkpoint_path=="":
            self.bce_logit_loss=nn.BCEWithLogitsLoss

            self.ray=True

            if self.ray:
                from navsim.planning.utils.multithreading.worker_ray_no_torch import RayDistributedNoTorch
                from nuplan.planning.utils.multithreading.worker_utils import worker_map
                self.worker = RayDistributedNoTorch(threads_per_node=8)
                self.worker_map=worker_map

            from .score_module.compute_navsim_score import get_scores

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

            self.loss = loss

        self._trajectory_sampling = trajectory_sampling
        self.backbone = None
        
        local_rank = int(os.getenv("LOCAL_RANK", "0"))
        device = f"cuda:{local_rank}"
        self.device = device
        if not self.vlm_config.cache_hidden_state and not self.vlm_config.cache_mode:
            print("Agent running in 'no-cache' mode. Initializing internal backbone.")
            if not self.vlm_config.vlm_path or not self.vlm_config.vlm_type:
                raise ValueError("In 'no-cache' mode, vlm_path and vlm_type are required.")
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

    def initialize(self) -> None:
        if self.checkpoint_path:
            ckpt = torch.load(self.checkpoint_path, map_location="cpu")["state_dict"]
            model_dict = self.state_dict()
            filtered_ckpt = {}
            for k, v in ckpt.items():
                k2 = k[len("agent."):] if k.startswith("agent.") else k
                if k2 in model_dict and v.shape == model_dict[k2].shape:
                    filtered_ckpt[k2] = v
            self.load_state_dict(filtered_ckpt, strict=True)
            print(f"✅ Agent loaded from checkpoint: {self.checkpoint_path}")
            
        if self.vlm_config.freeze_backbone:
            self._freeze_backbone()

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
        for key, tensor in features.items():
            if isinstance(tensor, torch.Tensor):
                features[key] = tensor.cuda()

        model_dtype = next(self.action_head.parameters()).dtype

        history_trajectory = features["history_trajectory"].cuda()
        high_command_one_hot = features["high_command_one_hot"].cuda()
        
        
        if history_trajectory.ndim == 2: history_trajectory = history_trajectory.unsqueeze(0)
        if high_command_one_hot.ndim == 1: high_command_one_hot = high_command_one_hot.unsqueeze(0)

        if self.vlm_config.cache_hidden_state:
            last_hidden_state = features["last_hidden_state"].cuda()
        else:
            if self.backbone is None:
                raise RuntimeError("Agent is in 'no-cache' mode, but backbone is not initialized.")
            image_path_tensor = features["image_path_tensor"]
            if image_path_tensor.ndim == 1: image_path_tensor = image_path_tensor.unsqueeze(0)
            image_paths = self._decode_paths_from_tensor(image_path_tensor)
            
            if self.vlm_config.vlm_type == "internvl":
                pixel_values_list = [load_image(path) for path in image_paths]
            
                num_patches_list = [p.shape[0] for p in pixel_values_list]
                pixel_values_cat = torch.cat(pixel_values_list, dim=0).cuda()
                

                navigation_commands = ['turn left', 'go straight', 'turn right', 'unknown']
                command_indices = torch.argmax(high_command_one_hot, dim=-1)
                command_str_list = [navigation_commands[idx.item()] for idx in command_indices]

                questions = []
                batch_size = high_command_one_hot.shape[0]
                for i in range(batch_size):
                    history_trajectory_sample = history_trajectory[i]
                    command_str_sample = command_str_list[i]

                    history_str = ' '.join([
                        f'   - t-{3-j}: ({format_number(history_trajectory_sample[j, 0].item())}, '
                        f'{format_number(history_trajectory_sample[j, 1].item())}, '
                        f'{format_number(history_trajectory_sample[j, 2].item())})'
                        for j in range(history_trajectory_sample.shape[0])
                    ])
                    
                    prompt = (
                        "<image>\nAs an autonomous driving system, predict the vehicle's trajectory based on:\n"
                        "1. Visual perception from front camera view\n"
                        f"2. Historical motion context (last 4 timesteps):{history_str}\n"
                        f"3. Active navigation command: [{command_str_sample.upper()}]"
                    )
                    output_requirements = (
                        "\nOutput requirements:\n- Predict 8 future trajectory points\n"
                        "- Each point format: (x:float, y:float, heading:float)\n"
                        "- Use [PT, ...] to encapsulate the trajectory\n"
                        "- Maintain numerical precision to 2 decimal places"
                    )
                    questions.append(f"{prompt}{output_requirements}")

                outputs = self.backbone(pixel_values_cat, questions, num_patches_list=num_patches_list)
                last_hidden_state = outputs.hidden_states[-1]
            
            elif self.vlm_config.vlm_type == "qwen3vl":
                pixel_values_list = image_paths
                
                navigation_commands = ['turn left', 'go straight', 'turn right', 'unknown']
                command_indices = torch.argmax(high_command_one_hot, dim=-1)
                command_str_list = [navigation_commands[idx.item()] for idx in command_indices]

                questions = []
                batch_size = high_command_one_hot.shape[0]
                for i in range(batch_size):
                    history_trajectory_sample = history_trajectory[i]
                    command_str_sample = command_str_list[i]

                    history_str = ' '.join([
                        f'   - t-{3-j}: ({format_number(history_trajectory_sample[j, 0].item())}, '
                        f'{format_number(history_trajectory_sample[j, 1].item())}, '
                        f'{format_number(history_trajectory_sample[j, 2].item())})'
                        for j in range(history_trajectory_sample.shape[0])
                    ])
                    
                    prompt = (
                        "<image>\nAs an autonomous driving system, predict the vehicle's trajectory based on:\n"
                        "1. Visual perception from front camera view\n"
                        f"2. Historical motion context (last 4 timesteps):{history_str}\n"
                        f"3. Active navigation command: [{command_str_sample.upper()}]"
                    )
                    output_requirements = (
                        "\nOutput requirements:\n- Predict 8 future trajectory points\n"
                        "- Each point format: (x:float, y:float, heading:float)\n"
                        "- Use [PT, ...] to encapsulate the trajectory\n"
                        "- Maintain numerical precision to 2 decimal places"
                    )
                    questions.append(f"{prompt}{output_requirements}")

                outputs, visual_feature_idx = self.backbone(pixel_values_list, questions)
                last_hidden_state = outputs.hidden_states[-1]
                
                # Get the alignment feature (index the visual token)
                start_index = visual_feature_idx[0]
                end_index = visual_feature_idx[-1]
                alignment_feature = outputs.hidden_states[-7][:, start_index:end_index+1, :]  # align the 3/4 layer (21/28) with the geometry feature

        status_feature = features["status_feature"].cuda()
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
        action_head_params = []
        backbone_params = []
        lora_params = []

        global_batchsize = self.batch_size * self.num_gpus
        if self._lr_args["name"] == "Adam":
            lr = self._lr_args["base_lr"] * math.sqrt(global_batchsize / self._lr_args["base_batch_size"])
        elif self._lr_args["name"] == "AdamW":
            lr = self._lr_args["base_lr"] * math.sqrt(global_batchsize / self._lr_args["base_batch_size"])
        else:
            raise NotImplementedError
        
        for name, param in self.named_parameters():
            if param.requires_grad:
                if "lora" in name.lower():
                    lora_params.append(param)
                elif "backbone" in name:
                    backbone_params.append(param)
                elif "action_head" in name:
                    action_head_params.append(param)
        
        # 构建参数组（不同学习率）
        param_groups = []
        
        # ====== 新增：LoRA参数组（使用较高学习率） ======
        if lora_params:
            param_groups.append({
                'params': lora_params,
                'lr_scale': 1.0,  # LoRA通常使用较高学习率
                'weight_decay': 1e-4,
            })
            print(f"✅ LoRA 参数组: {len(lora_params)} 个参数，学习率={lr:.2e}")
        
        # Backbone 参数组（非LoRA部分，通常冻结）
        if backbone_params:
            param_groups.append({
                'params': backbone_params,
                'lr_scale': 0.1,
                'weight_decay': 1e-4,
            })
            print(f"✅ Backbone 参数组: {len(backbone_params)} 个参数，学习率={lr * 0.1:.2e}")
        
        # Action Head 参数组（使用基础学习率）
        if action_head_params:
            param_groups.append({
                'params': action_head_params,
                'lr_scale': 1.0,
                'weight_decay': 1e-4,
            })
            print(f"✅ Action Head 参数组: {len(action_head_params)} 个参数，学习率={lr:.2e}")
        
        # 如果没有可训练参数，则抛出异常（理论上不会发生）
        if not param_groups:
            raise RuntimeError("No trainable parameters found.")
        
        # 创建优化器
        optimizer = torch.optim.AdamW(
            param_groups,
            betas=(0.9, 0.95),
            lr=lr
        )
        
        if self.scheduler_args is not None:

            T_max = int(math.ceil(self.scheduler_args.dataset_size / global_batchsize) *  self.scheduler_args.num_epochs)

            # classic cosine
            # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            #     optimizer,
            #     T_max=T_max, 
            #     eta_min=0.0, last_epoch=-1
            # )

            # Ramp + cosine
            T_max_ramp = int(T_max * 0.1)
            scheduler_ramp = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-6, total_iters=T_max_ramp)
            T_max_cosine = T_max - T_max_ramp
            scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=T_max_cosine, 
                eta_min=0.0, last_epoch=-1
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[scheduler_ramp, scheduler_cosine],
                milestones=[T_max_ramp],
            )           
            
            return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
        
        else:
            return [optimizer]

    def get_training_callbacks(self):

        checkpoint_cb_best = ModelCheckpoint(save_top_k=1,
                                        monitor='val/score_epoch',
                                        filename='best-{epoch}-{step}',
                                        mode="max"
                                        )
        
        checkpoint_cb = ModelCheckpoint(save_last=True)

        lr_monitor = LearningRateMonitor(logging_interval="step", 
                                            log_momentum=False,
                                            log_weight_decay=False)
        
        if self.progress_bar:
            return [checkpoint_cb_best, checkpoint_cb, lr_monitor]
        else:
            progress_bar = LitProgressBar()
            return [checkpoint_cb_best, checkpoint_cb, progress_bar, lr_monitor]

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

# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by Jinhui YE / HKUST University] in [2025].
"""
Qwen-GROOT Framework
A lightweight implementation that Qwen2.5-vl + Flow-matching head to directly predict continuous actions
Flow-matching header is copyright from GR00T N1.5, but a sample MoE inspired by PI_0
"""
import time
from tqdm import tqdm
from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image



from starVLA.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100

from starVLA.model.framework.base_framework import baseframework
from starVLA.model.modules.vlm import get_vlm_model
from starVLA.model.modules.action_model.LayerwiseFM_ActionHeader import get_action_model, LayerwiseFlowmatchingActionHead
from starVLA.model.modules.action_model.multi_trajectory.config import (
    MultiTrajectoryConfig,
    multi_trajectory_enabled,
)
from starVLA.training.trainer_utils.trainer_tools import resize_images
from starVLA.model.tools import FRAMEWORK_REGISTRY


def _collate_multi_trajectory_targets(values, device, dtype):
    """Stack explicit offline DDP-DRS labels; never called by inference."""
    first = values[0]
    if isinstance(first, dict):
        expected_keys = set(first)
        if any(set(value) != expected_keys for value in values):
            raise ValueError("DDP-DRS target dictionaries have inconsistent keys")
        return {
            key: _collate_multi_trajectory_targets(
                [value[key] for value in values], device, dtype
            )
            for key in first
        }
    return torch.as_tensor(np.asarray(values), device=device, dtype=dtype)

@FRAMEWORK_REGISTRY.register("QwenFM")
@FRAMEWORK_REGISTRY.register("QwenPI")
class Qwen_PI(baseframework):
    """
    Multimodal vision-language-action model.

    Components:
      - Qwen2.5 VL interface for fused language/vision token embeddings
      - Layer-wise cross DiT diffusion head 
      

    Focus: Predict future continuous actions conditioned on images + instruction.
    """
# 
    def __init__(
        self,
        config: Optional[dict] = None,
        **kwargs,
    ) -> None:
        """
        Construct all submodules and cache key configuration values.

        Args:
            config: Hierarchical configuration (OmegaConf/dict) containing framework + trainer sections.
            **kwargs: Reserved for future overrides (unused).
        """

        super().__init__()
        self.config = config
        self.qwen_vl_interface = get_vlm_model(config=self.config)

        # dynamic get llm config
        llm_layers, llm_hidden_size = self.config.framework.action_model.diffusion_model_cfg.num_layers, self.qwen_vl_interface.model.config.hidden_size

        DiTConfig = {
            "num_layers": llm_layers, 
            "input_embedding_dim": self.config.framework.action_model.hidden_size, 
            "attention_head_dim": 64, 
            "num_attention_heads": self.config.framework.action_model.hidden_size // 64}
        # self.config.framework.action_model.hidden_size = 1024 #check what this for?
        # self.config.framework.action_model.diffusion_model_cfg.cross_attention_dim = llm_hidden_size

        self.config.framework.action_model.DiTConfig = DiTConfig
        self.action_model: LayerwiseFlowmatchingActionHead = get_action_model(config=self.config)  # 修复后续引用

        # Keep the original model graph and state_dict byte-for-byte compatible
        # unless the new baseline is explicitly enabled.  Imports requiring
        # DrivoR/DriveSuprim assets and all new parameter construction happen
        # only inside this branch.
        if multi_trajectory_enabled(self.config):
            from starVLA.model.modules.action_model.multi_trajectory.planner import (
                DDPDrivoRSuprimPlanner,
            )

            self.multi_trajectory_config = MultiTrajectoryConfig.from_full_config(
                self.config
            )
            self.multi_trajectory_planner = DDPDrivoRSuprimPlanner(
                action_head=self.action_model,
                config=self.multi_trajectory_config,
                qwen_hidden_dim=llm_hidden_size,
            )
            for parameter in self.qwen_vl_interface.parameters():
                parameter.requires_grad = False
            self.qwen_vl_interface.eval()
            self.action_model.eval()

        self.future_action_window_size = config.framework.action_model.future_action_window_size
        self.past_action_window_size = config.framework.action_model.past_action_window_size
        self.chunk_len = self.past_action_window_size + 1 + self.future_action_window_size

    def train(self, mode: bool = True) -> "Qwen_PI":
        """Keep the frozen Qwen+DDP proposal generator deterministic when enabled."""

        super().train(mode)
        if hasattr(self, "multi_trajectory_planner"):
            self.qwen_vl_interface.eval()
            self.action_model.eval()
        return self

    def forward(
        self,
        examples: List[dict] = None,
        **kwargs,
    ) -> Tuple:
        """
        训练前向：直接回归未来动作（无扩散）。

        Flow:
          1. Build QwenVL inputs (images + instruction tokens)
          2. Extract hidden states from configured layer range
          7. Predict action and compute L1 loss

        Args:
            examples: List[dict], each dict requires:
                - image: List[PIL.Image] (multi-view)
                - lang: str instruction
                - action: np.ndarray or list shaped [T, action_dim]
            **kwargs: Reserved.

        Returns:
            dict:
                action_loss (torch.Tensor): Scalar diffusion noise prediction loss.
        """
        batch_images = [example["image"] for example in examples]  #  [B，[PLT]]
        instructions = [example["lang"] for example in examples]  # [B, str]
        actions = [example["action"] for example in examples]  # label [B， len, 7]
        
        state = [example["state"] for example in examples] if "state" in examples[0] else None  # [B, 1, state_dim]
        

        # Step 1: QWenVL input format
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            # 取与 DiT 层数匹配的最后 N 层隐藏态，按层喂给 DiT
            all_hidden = qwenvl_outputs.hidden_states
            expected_layers = len(self.action_model.model.transformer_blocks)
            vl_embs_list = list(all_hidden[-expected_layers:])
            base_hidden = vl_embs_list[-1]

        if hasattr(self, "multi_trajectory_planner"):
            stage = self.multi_trajectory_config.training_stage
            full_last_hidden_state = all_hidden[
                self.multi_trajectory_config.scene_compressor.source_layer
            ]
            attention_mask = qwen_inputs.get("attention_mask")
            if attention_mask is None:
                raise KeyError(
                    "DDP-DRS scene compressor requires the Qwen attention_mask"
                )
            state_tensor = (
                torch.as_tensor(
                    np.asarray(state),
                    device=base_hidden.device,
                    dtype=base_hidden.dtype,
                )
                if state is not None
                else None
            )
            if stage == "cache_candidates":
                candidates = self.multi_trajectory_planner.sample_physical_candidates(
                    vl_embs_list, state_tensor
                )
                return {"candidate_trajectories": candidates}
            raw_targets = [
                example.get("multi_trajectory_targets") for example in examples
            ]
            if any(target is None for target in raw_targets):
                raise KeyError(
                    "DDP-DRS training requires validated multi_trajectory_targets"
                )
            targets = _collate_multi_trajectory_targets(
                raw_targets, base_hidden.device, base_hidden.dtype
            )
            cached_dynamic_trajectories = None
            if stage in {
                "train_drivor",
                "train_suprim_joint",
                "joint_finetune",
            }:
                raw_candidates = [
                    example.get("multi_trajectory_candidates")
                    for example in examples
                ]
                if any(candidate is None for candidate in raw_candidates):
                    raise KeyError(
                        f"DDP-DRS {stage} requires validated cached candidates"
                    )
                cached_dynamic_trajectories = _collate_multi_trajectory_targets(
                    raw_candidates, base_hidden.device, base_hidden.dtype
                )
            with torch.autocast(
                device_type=base_hidden.device.type,
                dtype=torch.bfloat16,
                enabled=base_hidden.device.type == "cuda",
            ):
                training_output = self.multi_trajectory_planner.compute_training_loss(
                    vl_embs_list=vl_embs_list,
                    state=state_tensor,
                    full_hidden_state=full_last_hidden_state,
                    attention_mask=attention_mask,
                    targets=targets,
                    cached_dynamic_trajectories=cached_dynamic_trajectories,
                )
            return {
                "action_loss": training_output["loss"],
                "multi_trajectory_loss": training_output["loss"],
                "multi_trajectory_loss_details": training_output,
            }
        # Step 4: Action Expert Forward and Loss
        with torch.autocast("cuda", dtype=torch.float32):
            # 标签对齐：取最后 chunk_len 段
            actions = torch.tensor(
                np.array(actions), device=base_hidden.device, dtype=base_hidden.dtype
            )  # [B, T_full, action_dim]
            actions_target = actions[:, -(self.future_action_window_size):, :]  # (B, chunk_len, action_dim)

            repeated_diffusion_steps = (
                self.config.trainer.get("repeated_diffusion_steps", 4) if self.config and self.config.trainer else 4
            )
            # repeated_diffusion_steps = 2 # NO repeat for big action FM
            actions_target_repeated = actions_target.repeat(repeated_diffusion_steps, 1, 1)
            # 对每层特征做 repeat
            vl_embs_list_repeated = [h.repeat(repeated_diffusion_steps, 1, 1) for h in vl_embs_list]
            
            state_repeated = None
            if state is not None:
                state = torch.tensor(
                    np.array(state), device=base_hidden.device, dtype=base_hidden.dtype
                )
                state_repeated = state.repeat(repeated_diffusion_steps, 1, 1)

            action_loss = self.action_model(vl_embs_list_repeated, actions_target_repeated, state_repeated)  # (B, chunk_len, action_dim)



        return {"action_loss": action_loss}

    @torch.inference_mode()
    def predict_action(
        self,
        batch_images: Optional[List[List[Image.Image]]] = None,
        instructions: Optional[List[str]] = None,
        state: Optional[np.ndarray] = None,
        examples: Optional[List[dict]] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        推理：单次前向直接回归未来动作（无扩散采样）。

        Steps:
          1. Resize images to training resolution (if specified)
          2. Encode with QwenVL (hidden states retained)
          6. Return normalized action trajectory

        Args:
            batch_images: List of samples; each sample is List[PIL.Image] (multi-view).
            instructions: List[str] natural language task instructions.
            cfg_scale: >1 enables classifier-free guidance (scales conditional vs unconditional).
            use_ddim: Whether to use DDIM deterministic sampling.
            num_ddim_steps: Number of DDIM steps if enabled.
            **kwargs: Reserved.

        Returns:
            dict:
                normalized_actions (np.ndarray): Shape [B, T, action_dim], diffusion-sampled normalized actions.
        """
        # The native trainer/inference entry points pass sample dictionaries,
        # while older direct callers pass the three arrays separately.  Both
        # routes feed the exact same Qwen/action-head implementation.
        if examples is None and batch_images and isinstance(batch_images[0], dict):
            examples = batch_images
            batch_images = None
        if examples is not None:
            if batch_images is not None or instructions is not None or state is not None:
                raise ValueError(
                    "pass either examples or batch_images/instructions/state, not both"
                )
            if not examples:
                raise ValueError("predict_action examples cannot be empty")
            batch_images = [example["image"] for example in examples]
            instructions = [example["lang"] for example in examples]
            state_values = [example.get("state") for example in examples]
            state = None if all(value is None for value in state_values) else state_values
            if any(value is None for value in state_values) and state is not None:
                raise ValueError("state must be present for every sample or none")
        if batch_images is None or instructions is None:
            raise ValueError("predict_action requires images and instructions")

        train_obs_image_size = getattr(self.config.datasets.vla_data, "image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)

        use_multi_trajectory = hasattr(self, "multi_trajectory_planner")
        profile_multi_trajectory = (
            use_multi_trajectory
            and self.multi_trajectory_config.diagnostics_enabled
        )
        if profile_multi_trajectory and torch.cuda.is_available():
            torch.cuda.synchronize()
        qwen_start = time.perf_counter() if use_multi_trajectory else None
    
        # Step 1: QWenVL input format
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            all_hidden = qwenvl_outputs.hidden_states
            expected_layers = len(self.action_model.model.transformer_blocks)
            vl_embs_list = list(all_hidden[-expected_layers:])
            base_hidden = vl_embs_list[-1]
            if use_multi_trajectory:
                full_last_hidden_state = all_hidden[
                    self.multi_trajectory_config.scene_compressor.source_layer
                ]
                attention_mask = qwen_inputs.get("attention_mask")
                if attention_mask is None:
                    raise KeyError(
                        "DDP-DRS scene compressor requires the Qwen attention_mask"
                    )
        if profile_multi_trajectory and torch.cuda.is_available():
            torch.cuda.synchronize()
        qwen_latency = (
            time.perf_counter() - qwen_start if use_multi_trajectory else None
        )

        state = torch.from_numpy(np.array(state)).to(base_hidden.device, dtype=base_hidden.dtype) if state is not None else None
        # Step 4: Action Expert Forward and Loss
        if use_multi_trajectory:
            with torch.autocast(
                device_type=base_hidden.device.type,
                dtype=torch.bfloat16,
                enabled=base_hidden.device.type == "cuda",
            ):
                selected_trajectory = self.multi_trajectory_planner(
                    vl_embs_list=vl_embs_list,
                    state=state,
                    full_hidden_state=full_last_hidden_state,
                    attention_mask=attention_mask,
                )
                # The DDP-DRS planner returns the selected metric SE(2)
                # trajectory.  Convert at the framework boundary so the
                # unchanged evaluator still receives normalized action deltas.
                from starVLA.model.modules.action_model.multi_trajectory.trajectory_codec import (
                    poses_to_normalized_deltas,
                )

                pred_actions = poses_to_normalized_deltas(selected_trajectory)
        else:
            # Preserve the original single-DDP autocast context exactly.
            with torch.autocast("cuda", dtype=torch.float32):
                pred_actions = self.action_model.predict_action(vl_embs_list, state)  # (B, chunk_len, action_dim)

        if use_multi_trajectory:
            # NumPy has no bfloat16 dtype.  Keep the planner in AMP/bfloat16
            # and convert only the existing public API payload to float32.
            normalized_actions = (
                pred_actions.detach().to(dtype=torch.float32).cpu().numpy()
            )
        else:
            normalized_actions = pred_actions.detach().cpu().numpy()
        output = {"normalized_actions": normalized_actions}
        if use_multi_trajectory and self.multi_trajectory_config.diagnostics_enabled:
            diagnostics = self.multi_trajectory_planner.last_diagnostics
            if diagnostics is not None:
                diagnostics.latency_qwen = qwen_latency
                if diagnostics.latency_total_inference is not None:
                    diagnostics.latency_total_inference += qwen_latency
                output["multi_trajectory_diagnostics"] = diagnostics
        return output



if __name__ == "__main__":
    from omegaconf import OmegaConf
    import debugpy
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", type=str, default="./starVLA/config/training/starvla_cotrain_oxe.yaml", help="Path to YAML config")
    args, clipargs = parser.parse_known_args()

    debugpy.listen(("0.0.0.0", 10092))
    print("🔍 Rank 0 waiting for debugger attach on port 10092...")
    debugpy.wait_for_client()

    cfg = OmegaConf.load(args.config_yaml)
    # try get model
    cfg.framework.qwenvl.base_vlm = "./playground/Pretrained_models/Qwen3-VL-4B-Instruct"
    

    model = Qwen_PI(cfg)
    print(model)


    # fake sample 
    image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    # Create a sample
    sample = {
        "action": np.random.uniform(-1, 1, size=(16, 7)).astype(np.float16), # action_chunk, action_dim
        "image": [image, image], # two views
        "lang": "This is a fake instruction for testing.",
        "state" : np.random.uniform(-1, 1, size=(1, 7)).astype(np.float16), # chunk, state_dim
    }

    batch  = [sample, sample]  # batch size 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    forward_output = model(batch)
    action_loss = forward_output['action_loss']
    print(f"Action Loss: {action_loss.item()}")

    # test predict action
    predict_output = model.predict_action(batch_images=[batch[0]["image"]], instructions=[batch[0]["lang"]], state=[batch[0]["state"]])
    normalized_actions = predict_output['normalized_actions']
    print(f"Unnormalized Action: {normalized_actions}")

    # # Advance: try forward model with dataloader
    # # can be fake sample， but here get from dataloader for simpler
    # from starVLA.dataloader.lerobot_datasets import get_vla_dataset, collate_fn

    # vla_dataset_cfg = cfg.datasets.vla_data
    # dataset = get_vla_dataset(data_cfg=vla_dataset_cfg)

    # from torch.utils.data import DataLoader

    # train_dataloader = DataLoader(
    #     dataset,
    #     batch_size=2,
    #     num_workers=1,  # For Debug
    #     collate_fn=collate_fn,
    # )
    # # 
    # for batch in tqdm(train_dataloader, desc="Processing Batches"):
    #     batch
    #     break

    # # try get model
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = model.to(device)
    # model(batch)

    # action = model.predict_action(batch_images=[batch[0]["image"]], instructions=[batch[0]["lang"]])

    # # fake state
    # for ba in batch:
    #     ba["state"] = ba["action"][0][None]

    # model(batch)
    # action = model.predict_action(batch_images=[batch[0]["image"]], instructions=[batch[0]["lang"]], state=[batch[0]["state"]])

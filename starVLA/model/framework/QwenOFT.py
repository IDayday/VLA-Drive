# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
# Implemented by [Jinhui YE / HKUST University] in [2025]. 

"""
Qwen-OFT Framework

A lightweight implementation that uses an action special token to parallelly predict continuous actions
conditioned on multi-view images plus a language instruction (shares parameters with the VLM).
Inspired by OpenVLA-OFT
Key Points:
  - Qwen2.5 vision-language backbone
  - Injects an action special token into the VLM
  - Continuous action prediction via L1 regression over the action special token hidden states


Note: How to add special tokens to Qwen2.5:
  download our model checkpoint with special tokens added: https://huggingface.co/StarVLA/Qwen2.5-VL-3B-Instruct-Action
  or /starVLA/model/modules/vlm/tools/add_qwen_special_tokens/README.md （adpat a little code)
  
"""
from typing import List
from tqdm import tqdm
from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image



from starVLA.training.trainer_utils import initialize_overwatch
from starVLA.model.tools import FRAMEWORK_REGISTRY


logger = initialize_overwatch(__name__)

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100

from starVLA.model.framework.base_framework import baseframework
from starVLA.model.modules.vlm import get_vlm_model
# from starVLA.model.modules.action_model.MLP_ActionHeader import get_action_model
from starVLA.model.modules.action_model.GR00T_ActionHeader import get_action_model, FlowmatchingActionHead, MLP, FlowmatchingRewardHead, get_reward_model
from starVLA.training.trainer_utils.trainer_tools import resize_images
from starVLA.model.modules.video_model.wan_i2v_header import WanWorldHead
import time
from omegaconf import OmegaConf

##### depth ppd
from starVLA.model.modules.depth_model.models.ppd_train import PixelPerfectDepth
from starVLA.cache.navsim_feature_cache import (
    GS_QUERY_TOKENS,
    MINE_AGENT_QUERY_TOKENS,
    REWARD_QUERY_TOKENS,
    RGB_QUERY_TOKENS,
    ROBOT_HISTORY_TOKEN,
    action_query_tokens,
)

import torch
import torch.nn as nn
import torch.nn.functional as F
class TinyDepthAdapter(nn.Module):
    def __init__(self, in_c=128, hidden=2048, grid=(8, 8)):
        super().__init__()
        self.grid = grid
        self.ln = nn.LayerNorm(in_c)
        self.proj = nn.Linear(in_c, hidden, bias=False)

    def forward(self, feat):  # [B,256,H,W]
        x = F.adaptive_avg_pool2d(feat, self.grid)          # [B,256,8,8]
        x = x.flatten(2).transpose(1, 2).contiguous()       # [B,64,256]
        x = self.ln(x)
        x = self.proj(x)                                    # [B,64,2048]
        return x

class RGBLatentAdapter(nn.Module):
    def __init__(self, in_c=16, hidden=1024, grid=(1,4,8), n_view=3):
        super().__init__()
        self.grid = grid          # (Ft, Ht, Wt_per_view)
        self.n_view = n_view
        self.ln = nn.LayerNorm(in_c)
        self.proj = nn.Linear(in_c, hidden, bias=False)

    def forward(self, x):  # x: [B,16,F,H,W]  where W = n_view * Wv


        B, C, f, H, W = x.shape
        assert W % self.n_view == 0
        Wv = W // self.n_view

        # split views on width
        x = x.view(B, C, f, H, self.n_view, Wv).permute(0,4,1,2,3,5).contiguous()
        # x: [B, V, C, F, H, Wv]
        x = x.view(B*self.n_view, C, f, H, Wv)

        # pool per-view
        Ft, Ht, Wt = self.grid
        x = F.adaptive_avg_pool3d(x, (Ft, Ht, Wt))          # [B*V, C, Ft, Ht, Wt]
        x = x.flatten(2).transpose(1, 2).contiguous()       # [B*V, N, C], N=Ft*Ht*Wt
        x = self.ln(x)
        x = self.proj(x)                                    # [B*V, N, hidden]

        # merge views back: [B, V*N, hidden]
        x = x.view(B, self.n_view * x.shape[1], -1)
        return x



@FRAMEWORK_REGISTRY.register("QwenOFT")
class Qwenvl_OFT(baseframework):
    """
    Multimodal vision-language-action model.

    Components:
      - Qwen2.5 VL interface for fused language/vision token embeddings
      - Layer-wise QFormer for multi-layer feature aggregation
      - DINO encoder for dense multi-view spatial tokens
      - DiT diffusion head for future action sequence modeling

    Focus: Predict future continuous actions conditioned on images + instruction.
    """

    def __init__(
        self,
        config: Optional[dict] = None,
        accelerator = None,
        infer_not_load_wan=0,
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

        # 历史轨迹 token（单个 repeated K 次）
        self.robot_history_token = ROBOT_HISTORY_TOKEN

        # rgb（2d）token 64 个
        self.rgb_query_tokens = list(RGB_QUERY_TOKENS)

        # gs（3d）token 64 个
        self.gs_query_tokens = list(GS_QUERY_TOKENS)

        # act token 8 个
        self.act_tok = OmegaConf.select(config, "act_tok", default=8)
        self.act_query_tokens = list(action_query_tokens(self.act_tok))

        # reward token
        self.reward_query_tokens = list(REWARD_QUERY_TOKENS)
        self.mine_agent_query_tokens = list(MINE_AGENT_QUERY_TOKENS)

        self.action_prompt_mode = str(
            OmegaConf.select(self.config, "framework.action_prompt_mode", default="full")
        ).lower()

        tokenizer = self.qwen_vl_interface.processor.tokenizer
        self._special_token_ids = {
            "history": (tokenizer.convert_tokens_to_ids(self.robot_history_token),),
            "action": tuple(tokenizer.convert_tokens_to_ids(self.act_query_tokens)),
        }
        if self.action_prompt_mode != "minimal":
            self._special_token_ids.update(
                {
                    "rgb": tuple(tokenizer.convert_tokens_to_ids(self.rgb_query_tokens)),
                    "gs": tuple(tokenizer.convert_tokens_to_ids(self.gs_query_tokens)),
                    "reward": tuple(tokenizer.convert_tokens_to_ids(self.reward_query_tokens)),
                }
            )
        if self.action_prompt_mode == "minimal_agent":
            self._special_token_ids.update(
                {
                    "mine_agent": tuple(tokenizer.convert_tokens_to_ids(self.mine_agent_query_tokens)),
                }
            )

        # if self.config.datasets.vla_data.load_act_data:
        self.action_input_model = MLP(
            input_dim=config.framework.action_model.action_dim,
            hidden_dim=self.qwen_vl_interface.model.config.hidden_size,
            output_dim=self.qwen_vl_interface.model.config.hidden_size,
        )

        self.infer_not_load_wan = infer_not_load_wan

        self.agent_dino_loss_weight = float(OmegaConf.select(self.config, "framework.agent_dino.loss_weight", default=0.1))
        self.agent_dino_dim = int(OmegaConf.select(self.config, "framework.agent_dino.feature_dim", default=384))
        self.agent_dino_head = nn.Sequential(
            nn.LayerNorm(self.qwen_vl_interface.model.config.hidden_size),
            nn.Linear(self.qwen_vl_interface.model.config.hidden_size, self.qwen_vl_interface.model.config.hidden_size),
            nn.GELU(),
            nn.Linear(self.qwen_vl_interface.model.config.hidden_size, self.agent_dino_dim),
        )

        llm_layers, llm_hidden_size = self.config.framework.action_model.diffusion_model_cfg.num_layers, self.qwen_vl_interface.model.config.hidden_size

        DiTConfig = {
            "num_layers": llm_layers, 
            "input_embedding_dim": self.config.framework.action_model.hidden_size, 
            "attention_head_dim": 64, 
            "num_attention_heads": self.config.framework.action_model.hidden_size // 64}
        self.config.framework.action_model.DiTConfig = DiTConfig
        if self.config.datasets.vla_data.load_act_data:
            if self.config.framework.action_model.mlp_head == 0:
                self.action_model: FlowmatchingActionHead = get_action_model(config=self.config)
                self.mlp_head = 0
            else:
                self.action_model = nn.Sequential(
                    nn.Linear(self.qwen_vl_interface.model.config.hidden_size*8, self.qwen_vl_interface.model.config.hidden_size*4),
                    nn.LayerNorm(self.qwen_vl_interface.model.config.hidden_size*4),
                    nn.ReLU(inplace=True),
                    nn.Linear(self.qwen_vl_interface.model.config.hidden_size*4, self.qwen_vl_interface.model.config.hidden_size*2),
                    nn.LayerNorm(self.qwen_vl_interface.model.config.hidden_size*2),
                    nn.ReLU(inplace=True),
                    nn.Linear(self.qwen_vl_interface.model.config.hidden_size*2, self.qwen_vl_interface.model.config.hidden_size),
                    nn.LayerNorm(self.qwen_vl_interface.model.config.hidden_size),
                    nn.ReLU(inplace=True),
                    nn.Linear(self.qwen_vl_interface.model.config.hidden_size, 1024),
                    nn.LayerNorm(1024),
                    nn.ReLU(inplace=True),
                    nn.Linear(1024, config.framework.action_model.action_dim*8),
                )
                self.mlp_head = 1

        ## 2d gen
        if self.config.datasets.video_data.load_2d_data:
            if not infer_not_load_wan:
                self.rgb_model = WanWorldHead(self.config, accelerator)

        if self.config.datasets.video_data.load_2d_data:
            self.rgb_query = nn.Parameter(torch.randn(64, self.qwen_vl_interface.model.config.hidden_size) * 0.02)

            self.rgb_query_loss = OmegaConf.select(self.config, "rgb_query_loss", default=0)
            if self.rgb_query_loss:
                hidden_dim = self.qwen_vl_interface.model.config.hidden_size
                self.traj_emb_h0 = nn.Parameter(torch.zeros(1, hidden_dim))
                self.traj_emb = nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=1, bias=True,
                                batch_first=True, dropout=0.0, bidirectional=False)
                self.rgb_act_pre = nn.Sequential(
                        nn.Linear(self.qwen_vl_interface.model.config.hidden_size, self.qwen_vl_interface.model.config.hidden_size//4),
                        nn.LayerNorm(self.qwen_vl_interface.model.config.hidden_size//4),
                        nn.ReLU(inplace=True),
                        nn.Linear(self.qwen_vl_interface.model.config.hidden_size//4, config.framework.action_model.action_dim*8),
                )

        if self.config.datasets.gs_data.load_3d_data:
            self.gs_model = StormWorldHead(self.config, accelerator)

        if self.config.datasets.reward_data.load_reward_data:
            llm_layers, llm_hidden_size = self.config.framework.reward_model.diffusion_model_cfg.num_layers, self.qwen_vl_interface.model.config.hidden_size

            DiTConfig = {
                "num_layers": llm_layers,
                "input_embedding_dim": self.config.framework.reward_model.hidden_size,
                "attention_head_dim": 64,
                "num_attention_heads": self.config.framework.reward_model.hidden_size // 64}
            self.config.framework.reward_model.DiTConfig = DiTConfig
            self.reward_model: FlowmatchingRewardHead = get_reward_model(config=self.config)

        self.w_depth = OmegaConf.select(self.config, "w_depth", default=0)

        if self.w_depth:
            depth_ppd_path = 'starVLA/model/modules/depth_model/configs/train_finetune.yaml'
            self.depth_ppd_cfg = OmegaConf.load(depth_ppd_path)
            self.gs_model = PixelPerfectDepth(self.depth_ppd_cfg.model.pipeline.config)
            missing, unexpected = self.gs_model.load_state_dict(torch.load(self.depth_ppd_cfg.model.pipeline.config.ckpt_path, map_location='cpu'), strict=False)
            print(f'[PPD] missing keys: {len(missing)} {missing[:8]}')
            print(f'[PPD] unexpected keys: {len(unexpected)} {unexpected[:8]}')
        
        if self.config.datasets.gs_data.load_3d_data or self.w_depth:
            self.gs_query = nn.Parameter(torch.randn(64, self.qwen_vl_interface.model.config.hidden_size) * 0.02)

            self.gs_query_loss = OmegaConf.select(self.config, "gs_query_loss", default=0)
            if self.gs_query_loss:
                hidden_dim = self.qwen_vl_interface.model.config.hidden_size
                self.gs_traj_emb_h0 = nn.Parameter(torch.zeros(1, hidden_dim))
                self.gs_traj_emb = nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=1, bias=True,
                                batch_first=True, dropout=0.0, bidirectional=False)
                self.gs_act_pre = nn.Sequential(
                        nn.Linear(self.qwen_vl_interface.model.config.hidden_size, self.qwen_vl_interface.model.config.hidden_size//4),
                        nn.LayerNorm(self.qwen_vl_interface.model.config.hidden_size//4),
                        nn.ReLU(inplace=True),
                        nn.Linear(self.qwen_vl_interface.model.config.hidden_size//4, config.framework.action_model.action_dim*8),
                )

        
        self.w_video_latent = OmegaConf.select(self.config, "w_video_latent", default=0)
        
        if self.w_video_latent:
            self.rgb_latent_adapter = RGBLatentAdapter(
                in_c=16,
                hidden=self.config.framework.action_model.hidden_size,
                grid=(1,1,2),
                n_view=3,
            )
            self.rgb_latent_type = nn.Parameter(torch.randn(1, 1, self.config.framework.action_model.hidden_size) * 0.02)


    @staticmethod
    def _find_token_positions(input_ids, token_ids):
        """Find ordered special-token positions without Python/CUDA scalar syncs."""
        ids = torch.as_tensor(token_ids, device=input_ids.device, dtype=input_ids.dtype)
        matches = input_ids.unsqueeze(-1).eq(ids.view(1, 1, -1))
        return matches.to(torch.int8).argmax(dim=1)

    def _build_action_prompt_suffix(self) -> str:
        """Build the instruction suffix used by action-only prompting."""
        hist_str = self.robot_history_token
        act_str = "".join(self.act_query_tokens)
        if self.action_prompt_mode == "minimal":
            return f" {hist_str}{act_str}"
        if self.action_prompt_mode == "minimal_agent":
            mine_agent_str = "".join(self.mine_agent_query_tokens)
            return f" {hist_str}{mine_agent_str}{act_str}"

        rgb_str = "".join(self.rgb_query_tokens)
        gs_str = "".join(self.gs_query_tokens)
        rew_str = "".join(self.reward_query_tokens)
        if self.w_depth:
            return f" {hist_str}{gs_str}{rgb_str}{act_str}{rew_str}"
        return f" {hist_str}{rgb_str}{gs_str}{act_str}{rew_str}"

    def _build_qwen_batch(self, examples, instructions):
        """Build either cached or ordinary Qwen inputs for one training batch."""
        cached = [example.get("qwen_feature_cache") for example in examples]
        if any(payload is not None for payload in cached) and not all(payload is not None for payload in cached):
            raise RuntimeError("A batch cannot mix cached and uncached Qwen samples")

        device = self.qwen_vl_interface.model.device
        if all(payload is not None for payload in cached):
            lengths = [int(payload["input_ids"].numel()) for payload in cached]
            max_length = max(lengths)
            batch_size = len(cached)
            tokenizer = self.qwen_vl_interface.processor.tokenizer
            input_ids = torch.full(
                (batch_size, max_length),
                int(tokenizer.pad_token_id),
                dtype=torch.long,
                device=device,
            )
            attention_mask = torch.zeros(
                (batch_size, max_length), dtype=torch.long, device=device
            )
            position_ids = torch.ones(
                (3, batch_size, max_length), dtype=torch.long, device=device
            )
            position_names = tuple(self._special_token_ids.keys())
            positions = {name: [] for name in position_names}
            for batch_index, (payload, length) in enumerate(zip(cached, lengths)):
                offset = max_length - length
                input_ids[batch_index, offset:] = payload["input_ids"].to(device=device, dtype=torch.long)
                attention_mask[batch_index, offset:] = payload["attention_mask"].to(
                    device=device, dtype=torch.long
                )
                position_ids[:, batch_index, offset:] = payload["position_ids"].to(
                    device=device, dtype=torch.long
                )
                for name in position_names:
                    positions[name].append(
                        payload[f"{name}_positions"].to(device=device, dtype=torch.long) + offset
                    )

            deepstack_keys = sorted(
                key for key in cached[0] if key.startswith("deepstack_")
            )
            image_embeds = torch.cat([payload["image_embeds"] for payload in cached], dim=0)
            deepstack_embeds = [
                torch.cat([payload[key] for payload in cached], dim=0)
                for key in deepstack_keys
            ]
            return (
                input_ids,
                attention_mask,
                position_ids,
                {name: torch.stack(values) for name, values in positions.items()},
                image_embeds.to(device, non_blocking=True),
                [value.to(device, non_blocking=True) for value in deepstack_embeds],
            )

        batch_images = [example["image"] for example in examples]
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=batch_images,
            instructions=instructions,
        )
        input_ids = qwen_inputs["input_ids"]
        attention_mask = qwen_inputs["attention_mask"]
        with torch.no_grad():
            position_ids, _ = self.qwen_vl_interface.model.model.get_rope_index(
                input_ids=input_ids,
                image_grid_thw=qwen_inputs["image_grid_thw"],
                video_grid_thw=qwen_inputs.get("video_grid_thw", None),
                attention_mask=attention_mask,
            )
            image_parts, deepstack_embeds = self.qwen_vl_interface.model.model.get_image_features(
                qwen_inputs["pixel_values"],
                qwen_inputs["image_grid_thw"],
            )
            image_embeds = torch.cat(image_parts, dim=0)
        positions = {
            name: self._find_token_positions(input_ids, token_ids)
            for name, token_ids in self._special_token_ids.items()
        }
        return (
            input_ids,
            attention_mask,
            position_ids,
            positions,
            image_embeds,
            deepstack_embeds,
        )

    def _qwen_language_forward(
        self,
        input_ids,
        inputs_embeds,
        attention_mask,
        position_ids,
        image_embeds,
        deepstack_embeds,
    ):
        """Run only the trainable Qwen backbone, skipping the unused LM head."""
        image_mask = input_ids.eq(self.qwen_vl_interface.model.config.image_token_id)
        expanded_mask = image_mask.unsqueeze(-1).expand_as(inputs_embeds)
        image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        # Cache generation validates the placeholder/token contract.  Avoid a
        # boolean gather of every hidden element here; it allocated a temporary
        # tensor of the same size as all visual embeddings on every step.
        inputs_embeds = inputs_embeds.masked_scatter(expanded_mask, image_embeds)
        outputs = self.qwen_vl_interface.model.model.language_model(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            visual_pos_masks=image_mask,
            deepstack_visual_embeds=[
                value.to(inputs_embeds.device, inputs_embeds.dtype)
                for value in deepstack_embeds
            ],
            use_cache=False,
        )
        return outputs.last_hidden_state

    def forward(
        self,
        examples: List[dict] = None,
        accelerator = None,
        **kwargs,
    ) -> Tuple:

        instructions = [example["lang"] for example in examples]  # [B, str]
        try:
            actions = [example["action"] for example in examples]  # label [B， len, 7]
        except:
            actions = None
        try:
            states = [example["state"] for example in examples]
        except:
            states = None

        if self.w_depth:
            depth_data = [example['depth_data'] for example in examples]

        
        suffix = self._build_action_prompt_suffix()
        instructions = [instruction + suffix for instruction in instructions]
        (
            input_ids,
            attention_mask,
            position_ids,
            token_positions,
            image_embeds,
            deepstack_embeds,
        ) = self._build_qwen_batch(examples, instructions)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            text_embeds = self.qwen_vl_interface.model.get_input_embeddings()(input_ids)  # [B, L, H]


        # if self.config.datasets.vla_data.load_act_data: 
        state_device = next(self.action_input_model.parameters()).device
        with torch.autocast("cuda", dtype=torch.float32):
            # 映射到 hidden 维: [B, H]
            states = torch.as_tensor(np.asarray(states), dtype=torch.float32).to(state_device)[:, 0, :]
            states_embed = self.action_input_model(states)  # [B, H]
        states_embed = states_embed.to(dtype=text_embeds.dtype, device=text_embeds.device)

        # if self.w_depth:
        #     with torch.autocast("cuda", dtype=torch.float32):
        #         # 映射到 hidden 维: [B, H]
        #         depth_feats = torch.stack(depth_feats)
        #         bz, n_cam, n_channel, n_h, n_w = depth_feats.shape
        #         depth_feats = depth_feats.reshape(bz*n_cam, n_channel, n_h, n_w)
        #         depth_token = self.depth_adapter(depth_feats)
        #         n_token = depth_token.shape[1]
        #         depth_token = depth_token.reshape(bz, n_cam, n_token, -1)   # 3*64
        #         depth_token = depth_token[:, [1,2,0]]   # l,r,f
        #         depth_token = depth_token.reshape(bz, n_cam*n_token, -1)
        #     depth_token = depth_token.to(dtype=text_embeds.dtype)
        #     depth_token = depth_token + self.depth_type.to(depth_token.dtype)

        B, L, H = text_embeds.shape
        batch_indices = torch.arange(B, device=text_embeds.device)
        text_embeds[batch_indices, token_positions["history"][:, 0], :] = states_embed
        if self.config.datasets.video_data.load_2d_data:
            text_embeds[
                batch_indices[:, None], token_positions["rgb"], :
            ] = self.rgb_query.unsqueeze(0).to(text_embeds.dtype)
        if self.config.datasets.gs_data.load_3d_data or self.w_depth:
            text_embeds[
                batch_indices[:, None], token_positions["gs"], :
            ] = self.gs_query.unsqueeze(0).to(text_embeds.dtype)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            last_hidden = self._qwen_language_forward(
                input_ids=input_ids,
                inputs_embeds=text_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                image_embeds=image_embeds,
                deepstack_embeds=deepstack_embeds,
            )

        agent_dino_loss = torch.tensor(0.).to(text_embeds.device)
        if self.action_prompt_mode == "minimal_agent":
            mine_agent_g_idx = token_positions["mine_agent"].unsqueeze(-1).expand(-1, -1, H)
            mine_agent_queries = last_hidden.gather(dim=1, index=mine_agent_g_idx)  # [B, 4, H]
            agent_dino_payloads = [example.get("agent_dino_feature_cache") for example in examples]
            if all(payload is not None for payload in agent_dino_payloads):
                teacher_features = []
                teacher_masks = []
                for payload in agent_dino_payloads:
                    feats = payload["agent_features"].to(device=mine_agent_queries.device, dtype=torch.float32)
                    valid_count = int(feats.shape[0])
                    pad_count = max(0, 4 - valid_count)
                    if pad_count:
                        pad = torch.zeros((pad_count, feats.shape[1]), device=feats.device, dtype=feats.dtype)
                        feats = torch.cat([feats, pad], dim=0)
                    teacher_features.append(feats[:4])
                    mask = torch.zeros((4,), device=mine_agent_queries.device, dtype=torch.bool)
                    mask[:min(valid_count, 4)] = True
                    teacher_masks.append(mask)
                teacher_features = torch.stack(teacher_features, dim=0)  # [B,4,D]
                teacher_masks = torch.stack(teacher_masks, dim=0)        # [B,4]
                if next(self.agent_dino_head.parameters()).device != mine_agent_queries.device:
                    self.agent_dino_head = self.agent_dino_head.to(mine_agent_queries.device)
                pred_features = self.agent_dino_head(mine_agent_queries.float())  # [B,4,D]
                if teacher_masks.any():
                    agent_dino_loss = F.smooth_l1_loss(
                        pred_features[teacher_masks],
                        teacher_features[teacher_masks],
                    )
            agent_dino_loss = agent_dino_loss * self.agent_dino_loss_weight

        #### video gen ####
        if self.config.datasets.video_data.load_2d_data:

            rgb_data = [example['2d_gen_data'] for example in examples]

            g_idx = token_positions["rgb"].unsqueeze(-1).expand(-1, -1, H)
            rgb_queries = last_hidden.gather(dim=1, index=g_idx)                     # [B, T, H]

            if self.rgb_query_loss:
                m = self.traj_emb(
                    rgb_queries, 
                    self.traj_emb_h0.unsqueeze(0).repeat(1, rgb_queries.shape[0], 1)
                )[1].squeeze(1)

                # m for action prediction
                actions = torch.tensor(
                        np.array(actions), device=rgb_queries.device, dtype=torch.float32
                    )
                rgb_query_loss = nn.L1Loss()(self.rgb_act_pre(m).reshape(B, 8, 4), actions)
            

            with torch.autocast("cuda", dtype=torch.bfloat16):
                rgb_loss, video_latent = self.rgb_model(rgb_data, rgb_queries)
            if self.rgb_query_loss:
                rgb_loss += rgb_query_loss
        else:
            rgb_loss = torch.tensor(0.).cuda()


        # Step 4: Action Expert Forward and Loss
        if self.config.datasets.vla_data.load_act_data == 1:
            g_idx = token_positions["action"].unsqueeze(-1).expand(-1, -1, H)
            action_queries = last_hidden.gather(dim=1, index=g_idx)                     # [B, T, H]

            with torch.autocast("cuda", dtype=torch.float32):
                if type(actions) == list:
                    actions = torch.tensor(
                        np.array(actions), device=action_queries.device, dtype=torch.float32
                    )  # [B, T_full, action_dim]
                ####### repeat  ###
                repeated_diffusion_steps = (
                    self.config.framework.action_model.get("repeated_diffusion_steps", 1) if self.config else 1
                )
                repeat_actions = actions.repeat(repeated_diffusion_steps, 1, 1)
                # 对每层特征做 repeat
                repeat_action_queries = action_queries.repeat(repeated_diffusion_steps, 1, 1)

                if self.w_video_latent:
                    video_token = self.rgb_latent_adapter(video_latent)
                    video_token = video_token + self.rgb_latent_type.to(video_token.dtype)
                else:
                    video_token = None

                if self.mlp_head == 0:
                    action_loss = self.action_model(repeat_action_queries, repeat_actions, video_token)  # (B, chunk_len, action_dim)
                else:
                    b, l, h = action_queries.shape
                    pred_action = self.action_model(action_queries.reshape(b, l*h)).reshape(b, l, -1)
                    action_loss = nn.SmoothL1Loss()(pred_action, actions)
        else:
            action_loss = torch.tensor(0.).cuda()


        if self.config.datasets.gs_data.load_3d_data or self.w_depth:

            if self.config.datasets.gs_data.load_3d_data:
                gs_data = [example['3d_gs_data'] for example in examples]
            else:
                data_img = torch.stack([d['image'] for d in depth_data])
                data_depth = torch.stack([d['depth'] for d in depth_data])
                data_mask = torch.stack([d['mask'] for d in depth_data])
                cached_semantics = None
                if all('semantics' in value for value in depth_data):
                    cached_semantics = torch.stack([value['semantics'] for value in depth_data])

                # 合并 batch 和 view 维度
                B, V = data_img.shape[:2]
                data_img = data_img.reshape(B * V, *data_img.shape[2:])
                data_depth = data_depth.reshape(B * V, *data_depth.shape[2:])
                data_mask = data_mask.reshape(B * V, *data_mask.shape[2:])

                depth_data = {
                    'image': data_img,
                    'depth': data_depth,
                    'mask': data_mask,
                }
                if cached_semantics is not None:
                    depth_data['semantics'] = cached_semantics.reshape(
                        B * V, *cached_semantics.shape[2:]
                    )

            g_idx = token_positions["gs"].unsqueeze(-1).expand(-1, -1, H)
            gs_queries = last_hidden.gather(dim=1, index=g_idx)                     # [B, T, H]

            if self.config.datasets.gs_data.load_3d_data:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    gs_loss = self.gs_model(gs_data, gs_queries)
            else:
                depth_data['qwen_token'] = gs_queries.repeat_interleave(3, dim=0)

                if self.gs_query_loss:
                    m = self.gs_traj_emb(gs_queries, self.gs_traj_emb_h0.unsqueeze(0).repeat(1, gs_queries.shape[0], 1))[1].squeeze(0)

                    # m for action prediction
                    if type(actions) == list:
                        actions = torch.tensor(
                                np.array(actions), device=gs_queries.device, dtype=torch.float32
                            )
                    gs_query_loss = nn.L1Loss()(self.gs_act_pre(m).reshape(B, 8, 4), actions)

                    # use m for easy aggr: use cross attn
                    # depth_data['qwen_token'] = m

                with torch.autocast("cuda", dtype=torch.bfloat16):
                    output = self.gs_model.forward_train(depth_data)
                    # a bit huge
                    gs_loss = output['loss']
                    if self.gs_query_loss:
                        gs_loss += gs_query_loss
            
            # return {"action_loss": action_loss, "rgb_loss": rgb_loss, "gs_loss": gs_loss}
        else:
            gs_loss = torch.tensor(0.).cuda()

        if self.config.datasets.reward_data.load_reward_data:

            reward_data = np.array([example['reward_data'] for example in examples])  # list of reward (B)

            g_idx = token_positions["reward"].unsqueeze(-1).expand(-1, -1, H)
            reward_queries = last_hidden.gather(dim=1, index=g_idx)                     # [B, T, H]

            # debug here
            # reward = self.reward_model.predict_action(reward_queries)

            with torch.autocast("cuda", dtype=torch.float32):
                reward_loss = self.reward_model(reward_queries, reward_data)
            
            return {"action_loss": action_loss, "rgb_loss": rgb_loss, "gs_loss": gs_loss, "reward_loss": reward_loss, "agent_dino_loss": agent_dino_loss}
        else:
            reward_loss = torch.tensor(0.).cuda()

        return {"action_loss": action_loss, "rgb_loss": rgb_loss, "gs_loss": gs_loss*0.1, "reward_loss": reward_loss, "agent_dino_loss": agent_dino_loss}

    @torch.inference_mode()
    def predict_action(
        self,
        examples,
        **kwargs: str,
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
        # train_obs_image_size = getattr(self.config.datasets.vla_data, "image_size", None)
        # if train_obs_image_size:
        #     batch_images = resize_images(batch_images, target_size=train_obs_image_size)
    
        # step 0: add special action token to instruction
        # action_tokens = self.action_token* self.chunk_len #can't add " " between two tokens, otherwise will be tokenized to multiple tokens
        # prompt_suffix = f" Please predict the next {self.chunk_len} robot actions: <action>{action_tokens}<action>."
        # instructions = [instruction + prompt_suffix for instruction in instructions]

        batch_images = [example["image"] for example in examples]  #  [B，[PLT]]
        instructions = [example["lang"] for example in examples]  # [B, str]
        # actions = [example["action"] for example in examples]  # label [B， len, 7]
        try:
            states = [example["state"] for example in examples]
        except:
            state = None

        suffix = self._build_action_prompt_suffix()
        instructions = [instruction + suffix for instruction in instructions]

        # Step 1: QWenVL input format
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)

        # —— 覆盖 <robot_history_action_0> 的 embedding ——
        tok   = self.qwen_vl_interface.processor.tokenizer
        if self.config.datasets.vla_data.load_act_data:
            hist_id = tok.convert_tokens_to_ids(self.robot_history_token)  # "<robot_history_action_0>"

        if self.config.datasets.video_data.load_2d_data:
            rgb_ids = tok.convert_tokens_to_ids(self.rgb_query_tokens)
        
        if self.config.datasets.gs_data.load_3d_data:
            gs_ids = tok.convert_tokens_to_ids(self.gs_query_tokens)
        
        if self.config.datasets.reward_data.load_reward_data:
            # one token
            reward_ids = tok.convert_tokens_to_ids(self.reward_query_tokens)
        input_ids      = qwen_inputs["input_ids"]          # [B, L]
        attention_mask = qwen_inputs["attention_mask"]     # [B, L]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            text_embeds = self.qwen_vl_interface.model.get_input_embeddings()(input_ids)  # [B, L, H]


        # if self.config.datasets.vla_data.load_act_data:
        with torch.autocast("cuda", dtype=torch.float32):
            # 映射到 hidden 维: [B, H]
            states = torch.from_numpy(np.array(states)).cuda()[:, 0, :]
            states_embed = self.action_input_model(states)  # [B, H]
        states_embed = states_embed.to(dtype=text_embeds.dtype)

        # 逐样本把 hist_id 的那个位置替换成对应的 states_embed[b]
        B, L, H = text_embeds.shape
        if self.config.datasets.vla_data.load_act_data:
            for b in range(B):
                where = (input_ids[b] == hist_id).nonzero(as_tuple=False)
                if where.numel() == 0:
                    raise RuntimeError(f"Sample {b}: robot_history token not found in input_ids.")
                if where.numel() > 1:
                    # 如果你只想覆盖第一个出现的位置，就取 where[0]
                    # 这里严格要求只有一个
                    raise RuntimeError(f"Sample {b}: found multiple robot_history tokens: {where.squeeze(-1).tolist()}")
                pos = int(where[0])
                text_embeds[b, pos, :] = states_embed[b]

                # replace rgb token
                if self.config.datasets.video_data.load_2d_data:
                    # where = (input_ids[b] == rgb_ids).nonzero(as_tuple=False)
                    rgb_ids_tensor = torch.tensor(rgb_ids, device=input_ids.device)
                    where = torch.isin(input_ids[b], rgb_ids_tensor).nonzero(as_tuple=False).squeeze(1)
                    _, order = torch.sort(where)
                    rgb_query_reordered = self.rgb_query[order]    # [64, H]

                    text_embeds[b, where, :] = rgb_query_reordered

            # # replace 3d gs token
            # if self.config.datasets.gs_data.load_3d_data:
            #     # where = (input_ids[b] == rgb_ids).nonzero(as_tuple=False)
            #     gs_ids_tensor = torch.tensor(gs_ids, device=input_ids.device)
            #     where = torch.isin(input_ids[b], gs_ids_tensor).nonzero(as_tuple=False).squeeze(1)
            #     _, order = torch.sort(where)
            #     gs_query_reordered = self.gs_query[order]    # [64, H]

            #     text_embeds[b, where, :] = gs_query_reordered

            # # replace reward token
            # if self.config.datasets.reward_data.load_reward_data:
            #     # where = (input_ids[b] == rgb_ids).nonzero(as_tuple=False)
            #     reward_ids_tensor = torch.tensor(reward_ids, device=input_ids.device)
            #     where = torch.isin(input_ids[b], reward_ids_tensor).nonzero(as_tuple=False).squeeze(1)
            #     _, order = torch.sort(where)
            #     reward_query_reordered = self.reward_query[order]    # [64, H]

            #     text_embeds[b, where, :] = reward_query_reordered

        # 前向：用 inputs_embeds（不要再传 input_ids）
        # position_ids = (attention_mask.long().cumsum(-1) - 1).clamp(min=0)
        with torch.no_grad():
            # 注意：这里用的是底层 Qwen3VLModel 的 get_rope_index
            position_ids, _ = self.qwen_vl_interface.model.model.get_rope_index(
                input_ids=qwen_inputs["input_ids"],
                image_grid_thw=qwen_inputs["image_grid_thw"],
                video_grid_thw=qwen_inputs.get("video_grid_thw", None),
                attention_mask=attention_mask,   # 2D mask 就行
            )

        qwen_forward_mode = getattr(
            self, "_inference_qwen_forward_mode", "legacy"
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if qwen_forward_mode == "optimized":
                image_parts, deepstack_embeds = (
                    self.qwen_vl_interface.model.model.get_image_features(
                        qwen_inputs["pixel_values"],
                        qwen_inputs["image_grid_thw"],
                    )
                )
                image_embeds = torch.cat(image_parts, dim=0)
                last_hidden = self._qwen_language_forward(
                    input_ids=input_ids,
                    inputs_embeds=text_embeds,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    image_embeds=image_embeds,
                    deepstack_embeds=deepstack_embeds,
                )
            else:
                qw_out = self.qwen_vl_interface(
                    inputs_embeds=text_embeds,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    # 视觉侧保持不变
                    pixel_values=qwen_inputs.get("pixel_values", None),
                    image_grid_thw=qwen_inputs.get("image_grid_thw", None),
                    output_hidden_states=True,
                    return_dict=True,
                )
                last_hidden = qw_out.hidden_states[-1]   # [B, L, H]

        # Step 1: QWenVL input format
        # qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)
        # with torch.autocast("cuda", dtype=torch.bfloat16):
        #     qwenvl_outputs = self.qwen_vl_interface(
        #         **qwen_inputs,
        #         output_attentions=False,
        #         output_hidden_states=True,
        #         return_dict=True,
        #     )
        #     # last_hidden_state: [B, seq_len, H]
        #     last_hidden = qwenvl_outputs.hidden_states[-1]   # [B, L, H]

        # Step 4: Action Expert Forward and Loss
        # …接下来的流程保持你原来的：从动作 query 位置 gather hidden，过 action head，算 L1 loss …
        # 例如（如果你仍然用多个 <robot_action_*>）：
        
        # if self.config.datasets.vla_data.load_act_data == 1:
        if self.config.datasets.vla_data.load_act_data == 1:
            act_ids = [tok.convert_tokens_to_ids(t) for t in self.act_query_tokens]  # 长度 T
            act_pos_idx = []
            for b in range(B):
                pos_list = []
                for tid in act_ids:
                    w = (input_ids[b] == tid).nonzero(as_tuple=False)
                    if w.numel() == 0:
                        raise RuntimeError(f"Sample {b}: action token {tid} not found.")
                    pos_list.append(int(w[0]))
                act_pos_idx.append(torch.tensor(pos_list, device=last_hidden.device))
            act_pos_idx = torch.stack(act_pos_idx, dim=0)                            # [B, T]
            g_idx = act_pos_idx.unsqueeze(-1).expand(-1, -1, H)                      # [B, T, H]
            action_queries = last_hidden.gather(dim=1, index=g_idx)                     # [B, T, H]

            with torch.autocast("cuda", dtype=torch.float32):
                # 提取动作 token embedding 作为动作预测查询
                # input_ids = qwen_inputs.get("input_ids", None)
                # action_queries = self._gather_action_token_embeddings(last_hidden, input_ids, action_token_id=self.action_token_id)  # [B, chunk_len, H]
                if self.mlp_head == 0:
                    pred_actions = self.action_model.predict_action(action_queries)  # (B, chunk_len, action_dim)
                else:
                    pred_actions = self.action_model(action_queries)

            normalized_actions = pred_actions.detach().cpu().numpy()
        else:
            normalized_actions = None

        if self.config.datasets.video_data.load_2d_data and not self.infer_not_load_wan:
            rgb_data = [example['2d_gen_data'] for example in examples]

            rgb_ids = [tok.convert_tokens_to_ids(t) for t in self.rgb_query_tokens]  # 长度 T
            rgb_pos_idx = []
            for b in range(B):
                pos_list = []
                for tid in rgb_ids:
                    w = (input_ids[b] == tid).nonzero(as_tuple=False)
                    if w.numel() == 0:
                        raise RuntimeError(f"Sample {b}: action token {tid} not found.")
                    pos_list.append(int(w[0]))
                rgb_pos_idx.append(torch.tensor(pos_list, device=last_hidden.device))
            rgb_pos_idx = torch.stack(rgb_pos_idx, dim=0)                            # [B, T]
            g_idx = rgb_pos_idx.unsqueeze(-1).expand(-1, -1, H)                      # [B, T, H]
            rgb_queries = last_hidden.gather(dim=1, index=g_idx)                     # [B, T, H]

            with torch.autocast("cuda", dtype=torch.bfloat16):
                rgbs = self.rgb_model.predict_rgb(rgb_data, rgb_queries)
            
            return {"normalized_actions": normalized_actions, "rgbs": rgbs}
        
        if self.config.datasets.gs_data.load_3d_data:

            gs_data = [example['3d_gs_data'] for example in examples]

            gs_ids = [tok.convert_tokens_to_ids(t) for t in self.gs_query_tokens]  # 长度 T
            gs_pos_idx = []
            for b in range(B):
                pos_list = []
                for tid in gs_ids:
                    w = (input_ids[b] == tid).nonzero(as_tuple=False)
                    if w.numel() == 0:
                        raise RuntimeError(f"Sample {b}: action token {tid} not found.")
                    pos_list.append(int(w[0]))
                gs_pos_idx.append(torch.tensor(pos_list, device=last_hidden.device))
            gs_pos_idx = torch.stack(gs_pos_idx, dim=0)                            # [B, T]
            g_idx = gs_pos_idx.unsqueeze(-1).expand(-1, -1, H)                      # [B, T, H]
            gs_queries = last_hidden.gather(dim=1, index=g_idx)                     # [B, T, H]

            with torch.autocast("cuda", dtype=torch.bfloat16):
                gs = self.gs_model.predict_gs(gs_data, gs_queries)

            return {"normalized_actions": normalized_actions, "gs": gs}
        
        if self.config.datasets.reward_data.load_reward_data:

            # reward_data = np.array([example['reward_data'] for example in examples])  # list of reward (B)

            reward_ids = [tok.convert_tokens_to_ids(t) for t in self.reward_query_tokens]  # 长度 T
            reward_pos_idx = []
            for b in range(B):
                pos_list = []
                for tid in reward_ids:
                    w = (input_ids[b] == tid).nonzero(as_tuple=False)
                    if w.numel() == 0:
                        raise RuntimeError(f"Sample {b}: action token {tid} not found.")
                    pos_list.append(int(w[0]))
                reward_pos_idx.append(torch.tensor(pos_list, device=last_hidden.device))
            reward_pos_idx = torch.stack(reward_pos_idx, dim=0)                            # [B, T]
            g_idx = reward_pos_idx.unsqueeze(-1).expand(-1, -1, H)                      # [B, T, H]
            reward_queries = last_hidden.gather(dim=1, index=g_idx)                     # [B, T, H]

            with torch.autocast("cuda", dtype=torch.float32):
                reward = self.reward_model.predict_action(reward_queries)
            
            return {"normalized_actions": normalized_actions, "reward": reward}

        return {"normalized_actions": normalized_actions}

    @torch.inference_mode()
    def predict_action_infer_1d(
        self,
        examples,
        **kwargs: str,
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
        # train_obs_image_size = getattr(self.config.datasets.vla_data, "image_size", None)
        # if train_obs_image_size:
        #     batch_images = resize_images(batch_images, target_size=train_obs_image_size)
    
        # step 0: add special action token to instruction
        # action_tokens = self.action_token* self.chunk_len #can't add " " between two tokens, otherwise will be tokenized to multiple tokens
        # prompt_suffix = f" Please predict the next {self.chunk_len} robot actions: <action>{action_tokens}<action>."
        # instructions = [instruction + prompt_suffix for instruction in instructions]

        batch_images = [example["image"] for example in examples]  #  [B，[PLT]]
        instructions = [example["lang"] for example in examples]  # [B, str]
        # actions = [example["action"] for example in examples]  # label [B， len, 7]
        
        states = [example["state"] for example in examples]

        if self.w_depth:
            pass
            # depth_feats = [example['depth_feat'] for example in examples]

        suffix = self._build_action_prompt_suffix()
        instructions = [instruction + suffix for instruction in instructions]

        # Step 1: QWenVL input format
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)

        # —— 覆盖 <robot_history_action_0> 的 embedding ——
        tok   = self.qwen_vl_interface.processor.tokenizer
        hist_id = tok.convert_tokens_to_ids(self.robot_history_token)  # "<robot_history_action_0>"

        if self.config.datasets.video_data.load_2d_data:
            rgb_ids = tok.convert_tokens_to_ids(self.rgb_query_tokens)
        
        if self.config.datasets.gs_data.load_3d_data or self.w_depth:
            gs_ids = tok.convert_tokens_to_ids(self.gs_query_tokens)

        # if self.w_depth:
        #     depth_ids = tok.convert_tokens_to_ids(self.robot_history_token)[:-1]
        #     hist_id = hist_id[-1]
        
        if self.config.datasets.reward_data.load_reward_data:
            # one token
            reward_ids = tok.convert_tokens_to_ids(self.reward_query_tokens)
        input_ids      = qwen_inputs["input_ids"]          # [B, L]
        attention_mask = qwen_inputs["attention_mask"]     # [B, L]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            text_embeds = self.qwen_vl_interface.model.get_input_embeddings()(input_ids)  # [B, L, H]


        with torch.autocast("cuda", dtype=torch.float32):
            # 映射到 hidden 维: [B, H]
            states = torch.from_numpy(np.array(states)).cuda()[:, 0, :]
            states_embed = self.action_input_model(states)  # [B, H]
        states_embed = states_embed.to(dtype=text_embeds.dtype)

        # if self.w_depth:
        #     with torch.autocast("cuda", dtype=torch.float32):
        #         # 映射到 hidden 维: [B, H]
        #         depth_feats = torch.stack(depth_feats)
        #         bz, n_cam, n_channel, n_h, n_w = depth_feats.shape
        #         depth_feats = depth_feats.reshape(bz*n_cam, n_channel, n_h, n_w)
        #         depth_token = self.depth_adapter(depth_feats)
        #         n_token = depth_token.shape[1]
        #         depth_token = depth_token.reshape(bz, n_cam, n_token, -1)   # 3*64
        #         depth_token = depth_token[:, [1,2,0]]   # l,r,f
        #         depth_token = depth_token.reshape(bz, n_cam*n_token, -1)
        #     depth_token = depth_token.to(dtype=text_embeds.dtype)
        #     depth_token = depth_token + self.depth_type.to(depth_token.dtype)

        # 逐样本把 hist_id 的那个位置替换成对应的 states_embed[b]
        B, L, H = text_embeds.shape
        for b in range(B):
            where = (input_ids[b] == hist_id).nonzero(as_tuple=False)
            if where.numel() == 0:
                raise RuntimeError(f"Sample {b}: robot_history token not found in input_ids.")
            if where.numel() > 1:
                # 如果你只想覆盖第一个出现的位置，就取 where[0]
                # 这里严格要求只有一个
                # raise RuntimeError(f"Sample {b}: found multiple robot_history tokens: {where.squeeze(-1).tolist()}")
                pass
            pos = int(where[0])
            # if self.w_depth:
            #         pos = int(where[-1])
            #         dep_where = where.squeeze(1)
            text_embeds[b, pos, :] = states_embed[b]

            # replace rgb token
            if self.config.datasets.video_data.load_2d_data:
                # where = (input_ids[b] == rgb_ids).nonzero(as_tuple=False)
                rgb_ids_tensor = torch.tensor(rgb_ids, device=input_ids.device)
                where = torch.isin(input_ids[b], rgb_ids_tensor).nonzero(as_tuple=False).squeeze(1)
                _, order = torch.sort(where)

                rgb_query_reordered = self.rgb_query[order]    # [64, H]

                # why issues here???
                text_embeds[b, where, :] = rgb_query_reordered.to(text_embeds.dtype)

            # replace 3d gs token
            if self.config.datasets.gs_data.load_3d_data or self.w_depth:
                # where = (input_ids[b] == rgb_ids).nonzero(as_tuple=False)
                gs_ids_tensor = torch.tensor(gs_ids, device=input_ids.device)
                where = torch.isin(input_ids[b], gs_ids_tensor).nonzero(as_tuple=False).squeeze(1)
                _, order = torch.sort(where)
                gs_query_reordered = self.gs_query[order]    # [64, H]

                text_embeds[b, where, :] = gs_query_reordered.to(text_embeds.dtype)

            # # replace 3d gs token
            # if self.config.datasets.gs_data.load_3d_data:
            #     # where = (input_ids[b] == rgb_ids).nonzero(as_tuple=False)
            #     gs_ids_tensor = torch.tensor(gs_ids, device=input_ids.device)
            #     where = torch.isin(input_ids[b], gs_ids_tensor).nonzero(as_tuple=False).squeeze(1)
            #     _, order = torch.sort(where)
            #     gs_query_reordered = self.gs_query[order]    # [64, H]

            #     text_embeds[b, where, :] = gs_query_reordered

            # # replace reward token
            # if self.config.datasets.reward_data.load_reward_data:
            #     # where = (input_ids[b] == rgb_ids).nonzero(as_tuple=False)
            #     reward_ids_tensor = torch.tensor(reward_ids, device=input_ids.device)
            #     where = torch.isin(input_ids[b], reward_ids_tensor).nonzero(as_tuple=False).squeeze(1)
            #     _, order = torch.sort(where)
            #     reward_query_reordered = self.reward_query[order]    # [64, H]

            #     text_embeds[b, where, :] = reward_query_reordered

        # 前向：用 inputs_embeds（不要再传 input_ids）
        # position_ids = (attention_mask.long().cumsum(-1) - 1).clamp(min=0)
        with torch.no_grad():
            # 注意：这里用的是底层 Qwen3VLModel 的 get_rope_index
            position_ids, _ = self.qwen_vl_interface.model.model.get_rope_index(
                input_ids=qwen_inputs["input_ids"],
                image_grid_thw=qwen_inputs["image_grid_thw"],
                video_grid_thw=qwen_inputs.get("video_grid_thw", None),
                attention_mask=attention_mask,   # 2D mask 就行
            )

        qwen_forward_mode = getattr(
            self, "_inference_qwen_forward_mode", "legacy"
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if qwen_forward_mode == "optimized":
                image_parts, deepstack_embeds = (
                    self.qwen_vl_interface.model.model.get_image_features(
                        qwen_inputs["pixel_values"],
                        qwen_inputs["image_grid_thw"],
                    )
                )
                image_embeds = torch.cat(image_parts, dim=0)
                last_hidden = self._qwen_language_forward(
                    input_ids=input_ids,
                    inputs_embeds=text_embeds,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    image_embeds=image_embeds,
                    deepstack_embeds=deepstack_embeds,
                )
            else:
                qw_out = self.qwen_vl_interface(
                    inputs_embeds=text_embeds,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    # 视觉侧保持不变
                    pixel_values=qwen_inputs.get("pixel_values", None),
                    image_grid_thw=qwen_inputs.get("image_grid_thw", None),
                    output_hidden_states=True,
                    return_dict=True,
                )
                last_hidden = qw_out.hidden_states[-1]   # [B, L, H]

        # Step 1: QWenVL input format
        # qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)
        # with torch.autocast("cuda", dtype=torch.bfloat16):
        #     qwenvl_outputs = self.qwen_vl_interface(
        #         **qwen_inputs,
        #         output_attentions=False,
        #         output_hidden_states=True,
        #         return_dict=True,
        #     )
        #     # last_hidden_state: [B, seq_len, H]
        #     last_hidden = qwenvl_outputs.hidden_states[-1]   # [B, L, H]

        # Step 4: Action Expert Forward and Loss
        # …接下来的流程保持你原来的：从动作 query 位置 gather hidden，过 action head，算 L1 loss …
        # 例如（如果你仍然用多个 <robot_action_*>）：
        
        # if self.config.datasets.vla_data.load_act_data == 1:
        act_ids = [tok.convert_tokens_to_ids(t) for t in self.act_query_tokens]  # 长度 T
        act_pos_idx = []
        for b in range(B):
            pos_list = []
            for tid in act_ids:
                w = (input_ids[b] == tid).nonzero(as_tuple=False)
                if w.numel() == 0:
                    raise RuntimeError(f"Sample {b}: action token {tid} not found.")
                pos_list.append(int(w[0]))
            act_pos_idx.append(torch.tensor(pos_list, device=last_hidden.device))
        act_pos_idx = torch.stack(act_pos_idx, dim=0)                            # [B, T]
        g_idx = act_pos_idx.unsqueeze(-1).expand(-1, -1, H)                      # [B, T, H]
        action_queries = last_hidden.gather(dim=1, index=g_idx)                     # [B, T, H]

        with torch.autocast("cuda", dtype=torch.float32):
            # 提取动作 token embedding 作为动作预测查询
            # input_ids = qwen_inputs.get("input_ids", None)
            # action_queries = self._gather_action_token_embeddings(last_hidden, input_ids, action_token_id=self.action_token_id)  # [B, chunk_len, H]
            if self.mlp_head == 0:
                pred_actions = self.action_model.predict_action(action_queries)  # (B, chunk_len, action_dim)
            else:
                pred_actions = self.action_model(action_queries)

        normalized_actions = pred_actions.detach().cpu().numpy()

        # if self.config.datasets.video_data.load_2d_data:
        if False:
            rgb_data = [example['2d_gen_data'] for example in examples]

            rgb_ids = [tok.convert_tokens_to_ids(t) for t in self.rgb_query_tokens]  # 长度 T
            rgb_pos_idx = []
            for b in range(B):
                pos_list = []
                for tid in rgb_ids:
                    w = (input_ids[b] == tid).nonzero(as_tuple=False)
                    if w.numel() == 0:
                        raise RuntimeError(f"Sample {b}: action token {tid} not found.")
                    pos_list.append(int(w[0]))
                rgb_pos_idx.append(torch.tensor(pos_list, device=last_hidden.device))
            rgb_pos_idx = torch.stack(rgb_pos_idx, dim=0)                            # [B, T]
            g_idx = rgb_pos_idx.unsqueeze(-1).expand(-1, -1, H)                      # [B, T, H]
            rgb_queries = last_hidden.gather(dim=1, index=g_idx)                     # [B, T, H]

            with torch.autocast("cuda", dtype=torch.bfloat16):
                rgbs = self.rgb_model.predict_rgb(rgb_data, rgb_queries)
            
            return {"normalized_actions": normalized_actions, "rgbs": rgbs}
        
        # if self.config.datasets.gs_data.load_3d_data:
        if False:

            gs_data = [example['3d_gs_data'] for example in examples]

            gs_ids = [tok.convert_tokens_to_ids(t) for t in self.gs_query_tokens]  # 长度 T
            gs_pos_idx = []
            for b in range(B):
                pos_list = []
                for tid in gs_ids:
                    w = (input_ids[b] == tid).nonzero(as_tuple=False)
                    if w.numel() == 0:
                        raise RuntimeError(f"Sample {b}: action token {tid} not found.")
                    pos_list.append(int(w[0]))
                gs_pos_idx.append(torch.tensor(pos_list, device=last_hidden.device))
            gs_pos_idx = torch.stack(gs_pos_idx, dim=0)                            # [B, T]
            g_idx = gs_pos_idx.unsqueeze(-1).expand(-1, -1, H)                      # [B, T, H]
            gs_queries = last_hidden.gather(dim=1, index=g_idx)                     # [B, T, H]

            with torch.autocast("cuda", dtype=torch.bfloat16):
                gs = self.gs_model.predict_gs(gs_data, gs_queries)

            return {"normalized_actions": normalized_actions, "gs": gs}
        
        # if self.config.datasets.reward_data.load_reward_data:
        if False:

            # reward_data = np.array([example['reward_data'] for example in examples])  # list of reward (B)

            reward_ids = [tok.convert_tokens_to_ids(t) for t in self.reward_query_tokens]  # 长度 T
            reward_pos_idx = []
            for b in range(B):
                pos_list = []
                for tid in reward_ids:
                    w = (input_ids[b] == tid).nonzero(as_tuple=False)
                    if w.numel() == 0:
                        raise RuntimeError(f"Sample {b}: action token {tid} not found.")
                    pos_list.append(int(w[0]))
                reward_pos_idx.append(torch.tensor(pos_list, device=last_hidden.device))
            reward_pos_idx = torch.stack(reward_pos_idx, dim=0)                            # [B, T]
            g_idx = reward_pos_idx.unsqueeze(-1).expand(-1, -1, H)                      # [B, T, H]
            reward_queries = last_hidden.gather(dim=1, index=g_idx)                     # [B, T, H]

            with torch.autocast("cuda", dtype=torch.float32):
                reward = self.reward_model.predict_action(reward_queries)
            
            return {"normalized_actions": normalized_actions, "reward": reward}

        return {"normalized_actions": normalized_actions}

    
    @torch.inference_mode()
    def forward_act_embedding(
        self,
        examples,
        **kwargs: str,
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
        # train_obs_image_size = getattr(self.config.datasets.vla_data, "image_size", None)
        # if train_obs_image_size:
        #     batch_images = resize_images(batch_images, target_size=train_obs_image_size)
    
        # step 0: add special action token to instruction
        # action_tokens = self.action_token* self.chunk_len #can't add " " between two tokens, otherwise will be tokenized to multiple tokens
        # prompt_suffix = f" Please predict the next {self.chunk_len} robot actions: <action>{action_tokens}<action>."
        # instructions = [instruction + prompt_suffix for instruction in instructions]

        batch_images = [example["image"] for example in examples]  #  [B，[PLT]]
        instructions = [example["lang"] for example in examples]  # [B, str]
        # actions = [example["action"] for example in examples]  # label [B， len, 7]
        
        states = [example["state"] for example in examples]

        suffix = self._build_action_prompt_suffix()
        instructions = [instruction + suffix for instruction in instructions]

        # Step 1: QWenVL input format
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)

        # —— 覆盖 <robot_history_action_0> 的 embedding ——
        tok   = self.qwen_vl_interface.processor.tokenizer
        hist_id = tok.convert_tokens_to_ids(self.robot_history_token)  # "<robot_history_action_0>"

        if self.config.datasets.video_data.load_2d_data:
            rgb_ids = tok.convert_tokens_to_ids(self.rgb_query_tokens)
        
        if self.config.datasets.gs_data.load_3d_data:
            gs_ids = tok.convert_tokens_to_ids(self.gs_query_tokens)
        
        if self.config.datasets.reward_data.load_reward_data:
            # one token
            reward_ids = tok.convert_tokens_to_ids(self.reward_query_tokens)
        input_ids      = qwen_inputs["input_ids"]          # [B, L]
        attention_mask = qwen_inputs["attention_mask"]     # [B, L]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            text_embeds = self.qwen_vl_interface.model.get_input_embeddings()(input_ids)  # [B, L, H]


        with torch.autocast("cuda", dtype=torch.float32):
            # 映射到 hidden 维: [B, H]
            states = torch.from_numpy(np.array(states)).cuda()[:, 0, :]
            states_embed = self.action_input_model(states)  # [B, H]
        states_embed = states_embed.to(dtype=text_embeds.dtype)

        # 逐样本把 hist_id 的那个位置替换成对应的 states_embed[b]
        B, L, H = text_embeds.shape
        for b in range(B):
            where = (input_ids[b] == hist_id).nonzero(as_tuple=False)
            if where.numel() == 0:
                raise RuntimeError(f"Sample {b}: robot_history token not found in input_ids.")
            if where.numel() > 1:
                # 如果你只想覆盖第一个出现的位置，就取 where[0]
                # 这里严格要求只有一个
                raise RuntimeError(f"Sample {b}: found multiple robot_history tokens: {where.squeeze(-1).tolist()}")
            pos = int(where[0])
            text_embeds[b, pos, :] = states_embed[b]

            # replace rgb token
            if self.config.datasets.video_data.load_2d_data:
                # where = (input_ids[b] == rgb_ids).nonzero(as_tuple=False)
                rgb_ids_tensor = torch.tensor(rgb_ids, device=input_ids.device)
                where = torch.isin(input_ids[b], rgb_ids_tensor).nonzero(as_tuple=False).squeeze(1)
                _, order = torch.sort(where)

                rgb_query_reordered = self.rgb_query[order]    # [64, H]

                # why issues here???
                text_embeds[b, where, :] = rgb_query_reordered.to(text_embeds.dtype)

            # replace 3d gs token
            if self.config.datasets.gs_data.load_3d_data:
                # where = (input_ids[b] == rgb_ids).nonzero(as_tuple=False)
                gs_ids_tensor = torch.tensor(gs_ids, device=input_ids.device)
                where = torch.isin(input_ids[b], gs_ids_tensor).nonzero(as_tuple=False).squeeze(1)
                _, order = torch.sort(where)
                gs_query_reordered = self.gs_query[order]    # [64, H]

                text_embeds[b, where, :] = gs_query_reordered

            if self.action_prompt_mode == "minimal_agent":
                mine_agent_ids_tensor = torch.tensor(mine_agent_ids, device=input_ids.device)
                where = torch.isin(input_ids[b], mine_agent_ids_tensor).nonzero(as_tuple=False).squeeze(1)
                _, order = torch.sort(where)
                mine_agent_query_reordered = self.mine_agent_query[order] if hasattr(self, "mine_agent_query") else None

            # replace reward token
            if self.config.datasets.reward_data.load_reward_data:
                # where = (input_ids[b] == rgb_ids).nonzero(as_tuple=False)
                reward_ids_tensor = torch.tensor(reward_ids, device=input_ids.device)
                where = torch.isin(input_ids[b], reward_ids_tensor).nonzero(as_tuple=False).squeeze(1)
                _, order = torch.sort(where)
                reward_query_reordered = self.reward_query[order]    # [64, H]

                text_embeds[b, where, :] = reward_query_reordered

        # 前向：用 inputs_embeds（不要再传 input_ids）
        # position_ids = (attention_mask.long().cumsum(-1) - 1).clamp(min=0)
        with torch.no_grad():
            # 注意：这里用的是底层 Qwen3VLModel 的 get_rope_index
            position_ids, _ = self.qwen_vl_interface.model.model.get_rope_index(
                input_ids=qwen_inputs["input_ids"],
                image_grid_thw=qwen_inputs["image_grid_thw"],
                video_grid_thw=qwen_inputs.get("video_grid_thw", None),
                attention_mask=attention_mask,   # 2D mask 就行
            )

        with torch.autocast("cuda", dtype=torch.bfloat16):
            qw_out = self.qwen_vl_interface(
                inputs_embeds=text_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                # 视觉侧保持不变
                pixel_values=qwen_inputs.get("pixel_values", None),
                image_grid_thw=qwen_inputs.get("image_grid_thw", None),
                output_hidden_states=True,
                return_dict=True,
            )
            last_hidden = qw_out.hidden_states[-1]   # [B, L, H]

        # Step 1: QWenVL input format
        # qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)
        # with torch.autocast("cuda", dtype=torch.bfloat16):
        #     qwenvl_outputs = self.qwen_vl_interface(
        #         **qwen_inputs,
        #         output_attentions=False,
        #         output_hidden_states=True,
        #         return_dict=True,
        #     )
        #     # last_hidden_state: [B, seq_len, H]
        #     last_hidden = qwenvl_outputs.hidden_states[-1]   # [B, L, H]

        # Step 4: Action Expert Forward and Loss
        # …接下来的流程保持你原来的：从动作 query 位置 gather hidden，过 action head，算 L1 loss …
        # 例如（如果你仍然用多个 <robot_action_*>）：
        
        # if self.config.datasets.vla_data.load_act_data == 1:
        act_ids = [tok.convert_tokens_to_ids(t) for t in self.act_query_tokens]  # 长度 T
        act_pos_idx = []
        for b in range(B):
            pos_list = []
            for tid in act_ids:
                w = (input_ids[b] == tid).nonzero(as_tuple=False)
                if w.numel() == 0:
                    raise RuntimeError(f"Sample {b}: action token {tid} not found.")
                pos_list.append(int(w[0]))
            act_pos_idx.append(torch.tensor(pos_list, device=last_hidden.device))
        act_pos_idx = torch.stack(act_pos_idx, dim=0)                            # [B, T]
        g_idx = act_pos_idx.unsqueeze(-1).expand(-1, -1, H)                      # [B, T, H]
        action_queries = last_hidden.gather(dim=1, index=g_idx)                     # [B, T, H]

        with torch.autocast("cuda", dtype=torch.float32):
            # 提取动作 token embedding 作为动作预测查询
            # input_ids = qwen_inputs.get("input_ids", None)
            # action_queries = self._gather_action_token_embeddings(last_hidden, input_ids, action_token_id=self.action_token_id)  # [B, chunk_len, H]
            if self.mlp_head == 0:
                prompt_embeds = self.action_model.qwen_proj(action_queries)  # (B, chunk_len, action_dim)
            else:
                assert False
                pred_actions = self.action_model(action_queries)

        return prompt_embeds

    def _gather_action_token_embeddings(
        self,
        last_hidden: torch.Tensor,   # [B, L, H]
        input_ids: torch.Tensor,     # [B, L]
        action_token_id=None,        # 可为 int 或 List[int]
    ) -> torch.Tensor:
        """
        向量化批量提取动作 token embedding:
          - 不再逐样本 for 循环
          - 取每个样本里最靠后的 chunk_len 个动作占位 token
        Args:
            last_hidden: [B, L, H]
            input_ids:   [B, L]
            action_token_id: int 或 List[int]
        Returns:
            action_queries: [B, chunk_len, H]
        """
        if action_token_id is None:
            raise ValueError("action_token_id 不能为空")

        device = input_ids.device
        B, L, H = last_hidden.shape

        # 支持多 id（如多个变体）
        if isinstance(action_token_id, (list, tuple, set)):
            id_list = torch.tensor(list(action_token_id), device=device, dtype=input_ids.dtype)
            # torch.isin 需要 PyTorch >=1.10
            mask = torch.isin(input_ids, id_list)
        else:
            mask = (input_ids == action_token_id)  # [B, L]

        counts = mask.sum(dim=1)  # [B]
        if (counts < self.chunk_len).any():
            insufficient = (counts < self.chunk_len).nonzero(as_tuple=False).flatten().tolist()
            raise RuntimeError(
                f"以下样本动作 token 数量不足 {self.chunk_len}: {insufficient} | counts={counts.tolist()}"
            )

        # 位置索引
        idx = torch.arange(L, device=device).unsqueeze(0).expand(B, L)  # [B, L]
        masked_pos = torch.where(mask, idx, torch.full_like(idx, -1))   # 非动作位置置 -1

        # 取最后 chunk_len 个（索引大的在序列靠后）
        # 注意: 已确保数量足够，不会出现 -1 被错误选中的问题
        topk_pos = masked_pos.topk(k=self.chunk_len, dim=-1).values     # [B, chunk_len] 未排序
        # 时间顺序排序
        selected_pos = topk_pos.sort(dim=-1).values                     # [B, chunk_len]

        # Gather
        expanded_index = selected_pos.unsqueeze(-1).expand(-1, -1, H)   # [B, chunk_len, H]
        action_queries = last_hidden.gather(dim=1, index=expanded_index)  # [B, chunk_len, H]
        return action_queries


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
    cfg.framework.action_model.action_hidden_dim = 2048

    cfg.framework.qwenvl.base_vlm = "./playground/Pretrained_models/Qwen3-VL-4B-Instruct"
    

    # try get model
    model = Qwenvl_OFT(cfg)
    print(model)

    # fake sample 
    image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    # Create a sample
    sample = {
        "action": np.random.uniform(-1, 1, size=(16, 7)).astype(np.float16), # action_chunk, action_dim
        "image": [image, image], # two views
        "lang": "This is a fake instruction for testing.",
        # "state" : np.random.uniform(-1, 1, size=(1, 7)).astype(np.float16), # chunk, state_dim
    }

    sample2 = {
        "action": np.random.uniform(-1, 1, size=(16, 7)).astype(np.float16), # action_chunk, action_dim
        "image": [image, image], # two views
        "lang": "For testing.",
        # "state" : np.random.uniform(-1, 1, size=(1, 7)).astype(np.float16), # chunk, state_dim
    }

    batch  = [sample, sample2]  # batch size 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    forward_output = model(batch)
    action_loss = forward_output['action_loss']
    print(f"Action Loss: {action_loss.item()}")

    # test predict action
    predict_output = model.predict_action(batch_images=[batch[0]["image"]], instructions=[batch[0]["lang"]])
    normalized_actions = predict_output['normalized_actions']
    print(f"Unnormalized Action: {normalized_actions}")


    # # try forward model
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
    # # zhe
    # for batch in tqdm(train_dataloader, desc="Processing Batches"):
    #     batch
    #     break

    # # try get model
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = model.to(device)
    # model(batch)
    # pass
    # action = model.predict_action(batch_images=[batch[0]["image"]], instructions=[batch[0]["lang"]])

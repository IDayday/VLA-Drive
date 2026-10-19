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
import time
from omegaconf import OmegaConf

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


class BevHeatmapHead(nn.Module):
    def __init__(self, hidden_size: int, num_classes: int, bev_hw=(50, 50)):
        super().__init__()
        self.bev_h, self.bev_w = bev_hw
        self.norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.linear = nn.Linear(hidden_size, num_classes, bias=True)

    def forward(self, bev_feat: torch.Tensor):
        """
        bev_feat: [B, C, Hb, Wb]  (你后面自己保证它已经是 50x50 或者先 resize)
        return:  logits [B, num_classes, Hb, Wb]
        """
        B, C, H, W = bev_feat.shape
        x = bev_feat.flatten(2).transpose(1, 2).contiguous()   # [B, HW, C]
        x = self.norm(x)
        x = self.linear(x)                                     # [B, HW, num_classes]
        x = x.transpose(1, 2).reshape(B, -1, H, W).contiguous()
        return x


import torch
import torch.nn as nn
import torch.nn.functional as F


class BEVUpHead18x96(nn.Module):
    """
    Input : (B, 2048, 18, 96)
    Output: (B, out_ch, 500, 500)
    """
    def __init__(self, out_ch=1, mid_ch=256):
        super().__init__()

        # 1. 压通道（非常重要）
        self.proj = nn.Sequential(
            nn.Conv2d(2048, mid_ch, 1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.SiLU(inplace=True),
        )

        # 2. BEV-space refine（在 500x500 上学形状）
        def block(ch):
            return nn.Sequential(
                nn.Conv2d(ch, ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(ch),
                nn.SiLU(inplace=True),
            )

        self.refine = nn.Sequential(
            block(mid_ch),
            block(mid_ch),
            block(mid_ch),
        )

        # 3. 输出
        self.head = nn.Conv2d(mid_ch, out_ch, 1)

    def forward(self, feat):
        """
        feat: (B,2048,18,96)
        """
        x = self.proj(feat)  # (B,256,18,96)

        # ⭐ 核心：直接 resize 到 500×500
        x = F.interpolate(
            x, size=(496, 496),
            mode="bilinear",
            align_corners=False
        )

        x = self.refine(x)
        out = self.head(x)
        return out


import torch.nn.functional as F

def to_bev_50x50(featmap: torch.Tensor):
    # featmap: [B, C, Hm, Wm]
    return F.interpolate(featmap, size=(50, 50), mode="bilinear", align_corners=False)

class EgoTrajRegHead(nn.Module):
    """
    regress future trajectory in ego frame: (x forward, y left)
    """
    def __init__(self, in_ch=2048, T=12, mid=1024, use_avgmax=True):
        super().__init__()
        self.T = T
        self.use_avgmax = use_avgmax
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.max = nn.AdaptiveMaxPool2d(1)

        in_dim = in_ch * (2 if use_avgmax else 1)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, mid),
            nn.SiLU(inplace=True),
            nn.Linear(mid, mid),
            nn.SiLU(inplace=True),
            nn.Linear(mid, T * 4),
        )

    def forward(self, feat):  # feat: [B,C,H,W]
        a = self.avg(feat).flatten(1)  # [B,C]
        if self.use_avgmax:
            m = self.max(feat).flatten(1)
            x = torch.cat([a, m], dim=1)  # [B,2C]
        else:
            x = a
        out = self.mlp(x).view(-1, self.T, 4)  # [B,T,2]
        return out


@FRAMEWORK_REGISTRY.register("QwenVision")
class Qwenvl_Vision(baseframework):
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

        self.hidden_size = 2048
        
        self.bev_adap = nn.Conv2d(self.hidden_size, self.hidden_size, kernel_size=3, padding=1)

        self.traj_head = EgoTrajRegHead(in_ch=2048, T=8)

        self.bev_head = BEVUpHead18x96()

        try:
            self.w_depth = self.config.w_depth
        except:
            self.w_depth = 0

    def forward(
        self,
        examples: List[dict] = None,
        accelerator = None,
        **kwargs,
    ) -> Tuple:

        batch_images = [example["image"] for example in examples]  #  [B，[PLT]]
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
            depth_feats = [example['depth_feat'] for example in examples]

        gt_bev = [example['bev'] for example in examples]

        # Step 1: QWenVL input format
        pixel_values, image_grid_thw = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            qw_feat_map, flr = self.qwen_vl_interface(
                # 视觉侧保持不变
                pixel_values=pixel_values.cuda(),
                image_grid_thw=image_grid_thw.cuda(),
            )
            traj_pred = self.traj_head(qw_feat_map)
            # to bev size
            bev = to_bev_50x50(qw_feat_map)                # [1, C, 50, 50]
            logits = self.bev_head(bev)                 # [1, num_classes, 50, 50]
        
        gt_mask = torch.from_numpy(np.array(gt_bev)).cuda() # b, h, w
        gt = gt_mask.float().unsqueeze(1)  # [B,1,500,500]
        bev_loss = F.binary_cross_entropy_with_logits(logits, gt)
        actions = torch.tensor(
            np.array(actions), device=logits.device, dtype=torch.float32
        )
        traj_loss = nn.SmoothL1Loss()(traj_pred.float(), actions)

        loss = bev_loss + traj_loss

        return {'action_loss': loss}


        #### video gen ####
        if self.config.datasets.video_data.load_2d_data:

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

            ###### debug pipeline
            # rgbs = self.rgb_model.predict_rgb(rgb_data, rgb_queries)

            with torch.autocast("cuda", dtype=torch.bfloat16):
                rgb_loss, video_latent = self.rgb_model(rgb_data, rgb_queries)
        else:
            rgb_loss = torch.tensor(0.).cuda()


        # Step 4: Action Expert Forward and Loss
        if self.config.datasets.vla_data.load_act_data == 1:
            # …接下来的流程保持你原来的：从动作 query 位置 gather hidden，过 action head，算 L1 loss …
            # 例如（如果你仍然用多个 <robot_action_*>）：
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

                # 标签对齐：取最后 chunk_len 段
                actions = torch.tensor(
                    np.array(actions), device=action_queries.device, dtype=torch.float32
                )  # [B, T_full, action_dim]
                # actions_target = actions[:, -(self.future_action_window_size+1):, :]  # (B, chunk_len, action_dim)

                ####### repeat  ###
                repeated_diffusion_steps = (
                    self.config.framework.action_model.get("repeated_diffusion_steps", 1) if self.config else 1
                )
                # repeated_diffusion_steps = 2 # NO repeat for big action FM
                actions= actions.repeat(repeated_diffusion_steps, 1, 1)
                # 对每层特征做 repeat
                action_queries = action_queries.repeat(repeated_diffusion_steps, 1, 1)

                if self.w_video_latent:
                    video_token = self.rgb_latent_adapter(video_latent)
                    video_token = video_token + self.rgb_latent_type.to(video_token.dtype)
                else:
                    video_token = None

                if self.mlp_head == 0:
                    action_loss = self.action_model(action_queries, actions, video_token)  # (B, chunk_len, action_dim)
                else:
                    b, l, h = action_queries.shape
                    pred_action = self.action_model(action_queries.reshape(b, l*h)).reshape(b, l, -1)
                    action_loss = nn.SmoothL1Loss()(pred_action, actions)
        else:
            action_loss = torch.tensor(0.).cuda()


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

            # debug here
            # gs = self.gs_model.predict_gs(gs_data, gs_queries)

            with torch.autocast("cuda", dtype=torch.bfloat16):
                gs_loss = self.gs_model(gs_data, gs_queries)
            
            return {"action_loss": action_loss, "rgb_loss": rgb_loss, "gs_loss": gs_loss}
        else:
            gs_loss = torch.tensor(0.).cuda()

        if self.config.datasets.reward_data.load_reward_data:

            reward_data = np.array([example['reward_data'] for example in examples])  # list of reward (B)

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

            # debug here
            # reward = self.reward_model.predict_action(reward_queries)

            with torch.autocast("cuda", dtype=torch.float32):
                reward_loss = self.reward_model(reward_queries, reward_data)
            
            return {"action_loss": action_loss, "rgb_loss": rgb_loss, "gs_loss": gs_loss, "reward_loss": reward_loss}
        else:
            reward_loss = torch.tensor(0.).cuda()

        return {"action_loss": action_loss, "rgb_loss": rgb_loss, "gs_loss": gs_loss, "reward_loss": reward_loss}

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

        # step 0: add special action token to instruction
        hist_str  = self.robot_history_token
        rgb_str   = "".join(self.rgb_query_tokens)
        gs_str    = "".join(self.gs_query_tokens)
        act_str   = "".join(self.act_query_tokens)
        rew_str   = "".join(self.reward_query_tokens)

        suffix = f" {hist_str}{rgb_str}{gs_str}{act_str}{rew_str}"
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
                if self.config.datasets.video_data.load_2d_data and self.doing_v_pre:
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
            depth_feats = [example['depth_feat'] for example in examples]

        # step 0: add special action token to instruction
        hist_str  = self.robot_history_token
        rgb_str   = "".join(self.rgb_query_tokens)
        gs_str    = "".join(self.gs_query_tokens)
        act_str   = "".join(self.act_query_tokens)
        rew_str   = "".join(self.reward_query_tokens)

        # suffix = f" {hist_str}{rgb_str}{gs_str}{act_str}{rew_str}"
        # instructions = [instruction + suffix for instruction in instructions]

        if not self.w_depth:
            suffix = f" {hist_str}{rgb_str}{gs_str}{act_str}{rew_str}"
            instructions = [instruction + suffix for instruction in instructions]
        else:
            hist_str = "".join(self.robot_history_token)
            suffix = f" {hist_str}{rgb_str}{gs_str}{act_str}{rew_str}"
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

        if self.w_depth:
            depth_ids = tok.convert_tokens_to_ids(self.robot_history_token)[:-1]
            hist_id = hist_id[-1]
        
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

        if self.w_depth:
            with torch.autocast("cuda", dtype=torch.float32):
                # 映射到 hidden 维: [B, H]
                depth_feats = torch.stack(depth_feats)
                bz, n_cam, n_channel, n_h, n_w = depth_feats.shape
                depth_feats = depth_feats.reshape(bz*n_cam, n_channel, n_h, n_w)
                depth_token = self.depth_adapter(depth_feats)
                n_token = depth_token.shape[1]
                depth_token = depth_token.reshape(bz, n_cam, n_token, -1)   # 3*64
                depth_token = depth_token[:, [1,2,0]]   # l,r,f
                depth_token = depth_token.reshape(bz, n_cam*n_token, -1)
            depth_token = depth_token.to(dtype=text_embeds.dtype)
            depth_token = depth_token + self.depth_type.to(depth_token.dtype)

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
            if self.w_depth:
                    pos = int(where[-1])
                    dep_where = where.squeeze(1)
            text_embeds[b, pos, :] = states_embed[b]

            # replace rgb token
            if self.config.datasets.video_data.load_2d_data and self.doing_v_pre:
                # where = (input_ids[b] == rgb_ids).nonzero(as_tuple=False)
                rgb_ids_tensor = torch.tensor(rgb_ids, device=input_ids.device)
                where = torch.isin(input_ids[b], rgb_ids_tensor).nonzero(as_tuple=False).squeeze(1)
                _, order = torch.sort(where)

                rgb_query_reordered = self.rgb_query[order]    # [64, H]

                # why issues here???
                text_embeds[b, where, :] = rgb_query_reordered.to(text_embeds.dtype)

            if self.w_depth:
                # where = (input_ids[b] == rgb_ids).nonzero(as_tuple=False)
                # depth_ids_tensor = torch.tensor(depth_ids, device=input_ids.device)
                # where = torch.isin(input_ids[b], depth_ids_tensor).nonzero(as_tuple=False).squeeze(1)

                text_embeds[b, dep_where[:-1], :] = depth_token[b]

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

        # step 0: add special action token to instruction
        hist_str  = self.robot_history_token
        rgb_str   = "".join(self.rgb_query_tokens)
        gs_str    = "".join(self.gs_query_tokens)
        act_str   = "".join(self.act_query_tokens)
        rew_str   = "".join(self.reward_query_tokens)

        suffix = f" {hist_str}{rgb_str}{gs_str}{act_str}{rew_str}"
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
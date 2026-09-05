from typing import Dict, Optional
import math
import numpy as np
import torch
import torch.nn as nn
from .score_module.scorer import (
    Scorer,
    aggregate_drivor_pdm_score,
    normalized_drivor_pred_pdms,
)
from .transformer_decoder import TransformerDecoder, TransformerDecoderScorer
from .layers.image_encoder.dinov2_lora import ImgEncoder
from .layers.utils.mlp import MLP
from .utils import pylogger
from .layers.q_former.q_former import VisionOnlyQFormer
log = pylogger.get_pylogger(__name__)
import logging
# log.setLevel(logging.DEBUG)

# Scorer path adapted from valeoai/DrivoR by its original authors, pinned to
# commit fc6e5aa144bbcb5a046e22c18f1bd5cf3af8634a.

class ActionDecoder(nn.Module):
    def __init__(
        self,
        config,
        scene_fusion_config=None,
        total_optimizer_steps: Optional[int] = None,
    ):
        super().__init__()
        self._config = config
        self.poses_num=config.num_poses
        self.state_size=3
        self.embed_dims = self._config.tf_d_model

        ###########################################
        # camera embedding
        self.num_cams = 0
        if len(self._config["cam_f0"]) > 0:
            self.num_cams += 1
        if len(self._config["cam_l0"]) > 0:
            self.num_cams += 1
        if len(self._config["cam_l1"]) > 0:
            self.num_cams += 1
        if len(self._config["cam_l2"]) > 0:
            self.num_cams += 1
        if len(self._config["cam_r0"]) > 0:
            self.num_cams += 1
        if len(self._config["cam_r1"]) > 0:
            self.num_cams += 1
        if len(self._config["cam_r2"]) > 0:
            self.num_cams += 1
        if len(self._config["cam_b0"]) > 0:
            self.num_cams += 1

        ############################################
        # lidar embedding
        self.num_lidar = 0
        if len(self._config["lidar_pc"]) > 0:
            self.num_lidar += 1


        self.q_former=VisionOnlyQFormer(vision_dim=1536,hidden_dim=256)
        self.semantic_query_init_std = float(getattr(config, "semantic_query_init_std", 1e-6))
        self.semantic_use_padding_mask = bool(getattr(config, "semantic_use_padding_mask", False))
        self.scene_embeds = nn.Parameter(torch.randn(1, self._config.num_scene_tokens, config.tf_d_model)*self.semantic_query_init_std, requires_grad=True)

        # print("self.scene_embeds ", self.scene_embeds)

        # ego status encoder
        if self._config.full_history_status:
            self.hist_encoding = nn.Linear(11*4, config.tf_d_model)
        else:
            self.hist_encoding = nn.Linear(11, config.tf_d_model)

        # trajectory embdedding
        if self._config.one_token_per_traj:
            self.init_feature = nn.Embedding(config.proposal_num, config.tf_d_model)
            traj_head_output_size = self.poses_num*self.state_size
        else:
            self.init_feature = nn.Embedding(self.poses_num * config.proposal_num, config.tf_d_model)
            traj_head_output_size =self.state_size

        # trajectory decoder
        self.trajectory_decoder = TransformerDecoder(proj_drop=0.1, drop_path=0.2, config=config)

        # scorer decoder
        self.scorer_attention = TransformerDecoderScorer(num_layers=config.scorer_ref_num, d_model=config.tf_d_model, proj_drop=0.1, drop_path=0.2, config=config)

        self.pos_embed = nn.Sequential(
                nn.Linear(self.poses_num * 3, config.tf_d_ffn),
                nn.ReLU(),
                nn.Linear(config.tf_d_ffn, config.tf_d_model),
            )


        # get the trajectory decoders
        self.poses_num=config.num_poses
        self.state_size=3
        ref_num=config.ref_num
        self.traj_head = nn.ModuleList([MLP(config.tf_d_model, config.tf_d_ffn,  traj_head_output_size) for _ in range(ref_num+1)])

        # scorer
        self.scorer = Scorer(config)
        self.b2d = config.b2d

        # Keep the legacy topology byte-for-byte compatible when no scene
        # fusion config is supplied. PlanReg configurations opt in explicitly.
        self.scene_fusion_enabled = scene_fusion_config is not None
        self.scene_feature_mode = "semantic_only"
        self.scene_transition_fraction = 0.20
        if self.scene_fusion_enabled:
            self.scene_feature_mode = str(
                getattr(scene_fusion_config, "mode", "planning_plus_semantic")
            )
            if self.scene_feature_mode not in {
                "semantic_only",
                "planning_only",
                "planning_plus_semantic",
                "planning_primary_semantic_xattn",
            }:
                raise ValueError(
                    "scene_fusion.mode must be semantic_only, planning_only, "
                    "planning_plus_semantic, or planning_primary_semantic_xattn; "
                    f"got {self.scene_feature_mode!r}"
                )
            self.scene_transition_fraction = float(
                getattr(scene_fusion_config, "transition_fraction", 0.20)
            )
            if not 0.0 < self.scene_transition_fraction <= 1.0:
                raise ValueError("scene_fusion.transition_fraction must be in (0,1]")
            # semantic_only is a strict legacy bypass: its module topology and
            # scene tensor are identical to ActionDecoder without scene fusion.
            if self.scene_feature_mode in {"planning_only", "planning_plus_semantic"}:
                self.scene_norm = nn.LayerNorm(config.tf_d_model)
            if self.scene_feature_mode == "planning_plus_semantic":
                gate_init = float(
                    getattr(scene_fusion_config, "semantic_gate_init", 0.549306)
                )
                self.semantic_gate = nn.Parameter(
                    torch.full((1, 1, config.tf_d_model), gate_init)
                )
                self.register_buffer(
                    "_optimizer_step",
                    torch.zeros((), dtype=torch.long),
                    persistent=True,
                )
                self.register_buffer(
                    "_total_optimizer_steps",
                    torch.tensor(
                        0
                        if total_optimizer_steps is None
                        else int(total_optimizer_steps),
                        dtype=torch.long,
                    ),
                    persistent=True,
                )
            elif self.scene_feature_mode == "planning_primary_semantic_xattn":
                fusion_layers = int(getattr(scene_fusion_config, "layers", 1))
                if fusion_layers != 1:
                    raise ValueError(
                        "PlanReg-WM-V1 formal semantic fusion is exactly one "
                        f"cross-attention layer, got layers={fusion_layers}"
                    )
                num_heads = int(getattr(scene_fusion_config, "num_heads", 8))
                dropout = float(getattr(scene_fusion_config, "dropout", 0.0))
                if self.embed_dims % num_heads != 0:
                    raise ValueError(
                        f"Fusion embed dim {self.embed_dims} is not divisible by "
                        f"num_heads={num_heads}"
                    )
                if dropout != 0.0:
                    raise ValueError(
                        "Formal planning-primary semantic fusion requires dropout=0.0"
                    )
                initial_probability = float(
                    getattr(
                        scene_fusion_config,
                        "semantic_gate_init_probability",
                        0.20,
                    )
                )
                if not 0.0 < initial_probability < 1.0:
                    raise ValueError(
                        "semantic_gate_init_probability must be strictly between 0 and 1"
                    )
                gate_logit = math.log(
                    initial_probability / (1.0 - initial_probability)
                )
                self.planning_norm = nn.LayerNorm(self.embed_dims)
                self.semantic_norm = nn.LayerNorm(self.embed_dims)
                self.semantic_cross_attention = nn.MultiheadAttention(
                    embed_dim=self.embed_dims,
                    num_heads=num_heads,
                    dropout=dropout,
                    batch_first=True,
                )
                self.output_norm = nn.LayerNorm(self.embed_dims)
                self.semantic_gate = nn.Parameter(torch.tensor(gate_logit))
        # Attached only after loading the original Base checkpoint. Keeping this
        # as None preserves the exact legacy state_dict and inference behavior.
        self.memory_attention = None
        self.memory_injection_mode = "shared"

    def set_optimizer_step(self, optimizer_step: int) -> None:
        self.collect_content_diagnostics = optimizer_step % 500 == 0
        if self.scene_feature_mode == "planning_plus_semantic":
            self._optimizer_step.fill_(int(optimizer_step))

    def apply_refinement_training_policy(self):
        """Freeze inactive output heads after shared init/checkpoint restore.

        All forward computations are retained, preserving dropout RNG order
        and old checkpoint topology. Head0 never gets visual deep supervision.
        """
        policy = str(getattr(self._config, 'refinement_training_policy', 'legacy'))
        if policy not in {'legacy', 'final_only', 'light_deep_supervision'}:
            raise ValueError(f'Unknown refinement training policy: {policy}')
        if policy == 'legacy':
            return
        for index, head in enumerate(self.traj_head):
            trainable = index == len(self.traj_head)-1 or (policy == 'light_deep_supervision' and index > 0)
            head.requires_grad_(trainable)

    def configure_total_optimizer_steps(self, total_optimizer_steps: int) -> None:
        if total_optimizer_steps <= 0:
            raise ValueError("total_optimizer_steps must be positive")
        if self.scene_feature_mode == "planning_plus_semantic":
            self._total_optimizer_steps.fill_(int(total_optimizer_steps))

    def scene_mix_ratio(self, reference: torch.Tensor) -> torch.Tensor:
        if self.scene_feature_mode == "semantic_only":
            return reference.new_zeros(())
        if self.scene_feature_mode == "planning_primary_semantic_xattn":
            return torch.sigmoid(self.semantic_gate).to(
                device=reference.device, dtype=reference.dtype
            )
        if self.scene_feature_mode == "planning_only" or not self.training:
            return reference.new_ones(())
        total_steps = int(self._total_optimizer_steps.item())
        if total_steps <= 0:
            raise RuntimeError(
                "planning_plus_semantic training requires the total optimizer "
                "step count for the configured rho transition"
            )
        transition_steps = max(
            1, int(math.ceil(total_steps * self.scene_transition_fraction))
        )
        ratio = min(1.0, int(self._optimizer_step.item()) / transition_steps)
        return reference.new_tensor(ratio)

    def fuse_scene_features(
        self,
        semantic_features: torch.Tensor,
        planning_features: Optional[torch.Tensor],
    ):
        if not self.scene_fusion_enabled:
            return semantic_features, semantic_features.new_zeros(())
        rho = self.scene_mix_ratio(semantic_features)
        if self.scene_feature_mode == "semantic_only":
            # Do not normalize or otherwise perturb the exact legacy path.
            return semantic_features, rho
        else:
            if planning_features is None:
                raise KeyError(
                    f"scene_fusion.mode={self.scene_feature_mode} requires "
                    "features['planning_registers']"
                )
            if planning_features.shape != semantic_features.shape:
                raise ValueError(
                    "Planning and semantic scene features must both be [B,16,256]; "
                    f"got {tuple(planning_features.shape)} and "
                    f"{tuple(semantic_features.shape)}"
                )
            planning_features = planning_features.to(
                device=semantic_features.device,
                dtype=semantic_features.dtype,
            )
            if self.scene_feature_mode == "planning_only":
                return self.scene_norm(planning_features), rho
            elif self.scene_feature_mode == "planning_primary_semantic_xattn":
                planning = self.planning_norm(planning_features)
                semantic = self.semantic_norm(semantic_features)
                semantic_context, _ = self.semantic_cross_attention(
                    query=planning,
                    key=semantic,
                    value=semantic,
                    need_weights=False,
                )
                if getattr(self, "collect_content_diagnostics", False):
                    with torch.no_grad():
                        _, attention = self.semantic_cross_attention(
                            planning.detach(), semantic.detach(), semantic.detach(),
                            need_weights=True, average_attn_weights=False)
                        self._fusion_entropy = -(attention.float() * attention.float().clamp_min(1e-30).log()).sum(-1).mean()
                scene_tokens = self.output_norm(
                    planning
                    + torch.sigmoid(self.semantic_gate).to(
                        dtype=semantic_context.dtype
                    )
                    * semantic_context
                )
                return scene_tokens, rho
            else:
                planning_target = self.scene_norm(
                    planning_features
                    + torch.tanh(self.semantic_gate).to(semantic_features.dtype)
                    * semantic_features
                )
                combined = (
                    (1.0 - rho) * semantic_features + rho * planning_target
                )
                return combined, rho

    def set_memory_attention(
        self,
        memory_attention: Optional[nn.Module],
        *,
        injection_mode: str = "shared",
    ) -> None:
        if injection_mode not in {"shared", "scorer_only"}:
            raise ValueError(
                "injection_mode must be 'shared' or 'scorer_only'"
            )
        self.memory_attention = memory_attention
        self.memory_injection_mode = injection_mode


    def _apply_memory_attention(
        self,
        scene_features: torch.Tensor,
        ego_token: torch.Tensor,
        features: Dict[str, torch.Tensor],
    ):
        memory_attention_output = None
        if self.memory_attention is not None:
            required_keys = (
                "memory_map_query_key",
                "memory_agent_query_key",
            )
            missing_keys = [key for key in required_keys if key not in features]
            if missing_keys:
                raise KeyError(
                    "Memory Attention is attached but ActionDecoder inputs are "
                    f"missing: {missing_keys}"
                )
            memory_attention_output = self.memory_attention(
                scene_features,
                ego_token,
                features["memory_map_query_key"],
                features["memory_agent_query_key"],
                excluded_memory_indices=features.get(
                    "memory_excluded_index"
                ),
                profile_latency=bool(
                    features.get("memory_profile_latency", False)
                ),
                return_attention_weights=bool(
                    features.get("memory_return_attention_weights", False)
                ),
            )
            scene_features = memory_attention_output["scene_features"]
        return scene_features, memory_attention_output

    def _decode_scene(
        self,
        scene_features: torch.Tensor,
        ego_token: torch.Tensor,
        memory_attention_output=None,
        scorer_scene_features: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if scorer_scene_features is None:
            scorer_scene_features = scene_features
        traj_tokens = ego_token + self.init_feature.weight[None]
        log.debug(f"Traj tokens initial - {traj_tokens.shape}")
        batch_size = scene_features.shape[0]

        # initial trajectories
        proposals = self.traj_head[0](traj_tokens).reshape(traj_tokens.shape[0], -1, self.poses_num, self.state_size)
        proposal_list = [proposals]
        log.debug(f"Proposals initial - {proposals.shape}")

        # decode the trajectories at each step of the decoder
        token_list = self.trajectory_decoder(traj_tokens, scene_features)
        log.debug(f"Trajectory decoder - {len(token_list)}")
        for i in range(self._config.ref_num):
            tokens = token_list[i]
            proposals = self.traj_head[i+1](tokens).reshape(tokens.shape[0], -1, self.poses_num, self.state_size)
            proposal_list.append(proposals)
        
        traj_tokens = token_list[-1]
        proposals=proposal_list[-1]
        

        output={}
        output["proposals"] = proposals
        output["proposal_list"] = proposal_list

        # scoring
        B,N,_,_=proposals.shape

        embedded_traj = self.pos_embed(proposals.reshape(B, N, -1).detach())  # (B, N, d_model)
        tr_out = self.scorer_attention(
            embedded_traj,
            scorer_scene_features,
        )  # (B, N, d_model)
        tr_out = tr_out+ego_token
        pred_logit,pred_logit2, pred_agents_states, pred_area_logit ,bev_semantic_map,agent_states,agent_labels= self.scorer(proposals, tr_out)

        output["pred_logit"]=pred_logit
        output["pred_logit2"]=pred_logit2
        output["pred_agents_states"]=pred_agents_states
        output["pred_area_logit"]=pred_area_logit
        output["bev_semantic_map"]=bev_semantic_map
        output["agent_states"]=agent_states
        output["agent_labels"]=agent_labels

        pdm_score = aggregate_drivor_pdm_score(pred_logit, self._config)

        token = torch.argmax(pdm_score, dim=1)
        trajectory = proposals[
            torch.arange(batch_size, device=proposals.device),
            token,
        ]

        output["trajectory"] = trajectory
        output["pdm_score"] = pdm_score
        output["pred_pdms"] = normalized_drivor_pred_pdms(
            pdm_score, self._config
        )
        if bool(getattr(self._config, "return_memory_fields", False)):
            output["language_feature"] = scene_features
            output["ego_feature"] = ego_token
            output["selected_proposal_index"] = token
        if memory_attention_output is not None:
            output["memory_attention"] = {
                key: value
                for key, value in memory_attention_output.items()
                if key != "scene_features"
            }

        return output

    def forward_from_scene_features(
        self,
        scene_features: torch.Tensor,
        ego_feature: torch.Tensor,
        map_query_key: Optional[torch.Tensor] = None,
        agent_query_key: Optional[torch.Tensor] = None,
        excluded_memory_indices: Optional[torch.Tensor] = None,
        profile_latency: bool = False,
        return_attention_weights: bool = False,
        use_memory: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Continue decoding from cached Q-Former and ego features."""
        if use_memory and self.memory_attention is not None:
            if map_query_key is None or agent_query_key is None:
                raise ValueError(
                    "Map and agent query keys are required when memory is used"
                )
            memory_inputs = {
                "memory_map_query_key": map_query_key,
                "memory_agent_query_key": agent_query_key,
                "memory_excluded_index": excluded_memory_indices,
                "memory_profile_latency": profile_latency,
                "memory_return_attention_weights": return_attention_weights,
            }
            original_scene_features = scene_features
            enhanced_scene_features, memory_output = self._apply_memory_attention(
                scene_features,
                ego_feature,
                memory_inputs,
            )
            if self.memory_injection_mode == "scorer_only":
                scene_features = original_scene_features
                scorer_scene_features = enhanced_scene_features
            else:
                scene_features = enhanced_scene_features
                scorer_scene_features = enhanced_scene_features
        else:
            memory_output = None
            scorer_scene_features = scene_features
        return self._decode_scene(
            scene_features,
            ego_feature,
            memory_output,
            scorer_scene_features,
        )

    def forward(
        self,
        features: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        ego_status = features["status_feature"]
        ego_status = torch.cat(
            [torch.zeros_like(ego_status)[:, :3], ego_status],
            dim=1,
        )
        ego_token = self.hist_encoding(ego_status)[:, None]
        log.debug(f"Ego features - {ego_token.shape}")

        semantic_features = self.q_former(
            self.scene_embeds,
            features["last_hidden_state"],
            semantic_token_valid_mask=(features.get("semantic_token_valid_mask")
                                       if self.semantic_use_padding_mask else None),
        )
        self._latest_semantic_diagnostics = semantic_features.detach()
        planning_features = features.get("planning_registers")
        scene_features, scene_mix_ratio = self.fuse_scene_features(
            semantic_features,
            planning_features,
        )
        original_scene_features = scene_features
        enhanced_scene_features, memory_output = self._apply_memory_attention(
            scene_features,
            ego_token,
            features,
        )
        if (
            memory_output is not None
            and self.memory_injection_mode == "scorer_only"
        ):
            scene_features = original_scene_features
            scorer_scene_features = enhanced_scene_features
        else:
            scene_features = enhanced_scene_features
            scorer_scene_features = enhanced_scene_features
        output = self._decode_scene(
            scene_features,
            ego_token,
            memory_output,
            scorer_scene_features,
        )
        if self.scene_fusion_enabled:
            output.update(
                {
                    "planning_registers": planning_features,
                    "semantic_scene_features": semantic_features,
                    "planning_scene_features": planning_features,
                    "scene_mix_ratio": scene_mix_ratio,
                }
            )
        return output

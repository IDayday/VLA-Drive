# Copyright 2025 NVIDIA Corp. and affiliates. All rights reserved.
# Modified by [Junqiu YU/ Fudan University] in [2025]. 
# Modification: [rm and add some connect adapter to match with starVLA, e.g., "rm "].



from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Beta
from transformers import PretrainedConfig
from transformers.feature_extraction_utils import BatchFeature

from starVLA.model.modules.action_model.flow_matching_head.action_encoder import (
    SinusoidalPositionalEncoding,
    swish,
)

from starVLA.model.modules.action_model.flow_matching_head.cross_attention_dit import DiT, SelfAttentionTransformer

# TODO try to meger DiT Modules with follow_match_head, they are just the same arch, but diff loss, use diffusers package will be simple

class CategorySpecificLinear(nn.Module):
    def __init__(self, num_categories, input_dim, hidden_dim):
        super().__init__()
        self.num_categories = num_categories
        # For each category, we have separate weights and biases.
        self.W = nn.Parameter(0.02 * torch.randn(num_categories, input_dim, hidden_dim))
        self.b = nn.Parameter(torch.zeros(num_categories, hidden_dim))

    def forward(self, x, cat_ids):
        selected_W = self.W[cat_ids]
        selected_b = self.b[cat_ids]
        # import ipdb; ipdb.set_trace()
        return torch.bmm(x, selected_W) + selected_b.unsqueeze(1)


class CategorySpecificMLP(nn.Module):
    def __init__(self, num_categories, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.num_categories = num_categories
        self.layer1 = CategorySpecificLinear(num_categories, input_dim, hidden_dim)
        self.layer2 = CategorySpecificLinear(num_categories, hidden_dim, output_dim)

    def forward(self, x, cat_ids):
        hidden = F.relu(self.layer1(x, cat_ids))
        return self.layer2(hidden, cat_ids)



class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        return self.layer2(F.relu(self.layer1(x)))


class ActionEncoder(nn.Module):
    def __init__(self, action_dim, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.action_dim = action_dim
        self.layer1 = nn.Linear(action_dim, hidden_size)
        self.layer2 = nn.Linear(2 * hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, hidden_size)
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_size)

    def forward(self, actions, timesteps):
        """
        actions:   shape (B, T, action_dim)
        timesteps: shape (B,)  -- a single scalar per batch item
        returns:   shape (B, T, hidden_size)
        """
        B, T, _ = actions.shape

        # 1) Expand each batch's single scalar time 'tau' across all T steps
        #    so that shape => (B, T)
        #    e.g. if timesteps is (B,), replicate across T
        if timesteps.dim() == 1 and timesteps.shape[0] == B:
            # shape (B,) => (B,T)
            timesteps = timesteps.unsqueeze(1).expand(-1, T)
        else:
            raise ValueError(
                "Expected `timesteps` to have shape (B,) so we can replicate across T."
            )

        # 2) Standard action MLP step for shape => (B, T, w)
        a_emb = self.layer1(actions)

        # 3) Get the sinusoidal encoding (B, T, w)
        tau_emb = self.pos_encoding(timesteps).to(dtype=a_emb.dtype)

        # 4) Concat along last dim => (B, T, 2w), then layer2 => (B, T, w), swish
        x = torch.cat([a_emb, tau_emb], dim=-1)
        x = swish(self.layer2(x))

        # 5) Finally W3 => (B, T, w)
        x = self.layer3(x)
        return x



class MultiEmbodimentActionEncoder(nn.Module):
    def __init__(self, action_dim, hidden_size, num_embodiments):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_embodiments = num_embodiments

        # W1: R^{w x d}, W2: R^{w x 2w}, W3: R^{w x w}
        self.W1 = CategorySpecificLinear(num_embodiments, action_dim, hidden_size)  # (d -> w)
        self.W2 = CategorySpecificLinear(num_embodiments, 2 * hidden_size, hidden_size)  # (2w -> w)
        self.W3 = CategorySpecificLinear(num_embodiments, hidden_size, hidden_size)  # (w -> w)
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_size)

    def forward(self, actions, timesteps, cat_ids):
        """
        actions:   shape (B, T, action_dim)
        timesteps: shape (B,)  -- a single scalar per batch item
        cat_ids:   shape (B,)
        returns:   shape (B, T, hidden_size)
        """
        B, T, _ = actions.shape

        # 1) Expand each batch's single scalar time 'tau' across all T steps
        #    so that shape => (B, T)
        #    e.g. if timesteps is (B,), replicate across T
        if timesteps.dim() == 1 and timesteps.shape[0] == B:
            # shape (B,) => (B,T)
            timesteps = timesteps.unsqueeze(1).expand(-1, T)
        else:
            raise ValueError(
                "Expected `timesteps` to have shape (B,) so we can replicate across T."
            )

        # 2) Standard action MLP step for shape => (B, T, w)
        a_emb = self.W1(actions, cat_ids)

        # 3) Get the sinusoidal encoding (B, T, w)
        tau_emb = self.pos_encoding(timesteps).to(dtype=a_emb.dtype)

        # 4) Concat along last dim => (B, T, 2w), then W2 => (B, T, w), swish
        x = torch.cat([a_emb, tau_emb], dim=-1)
        x = swish(self.W2(x, cat_ids))

        # 5) Finally W3 => (B, T, w)
        x = self.W3(x, cat_ids)
        return x


@dataclass
class FlowmatchingActionHeadConfig(PretrainedConfig):
    """NOTE: N1.5 uses XEmbFlowmatchingPolicyHeadConfig as action head"""

    add_pos_embed: bool = field(
        default=True, metadata={"help": "Whether to add positional embedding"}
    )
    diffusion_model_cfg: dict = field(
        default=None, metadata={"help": "Diffusion model configuration."}
    )
    input_embedding_dim: int = field(
        default=1536, metadata={"help": "Input embedding channel dimension."}
    )

    hidden_size: int = field(default=1024, metadata={"help": "Input embedding dimension."})
    max_seq_len: int = field(default=1024, metadata={"help": "Maxium Sequence Length"})
    action_dim: int = field(default=None, metadata={"help": "Action dimension."})
    action_horizon: int = field(default=None, metadata={"help": "Action horizon."})
    noise_beta_alpha: float = field(default=1.5, metadata={"help": ""})
    noise_beta_beta: float = field(default=1.0, metadata={"help": ""})
    noise_s: float = field(
        default=0.999, metadata={"help": "Flow matching noise Beta distribution s."}
    )
    num_timestep_buckets: int = field(
        default=1000, metadata={"help": "Number of timestep discretization buckets."}
    )
    num_inference_timesteps: int = field(
        default=None,
        metadata={"help": "Number of inference steps for noise diffusion."},
    )
    max_num_embodiments: int = field(default=32, metadata={"help": "Number of embodiments."})
    tune_projector: bool = field(default=True, metadata={"help": "Whether to tune the projector."})
    tune_diffusion_model: bool = field(
        default=True, metadata={"help": "Whether to tune the diffusion model."}
    )
    load_pretrained_det_decode_layer_path: str = field(
        default=None, metadata={"help": "Path to pretrained detection model."}
    )
    detection_coeff: float = field(default=1.0, metadata={"help": "Detection coefficient."})

    freeze_decode_layer: bool = field(default=False)
    expand_batch: int = field(default=None)
    use_vlln: bool = field(default=True)

    vl_self_attention_cfg: dict = field(default=None)
    num_target_vision_tokens: int = field(
        default=32, metadata={"help": "Number of target vision tokens."}
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)


DiTConfig = {
    "DiT-B": {"input_embedding_dim": 768, "attention_head_dim": 64, "num_attention_heads": 12},
    "DiT-L": {"input_embedding_dim": 1536, "attention_head_dim": 48, "num_attention_heads": 32},
}

class LayerwiseFlowmatchingActionHead(nn.Module):
    def __init__(
        self,
        full_config,
        **kwargs,
    ):
        super().__init__()
        config = full_config.framework.action_model
        self.hidden_size = config.hidden_size
        self.full_config = full_config

        action_model_cfg = full_config.framework.action_model.DiTConfig
        self.input_embedding_dim = action_model_cfg["input_embedding_dim"]

        diffusion_model_cfg = config.diffusion_model_cfg
        diffusion_model_cfg = {**action_model_cfg, **diffusion_model_cfg}
        self.model = DiT(**diffusion_model_cfg) # TODO better way is copy LLM from VLM
        self.action_dim = config.action_dim
        self.action_horizon = config.action_horizon
        self.num_inference_timesteps = config.num_inference_timesteps

        qwen_input_dim = int(
            config.get(
                "qwen_input_dim",
                full_config.framework.qwenvl.get("vl_hidden_dim", 2048),
            )
        )

        self.qwen_proj = MLP(
            input_dim = qwen_input_dim,
            hidden_dim = self.hidden_size,
            output_dim = self.hidden_size
        )
        self.qwen_input_dim = qwen_input_dim
        self.use_global_scene_tokens = bool(
            config.get("use_global_scene_tokens", False)
        )
        self.scene_dim = int(config.get("scene_dim", 2048))
        # Old QwenPI/QwenFM configs do not construct this module, preserving
        # their state-dict and numerical behavior.  The new joint framework
        # enables it explicitly.
        self.scene_proj = (
            MLP(
                input_dim=self.scene_dim,
                hidden_dim=self.hidden_size,
                output_dim=self.hidden_size,
            )
            if self.use_global_scene_tokens
            else None
        )

        self.state_encoder = MLP(
            input_dim=config.state_dim,
            hidden_dim=self.hidden_size,
            output_dim=self.input_embedding_dim,
        ) if config.state_dim else None

        self.action_encoder = ActionEncoder(
            action_dim=config.action_dim,
            hidden_size=self.input_embedding_dim,
        )
        self.action_decoder = MLP(
            input_dim=self.hidden_size,
            hidden_dim=self.hidden_size,
            output_dim=self.action_dim,
        )
        self.future_tokens = nn.Embedding(config.num_target_vision_tokens, self.input_embedding_dim)
        nn.init.normal_(self.future_tokens.weight, mean=0.0, std=0.02)

        if config.add_pos_embed:
            self.position_embedding = nn.Embedding(config.max_seq_len, self.input_embedding_dim)
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

        self.beta_dist = Beta(config.noise_beta_alpha, config.noise_beta_beta)
        self.num_timestep_buckets = config.num_timestep_buckets
        self.config = config

    def sample_time(self, batch_size, device, dtype):
        sample = self.beta_dist.sample([batch_size]).to(device, dtype=dtype)
        return (self.config.noise_s - sample) / self.config.noise_s

    def prepare_input(self, batch: dict) -> BatchFeature:
        return BatchFeature(data=batch)


    def _validate_condition_inputs(
        self,
        vl_embs_list: list,
        global_scene_tokens: torch.Tensor = None,
    ) -> None:
        expected_layers = len(self.model.transformer_blocks)
        if len(vl_embs_list) != expected_layers:
            raise ValueError(
                f"vl_embs_list contains {len(vl_embs_list)} layers, expected "
                f"exactly {expected_layers} DiT conditioning layers"
            )
        if not vl_embs_list:
            raise ValueError("vl_embs_list cannot be empty")
        reference = vl_embs_list[0]
        if reference.ndim != 3 or reference.shape[-1] != self.qwen_input_dim:
            raise ValueError(
                f"vl_embs_list[0] must have shape [B,L,{self.qwen_input_dim}], "
                f"got {tuple(reference.shape)}"
            )
        for index, memory in enumerate(vl_embs_list):
            if memory.ndim != 3 or memory.shape[0] != reference.shape[0] or memory.shape[-1] != self.qwen_input_dim:
                raise ValueError(
                    f"vl_embs_list[{index}] has shape {tuple(memory.shape)}; "
                    f"expected [B,L,{self.qwen_input_dim}] with B={reference.shape[0]}"
                )
            if memory.device != reference.device or memory.dtype != reference.dtype:
                raise ValueError(
                    f"vl_embs_list[{index}] device/dtype {memory.device}/{memory.dtype} "
                    f"does not match {reference.device}/{reference.dtype}"
                )
        if global_scene_tokens is not None:
            expected = (reference.shape[0], self.scene_dim)
            if global_scene_tokens.ndim != 3 or (
                global_scene_tokens.shape[0] != expected[0]
                or global_scene_tokens.shape[-1] != expected[1]
            ):
                raise ValueError(
                    "global_scene_tokens must have shape "
                    f"[B,Q,{self.scene_dim}] with B={reference.shape[0]}, got "
                    f"{tuple(global_scene_tokens.shape)}"
                )
            if (
                global_scene_tokens.device != reference.device
                or global_scene_tokens.dtype != reference.dtype
            ):
                raise ValueError(
                    "global_scene_tokens and layerwise Qwen memory must share "
                    "device and dtype"
                )
            if not self.use_global_scene_tokens:
                raise ValueError(
                    "global_scene_tokens were provided but action_model."
                    "use_global_scene_tokens is false"
                )

    def _build_condition_memory(
        self,
        layerwise_memory: torch.Tensor,
        global_scene_tokens: torch.Tensor = None,
    ) -> torch.Tensor:
        """Append projected scene queries to one existing Qwen memory layer."""

        action_condition = self.qwen_proj(layerwise_memory)
        if global_scene_tokens is None or not self.use_global_scene_tokens:
            return action_condition
        if self.scene_proj is None:
            raise RuntimeError("scene projection is not initialized")
        scene_condition = self.scene_proj(global_scene_tokens)
        return torch.cat((action_condition, scene_condition), dim=1)

    def _project_condition_memories(
        self,
        vl_embs_list: list,
        global_scene_tokens: torch.Tensor = None,
    ) -> list:
        """Project time-invariant memories once for all DiT/Euler steps."""

        self._validate_condition_inputs(vl_embs_list, global_scene_tokens)
        qwen_conditions = [self.qwen_proj(memory) for memory in vl_embs_list]
        if global_scene_tokens is None or not self.use_global_scene_tokens:
            return qwen_conditions
        if self.scene_proj is None:
            raise RuntimeError("scene projection is not initialized")
        scene_condition = self.scene_proj(global_scene_tokens)
        return [
            torch.cat((action_condition, scene_condition), dim=1)
            for action_condition in qwen_conditions
        ]

    def forward(
        self,
        vl_embs_list: list,
        actions: torch.Tensor,
        state: torch.Tensor = None,
        global_scene_tokens: torch.Tensor = None,
    ):
        """
        vl_embs: list of torch.Tensor, each shape (B, seq_length, feature_dim)
        actions: shape (B, future_action_window_size, D_action)
        """
        device = actions.device
        self._validate_condition_inputs(vl_embs_list, global_scene_tokens)
        B = vl_embs_list[0].shape[0]
        if actions.ndim != 3 or actions.shape[0] != B or actions.shape[-1] != self.action_dim:
            raise ValueError(
                f"actions must have shape [B,T,{self.action_dim}] with B={B}, "
                f"got {tuple(actions.shape)}"
            )
        condition_memories = self._project_condition_memories(
            vl_embs_list, global_scene_tokens
        )
        # Embed noised action trajectory.
        noise = torch.randn(actions.shape, device=actions.device, dtype=actions.dtype)
        t = self.sample_time(actions.shape[0], device=actions.device, dtype=actions.dtype)
        t = t[:, None, None]  # shape (B,1,1) for broadcast

        noisy_trajectory = (1 - t) * noise + t * actions
        velocity = actions - noise

        # Convert (continuous) t -> discrete if needed
        t_discretized = (t[:, 0, 0] * self.num_timestep_buckets).long()
        action_features = self.action_encoder(noisy_trajectory, t_discretized)

        # Embed statex
        state_features = self.state_encoder(state) if state is not None else None

        # Maybe add position embedding.
        if self.config.add_pos_embed:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
            pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
            action_features = action_features + pos_embs

        # state and action embedding along sequence dimension.
        future_tokens = self.future_tokens.weight.unsqueeze(0).expand(B, -1, -1)
        sa_embs = torch.cat((state_features, future_tokens, action_features), dim=1) \
            if state_features is not None else torch.cat((future_tokens, action_features), dim=1)
        
        # Encode timesteps
        temb = self.model.timestep_encoder(t_discretized)

        # Layerwise cross-attention with vl_embs
        model_output = sa_embs
        # both length 16
        for layer_idx, layer in enumerate(self.model.transformer_blocks):
            model_output = layer(
                hidden_states=model_output,
                    encoder_hidden_states=condition_memories[layer_idx],
                temb=temb,
            )

            # hidden_states = block(
            #         hidden_states,
            #         attention_mask=None,
            #         encoder_hidden_states=encoder_hidden_states,
            #         encoder_attention_mask=None,
            #         temb=temb,
            #     )
            
        # Output processing
        conditioning = temb
        shift, scale = self.model.proj_out_1(F.silu(conditioning)).chunk(2, dim=1)
        model_output = self.model.norm_out(model_output) * (1 + scale[:, None]) + shift[:, None]
        model_output = self.model.proj_out_2(model_output)

        pred = self.action_decoder(model_output)
        pred_actions = pred[:, -actions.shape[1] :]

        # Slice out only the action portion of pred and target.
        loss = ((pred_actions - velocity) ** 2).mean()
        return loss

    def _euler_sample(
        self,
        actions: torch.Tensor,
        condition_memories: list,
        state_features: torch.Tensor = None,
    ) -> torch.Tensor:
        """Run the unchanged Euler integration from an explicit noise tensor."""

        batch_size = actions.shape[0]
        device = actions.device
        num_steps = self.num_inference_timesteps
        dt = 1.0 / num_steps

        # Run denoising steps.
        for t in range(num_steps):
            t_cont = t / float(num_steps)
            t_discretized_int = int(t_cont * self.num_timestep_buckets)
            timesteps_tensor = torch.full(
                size=(batch_size,), fill_value=t_discretized_int, device=device, dtype=torch.long
            )

            # Embed current action trajectory with timestep
            action_features = self.action_encoder(actions, timesteps_tensor)

            # Maybe add position embedding.
            if self.config.add_pos_embed:
                pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
                pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
                action_features = action_features + pos_embs

            future_tokens = self.future_tokens.weight.unsqueeze(0).expand(batch_size, -1, -1)
            sa_embs = (
                torch.cat((state_features, future_tokens, action_features), dim=1)
                if state_features is not None
                else torch.cat((future_tokens, action_features), dim=1)
            )

            # Encode timestep
            temb = self.model.timestep_encoder(timesteps_tensor)

            # Layerwise cross-attention with vl_embs_list
            model_output = sa_embs
            for layer_idx, layer in enumerate(self.model.transformer_blocks):
                model_output = layer(
                    hidden_states=model_output,
                    encoder_hidden_states=condition_memories[layer_idx],
                    temb=temb,
                )
            
            # Output processing
            conditioning = temb
            shift, scale = self.model.proj_out_1(F.silu(conditioning)).chunk(2, dim=1)
            model_output = self.model.norm_out(model_output) * (1 + scale[:, None]) + shift[:, None]
            model_output = self.model.proj_out_2(model_output)

            pred = self.action_decoder(model_output)
            pred_velocity = pred[:, -self.action_horizon :]

            # Euler integration
            actions = actions + dt * pred_velocity
        return actions

    @torch.no_grad()
    def predict_multi_action(
        self,
        vl_embs_list: list,
        state: torch.Tensor = None,
        global_scene_tokens: torch.Tensor = None,
        num_candidates: int = 64,
        candidate_chunk_size: int = 8,
        initial_noise: torch.Tensor = None,
    ) -> torch.Tensor:
        """Sample independent Flow trajectories with one condition projection.

        Args:
            vl_embs_list: Existing per-DiT-layer Qwen memories ``[B,L,Dq]``.
            state: Optional ego tensor ``[B,1,state_dim]``.
            global_scene_tokens: Optional trainable scene tokens ``[B,Q,Dscene]``.
            initial_noise: Optional deterministic ``[B,K,T,Daction]`` tensor.

        Returns:
            Normalized Flow actions with shape ``[B,K,T,Daction]``.
        """

        if num_candidates <= 0 or candidate_chunk_size <= 0:
            raise ValueError("num_candidates and candidate_chunk_size must be positive")
        self._validate_condition_inputs(vl_embs_list, global_scene_tokens)
        batch_size = vl_embs_list[0].shape[0]
        reference = vl_embs_list[0]
        expected_noise_shape = (
            batch_size,
            num_candidates,
            self.action_horizon,
            self.action_dim,
        )
        if initial_noise is None:
            initial_noise = torch.randn(
                expected_noise_shape,
                device=reference.device,
                dtype=reference.dtype,
            )
        elif tuple(initial_noise.shape) != expected_noise_shape:
            raise ValueError(
                f"initial_noise has shape {tuple(initial_noise.shape)}, "
                f"expected {expected_noise_shape}"
            )
        elif initial_noise.device != reference.device or initial_noise.dtype != reference.dtype:
            raise ValueError(
                "initial_noise and Qwen condition memory must share device and dtype"
            )

        condition_memories = self._project_condition_memories(
            vl_embs_list, global_scene_tokens
        )
        state_features = self.state_encoder(state) if state is not None else None

        def expand_candidates(value: torch.Tensor, count: int) -> torch.Tensor:
            return value[:, None].expand(
                batch_size, count, *value.shape[1:]
            ).reshape(batch_size * count, *value.shape[1:])

        chunks = []
        for start in range(0, num_candidates, candidate_chunk_size):
            stop = min(num_candidates, start + candidate_chunk_size)
            count = stop - start
            chunk_noise = initial_noise[:, start:stop].reshape(
                batch_size * count, self.action_horizon, self.action_dim
            )
            chunk_conditions = [
                expand_candidates(memory, count) for memory in condition_memories
            ]
            chunk_state = (
                None
                if state_features is None
                else expand_candidates(state_features, count)
            )
            sampled = self._euler_sample(
                chunk_noise, chunk_conditions, chunk_state
            ).reshape(
                batch_size, count, self.action_horizon, self.action_dim
            )
            chunks.append(sampled)
        return torch.cat(chunks, dim=1)

    @torch.no_grad()
    def predict_action(
        self,
        vl_embs_list: list,
        state: torch.Tensor = None,
        global_scene_tokens: torch.Tensor = None,
    ) -> torch.Tensor:
        """Backward-compatible single sample through the shared multi sampler."""

        return self.predict_multi_action(
            vl_embs_list=vl_embs_list,
            state=state,
            global_scene_tokens=global_scene_tokens,
            num_candidates=1,
            candidate_chunk_size=1,
        )[:, 0]

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype



def get_action_model(config=None):
    """
    Factory: build FlowmatchingActionHead from global framework config.
    
    Args:
        config: Global config (expects config.framework.action_model namespace).

    Returns:
        FlowmatchingActionHead: Initialized FlowMatchingActionHead.
    """
    return LayerwiseFlowmatchingActionHead(
        full_config=config
    )



if __name__ == "__main__":
    # TODO make each backbone.py can be debug independently

    pass

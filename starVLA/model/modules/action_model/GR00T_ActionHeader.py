# Copyright 2025 NVIDIA Corp. and affiliates. All rights reserved.
# Modified by [Junqiu YU/ Fudan University] in [2025]. 
# Modification: [rm and add some connect adapter to match with starVLA, e.g., "rm "].
# Action repeat is inspired by CogACT



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

from starVLA.model.modules.action_model.flow_matching_head.cross_attention_dit import DiT

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

class FlowmatchingActionHead(nn.Module):
    def __init__(
        self,
        full_config,
    ):
        super().__init__()
        config = full_config.framework.action_model
        self.hidden_size = config.hidden_size # 是不要和 Q对齐？
        self.full_config = full_config

        action_model_cfg = full_config.framework.action_model.DiTConfig
        self.input_embedding_dim = action_model_cfg["input_embedding_dim"]

        diffusion_model_cfg = config.diffusion_model_cfg
        diffusion_model_cfg = {**action_model_cfg, **diffusion_model_cfg}
        self.model = DiT(**diffusion_model_cfg)
        self.action_dim = config.action_dim
        self.action_horizon = config.action_horizon
        self.num_inference_timesteps = config.num_inference_timesteps

        # self.state_encoder = MLP(
        #     input_dim=config.state_dim,
        #     hidden_dim=self.hidden_size,
        #     output_dim=self.input_embedding_dim,
        # ) if config.state_dim else None

        self.qwen_proj = MLP(
            input_dim = full_config.framework.qwenvl.vl_hidden_dim,
            hidden_dim = self.hidden_size,
            output_dim = self.hidden_size
        )
        self.use_global_scene_tokens = bool(
            config.get("use_global_scene_tokens", False)
        )
        self.scene_dim = int(config.get("scene_dim", 256))
        # This module exists only for the baseline-matched scene variants.  It
        # is absent for the legacy QwenOFT configuration, preserving that
        # model's state dict and numerical path.
        self.scene_to_dit = (
            nn.Sequential(
                nn.Linear(self.scene_dim, self.hidden_size),
                nn.LayerNorm(self.hidden_size),
            )
            if self.use_global_scene_tokens
            else None
        )

        self.action_encoder = ActionEncoder(
            action_dim=config.action_dim,
            hidden_size=self.input_embedding_dim,
        )
        self.action_decoder = MLP(
            input_dim=self.hidden_size,
            hidden_dim=self.hidden_size,
            output_dim=self.action_dim,
        )
        # self.future_tokens = nn.Embedding(config.num_target_vision_tokens, self.input_embedding_dim)
        # nn.init.normal_(self.future_tokens.weight, mean=0.0, std=0.02)

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

    def _project_condition(
        self,
        vl_embs: torch.Tensor,
        global_scene_tokens: torch.Tensor = None,
        video_token: torch.Tensor = None,
    ) -> torch.Tensor:
        """Project the baseline action queries and append one shared scene memory."""

        if vl_embs.ndim != 3:
            raise ValueError("vl_embs must have shape [B,T,H]")
        condition = self.qwen_proj(vl_embs)
        if video_token is not None:
            if video_token.ndim != 3 or video_token.shape[0] != condition.shape[0]:
                raise ValueError("video_token must have shape [B,S,hidden_size]")
            condition = torch.cat((condition, video_token), dim=1)
        if self.use_global_scene_tokens:
            if global_scene_tokens is None:
                raise ValueError(
                    "use_global_scene_tokens=true requires global_scene_tokens"
                )
            if (
                global_scene_tokens.ndim != 3
                or global_scene_tokens.shape[0] != condition.shape[0]
                or global_scene_tokens.shape[-1] != self.scene_dim
            ):
                raise ValueError(
                    "global_scene_tokens must have shape "
                    f"[B,Q,{self.scene_dim}]"
                )
            if self.scene_to_dit is None:
                raise RuntimeError("scene_to_dit is not initialized")
            # Calculate once per Flow forward/Euler trajectory and reuse in all
            # 24 unchanged DiT blocks through encoder_hidden_states.
            scene_condition = self.scene_to_dit(global_scene_tokens)
            condition = torch.cat((condition, scene_condition), dim=1)
        elif global_scene_tokens is not None:
            raise ValueError(
                "global_scene_tokens were provided while scene injection is disabled"
            )
        return condition


    def forward(
        self,
        vl_embs: torch.Tensor,
        actions: torch.Tensor,
        video_token=None,
        state: torch.Tensor = None,
        global_scene_tokens: torch.Tensor = None,
    ):
        """
        vl_embs: shape (B, seq_length, feature_dim)
        actions: shape (B, future_action_window_size, D_action)
        """
        device = vl_embs.device

        # Embed noised action trajectory.
        noise = torch.randn(actions.shape, device=actions.device, dtype=actions.dtype)
        t = self.sample_time(actions.shape[0], device=actions.device, dtype=actions.dtype)
        t = t[:, None, None]  # shape (B,1,1) for broadcast

        noisy_trajectory = (1 - t) * noise + t * actions
        velocity = actions - noise

        # Convert (continuous) t -> discrete if needed
        t_discretized = (t[:, 0, 0] * self.num_timestep_buckets).long()
        action_features = self.action_encoder(noisy_trajectory, t_discretized)


        vl_embs = self._project_condition(
            vl_embs,
            global_scene_tokens=global_scene_tokens,
            video_token=video_token,
        )


        # Maybe add position embedding.
        if self.config.add_pos_embed:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
            pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
            action_features = action_features + pos_embs

        # state and action embedding along sequence dimension.
        # future_tokens = self.future_tokens.weight.unsqueeze(0).expand(vl_embs.shape[0], -1, -1)
        # sa_embs = torch.cat((state_features, future_tokens, action_features), dim=1) \
        #     if state_features is not None else torch.cat((future_tokens, action_features), dim=1)

        sa_embs = action_features

        # Join VLM features with state and action embedding along sequence dimension.
        model_output = self.model(
            hidden_states=sa_embs,
            encoder_hidden_states=vl_embs,
            timestep=t_discretized,
            return_all_hidden_states=False,  # NOTE (YL): not using flare now
        )
        pred = self.action_decoder(model_output)

        pred_actions = pred

        # Slice out only the action portion of pred and target.
        loss = ((pred_actions - velocity) ** 2).mean()
        return loss

    def _euler_sample(
        self,
        actions: torch.Tensor,
        projected_condition: torch.Tensor,
    ) -> torch.Tensor:
        """Run the baseline Euler integrator from an explicit noise tensor."""

        batch_size = actions.shape[0]
        device = actions.device
        dt = 1.0 / self.num_inference_timesteps
        for step in range(self.num_inference_timesteps):
            t_cont = step / float(self.num_inference_timesteps)
            t_discretized = int(t_cont * self.num_timestep_buckets)
            timesteps_tensor = torch.full(
                (batch_size,),
                fill_value=t_discretized,
                device=device,
                dtype=torch.long,
            )
            action_features = self.action_encoder(actions, timesteps_tensor)
            if self.config.add_pos_embed:
                pos_ids = torch.arange(
                    action_features.shape[1], dtype=torch.long, device=device
                )
                action_features = action_features + self.position_embedding(
                    pos_ids
                ).unsqueeze(0)
            model_output = self.model(
                hidden_states=action_features,
                encoder_hidden_states=projected_condition,
                timestep=timesteps_tensor,
            )
            actions = actions + dt * self.action_decoder(model_output)
        return actions

    @torch.no_grad()
    def predict_multi_action(
        self,
        vl_embs: torch.Tensor,
        state: torch.Tensor = None,
        global_scene_tokens: torch.Tensor = None,
        num_candidates: int = 64,
        candidate_chunk_size: int = 8,
        initial_noise: torch.Tensor = None,
    ) -> torch.Tensor:
        """Generate ``K`` full trajectories without changing results by chunking."""

        if state is not None:
            # Ego state already enters the local baseline through the Qwen
            # history token.  It is accepted only for call-site compatibility.
            if state.shape[0] != vl_embs.shape[0]:
                raise ValueError("state and Qwen condition batch sizes differ")
        if num_candidates <= 0 or candidate_chunk_size <= 0:
            raise ValueError("candidate counts and chunk size must be positive")
        batch_size = vl_embs.shape[0]
        if initial_noise is None:
            initial_noise = torch.randn(
                batch_size,
                num_candidates,
                self.action_horizon,
                self.action_dim,
                dtype=vl_embs.dtype,
                device=vl_embs.device,
            )
        expected = (
            batch_size,
            num_candidates,
            self.action_horizon,
            self.action_dim,
        )
        if tuple(initial_noise.shape) != expected:
            raise ValueError(
                f"initial_noise has shape {tuple(initial_noise.shape)}, expected {expected}"
            )
        projected = self._project_condition(
            vl_embs, global_scene_tokens=global_scene_tokens
        )
        outputs = []
        for start in range(0, num_candidates, candidate_chunk_size):
            stop = min(start + candidate_chunk_size, num_candidates)
            chunk_count = stop - start
            chunk_condition = projected[:, None].expand(
                -1, chunk_count, -1, -1
            ).reshape(
                batch_size * chunk_count,
                projected.shape[1],
                projected.shape[2],
            )
            chunk_noise = initial_noise[:, start:stop].reshape(
                batch_size * chunk_count,
                self.action_horizon,
                self.action_dim,
            )
            outputs.append(
                self._euler_sample(chunk_noise, chunk_condition).reshape(
                    batch_size,
                    chunk_count,
                    self.action_horizon,
                    self.action_dim,
                )
            )
        return torch.cat(outputs, dim=1)

    @torch.no_grad()
    def predict_action(
        self,
        vl_embs: torch.Tensor,
        state: torch.Tensor = None,
        global_scene_tokens: torch.Tensor = None,
    ) -> torch.Tensor:
        return self.predict_multi_action(
            vl_embs,
            state=state,
            global_scene_tokens=global_scene_tokens,
            num_candidates=1,
            candidate_chunk_size=1,
        )[:, 0]

    def predict_action_onestep(self, vl_embs: torch.Tensor, actions, timesteps_tensor) -> torch.Tensor:
        # Set initial actions as the sampled noise.
        batch_size = vl_embs.shape[0]
        device = vl_embs.device
        # actions = torch.randn(
        #     size=(batch_size, self.config.action_horizon, self.config.action_dim),
        #     dtype=vl_embs.dtype,
        #     device=device,
        # )

        # num_steps = self.num_inference_timesteps
        # dt = 1.0 / num_steps
        
        state_features = None

        # vl_embs = self.qwen_proj(vl_embs)

        # Run denoising steps.
        # for t in range(num_steps):

        # t_cont = t / float(num_steps)  # e.g. goes 0, 1/N, 2/N, ...
        # t_discretized = int(t_cont * self.num_timestep_buckets)

        # Embed noised action trajectory.
        # timesteps_tensor = torch.full(
        #     size=(batch_size,), fill_value=t_discretized, device=device
        # )
        action_features = self.action_encoder(actions, timesteps_tensor)
        # Maybe add position embedding.
        if self.config.add_pos_embed:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
            pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
            action_features = action_features + pos_embs

        # Join vision, language, state and action embedding along sequence dimension.
        # future_tokens = self.future_tokens.weight.unsqueeze(0).expand(vl_embs.shape[0], -1, -1)
        # sa_embs = torch.cat((state_features, future_tokens, action_features), dim=1) \
        #     if state_features is not None else torch.cat((future_tokens, action_features), dim=1)

        sa_embs = action_features

        # Run model forward.
        model_output = self.model(
            hidden_states=sa_embs,
            encoder_hidden_states=vl_embs,
            timestep=timesteps_tensor,
        )

        pred = self.action_decoder(model_output)

        return pred # will deal outside

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype

    def forward_onestep(
            self,
            vl_embs: torch.Tensor,
            # actions: torch.Tensor,
            # t,
            noisy_trajectory,
            input_t_discretized,
            state: torch.Tensor = None
        ):
        """
        vl_embs: shape (B, seq_length, feature_dim)
        actions: shape (B, future_action_window_size, D_action)
        """
        device = vl_embs.device

        # Embed noised action trajectory.
        # noise = torch.randn(actions.shape, device=actions.device, dtype=actions.dtype)
        # t = self.sample_time(actions.shape[0], device=actions.device, dtype=actions.dtype)
        # t = t[:, None, None]  # shape (B,1,1) for broadcast

        # noisy_trajectory = (1 - t) * noise + t * actions
        # velocity = actions - noise

        # Convert (continuous) t -> discrete if needed
        # t_discretized = (t[:, 0, 0] * self.num_timestep_buckets).long()
        action_features = self.action_encoder(noisy_trajectory, input_t_discretized)


        # embed state
        state_features = self.state_encoder(state) if state is not None else None


        # vl_embs = self.qwen_proj(vl_embs)


        # Maybe add position embedding.
        if self.config.add_pos_embed:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
            pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
            action_features = action_features + pos_embs

        # state and action embedding along sequence dimension.
        # future_tokens = self.future_tokens.weight.unsqueeze(0).expand(vl_embs.shape[0], -1, -1)
        # sa_embs = torch.cat((state_features, future_tokens, action_features), dim=1) \
        #     if state_features is not None else torch.cat((future_tokens, action_features), dim=1)

        sa_embs = action_features

        # Join VLM features with state and action embedding along sequence dimension.
        model_output = self.model(
            hidden_states=sa_embs,
            encoder_hidden_states=vl_embs,
            timestep=input_t_discretized,
            return_all_hidden_states=False,  # NOTE (YL): not using flare now
        )
        pred = self.action_decoder(model_output)

        pred_actions = pred

        # Slice out only the action portion of pred and target.
        # loss = ((pred_actions - velocity) ** 2).mean()
        # return loss

        return [pred_actions]   #for outside func



class FlowmatchingRewardHead(nn.Module):
    def __init__(
        self,
        full_config,
    ):
        super().__init__()
        config = full_config.framework.reward_model
        self.hidden_size = config.hidden_size # 是不要和 Q对齐？
        self.full_config = full_config

        action_model_cfg = full_config.framework.reward_model.DiTConfig
        self.input_embedding_dim = action_model_cfg["input_embedding_dim"]

        diffusion_model_cfg = config.diffusion_model_cfg
        diffusion_model_cfg = {**action_model_cfg, **diffusion_model_cfg}
        self.model = DiT(**diffusion_model_cfg)
        self.action_dim = config.action_dim
        self.action_horizon = config.action_horizon
        self.num_inference_timesteps = config.num_inference_timesteps

        # self.state_encoder = MLP(
        #     input_dim=config.state_dim,
        #     hidden_dim=self.hidden_size,
        #     output_dim=self.input_embedding_dim,
        # ) if config.state_dim else None

        self.qwen_proj = MLP(
            input_dim = 2048,
            hidden_dim = self.hidden_size,
            output_dim = self.hidden_size
        )

        self.action_encoder = ActionEncoder(
            action_dim=config.action_dim,
            hidden_size=self.input_embedding_dim,
        )
        self.action_decoder = MLP(
            input_dim=self.hidden_size,
            hidden_dim=self.hidden_size,
            output_dim=self.action_dim,
        )
        # self.future_tokens = nn.Embedding(config.num_target_vision_tokens, self.input_embedding_dim)
        # nn.init.normal_(self.future_tokens.weight, mean=0.0, std=0.02)

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


    def forward(self, vl_embs: torch.Tensor, actions: torch.Tensor, state: torch.Tensor = None):
        """
        vl_embs: shape (B, seq_length, feature_dim)
        actions: shape (B, future_action_window_size, D_action)
        """
        device = vl_embs.device

        actions = torch.from_numpy(actions).to(device).reshape(-1, self.config.action_horizon, 1)

        # Embed noised action trajectory.
        noise = torch.randn(actions.shape, device=actions.device, dtype=actions.dtype)
        t = self.sample_time(actions.shape[0], device=actions.device, dtype=actions.dtype)
        t = t[:, None, None]  # shape (B,1,1) for broadcast

        noisy_trajectory = (1 - t) * noise + t * actions
        velocity = actions - noise

        # Convert (continuous) t -> discrete if needed
        t_discretized = (t[:, 0, 0] * self.num_timestep_buckets).long()
        action_features = self.action_encoder(noisy_trajectory, t_discretized)


        # embed state
        state_features = self.state_encoder(state) if state is not None else None


        vl_embs = self.qwen_proj(vl_embs)


        # Maybe add position embedding.
        if self.config.add_pos_embed:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
            pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
            action_features = action_features + pos_embs

        # state and action embedding along sequence dimension.
        # future_tokens = self.future_tokens.weight.unsqueeze(0).expand(vl_embs.shape[0], -1, -1)
        # sa_embs = torch.cat((state_features, future_tokens, action_features), dim=1) \
        #     if state_features is not None else torch.cat((future_tokens, action_features), dim=1)

        sa_embs = action_features

        # Join VLM features with state and action embedding along sequence dimension.
        model_output = self.model(
            hidden_states=sa_embs,
            encoder_hidden_states=vl_embs,
            timestep=t_discretized,
            return_all_hidden_states=False,  # NOTE (YL): not using flare now
        )

        pred = self.action_decoder(model_output)

        pred_actions = pred

        # Slice out only the action portion of pred and target.
        loss = ((pred_actions - velocity) ** 2).mean()
        return loss

    @torch.no_grad()
    def predict_action(self, vl_embs: torch.Tensor, state: torch.Tensor = None) -> torch.Tensor:
        # Set initial actions as the sampled noise.
        batch_size = vl_embs.shape[0]
        device = vl_embs.device
        actions = torch.randn(
            size=(batch_size, self.config.action_horizon, self.config.action_dim),
            dtype=vl_embs.dtype,
            device=device,
        )

        num_steps = self.num_inference_timesteps
        dt = 1.0 / num_steps
        
        state_features = self.state_encoder(state) if state is not None else None

        vl_embs = self.qwen_proj(vl_embs)

        # Run denoising steps.
        for t in range(num_steps):
            t_cont = t / float(num_steps)  # e.g. goes 0, 1/N, 2/N, ...
            t_discretized = int(t_cont * self.num_timestep_buckets)

            # Embed noised action trajectory.
            timesteps_tensor = torch.full(
                size=(batch_size,), fill_value=t_discretized, device=device
            )
            action_features = self.action_encoder(actions, timesteps_tensor)
            # Maybe add position embedding.
            if self.config.add_pos_embed:
                pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
                pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
                action_features = action_features + pos_embs

            # Join vision, language, state and action embedding along sequence dimension.
            # future_tokens = self.future_tokens.weight.unsqueeze(0).expand(vl_embs.shape[0], -1, -1)
            # sa_embs = torch.cat((state_features, future_tokens, action_features), dim=1) \
            #     if state_features is not None else torch.cat((future_tokens, action_features), dim=1)

            sa_embs = action_features

            # Run model forward.
            model_output = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embs,
                timestep=timesteps_tensor,
            )
            pred = self.action_decoder(model_output)

            pred_velocity = pred

            # Update actions using euler integration.
            actions = actions + dt * pred_velocity
        return actions

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
    return FlowmatchingActionHead(
        full_config=config
    )

def get_reward_model(config=None):
    """
    Factory: build FlowmatchingActionHead from global framework config.
    
    Args:
        config: Global config (expects config.framework.action_model namespace).

    Returns:
        FlowmatchingActionHead: Initialized FlowMatchingActionHead.
    """
    return FlowmatchingRewardHead(
        full_config=config
    )


if __name__ == "__main__":
    # TODO make each backbone.py can be debug independently

    pass

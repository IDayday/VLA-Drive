"""Architecture D: native Qwen3-VL text layers are the clean-x denoiser.

The native layers, RMSNorms, Q/K RMSNorms, GQA and rotary implementation remain
the installed Transformers implementation.  The segmented execution below is an
explicit D adaptation: native C causal attention, bidirectional P/A/R blocks and
A-only near-identity time FiLM.  It does not alter a global attention processor,
or reuse the Qwen3.5 Norm/cache implementation from AUGUstus.

The two residual sublayers in ``_layer`` follow
transformers/models/qwen3_vl/modeling_qwen3_vl.py (Transformers 4.57.0).
Copyright 2025 The Qwen Team and The HuggingFace Inc. team. Apache-2.0.
Local changes: functional immutable prefix K/V, per-role masks and A-only FiLM.
"""
from dataclasses import dataclass, replace
from typing import Any, Mapping, Optional, Tuple

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from starVLA.model.modules.qwen_world_postcot.actor import timestep_embedding


KV = Tuple[torch.Tensor, torch.Tensor]


def _select(tensor, indices, batch_axis=0):
    """A repeated single-scene group broadcasts K/V without materializing G copies."""
    if indices.ndim != 1 or indices.dtype != torch.long:
        raise ValueError('candidate_to_scene must be a one-dimensional long tensor')
    if not len(indices):
        return tensor.index_select(batch_axis, indices)
    if torch.all(indices == indices[0]):
        shape = list(tensor.shape)
        shape[batch_axis] = len(indices)
        return tensor.narrow(batch_axis, int(indices[0]), 1).expand(*shape)
    return tensor.index_select(batch_axis, indices)


@dataclass(frozen=True)
class DCurrentEncoding:
    """Current-only, pre-language representation; IL retains its visual graph.

    During RL the image/merger and base embeddings are frozen, so ``detached``
    may be used once per fresh collection.  Text C/P is never stored here.
    """
    inputs_embeds: torch.Tensor
    valid: torch.Tensor
    positions: torch.Tensor  # native [3,B,C] mRoPE position IDs
    visual_mask: torch.Tensor
    deepstack: Tuple[torch.Tensor, ...]
    ego: torch.Tensor
    scene_identity: Tuple[str, ...] = ()
    execution_identity: str = ''
    position_mask_identity: str = ''

    def detached(self):
        return replace(self, inputs_embeds=self.inputs_embeds.detach(),
                       deepstack=tuple(x.detach() for x in self.deepstack),
                       ego=self.ego.detach())


@dataclass(frozen=True)
class DPrefix:
    layer_kv: Tuple[KV, ...]
    valid: torch.Tensor
    next_positions: torch.Tensor  # [B], before A0
    reasoning: torch.Tensor
    scene_identity: Tuple[str, ...] = ()
    policy_version: str = ''
    execution_identity: str = ''
    position_mask_identity: str = ''
    architecture_id: str = 'D'

    def index_select(self, indices):
        names = tuple(self.scene_identity[int(i)] for i in indices) if self.scene_identity else ()
        return replace(self,
            layer_kv=tuple((_select(k, indices), _select(v, indices)) for k, v in self.layer_kv),
            valid=_select(self.valid, indices), next_positions=_select(self.next_positions, indices),
            reasoning=_select(self.reasoning, indices), scene_identity=names)

    def detached(self):
        return replace(self, layer_kv=tuple((k.detach(), v.detach()) for k, v in self.layer_kv),
                       reasoning=self.reasoning.detach())

    @property
    def logical_nbytes(self):
        return sum(x.numel()*x.element_size() for pair in self.layer_kv for x in pair)


class _OneLayerCache:
    """Per-call native attention cache adapter; never mutates parent prefix tensors.

    Qwen3VLTextAttention only calls ``update``.  A new object is created inside
    every checkpoint invocation, so backward recomputation cannot append twice.
    """
    def __init__(self, layer_index, prefix=None):
        self.layer_index, self.prefix, self.result = layer_index, prefix, None

    def update(self, key, value, layer_idx, cache_kwargs=None):
        if layer_idx != self.layer_index or self.result is not None:
            raise RuntimeError('Native Qwen layer/cache identity mismatch or duplicate write')
        if self.prefix is not None:
            key = torch.cat((self.prefix[0], key), dim=-2)
            value = torch.cat((self.prefix[1], value), dim=-2)
        self.result = key, value
        return self.result


class QwenFlowAdapter(nn.Module):
    """Unified Qwen current/P/A/R backbone without an independent actor.

    ``qwen_model`` is a native Qwen3VLModel, not a vocabulary-head wrapper.
    Public initialization must use the same sealed generic 2B checkpoint as A-C.
    Small native Qwen configs are accepted only when ``qualification=True``.
    """
    architecture_id = 'D'
    prediction_type = 'x0'
    action_tokens = 8
    action_dim = 3

    def __init__(self, qwen_model, *, time_dim=256, p_tokens=32, r_tokens=16,
                 ego_dim=8, activation_checkpointing=True, qualification=False,
                 training_queries=True, pack_shared_candidates=True,
                 checkpoint_short_suffix=True):
        super().__init__()
        if not hasattr(qwen_model, 'language_model') or not hasattr(qwen_model, 'visual'):
            raise ValueError('D requires the native multimodal Qwen3VLModel')
        text = qwen_model.config.text_config
        if not qualification and (text.num_hidden_layers != 28 or text.hidden_size != 2048):
            raise ValueError('Formal D uses the sealed Qwen3-VL-2B 28x2048 model')
        if p_tokens != 32 or r_tokens != 16 or time_dim != 256 or ego_dim != 8:
            raise ValueError('Keep formal P32/A8/R16, time256 and legal ego8')
        if text.attention_dropout != 0:
            raise ValueError('Qwen policy execution must have deterministic dropout=0')
        if text._attn_implementation not in ('sdpa', 'eager'):
            raise ValueError('Directional suffix masks require qualified native SDPA/eager')
        self.qwen_model = qwen_model
        self.hidden_dim, self.num_layers = text.hidden_size, text.num_hidden_layers
        self.pre_width = self.post_width = self.hidden_dim
        self.time_dim, self.p_tokens, self.r_tokens, self.ego_dim = time_dim, p_tokens, r_tokens, ego_dim
        self.activation_checkpointing = bool(activation_checkpointing)
        self.checkpoint_short_suffix = bool(checkpoint_short_suffix)
        self.pack_shared_candidates = bool(pack_shared_candidates)
        self.action_input = nn.Linear(3, self.hidden_dim)
        self.action_output = nn.Linear(self.hidden_dim, 3)
        self.action_position = nn.Parameter(torch.empty(8, self.hidden_dim))
        self.p_embeddings = nn.Parameter(torch.empty(32, self.hidden_dim))
        self.r_embeddings = (nn.Parameter(torch.empty(16, self.hidden_dim))
                             if training_queries else None)
        # [current numeric ego, P, A, R]; never a learned scene identity.
        self.role_embedding = nn.Embedding(4, self.hidden_dim)
        self.ego_input = nn.Sequential(nn.Linear(ego_dim, self.hidden_dim), nn.SiLU(),
                                       nn.Linear(self.hidden_dim, self.hidden_dim))
        self.time_embedding = nn.Sequential(nn.Linear(time_dim, time_dim), nn.SiLU(),
                                           nn.Linear(time_dim, time_dim))
        self.time_films = nn.ModuleList(nn.ModuleList((
            nn.Linear(time_dim, 2*self.hidden_dim), nn.Linear(time_dim, 2*self.hidden_dim)))
            for _ in range(self.num_layers))
        for token in (self.action_position, self.p_embeddings, self.r_embeddings):
            if token is not None:
                nn.init.normal_(token, std=0.02)
        for pair in self.time_films:
            for film in pair:
                nn.init.zeros_(film.weight)
                nn.init.zeros_(film.bias)
        nn.init.xavier_uniform_(self.action_output.weight)
        nn.init.zeros_(self.action_output.bias)

    @property
    def language_model(self):
        return self.qwen_model.language_model

    @property
    def device(self):
        return self.action_input.weight.device

    def encode_current(self, current_inputs: Mapping[str, Any], ego=None):
        """Official image forward, placeholder scatter and native mRoPE IDs.

        Future/metric labels are deliberately outside this method's contract.
        The processor batch can carry camera_indices metadata; it is not an actor
        input.  Only the listed native tensor fields are consumed.
        """
        forbidden = {'supervision', 'GT', 'requested_GT', 'future_states', 'reward',
                     'candidate_labels', 'metric_cache', 'teacher_features'}
        if forbidden.intersection(current_inputs):
            raise ValueError('Training supervision is not a D current input')
        ids = current_inputs['input_ids'].to(self.device)
        valid = current_inputs.get('attention_mask', torch.ones_like(ids)).to(self.device).bool()
        if ids.ndim != 2 or valid.shape != ids.shape or not valid.any(1).all():
            raise ValueError('Invalid native current token/padding layout')
        if ego is None:
            ego = current_inputs.get('ego', current_inputs.get('proprio'))
        if ego is None or ego.shape != (len(ids), self.ego_dim):
            raise ValueError('D requires explicit legal current ego [B,8]')
        embeds = self.qwen_model.get_input_embeddings()(ids)
        grids = current_inputs.get('image_grid_thw')
        pixels = current_inputs.get('pixel_values')
        visual_mask = torch.zeros_like(valid)
        deepstack = ()
        if (pixels is None) != (grids is None):
            raise ValueError('Image values and native grid must appear together')
        if pixels is not None:
            grids = grids.to(self.device)
            image_embeds, deepstack = self.qwen_model.get_image_features(pixels.to(self.device), grids)
            image_embeds = torch.cat(image_embeds, dim=0).to(embeds.device, embeds.dtype)
            mask, _ = self.qwen_model.get_placeholder_mask(ids, embeds, image_features=image_embeds)
            embeds = embeds.masked_scatter(mask, image_embeds)
            visual_mask = mask[..., 0]
        elif (ids == self.qwen_model.config.image_token_id).any():
            raise ValueError('Image placeholder without a real current image')
        if (visual_mask & ~valid).any():
            raise ValueError('A current image cannot occupy padding')
        positions, _ = self.qwen_model.get_rope_index(ids, image_grid_thw=grids,
                                                    attention_mask=valid.long())
        names = tuple(current_inputs.get('scene_identity', ()))
        if names and len(names) != len(ids):
            raise ValueError('Scene identity must align with the current batch')
        return DCurrentEncoding(embeds, valid, positions, visual_mask, tuple(deepstack),
            ego.to(self.device).float(), names, str(current_inputs.get('execution_identity', '')),
            str(current_inputs.get('position_mask_identity', '')))

    @staticmethod
    def _additive(allowed, dtype):
        result = torch.zeros(allowed.shape, dtype=dtype, device=allowed.device)
        return result.masked_fill(~allowed, torch.finfo(dtype).min)[:, None]

    @staticmethod
    def _suffix_positions(start, length):
        values = start[:, None]+torch.arange(length, device=start.device)[None]
        return values.unsqueeze(0).expand(3, -1, -1)

    def _layer(self, index, hidden, positions, allowed, prefix_kv, time, action_rows,
               *, causal=False):
        layer = self.language_model.layers[index]
        rotation = self.language_model.rotary_emb(hidden, positions)
        # Constructing the mask after Qwen RMSNorm preserves the actual compute
        # dtype in FP32-storage / BF16-autocast execution.
        normed = layer.input_layernorm(hidden)
        if action_rows is not None:
            scale, shift = self.time_films[index][0](time).to(normed.dtype).chunk(2, -1)
            if time.ndim == 2:
                scale, shift = scale[:, None], shift[:, None]
            normed = normed*(1+action_rows[None, :, None]*scale)+action_rows[None, :, None]*shift
        cache = _OneLayerCache(index, prefix_kv)
        mask = None if allowed is None else self._additive(allowed, normed.dtype)
        value, _ = layer.self_attn(normed, position_embeddings=rotation,
            attention_mask=mask, past_key_values=cache, is_causal=causal)
        hidden = hidden+value
        normed = layer.post_attention_layernorm(hidden)
        if action_rows is not None:
            scale, shift = self.time_films[index][1](time).to(normed.dtype).chunk(2, -1)
            if time.ndim == 2:
                scale, shift = scale[:, None], shift[:, None]
            normed = normed*(1+action_rows[None, :, None]*scale)+action_rows[None, :, None]*shift
        hidden = hidden+layer.mlp(normed)
        return hidden, cache.result[0], cache.result[1]

    def _run(self, hidden, positions, allowed, *, prefix_kv=None, time=None,
             action_rows=None, visual_mask=None, deepstack=(), causal=False,
             record_layer_states=False):
        """Functional native blocks; explicit checkpoint works in eval()+grad too."""
        if prefix_kv is not None and len(prefix_kv) != self.num_layers:
            raise ValueError('Prefix belongs to another Qwen depth')
        kv, layer_states = [], []
        for i in range(self.num_layers):
            prefix = None if prefix_kv is None else prefix_kv[i]
            def run_layer(h, *, number=i, parent=prefix):
                return self._layer(number, h, positions, allowed, parent, time, action_rows,
                                   causal=causal)
            if (self.activation_checkpointing and torch.is_grad_enabled()
                    and (prefix_kv is None or self.checkpoint_short_suffix)):
                hidden, key, value = checkpoint(run_layer, hidden, use_reentrant=False)
            else:
                hidden, key, value = run_layer(hidden)
            if i < len(deepstack):
                if visual_mask is None or int(visual_mask.sum()) != len(deepstack[i]):
                    raise ValueError('DeepStack must align only with real current image positions')
                feature = deepstack[i].to(hidden.device, hidden.dtype)
                hidden = hidden+torch.zeros_like(hidden).masked_scatter(
                    visual_mask[..., None].expand_as(hidden), feature)
            kv.append((key, value))
            if record_layer_states:
                layer_states.append(hidden)
        return self.language_model.norm(hidden), tuple(kv), tuple(layer_states)

    def prepare_prefix(self, current: DCurrentEncoding, *, policy_version=''):
        if not isinstance(current, DCurrentEncoding):
            raise TypeError('D takes native current encoding, never C160 scene memory')
        b, length, _ = current.inputs_embeds.shape
        # The all-valid case uses the native SDPA causal fast path. Padding is
        # masked explicitly; padded query results are never read downstream.
        if current.valid.all() and self.qwen_model.config.text_config._attn_implementation == 'sdpa':
            allowed = None
        else:
            triangle = torch.arange(length, device=self.device)[:, None] >= torch.arange(length, device=self.device)[None]
            allowed = triangle[None]&current.valid[:, None, :]
        _, kv, _ = self._run(current.inputs_embeds, current.positions, allowed,
            visual_mask=current.visual_mask, deepstack=current.deepstack, causal=allowed is None)
        start = current.positions.masked_fill(~current.valid[None], -1).amax((0, 2))+1
        ego = self.ego_input(current.ego)[:, None]+self.role_embedding.weight[0][None, None]
        p = self.p_embeddings[None].expand(b, -1, -1)+self.role_embedding.weight[1][None, None]
        ep = torch.cat((ego, p), dim=1)
        n = 1+self.p_tokens
        allowed = torch.ones((b, n, length+n), dtype=torch.bool, device=self.device)
        allowed[:, :, :length] = current.valid[:, None]
        allowed[:, 0, length+1:] = False  # current numeric ego does not read P.
        reasoning, kv, _ = self._run(ep, self._suffix_positions(start, n), allowed, prefix_kv=kv)
        valid = torch.cat((current.valid, torch.ones((b, n), dtype=torch.bool, device=self.device)), dim=1)
        return DPrefix(kv, valid, start+n, reasoning[:, 1:], current.scene_identity,
            policy_version, current.execution_identity, current.position_mask_identity)

    def pre_reasoning_states(self, prefix):
        return prefix.reasoning

    def _action_embeds(self, state):
        if state.ndim != 3 or state.shape[1:] != (8, 3):
            raise ValueError('D uses normalized absolute native [B,8,3] actions')
        return self.action_input(state.float())+self.action_position[None]+self.role_embedding.weight[2][None, None]

    def _candidate_suffix(self, state, sigma, prefix, *, include_r=False, r_override=None):
        if len(state) != len(prefix.valid) or prefix.architecture_id != 'D':
            raise ValueError('D prefix scene/candidate layout mismatch')
        sigma = torch.as_tensor(sigma, device=state.device, dtype=torch.float32)
        if sigma.ndim == 0:
            sigma = sigma.expand(len(state))
        if sigma.shape != (len(state),) or not torch.isfinite(sigma).all():
            raise ValueError('Sigma must have one finite value per candidate')
        hidden = self._action_embeds(state)
        a = 8
        if include_r:
            if self.r_embeddings is None:
                raise RuntimeError('Deployment D has no training-only R queries')
            r = self.r_embeddings[None].expand(len(state), -1, -1) if r_override is None else r_override
            if r.shape != (len(state), self.r_tokens, self.hidden_dim):
                raise ValueError('R override must be an aligned R16 diagnostic tensor')
            hidden = torch.cat((hidden, r+self.role_embedding.weight[3][None, None]), dim=1)
        n, p = hidden.shape[1], prefix.valid.shape[1]
        allowed = torch.ones((len(state), n, p+n), dtype=torch.bool, device=state.device)
        allowed[:, :, :p] = prefix.valid[:, None]
        allowed[:, :a, p+a:] = False
        action_rows = (torch.arange(n, device=state.device) < a).to(hidden.dtype)
        time = self.time_embedding(timestep_embedding(sigma, self.time_dim))
        # One-scene G candidates need one CP, not G physical copies of its long
        # K/V. Pack independent candidate blocks into the query axis and mask
        # all cross-candidate suffix edges. This remains native SDPA/GQA/RoPE;
        # no custom attention kernel or changed proposal distribution is used.
        if (self.pack_shared_candidates and len(state) > 1
                and prefix.valid.stride(0) == 0
                and all(k.stride(0) == 0 and v.stride(0) == 0 for k, v in prefix.layer_kv)):
            group = len(state)
            own = torch.arange(group, device=state.device).repeat_interleave(n)
            suffix_allowed = own[:, None] == own[None, :]
            if include_r:
                in_block = torch.arange(group*n, device=state.device) % n
                suffix_allowed &= ~((in_block[:, None] < a) & (in_block[None, :] >= a))
            memory_allowed = prefix.valid[:1, None, :].expand(1, group*n, -1)
            packed_allowed = torch.cat((memory_allowed, suffix_allowed[None]), -1)
            packed_position = self._suffix_positions(prefix.next_positions[:1], n).repeat(1, 1, group)
            result, kv, layers = self._run(hidden.reshape(1, group*n, self.hidden_dim),
                packed_position, packed_allowed,
                prefix_kv=tuple((k[:1], v[:1]) for k, v in prefix.layer_kv),
                time=time[:, None, :].expand(-1, n, -1).reshape(1, group*n, self.time_dim),
                action_rows=action_rows.repeat(group))
            return result.reshape(group, n, self.hidden_dim), kv, layers
        return self._run(hidden, self._suffix_positions(prefix.next_positions, n), allowed,
                         prefix_kv=prefix.layer_kv, time=time, action_rows=action_rows)

    def denoise_x0(self, action_state_fp32, sigma_fp32, prefix):
        hidden, _, _ = self._candidate_suffix(action_state_fp32, sigma_fp32, prefix)
        return self.action_output(hidden).float()

    def replay_candidates(self, normalized_candidate_trajectory, candidate_to_scene, prefix):
        selected = prefix.index_select(candidate_to_scene)
        sigma = normalized_candidate_trajectory.new_zeros(len(normalized_candidate_trajectory))
        hidden, _, _ = self._candidate_suffix(normalized_candidate_trajectory, sigma, selected, include_r=True)
        return hidden[:, 8:]

    def named_parameter_roles(self):
        """Explicit grouping; newly added interfaces are not pretrained Qwen roles."""
        roles = {}
        for name, _ in self.named_parameters():
            if name.startswith('qwen_model.language_model.'):
                role = 'pretrained_language'
            elif name.startswith('qwen_model.visual.'):
                role = ('pretrained_merger' if '.merger.' in name or '.deepstack_merger_list.' in name
                        else 'pretrained_vision')
            elif name.startswith(('p_embeddings', 'r_embeddings', 'role_embedding')):
                role = 'new_reasoning_tokens'
            elif name.startswith(('time_embedding.', 'time_films.')):
                role = 'new_time_modulation'
            else:
                role = 'new_action_interface'
            roles[name] = role
        return roles

    def rl_attention_paths(self):
        return tuple(f'qwen_model.language_model.layers.{i}.self_attn.{projection}_proj'
                     for i in range(self.num_layers) for projection in ('q', 'k', 'v', 'o'))

    def discard_training_only(self):
        """Remove R query storage without touching any deployment computation."""
        removed = []
        if self.r_embeddings is not None:
            self.register_parameter('r_embeddings', None)
            removed.append('r_embeddings')
        return tuple(removed)

    def deployment_state_dict(self):
        return {name: value for name, value in self.state_dict().items()
                if name != 'r_embeddings'}

    def deployment_components(self):
        return {'architecture_id': 'D', 'prediction_type': 'x0', 'action_shape': [8, 3],
            'kept': ['qwen_model', 'action_input', 'action_output', 'action_position',
                     'p_embeddings', 'role_embedding', 'ego_input', 'time_embedding', 'time_films'],
            'train_only': ['r_embeddings', 'external_training_readouts'],
            'qwen_text_layers_per_action_step': self.num_layers,
            'immutable_prefix': 'current C + numeric ego + P32', 'rollout_prefix_policy_versioned': True}


ArchitectureD = QwenFlowAdapter

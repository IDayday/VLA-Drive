"""Joint vehicle dimension on the ORIGINAL DDP FlowmatchingActionHead/DiT.

Existing alternating cross/self-attention now exchanges all actor-time states.
No second ego generator. No historical JointSceneFlow model or weights used.
"""
import torch
from torch import nn
from starVLA.model.modules.action_model.GR00T_ActionHeader import FlowmatchingActionHead, MLP
from .initialization import initialization_seed


def modeled_mask(active, steps=8):
    if active.dtype != torch.bool or active.ndim != 2 or not active[:, 0].all():
        raise ValueError("Active actors require a boolean B,N mask with ego slot0")
    mask = active[..., None, None].expand(-1, -1, steps, 4).clone()
    mask[:, 1:, :, 2:] = False
    return mask


class VehicleJointActionHead(FlowmatchingActionHead):
    def __init__(self, full_config, *, extension_seed=1042, base_head=None):
        if base_head is None:
            super().__init__(full_config)
        else:
            # Reuse freshly initialized original submodules, not a driving
            # checkpoint. Avoid allocating a second 24-layer DiT temporarily.
            nn.Module.__init__(self)
            for name in ("hidden_size", "full_config", "input_embedding_dim", "action_dim",
                         "action_horizon", "num_inference_timesteps", "beta_dist",
                         "num_timestep_buckets", "config"):
                setattr(self, name, getattr(base_head, name))
            for name, module in base_head.named_children():
                self.add_module(name, module)
        if self.action_dim != 4 or self.action_horizon != 8:
            raise ValueError("Joint task requires 8 x (xy,sin yaw,cos yaw)")
        if not self.model.config.interleave_self_attention:
            raise ValueError("Joint actors require the original interleaved self-attention")
        with initialization_seed(extension_seed):
            self.actor_role = nn.Embedding(2, self.input_embedding_dim)
            self.current_encoder = MLP(8, self.input_embedding_dim, self.input_embedding_dim)
            self.vehicle_query_proj = nn.Linear(full_config.framework.qwenvl.vl_hidden_dim,
                                                self.input_embedding_dim)
            self.known_encoder = nn.Linear(4, self.input_embedding_dim, bias=False)
            nn.init.normal_(self.actor_role.weight, std=.02)

    def velocity(self, state, times, action_queries, actor_queries, current_boxes,
                 active, known_mask=None):
        mask = modeled_mask(active, self.action_horizon)
        if state.shape != mask.shape or times.shape != (state.shape[0],):
            raise ValueError("Joint state/time shape mismatch")
        if current_boxes.shape != (*active.shape, 8) or actor_queries.shape[:2] != active.shape:
            raise ValueError("Current actor condition shape mismatch")
        for value, valid in ((state, mask), (current_boxes, active[..., None].expand_as(current_boxes)),
                             (actor_queries, active[..., None].expand_as(actor_queries))):
            if not torch.isfinite(value[valid]).all():
                raise ValueError("Nonfinite modeled state or active current input")
        known_mask = torch.zeros_like(mask) if known_mask is None else known_mask
        if known_mask.dtype != torch.bool or known_mask.shape != mask.shape or (known_mask & ~mask).any():
            raise ValueError("Known coordinates must be active and modeled")
        state = torch.where(mask, state, 0.)
        boxes = torch.where(active[..., None], current_boxes, 0.)
        queries = torch.where(active[..., None], actor_queries, 0.)
        batch, actors, steps, _ = state.shape
        discrete = (times * self.num_timestep_buckets).long()
        encoded = self.action_encoder(state.flatten(1, 2), discrete).reshape(batch, actors, steps, -1)
        # Reuse the same original eight temporal positions for each actor.
        if self.config.add_pos_embed:
            encoded = encoded + self.position_embedding(torch.arange(steps, device=state.device))[None, None]
        roles = torch.ones(actors, device=state.device, dtype=torch.long)
        roles[0] = 0
        box_scale = boxes.new_tensor([20., 20., 5., 10., 10., 5., 1., 1.])
        current = self.current_encoder(boxes / box_scale) + self.vehicle_query_proj(queries)
        encoded = encoded + current[:, :, None] + self.actor_role(roles)[None, :, None]
        encoded = encoded + self.known_encoder(known_mask.to(encoded.dtype))
        token_mask = active[..., None].expand(-1, -1, steps).flatten(1)
        hidden = self.model(hidden_states=encoded.flatten(1, 2),
                            encoder_hidden_states=self.qwen_proj(action_queries), timestep=discrete,
                            active_token_mask=token_mask)
        result = self.action_decoder(hidden).reshape_as(state)
        return torch.where(mask, result, 0.)

    def loss(self, action_queries, actor_queries, current_boxes, active, targets,
             feature_valid, noise, times, known_mask=None):
        modeled = modeled_mask(active, self.action_horizon)
        if targets.shape != modeled.shape or feature_valid.shape != modeled.shape or feature_valid.dtype != torch.bool:
            raise ValueError("Label coordinates require a boolean feature_valid of the joint state shape")
        if (feature_valid & ~modeled).any() or not torch.isfinite(targets[feature_valid]).all():
            raise ValueError("Invalid supervised target channel")
        if noise.shape != modeled.shape or not torch.isfinite(noise[modeled]).all():
            raise ValueError("Invalid joint noise")
        known = torch.zeros_like(modeled) if known_mask is None else known_mask
        if known.shape != modeled.shape or known.dtype != torch.bool or (known & ~feature_valid).any():
            raise ValueError("Training known_mask must have valid labels and modeled active channels")
        labels = torch.where(feature_valid, targets, 0.)
        clean_noise = torch.where(modeled, noise, 0.)
        t = times[:, None, None, None]
        # Missing xy supervision stays uncertain noise here; it is NOT a
        # deployment integration mask and is never passed into velocity().
        mixed = torch.where(feature_valid, (1-t)*clean_noise + t*labels, clean_noise)
        mixed = torch.where(known, labels, mixed)
        pred = self.velocity(mixed, times, action_queries, actor_queries, current_boxes, active, known)
        valid = feature_valid & ~known
        errors = (pred.float() - (labels-clean_noise).float()).square()
        def average(part):
            weights = valid[:, part]
            numerator = torch.where(weights, errors[:, part], 0.).flatten(1).sum(-1)
            denominator = weights.flatten(1).sum(-1).clamp_min(1)
            return (numerator/denominator).mean()
        # Preserve full ego FM magnitude; additional vehicles do not dilute it.
        ego, vehicles = average(slice(0, 1)), average(slice(1, None))
        return ego + vehicles, {"ego_fm": ego.detach(), "vehicle_fm": vehicles.detach(),
                                 "ego_coordinates": int(valid[:, 0].sum()),
                                 "vehicle_coordinates": int(valid[:, 1:].sum())}

    @torch.no_grad()
    def sample(self, action_queries, actor_queries, current_boxes, active, noise,
               known_values=None, known_mask=None, return_history=False):
        modeled = modeled_mask(active, self.action_horizon)
        if noise.shape != modeled.shape or not torch.isfinite(noise[modeled]).all():
            raise ValueError("Invalid modeled sample noise")
        known = torch.zeros_like(modeled) if known_mask is None else known_mask
        if known.dtype != torch.bool or known.shape != modeled.shape or (known & ~modeled).any():
            raise ValueError("Known coordinates must be active and modeled")
        values = torch.zeros_like(noise) if known_values is None else known_values
        if values.shape != noise.shape or not torch.isfinite(values[known]).all():
            raise ValueError("Invalid known values")
        # Hidden GT/NaNs are removed before any encoding. Unmodeled yaw=0 sentinel.
        values = torch.where(known, values, 0.)
        def clamp(x): return torch.where(modeled, torch.where(known, values, x), 0.)
        state = clamp(noise)
        history = [state.clone()] if return_history else None
        for step in range(self.num_inference_timesteps):
            times = state.new_full((state.shape[0],), step/self.num_inference_timesteps)
            velocity = self.velocity(state, times, action_queries, actor_queries, current_boxes, active, known)
            # Clamp output too: even a modified decoder cannot integrate yaw.
            velocity = torch.where(modeled & ~known, velocity, 0.)
            state = clamp(state + velocity / self.num_inference_timesteps)
            if history is not None: history.append(state.clone())
        if not torch.isfinite(state).all(): raise FloatingPointError("Nonfinite joint sample")
        return (state, history) if return_history else state

    @staticmethod
    def executed_ego(joint_sample):
        return joint_sample[:, 0]

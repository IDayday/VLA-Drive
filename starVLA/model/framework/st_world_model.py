"""Explicit ST supervision around a strictly loaded original S0 DiT model.

The parent current encoder and Flow action implementation remain unchanged.
Labels reach only this loss wrapper. Parent four-task auxiliary heads are
removed after strict loading, and replaced by the registered sparse readout.
"""
from __future__ import annotations

import torch
from torch import nn

from starVLA.dataloader.st_world_schema import ModelInputs
from starVLA.model.modules.vehicle_joint.initialization import initialization_seed
from starVLA.model.modules.st_world.config import SpaceTimeConfig
from starVLA.model.modules.st_world.sparse_readout import SparseFeatureReadout
from starVLA.model.modules.st_world.losses import temporal_losses, weighted_temporal_loss
from starVLA.model.modules.st_world.requests import gather_current, gather_future
from tools.st_world_policy.world_expansion import expand_world_queries


class SpaceTimeStudent(nn.Module):
    def __init__(self, parent: nn.Module, options: dict | SpaceTimeConfig, *,
                 initialization_seed_value: int, history_encoder: nn.Module | None = None):
        super().__init__()
        self.st_config = options if isinstance(options, SpaceTimeConfig) else SpaceTimeConfig.from_dict(options)
        self.st_config.validate()
        self.base = parent
        self.history_encoder = history_encoder
        if self.st_config.history != (history_encoder is not None):
            raise ValueError("History cells require their explicitly configured causal history encoder")
        original = parent.config.framework.action_model
        if (original.action_dim, original.action_horizon, original.repeated_diffusion_steps,
                original.num_inference_timesteps) != (4, 8, 8, 10):
            raise ValueError("Original S0 Flow contract changed")
        if parent.foresight_config.planner_condition_mode != "action_only":
            raise ValueError("This study preserves the original action-only planner condition")
        removed = [name for name in ("dino_head", "future_head", "interaction_head", "spatiotemporal_head")
                   if hasattr(parent, name)]
        parent.strip_auxiliary_heads()
        self.migration = expand_world_queries(parent, self.st_config.world_queries_per_view,
                                              seed=initialization_seed_value + 3500)
        self.migration["removed_auxiliary_modules"] = removed
        hidden = parent.qwen_vl_interface.model.config.hidden_size
        with initialization_seed(initialization_seed_value + 3600):
            self.sparse_readout = SparseFeatureReadout(
                hidden, self.st_config.teacher.channels,
                views=len(self.st_config.teacher.view_names), dim=self.st_config.readout_dim,
                layers=self.st_config.readout_layers, heads=self.st_config.readout_heads)
        self._deployment_only = False
        # Constructing the wrapper must not accidentally enable a frozen visual.
        self.train(parent.training)

    def train(self, mode: bool = True):
        super().train(mode)
        self.base.qwen_vl_interface.model.model.visual.eval()
        return self

    @property
    def action_model(self):
        return self.base.action_model

    @property
    def last_sequence_lengths(self):
        return self.base.last_sequence_lengths

    def encode_inputs(self, observations):
        inputs = [x if isinstance(x, ModelInputs) else ModelInputs(x) for x in observations]
        if self.st_config.history:
            return self.history_encoder.encode(self.base, inputs)
        if any(x.history is not None for x in inputs):
            raise ValueError("A current-only control cannot silently consume history")
        return self.base.encode_current([x.legacy_current() for x in inputs])

    def forward(self, observations, targets, *, requests, global_counts,
                noise_generator, time_generator, completed_updates=0):
        if self._deployment_only:
            raise RuntimeError("Deployment export cannot silently drop registered training losses")
        cfg = self.st_config
        if cfg.future:
            actual_times = targets.get("future_times_s")
            if not isinstance(actual_times, torch.Tensor) or not actual_times.is_floating_point():
                raise ValueError("Future physical time table must accompany real teacher targets")
            expected_times = actual_times.new_tensor(cfg.future_times_s).expand(len(observations), -1)
            if actual_times.shape != expected_times.shape or not torch.equal(actual_times, expected_times):
                raise ValueError("Teacher physical time table differs from the registered horizon table")
        encoded = self.encode_inputs(observations)
        condition = self.base.build_planner_condition(encoded)
        device = condition.device
        ego = targets["ego"].to(device=device, dtype=torch.float32)
        if ego.shape != (len(observations), 8, 4) or not torch.isfinite(ego).all():
            raise ValueError("Original finite normalized 8x4 ego supervision is required")
        if len(requests.patch_indices) != len(observations):
            raise ValueError("Requests must be sliced with their original optimizer-batch samples")
        # Same repeat, bridge, independent random streams, and Beta(alpha,1)
        # transformation as the original S0 forward. This is not native-repeat1.
        action_cfg = self.base.config.framework.action_model
        if action_cfg.noise_beta_beta != 1.:
            raise ValueError("The registered original Beta sampler requires beta=1")
        repeat = int(action_cfg.repeated_diffusion_steps)
        y = ego.repeat(repeat, 1, 1)
        noise = torch.randn(y.shape, device=device, dtype=y.dtype, generator=noise_generator)
        u = torch.rand((len(y),), device=device, generator=time_generator)
        times = (action_cfg.noise_s - u.pow(1. / action_cfg.noise_beta_alpha)) / action_cfg.noise_s
        from torch import distributed as dist
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        denominator = int(global_counts["ego_scenes"])
        if denominator < len(observations):
            raise ValueError("Global original-scene denominator is invalid")
        with self.base.amp():
            fm = self.base.action_model(condition.repeat(repeat, 1, 1), y, noise=noise, times=times)
            view, xy, seconds = requests.query_layout(current=cfg.current, future=cfg.future, device=device)
            prediction = self.sparse_readout(encoded["W"], view, xy, seconds)
        fm = fm * (len(observations) * world_size / denominator)
        batch, views, positions = requests.patch_indices.shape
        prediction = prediction.reshape(batch, 1 + 2 * int(cfg.future), views, positions, -1)
        current, current_valid = gather_current(targets["current_dense_features"],
                                                targets["current_valid"], requests)
        kw = dict(current_prediction=prediction[:, 0], current_target=current.to(device),
                  current_valid=current_valid.to(device), delta=cfg.delta, world_size=world_size)
        if cfg.future:
            future, future_valid = gather_future(targets["future_dense_features"],
                                                 targets["future_valid"], requests)
            kw.update(future_prediction=prediction[:, 1:], future_target=future.to(device),
                      future_valid=future_valid.to(device))
        auxiliary_counts = {key: value for key, value in global_counts.items() if key != "ego_scenes"}
        feature_losses, diagnostics = temporal_losses(auxiliary_counts, **kw)
        feature_loss = weighted_temporal_loss(feature_losses, cfg)
        total = fm + feature_loss
        if not torch.isfinite(total):
            raise FloatingPointError("Nonfinite ST training objective")
        return {"loss": total, "losses": {"ego_fm": fm, **feature_losses},
                "metrics": diagnostics}

    @torch.no_grad()
    def predict_raw(self, observations, *, initial_noise):
        encoded = self.encode_inputs(observations)
        condition = self.base.build_planner_condition(encoded)
        if initial_noise.shape != (len(observations), 8, 4):
            raise ValueError("Explicit original trajectory sampling noise is required")
        with self.base.amp():
            return self.base.action_model.predict_action(condition, initial_noise=initial_noise)

    @torch.no_grad()
    def predict_action(self, observations, *, initial_noise):
        from starVLA.dataloader.foresight_dataset import decode_ego
        return decode_ego(self.predict_raw(observations, initial_noise=initial_noise).float())

    def strip_auxiliary_heads(self):
        if hasattr(self, "sparse_readout"):
            del self.sparse_readout
        self._deployment_only = True
        return self

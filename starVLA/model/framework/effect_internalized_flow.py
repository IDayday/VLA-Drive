"""One original Euler10 generation with an internal private velocity adapter."""
import torch
from torch import nn
from starVLA.model.modules.flow_posttraining.policy_snapshot import DiTFlowPolicy
from starVLA.model.modules.action_effect.effect_distillation import (
    corrected_residual_target, velocity_loss, effect_consistency_loss)


class EffectInternalizedFlow(nn.Module):
    ARMS = ('B0', 'B1-trajectory', 'B1-effect')

    def __init__(self, action_head, adapter, *, arm, enabled=True):
        super().__init__()
        if arm not in self.ARMS:
            raise ValueError('Explicit independent B0/B1 experiment arm required')
        if type(enabled) is not bool:
            raise ValueError('Adapter enabled must be a boolean')
        self.base = DiTFlowPolicy(action_head).requires_grad_(False).eval()
        self.adapter = adapter
        self.arm, self.enabled = arm, bool(enabled)
        base_storage = {p.untyped_storage().data_ptr() for p in self.base.parameters()}
        if any(p.untyped_storage().data_ptr() in base_storage for p in adapter.parameters()):
            raise ValueError('Adapter aliases Base storage')
        self.train(True)

    def train(self, mode=True):
        super().train(mode)
        self.base.eval()
        self.base.requires_grad_(False)
        return self

    def _base_projected(self, condition):
        with torch.no_grad():
            return self.base.project_condition(condition)

    def base_velocity(self, condition, state, times):
        with torch.no_grad():
            return self.base.velocity(self._base_projected(condition), state, times)

    def velocity(self, condition, state, times):
        base = self.base_velocity(condition, state, times)
        if not self.enabled:
            return base
        return base+self.adapter(condition.detach(), state, times)

    def forward(self, condition, pair, valid_scenes, *, teacher=None, true_delta=None,
                effect_valid=None, effect_scales=None, global_denominator=None,
                world_size=1, effect_global_denominator=None,
                direction_global_denominator=None, effect_context=None):
        pair.validate()
        endpoint = 'plus' if self.arm == 'B0' else 'minus'
        state = pair.state(endpoint)
        base = self.base_velocity(condition, state, pair.times)
        residual = self.adapter(condition.detach(), state, pair.times)
        prediction = base+residual
        losses = {'velocity': velocity_loss(prediction, pair.target_velocity(), valid_scenes,
                    global_denominator=global_denominator, world_size=world_size)}
        result = {'losses': losses, 'prediction': prediction,
                  'path': endpoint, 'base_velocity': base}
        if self.arm != 'B0':
            result['residual_target'] = corrected_residual_target(pair, base)
        if self.arm == 'B1-effect':
            if any(x is None for x in (teacher, true_delta, effect_valid, effect_scales, effect_context)):
                raise ValueError('B1-effect requires verified G1 teacher and real paired physical deltas')
            effect = effect_consistency_loss(teacher, effect_context, pair, prediction,
                true_delta, effect_valid, effect_scales, world_size=world_size,
                global_denominator=effect_global_denominator,
                direction_global_denominator=direction_global_denominator)
            losses.update({k: effect[k] for k in ('effect_consistency', 'effect_direction')})
            result.update({k: effect[k] for k in ('predicted_effect', 'endpoint_proxy')})
        elif teacher is not None or true_delta is not None or effect_context is not None:
            raise ValueError('B0/B1-trajectory cannot silently consume physical effect supervision')
        return result

    @torch.no_grad()
    def predict_action(self, condition, *, initial_noise):
        self.base.eval()
        if not self.enabled:
            return self.base.action_head.predict_action(condition.detach(), initial_noise=initial_noise)
        if initial_noise.shape != (len(condition), 8, 4) or not torch.isfinite(initial_noise).all():
            raise ValueError('Original finite raw8x4 initial noise required')
        # Exactly one full generation; no a_minus rollout/refiner/teacher/score.
        projected = self._base_projected(condition)
        actions = initial_noise.clone()
        was_training = self.adapter.training
        self.adapter.eval()
        try:
            for step in range(10):
                times = torch.full((len(actions),), step/10, device=actions.device)
                base = self.base.velocity(projected, actions, times)
                velocity = base+self.adapter(condition.detach(), actions, times)
                actions = actions+(1./10)*velocity
            return actions
        finally:
            self.adapter.train(was_training)

    def trainable_parameters(self):
        return self.adapter.parameters()

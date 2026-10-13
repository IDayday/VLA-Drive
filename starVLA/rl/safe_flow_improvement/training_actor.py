"""Full original action head: positive FM/NFT update and common retention."""
import torch
from torch import nn

from starVLA.rl.flow_grpo.rollout import velocity
from .contracts import seed_for
from .nft_adapter import nft_forward_loss
from .policy_adapter import encode_condition, original_fm_loss
from .reference_retention import fm_draws, original_replay_loss, safe_path_retention
from .weighted_flow_objective import interpolation


class ExtractionActor(nn.Module):
    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, *, raw, target, replay_raw, replay_target, retention_raw,
                retention_target, reference, seed, retention_coefficient,
                mode='positive_fm', old_behavior=None, advantage=None, nft_beta=1.):
        if retention_coefficient < 0 or mode not in ('positive_fm', 'nft'):
            raise ValueError('Invalid registered update objective')
        # All labels and observation-prefix inputs are data. Only the complete
        # action_model, including qwen_proj, is connected to trainable leaves.
        raw, target = raw.detach(), target.detach()
        y, noise, time = fm_draws(self.policy.action_model, target,
                                  seed_for('improvement_fm', seed, 'draw'), nft=mode == 'nft')
        repeated = raw.repeat(8, 1, 1)
        if mode == 'positive_fm':
            improve = original_fm_loss(self.policy, repeated, y, noise, time)
            extra = {}
        else:
            if old_behavior is None or advantage is None:
                raise ValueError('NFT requires its frozen actual ODE behavior and true reward advantage')
            x, _ = interpolation(y, noise, time)
            bucket = (time*self.policy.action_model.num_timestep_buckets).long()
            with torch.no_grad():
                old = velocity(old_behavior, x, bucket, encode_condition(old_behavior, repeated), use_graph=False)
            current = velocity(self.policy, x, bucket, encode_condition(self.policy, repeated), use_graph=False)
            improve, extra = nft_forward_loss(current, old, x, y, time,
                                               advantage.detach().repeat(8), beta=nft_beta)
        replay = original_replay_loss(self.policy, replay_raw, replay_target,
                                      seed_for('gt_replay_fm', seed, 'draw'))
        retain = safe_path_retention(self.policy, reference, retention_raw, retention_target,
                                      seed_for('reference_safe_path_fm', seed, 'draw'))
        loss = improve + .1*replay + retention_coefficient*retain
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite extraction update')
        return {'loss': loss, 'improve': improve, 'replay': replay, 'reference_velocity': retain,
                'nft_positive_fraction': extra['positive_fraction'].detach() if 'positive_fraction' in extra else None}

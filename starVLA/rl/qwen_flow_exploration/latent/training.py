"""Small root-policy update callback, independent of loading/scoring platforms."""
from dataclasses import dataclass
import copy
import torch
from .gaussian import GaussianParameters, joint_log_prob
from .objective import latent_ppo_loss
from starVLA.rl.simwam_recipe.recovery_state import tensor_digest


@dataclass(frozen=True)
class LatentBehaviorBatch:
    features: torch.Tensor
    roots: torch.Tensor
    old_parameters: GaussianParameters
    old_joint_logp: torch.Tensor
    rewards: torch.Tensor
    advantages: torch.Tensor
    behavior_digest: str

    def to(self, device):
        return LatentBehaviorBatch(self.features.to(device), self.roots.to(device),
            GaussianParameters(self.old_parameters.mean.to(device), self.old_parameters.log_std.to(device)),
            self.old_joint_logp.to(device), self.rewards.to(device), self.advantages.to(device), self.behavior_digest)

    @classmethod
    @torch.no_grad()
    def from_collected(cls, policy, features, roots, rewards, *, old_parameters,
                       old_joint_logp, advantage_callback):
        if features.shape[0] != roots.shape[0] or rewards.shape != roots.shape[:2]:
            raise ValueError('Collected scene/group association differs')
        params = policy(features).detached()
        if (not torch.equal(params.mean, old_parameters.mean) or
            not torch.equal(params.log_std, old_parameters.log_std)):
            raise ValueError('Actual old sampling mean/std differ from fixed-layout recomputation')
        if not torch.equal(joint_log_prob(old_parameters, roots), old_joint_logp):
            raise ValueError('Saved actual joint density differs from actual sampling Gaussian')
        result = advantage_callback(rewards)
        advantage = result['advantages'].reshape_as(rewards).detach()
        # The shared advantage mask removes numerical tied residuals, not bad
        # candidates. All legitimate candidates remain in the behavior batch.
        mask = result.get('adv_mask', result.get('mask'))
        if mask is not None:
            advantage = advantage*mask.reshape_as(rewards)
        return cls(features.detach().clone(), roots.detach().clone(), params,
                   old_joint_logp.detach().clone(), rewards.detach().clone(),
                   advantage, tensor_digest(policy.state_dict()))

    def digest(self):
        return tensor_digest(dict(features=self.features, roots=self.roots,
            old_mean=self.old_parameters.mean, old_log_std=self.old_parameters.log_std,
            old_joint_logp=self.old_joint_logp, rewards=self.rewards,
            advantages=self.advantages, behavior=self.behavior_digest))


def train_collected_batch(policy, optimizer, batch, *, inner_epochs=4,
                          start_inner=0, clip_range=.2, prior_coefficient=.01,
                          checkpoint_callback=None):
    """Four actual optimizer updates; callers own atomic run transactions.

    Resume supplies the same immutable batch and optimizer, plus start_inner.
    No generator operation, resampling, or reward recomputation occurs here.
    """
    if not 0 <= start_inner < inner_epochs or inner_epochs <= 0:
        raise ValueError('Invalid pending-inner state')
    if start_inner == 0 and tensor_digest(policy.state_dict()) != batch.behavior_digest:
        raise ValueError('Fresh collection policy changed before its first update')
    if any(t.requires_grad for t in [batch.features, batch.roots, batch.old_joint_logp,
                                     batch.rewards, batch.advantages]):
        raise ValueError('Behavior evidence must be detached')
    before = batch.digest()
    history = []
    for inner in range(start_inner, inner_epochs):
        optimizer.zero_grad(set_to_none=True)
        parameters = policy(batch.features)
        loss, metrics = latent_ppo_loss(parameters, batch.roots, batch.old_joint_logp,
            batch.advantages, old_parameters=batch.old_parameters,
            clip_range=clip_range, prior_coefficient=prior_coefficient)
        if inner == 0 and (metrics['ratio_min'] != 1 or metrics['ratio_max'] != 1):
            raise ValueError('Unchanged current/old root density is not ratio1')
        loss.backward()
        gradients = [p.grad for p in policy.parameters() if p.requires_grad]
        if not gradients or any(g is None or not torch.isfinite(g).all() for g in gradients):
            raise FloatingPointError('Root policy gradient contract failed')
        norm = torch.stack([g.detach().float().square().sum() for g in gradients]).sum().sqrt()
        per_parameter_norms = {name:float(parameter.grad.detach().float().norm())
                               for name,parameter in policy.named_parameters() if parameter.requires_grad}
        learning_rates = [float(group['lr']) for group in optimizer.param_groups]
        optimizer.step()
        if batch.digest() != before:
            raise ValueError('An inner update mutated saved behavior evidence')
        metrics.update(inner_completed=inner+1, gradient_norm=float(norm),
                       parameter_gradient_norms=per_parameter_norms,
                       learning_rates_used=learning_rates,
                       optimizer_updates=1, effective_PG=bool(batch.advantages.ne(0).any()),
                       behavior_digest=batch.behavior_digest, behavior_batch_digest=before)
        history.append(metrics)
        if checkpoint_callback is not None:
            checkpoint_callback(inner+1, batch, metrics)
    return history


def save_payload(policy, optimizer, batch=None, *, next_inner=0):
    """Serializable payload for the common atomic checkpoint publisher."""
    if bool(next_inner) != (batch is not None):
        raise ValueError('A pending inner checkpoint must carry its old samples')
    return copy.deepcopy(dict(policy=policy.state_dict(), optimizer=optimizer.state_dict(),
                              batch=batch, next_inner=next_inner, policy_identity=policy.identity()))

"""Exact root-density PPO, with fixed-prior KL separate from old-policy KL."""
import torch
from .gaussian import joint_log_prob, joint_kl, joint_entropy


def latent_ppo_loss(current, roots, old_joint_logp, advantages, *, old_parameters,
                    clip_range, prior_coefficient, scene_mask=None):
    if not 0 < clip_range < 1 or prior_coefficient < 0:
        raise ValueError('Register the joint-root PPO clip and prior coefficient explicitly')
    if roots.requires_grad or old_joint_logp.requires_grad or advantages.requires_grad:
        raise ValueError('Saved behavior samples/probabilities/advantages must be detached')
    if old_joint_logp.shape != roots.shape[:2] or advantages.shape != roots.shape[:2]:
        raise ValueError('Root/old density/advantage scene-candidate identity mismatch')
    if not torch.isfinite(old_joint_logp).all() or not torch.isfinite(advantages).all():
        raise FloatingPointError('Nonfinite behavior evidence')
    if old_parameters.mean.requires_grad or old_parameters.log_std.requires_grad:
        raise ValueError('Old behavior Gaussian must remain frozen during all inner updates')
    if scene_mask is None:
        scene_mask = torch.ones(len(roots), device=roots.device, dtype=torch.bool)
    if scene_mask.shape != (len(roots),) or scene_mask.dtype != torch.bool or not scene_mask.any():
        raise ValueError('An explicit nonempty mask is required only for padded SCENES')
    new_logp = joint_log_prob(current, roots)
    log_ratio = new_logp-old_joint_logp
    ratio = torch.exp(log_ratio)
    if not torch.isfinite(ratio).all():
        raise FloatingPointError('Joint root ratio overflow; preserve failed batch instead of clamping probability')
    candidate_loss = -torch.minimum(ratio*advantages,
                                    ratio.clamp(1-clip_range, 1+clip_range)*advantages)
    # Tied groups have exact zero A. They remain in the common scene average;
    # excluding them here would silently strengthen PG relative to prior KL.
    scene_pg = candidate_loss.mean(dim=1)
    prior_kl = joint_kl(current, current.standard_prior())
    old_kl = joint_kl(current, old_parameters)
    mean = lambda value: value[scene_mask].mean()
    pg, regularizer = mean(scene_pg), mean(prior_kl)
    total = pg+prior_coefficient*regularizer
    if not torch.isfinite(total):
        raise FloatingPointError('Nonfinite latent policy objective')
    with torch.no_grad():
        selected_ratio = ratio[scene_mask]
        std = current.log_std.exp()[scene_mask]
        metrics = dict(loss=float(total), PG=float(pg), prior_coefficient=float(prior_coefficient),
                       prior_KL_joint=float(regularizer), prior_KL_per_dimension=float(regularizer/24),
                       old_KL_joint=float(mean(old_kl)), old_KL_per_dimension=float(mean(old_kl)/24),
                       entropy_joint=float(mean(joint_entropy(current))),
                       ratio_min=float(selected_ratio.min()), ratio_max=float(selected_ratio.max()),
                       ratio_clip_fraction=float(((selected_ratio-1).abs()>clip_range).float().mean()),
                       mean_l2=float(current.mean[scene_mask].square().sum((-1, -2)).sqrt().mean()),
                       std_min=float(std.min()), std_max=float(std.max()), std_mean=float(std.mean()),
                       reduction='mean candidates then mean valid scenes; tied groups retained',
                       ratio_semantics='complete24D root Gaussian; not final trajectory likelihood')
    return total, metrics

"""One-factor RF++-Baseline advantage scaling for the World PPO arm.

Hu et al., https://arxiv.org/html/2501.03262v8, Eqs. 11--12, subtract
the group reward mean BEFORE normalizing the resulting advantages globally.
OpenRLHF's experience_maker.py implements the same order with population
variance. This differs from our source's global std of RAW rewards. DAPO
Eq. 9 still uses group std; this arm must not be described as DAPO.

For the registered rollout there are 8 ranks x 1 scene x 8 candidates.
Only the denominator changes. Keep the source FP64 centering, eps=1e-8,
tied-group mask, local effective mean, and all four updates on the SAME
finalized advantage tensor. This is the normalization ablation, not a full
RF++ implementation: the source transition ratio and IL mean anchor remain.
"""
from __future__ import annotations

import torch
import torch.distributed as dist

from starVLA.rl.simwam_recipe.recovery_state import validate_boundary as source_validate_boundary

NORMALIZATION_SCHEMA = 'group_centered_global_advantage_population_std_v1'


@torch.no_grad()
def centered_batch_advantages(rewards, *, all_rewards=None, eps=1e-8):
    """Normalize group-centered rewards over the full fresh global rollout.

    `all_rewards` is an explicit CPU/math-test injection; otherwise gather
    every rank once, before any inner update. Tied groups remain in the
    population variance even though their PG mask is zero. The historical
    `global_std` retains its RAW-reward meaning for existing exploration
    journals; `normalization_std` is the denominator actually used here.
    """
    if rewards.ndim != 2 or rewards.shape[1] != 8 or not torch.isfinite(rewards).all():
        raise ValueError('Finite [scene, 8 candidates] rewards required')
    if eps != 1e-8:
        raise ValueError('This one-factor arm preserves the source epsilon')
    values = rewards.detach().double()
    if all_rewards is None:
        if dist.is_available() and dist.is_initialized():
            gathered = [torch.empty_like(values) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered, values)
            all_rewards = torch.cat(gathered, dim=0)
        else:
            all_rewards = values
    all_rewards = all_rewards.detach().to(device=values.device, dtype=torch.float64)
    if (all_rewards.ndim != 2 or all_rewards.shape[1] != 8
            or not torch.isfinite(all_rewards).all()):
        raise ValueError('Finite global [scene, 8 candidates] rewards required')
    offsets = values-values[:, :1]
    centered = offsets-offsets.mean(dim=1, keepdim=True)
    all_offsets = all_rewards-all_rewards[:, :1]
    all_centered = all_offsets-all_offsets.mean(dim=1, keepdim=True)
    # Each group sums to zero algebraically. Avoid introducing tiny nonzero
    # advantages into constant groups by subtracting a rounded global mean.
    normalization_std = all_centered.std(unbiased=False)
    raw_std = all_rewards.std(unbiased=False)
    group_std = offsets.std(dim=1, unbiased=False, keepdim=True)
    a = centered/(normalization_std+eps)
    mask = (group_std>eps).expand_as(values).reshape(-1).float()
    return dict(advantages=a.reshape(-1).float().detach(), adv_mask=mask.detach(),
        global_std=raw_std.detach(), normalization_std=normalization_std.detach(),
        raw_group_std=group_std.detach(),
        all_full=int((values==1).all(dim=1).sum()),
        all_zero=int((values==0).all(dim=1).sum()), tied=int((group_std<=eps).sum()),
        effective_PG_candidate_fraction=float(mask.mean()),
        normalization_schema=NORMALIZATION_SCHEMA,
        normalization='group centered / population std of ALL global group-centered rewards',
        normalization_population_candidates=int(all_rewards.numel()),
        normalization_population_groups=int(all_rewards.shape[0]),
        reduction='source local effective mean, then equal mean across ranks')


def finalize_centered_batch_advantages(rollout):
    """Runtime finalizer hook: call ONCE after all ranks score successfully."""
    if 'advantages' in rollout:
        raise ValueError('A committed rollout must not be renormalized between inner updates')
    stats = centered_batch_advantages(rollout['rewards'])
    tensor_keys = ('advantages', 'adv_mask', 'global_std', 'raw_group_std', 'normalization_std')
    rollout.update({k: stats[k] for k in tensor_keys})
    rollout['advantage_statistics'] = {k:v for k,v in stats.items() if k not in tensor_keys}
    return rollout


def validate_centered_batch_boundary(state, rollout, identity, optimizer=None, scheduler=None, *, legacy=False):
    """Native transaction hook preserving every source structural assertion."""
    if identity['recipe'].get('research', {}).get('advantage_normalization') != NORMALIZATION_SCHEMA:
        raise ValueError('Centered-batch advantages require a separately registered scientific identity')
    summary = source_validate_boundary(state, rollout, identity, optimizer, scheduler, legacy=legacy)
    summary['advantage_normalization_schema'] = NORMALIZATION_SCHEMA
    if summary['pending']:
        meta = rollout['advantage_statistics']
        scale = rollout.get('normalization_std')
        if (meta.get('normalization_schema') != NORMALIZATION_SCHEMA
                or meta.get('normalization_population_candidates') != identity['world']*8
                or meta.get('normalization_population_groups') != identity['world']
                or not torch.is_tensor(scale) or scale.ndim != 0 or scale.requires_grad
                or not bool(torch.isfinite(scale)) or float(scale)<0):
            raise ValueError('Missing or changed global advantage-normalization material')
        summary['normalization_std'] = float(scale)
    return summary


def validate_centered_batch_ranks(summaries):
    """Check the actual saved advantages using all ranks' saved rewards.

    Replaces the source's RAW-std semantic check only for this new arm. The
    same summary feeds collection commits, checkpoint publication, and exact
    native restore; no old scientific identity is reinterpreted.
    """
    if not summaries:
        raise ValueError('No rank summaries')
    common = ('phase', 'updates', 'collections', 'scene_cursor', 'next_inner', 'run',
              'pending', 'advantage_normalization_schema')
    for s in summaries:
        if s.get('advantage_normalization_schema') != NORMALIZATION_SCHEMA:
            raise ValueError('Missing centered-batch rank protocol')
        if {k:s[k] for k in common} != {k:summaries[0][k] for k in common}:
            raise ValueError('Ranks disagree on recovery boundary')
    if summaries[0]['pending']:
        if any(s['behavior'] != summaries[0]['behavior'] for s in summaries):
            raise ValueError('Ranks disagree on collection-start behavior')
        rewards = torch.tensor([s['rewards'][0] for s in summaries], dtype=torch.float64)
        expected = centered_batch_advantages(rewards, all_rewards=rewards)
        adv = expected['advantages'].reshape(len(summaries),8)
        mask = expected['adv_mask'].reshape(len(summaries),8)
        for rank,s in enumerate(summaries):
            if (abs(s['global_std']-float(expected['global_std']))>1e-12
                    or abs(s['normalization_std']-float(expected['normalization_std']))>1e-12
                    or not torch.allclose(torch.tensor(s['advantages']), adv[rank], atol=1e-7, rtol=1e-6)):
                raise ValueError('Centered-batch advantage was not globally finalized')
            if s['mask'] != mask[rank].tolist():
                raise ValueError('PG mask mismatch')
    return summaries

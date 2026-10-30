"""Training-only safe-trajectory comfort alignment; canonical PDMS intact.

Only candidates passing NC/DAC/TTC all receive constant training comfort1.
Other candidates keep their canonical reward exactly. There is no component
whitening, safety dual, filter, sampling or normalization change.
"""
from __future__ import annotations

import copy
import math
import torch

from starVLA.rl.simwam_recipe.recovery_state import (
    validate_boundary as source_validate_boundary,
    validate_ranks as source_validate_ranks,
    tensor_digest,
)
from starVLA.rl.simwam_recipe.collection import finalize_advantages as source_finalize_advantages

COMFORT_SCHEMA = 'canonical_all3safe_constant_comfort1_train_only_v1'
COMFORT_PROTOCOL = dict(
    schema=COMFORT_SCHEMA,
    canonical_reward='NC*DAC*(5*EP+5*TTC+2*official_Comfort)/12',
    training_reward='if NC=DAC=TTC=1: (5*EP+5*TTC+2)/12; else: canonical_PDMS',
    safety_gate='exact official NC==1 and DAC==1 and TTC==1; no threshold relaxation',
    constant_training_comfort=1.0,
    canonical_feedback='unchanged original official records, score, protocol and readouts',
    canonical_rewards='separate detached FP32 [1,8] field; preserved in every behavior/checkpoint',
    advantage='unchanged group mean / global RAW training reward population std over all64',
    inner_updates='four updates reuse the same finalized advantage tensor',
    scope='training scalar reward only; official evaluation always canonical PDMS',
)


@torch.no_grad()
def comfort_aligned_rewards(records, *, device='cpu'):
    if len(records) != 8:
        raise ValueError('Exactly eight successful canonical feedback records required')
    canonical=[];training=[]
    names=('no_at_fault_collisions','drivable_area_compliance',
           'ego_progress','time_to_collision_within_bound','comfort','score')
    for record in records:
        if record.get('status') != 'OK':
            raise ValueError('Scoring failure must not become a transformed zero reward')
        m=record['metrics']
        if any(not math.isfinite(float(m[k])) or not 0<=float(m[k])<=1 for k in names):
            raise ValueError('Canonical components must be finite in [0,1]')
        nc,dac,ep,ttc,c=(float(m[k]) for k in names[:5])
        score=nc*dac*(5*ep+5*ttc+2*c)/12
        if abs(score-float(m['score']))>5e-15:
            raise ValueError('Canonical feedback aggregation changed')
        if abs(float(record['readouts']['canonical_PDMS'])-float(m['score']))>5e-15:
            raise ValueError('Canonical readout changed')
        canonical.append(float(m['score']))
        training.append((5*ep+5*ttc+2)/12 if nc==dac==ttc==1 else float(m['score']))
    return dict(canonical_rewards=torch.tensor([canonical],device=device,dtype=torch.float32),
                rewards=torch.tensor([training],device=device,dtype=torch.float32))


def apply_comfort_alignment(rollout):
    """Pre-finalization transform; no forward/RNG/score/feedback mutation."""
    if 'canonical_rewards' in rollout or 'reward_transform' in rollout or 'advantages' in rollout:
        raise ValueError('Transform exactly once, before advantage finalization')
    values=comfort_aligned_rewards(rollout['records'],device=rollout['rewards'].device)
    if not torch.equal(rollout['rewards'],values['canonical_rewards']):
        raise ValueError('Comfort alignment requires the original canonical training reward')
    rollout['canonical_rewards']=rollout['rewards'].detach().clone()
    rollout['rewards']=values['rewards'].detach()
    rollout['reward_transform']=copy.deepcopy(COMFORT_PROTOCOL)
    return rollout


def finalize_safe_comfort_advantages(rollout):
    """Runtime hook: transform once, then unchanged global RAW-std finalizer."""
    apply_comfort_alignment(rollout)
    return source_finalize_advantages(rollout)


def validate_comfort_boundary(state, rollout, identity, optimizer=None, scheduler=None, *, legacy=False):
    research=identity['recipe'].get('research',{})
    if (research.get('reward_transform') != COMFORT_SCHEMA
            or research.get('policy_objective') != 'ppo'
            or research.get('advantage_normalization') is not None):
        raise ValueError('Require separately registered PPO/raw-batchstd comfort identity')
    summary=source_validate_boundary(state,rollout,identity,optimizer,scheduler,legacy=legacy)
    summary['reward_transform_schema']=COMFORT_SCHEMA
    if summary['pending']:
        if rollout.get('reward_transform') != COMFORT_PROTOCOL:
            raise ValueError('Missing or changed training-only reward transform protocol')
        expected=comfort_aligned_rewards(rollout['records'])
        for key in ('canonical_rewards','rewards'):
            value=rollout.get(key)
            if (not torch.is_tensor(value) or value.shape!=(1,8) or value.requires_grad
                    or value.dtype!=torch.float32 or not torch.equal(value.cpu(),expected[key])):
                raise ValueError('Saved official/transformed reward material differs: '+key)
        summary['canonical_rewards']=rollout['canonical_rewards'].double().cpu().tolist()
        summary['canonical_rewards_digest']=tensor_digest(rollout['canonical_rewards'])
    return summary


def validate_comfort_ranks(summaries):
    if not summaries or any(s.get('reward_transform_schema')!=COMFORT_SCHEMA for s in summaries):
        raise ValueError('Ranks require the same explicit comfort reward protocol')
    # Standard source semantic check sees the actual transformed RAW rewards
    # and ensures their full-global std/A/masks are finalized consistently.
    return source_validate_ranks(summaries)

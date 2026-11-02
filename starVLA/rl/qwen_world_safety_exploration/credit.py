"""Original SafetyEP equations, once-finalized across world*G candidates.

Subclass GroupCredit.parts/protocol for a separately registered credit variant;
the same object recomputes its math during boundary and all-rank validation.
Historical behavior never enters this fresh global normalization.
"""
import copy
import math

import torch
import torch.distributed as dist

from starVLA.rl.qwen_world_research_algorithms.safety_efficiency import COMPONENT_NAMES

STRUCTURED_SCHEMA = 'world_SafetyEP_group_credit_v1'


def protocol(group_size):
    if type(group_size) is not int or group_size not in (8, 16, 32):
        raise ValueError('Only G8 or G16')
    return dict(schema=STRUCTURED_SCHEMA, group_size=group_size, world=8,
        normalization_population_candidates=8*group_size, kappa=0.0, lambda_EP=1.0,
        feasible='official NC == DAC == TTC == 1',
        cost='mean(1-NC,1-DAC,1-TTC); indicator deficit, not accident severity',
        safety_advantage='if any feasible: f-mean(f); else: mean(c)-c',
        efficiency_advantage='if >=2 feasible: f*(EP-mean_feasible EP); else 0',
        raw_advantage='A_safe+A_EP; group sum zero',
        normalization='population std of all world*G fresh raw advantages +1e-8, including zero groups',
        mask='all-zero raw group has zero PG mask, successful saturation is not failure',
        canonical_feedback='unchanged canonical rewards/records/readouts',
        inner_updates='reuse same detached advantages for all four updates',
        loss='unchanged source rank mean PPO, cap5, beta50 permanent IL70 physical mean anchor',
        component_names=list(COMPONENT_NAMES))


@torch.no_grad()
def canonical_components(records, group_size, *, device='cpu'):
    if len(records) != group_size:
        raise ValueError('Canonical record count must equal registered group size')
    rewards, components = [], []
    for record in records:
        if record.get('status') != 'OK':
            raise ValueError('Failed scoring cannot become a reward')
        m = record['metrics']
        names = COMPONENT_NAMES + ('comfort', 'score')
        if any(not math.isfinite(float(m[k])) or not 0 <= float(m[k]) <= 1 for k in names):
            raise ValueError('Canonical components outside finite [0,1]')
        nc, dac, ttc, ep, comfort, score = (float(m[k]) for k in names)
        if (abs(nc*dac*(5*ep+5*ttc+2*comfort)/12-score) > 5e-15
                or not math.isfinite(float(record['readouts']['canonical_PDMS']))
                or abs(float(record['readouts']['canonical_PDMS'])-score) > 5e-15):
            raise ValueError('Canonical aggregation/readout changed')
        rewards.append(score); components.append((nc, dac, ttc, ep))
    return dict(canonical_rewards=torch.tensor([rewards], device=device, dtype=torch.float32),
                components=torch.tensor([components], device=device, dtype=torch.float64))


@torch.no_grad()
def raw_advantage_parts(components):
    if (components.ndim != 3 or components.shape[1] not in (8, 16, 32) or components.shape[2] != 4
            or not torch.isfinite(components).all()
            or not bool(((components >= 0) & (components <= 1)).all())):
        raise ValueError('Finite [scene,G,NC/DAC/TTC/EP] material required')
    x = components.detach().double()
    feasible = (x[:, :, :3] == 1).all(-1); f = feasible.double()
    count = f.sum(1, keepdim=True); cost = (1-x[:, :, :3]).mean(-1)
    dc = cost-cost[:, :1]
    a_safe = torch.where(count > 0, f-f.mean(1, keepdim=True), dc.mean(1, keepdim=True)-dc)
    ep = x[:, :, 3]; first = feasible.long().argmax(1, keepdim=True)
    ep_offsets = (ep-ep.gather(1, first))*f
    a_ep = (ep_offsets-ep_offsets.sum(1, keepdim=True)/count.clamp_min(1))*f
    a_ep = torch.where(count >= 2, a_ep, torch.zeros_like(a_ep))
    return dict(feasible=feasible, cost=cost, A_safe=a_safe, A_EP=a_ep, A_raw=a_safe+a_ep)


class GroupCredit:
    parts = staticmethod(raw_advantage_parts)

    def __init__(self, group_size):
        self.group_size = group_size
        self.protocol = protocol(group_size)

    @torch.no_grad()
    def advantages(self, rewards, components, *, all_material=None):
        g = self.group_size
        if (rewards.ndim != 2 or rewards.shape[1] != g
                or tuple(components.shape) != (*rewards.shape, 4)
                or not torch.isfinite(rewards).all()
                or not bool(((rewards >= 0) & (rewards <= 1)).all())):
            raise ValueError('Canonical [scene,G] rewards and matching components required')
        local = self.parts(components)
        material = torch.cat((rewards.detach().double().unsqueeze(-1),
                              components.detach().to(rewards.device, torch.float64)), -1)
        if all_material is None:
            if dist.is_available() and dist.is_initialized():
                if dist.get_world_size() != 8:
                    raise ValueError('Fresh credit requires exactly eight ranks')
                gathered = [torch.empty_like(material) for _ in range(8)]
                dist.all_gather(gathered, material); all_material = torch.cat(gathered)
            else:
                all_material = material
        all_material = all_material.detach().to(rewards.device, torch.float64)
        if (all_material.ndim != 3 or tuple(all_material.shape[1:]) != (g, 5)
                or not torch.isfinite(all_material).all()
                or not bool(((all_material >= 0) & (all_material <= 1)).all())):
            raise ValueError('Invalid global reward/component material')
        global_parts = self.parts(all_material[:, :, 1:])
        scale = global_parts['A_raw'].std(unbiased=False)
        raw = rewards.detach().double(); offsets = raw-raw[:, :1]
        group_std = offsets.std(1, unbiased=False, keepdim=True)
        active = local['A_raw'].ne(0).any(1, keepdim=True)
        mask = active.expand_as(raw).reshape(-1).float()
        count = local['feasible'].sum(1)
        return dict(**local,
            advantages=(local['A_raw']/(scale+1e-8)).reshape(-1).float().detach(),
            adv_mask=mask.detach(), global_std=all_material[:, :, 0].std(unbiased=False),
            raw_group_std=group_std, normalization_std=scale,
            statistics=dict(structured_advantage_schema=self.protocol['schema'], group_size=g,
                normalization_population_candidates=int(all_material.shape[0]*g),
                normalization_population_groups=int(all_material.shape[0]),
                normalization=self.protocol['normalization'],
                all_full=int((raw == 1).all(1).sum()), all_zero=int((raw == 0).all(1).sum()),
                tied=int((group_std <= 1e-8).sum()), structured_zero_groups=int((~active).sum()),
                feasible_candidates=int(count.sum()), all_fail_groups=int((count == 0).sum()),
                singleton_feasible_groups=int((count == 1).sum()),
                effective_PG_candidate_fraction=float(mask.mean()),
                reduction='source local effective mean, then equal mean across ranks'))

    @torch.no_grad()
    def finalize(self, rollout):
        if any(k in rollout for k in ('advantages', 'structured_advantage', 'canonical_rewards', 'reward_transform')):
            raise ValueError('Finalize original canonical credit exactly once')
        material = canonical_components(rollout['records'], self.group_size, device=rollout['rewards'].device)
        if not torch.equal(rollout['rewards'], material['canonical_rewards']):
            raise ValueError('Original canonical rewards changed')
        stats = self.advantages(rollout['rewards'], material['components'])
        rollout['canonical_rewards'] = rollout['rewards'].detach().clone()
        for k in ('advantages', 'adv_mask', 'global_std', 'raw_group_std', 'normalization_std'):
            rollout[k] = stats[k]
        rollout['advantage_statistics'] = stats['statistics']
        rollout['structured_advantage'] = dict(protocol=copy.deepcopy(self.protocol),
            components=material['components'],
            **{k:stats[k] for k in ('feasible', 'cost', 'A_safe', 'A_EP', 'A_raw')})
        return rollout

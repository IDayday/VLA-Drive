"""Finite same-theta credit replay on already scored TRAIN P0 rollouts.

This is a diagnostic, not a trainer or a new policy. It creates no sampler,
reward service, optimizer, backward-to-.grad, or checkpoint. The actual source
ResearchLoss computes Safety PG and the physical IL anchor once. A forwarding
tap retains its live current logp for the alternative canonical-PPO credit.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import time

import torch
import torch.distributed as dist

from starVLA.rl.flow_grpo.transactions import atomic_json
from starVLA.rl.simwam_recipe import checkpoint
from starVLA.rl.simwam_recipe.source_advantages import advantages as source_advantages
from starVLA.rl.qwen_world_postcot.contracts import file_sha
from .gradient_probe import measure_components
from .policy import ResearchLoss, _masked_mean, policy_gradient_objective
from .safety_efficiency import canonical_components, safety_efficiency_advantages

SCHEMA = 'world_same_theta_Safety_PPO_credit_gradient_replay_v1'


def shared_branch_safe_objective(new_logp, old_logp, total_advantage,
                                 safe_advantage, mask, *, clip_range, cap):
    """Attribute a fixed share of the ACTUAL total-A objective to A_safe.

    cap_gain = capped_total / uncapped_total, zero when total is zero. A_safe
    is never independently clipped. The detached total-A branch multiplier
    reproduces torch.maximum's source PPO branch, including its half-gradient
    at a *strictly clipped* numerical tie. A_EP is defined by exact scalar
    complement in CreditReplayLoss, not by another independently clipped PPO.
    """
    if new_logp.shape != total_advantage.shape or safe_advantage.shape != new_logp.shape:
        raise ValueError('Aligned candidate/step advantages required')
    raw = total_advantage.detach().to(new_logp)
    capped = raw if cap is None else raw.clamp(-cap, cap)
    gain = torch.where(raw.ne(0), capped / torch.where(raw.ne(0), raw, torch.ones_like(raw)),
                       torch.zeros_like(raw))
    safe = safe_advantage.detach().to(new_logp) * gain
    ratio = torch.exp(new_logp-old_logp.detach())
    clipped_ratio = ratio.clamp(1-clip_range, 1+clip_range)
    u, c = -capped*ratio, -capped*clipped_ratio
    # Derivative of the exact source maximum+clamp, evaluated on total A.
    u_weight = torch.where(u > c, torch.ones_like(u),
                           torch.where(u < c, torch.zeros_like(u), torch.full_like(u, .5)))
    clamp_derivative = ((ratio >= 1-clip_range) & (ratio <= 1+clip_range)).to(ratio)
    branch = (u_weight + (1-u_weight)*clamp_derivative).detach()
    loss = -_masked_mean(ratio*safe*branch, mask.detach().to(new_logp))
    return loss, dict(cap_gain=gain, total_capped=capped, branch_derivative=branch,
                      safe_effective=safe)


class _LogprobTap:
    """Transparent forwarding view; never replaces or mutates an actor method."""
    def __init__(self, actor):
        self.actor, self.current = actor, None

    def __getattr__(self, name):
        return getattr(self.actor, name)

    def action_chain_logprobs(self, *args, **kwargs):
        value = self.actor.action_chain_logprobs(*args, **kwargs)
        if not kwargs.get('use_ref', False):
            if self.current is not None:
                raise ValueError('Expected exactly one source current logp recomputation')
            self.current = value[0] if isinstance(value, tuple) else value
        return value


class CreditReplayLoss(ResearchLoss):
    def record_policy_components(self, pg, anchor):
        self.parts = dict(safety_pg=pg, anchor=anchor)

    def _policy_loss(self, model, rollout):
        if (self.algorithm != 'ppo' or self.advantage_normalization
                or float(self._effective_kl_beta(model)) != 50
                or model.grpo_kl_type != 'x' or model.grpo_denoising_discount != 1):
            raise ValueError('Replay binds original Safety PPO/W3/beta50 physical mean anchor')
        tap = _LogprobTap(model)
        total, metrics = super()._policy_loss(tap, rollout)
        new = tap.current
        steps = new.shape[1]
        actual = rollout['advantages'].detach()[:, None].expand(-1, steps)
        ppo = rollout['credit_replay_ppo']
        ordinary = ppo['advantages'].to(new)[:, None].expand_as(new)
        ordinary = ordinary if self.adv_clip_max is None else ordinary.clamp(-self.adv_clip_max, self.adv_clip_max)
        ppo_pg, ppo_values = policy_gradient_objective(new, rollout['old_logp'], ordinary,
            ppo['adv_mask'].to(new), clip_range=self.ppo_clip_range)
        safe = (rollout['structured_advantage']['A_safe'] /
                (rollout['normalization_std']+1e-8)).reshape(-1).float().to(new)
        safe = safe[:, None].expand_as(new)
        safe_pg, attribution = shared_branch_safe_objective(new, rollout['old_logp'], actual, safe,
            rollout['adv_mask'], clip_range=self.ppo_clip_range, cap=self.adv_clip_max)
        self.parts.update(ppo_pg=ppo_pg, safety_safe=safe_pg,
                          safety_EP=self.parts['safety_pg']-safe_pg)
        self.diagnostics = dict(
            current_old_logp_bit_equal=torch.equal(new.detach(), rollout['old_logp']),
            current_old_logp_max_abs=float((new.detach()-rollout['old_logp']).abs().max()),
            total_advantage_cap_hits=int((actual != attribution['total_capped']).sum()),
            total_zero_candidate_steps=int(actual.eq(0).sum()),
            shared_total_branch_blocked_candidate_steps=int(attribution['branch_derivative'].eq(0).sum()),
            canonical_ppo_blocked_candidate_steps=int((ppo_values['ppo_blocked'] & ordinary.ne(0)
                & ppo['adv_mask'].to(new).bool()[:, None]).sum()),
            scalars={k:float(v.detach()) for k,v in self.parts.items()},
            scalar_complement_residual=float((self.parts['safety_safe']+self.parts['safety_EP']
                                              -self.parts['safety_pg']).detach()))
        return total, metrics


def candidate_update_geometry(report):
    """Five component sums double-count PG; report only meaningful candidates."""
    for scope in report['scopes'].values():
        scope.pop('summed_gradient_norm', None)
        norms, cosine = scope['global_mean_gradient_norm'], scope['cosine']

        def dot(a, b):
            c = cosine.get('__'.join(sorted((a,b))))
            return 0. if c is None else norms[a]*norms[b]*c

        def summed_norm(a, b):
            return math.sqrt(max(0., norms[a]**2+norms[b]**2+2*dot(a,b)))

        p, s = summed_norm('ppo_pg','anchor'), summed_norm('safety_pg','anchor')
        cross = (dot('ppo_pg','safety_pg')+dot('ppo_pg','anchor')
                 +dot('safety_pg','anchor')+norms['anchor']**2)
        scope['candidate_update_norms'] = dict(ppo_plus_beta50_anchor=p, safety_plus_beta50_anchor=s)
        scope['candidate_update_cosine'] = max(-1., min(1., cross/(p*s))) if p*s else None
    report['all_five_component_sum'] = 'UNDEFINED_AS_UPDATE: overlapping PG and its attribution parts; deliberately omitted'
    report['attribution_identity'] = 'safety_EP = source safety_pg - shared-total-cap/branch safety_safe'
    return report


def group_credit_material(rollout, ppo):
    parts = rollout['structured_advantage']
    feasible = parts['feasible'].reshape(-1)
    ep = parts['components'][..., 3].reshape(-1)
    n = int(feasible.sum())
    active = bool(rollout['adv_mask'].any())
    if not active and n == 8:
        classification = 'success_EP1_saturation' if bool(ep.eq(1).all()) else 'all_feasible_equal_EP_tie'
    elif not n:
        classification = 'all_fail_cost_varies' if active else 'all_fail_cost_tie'
    else:
        classification = 'mixed_feasible' if n < 8 else 'all_feasible_progress_signal'
    return dict(token=rollout['token'],log=rollout['log'],classification=classification,
        canonical_PPO_PG0=not bool(ppo['adv_mask'].any()),Safety_PG0=not active,
        feasible_count=n, rewards=rollout['rewards'].cpu().reshape(-1).tolist(),
        canonical_PPO_advantages=ppo['advantages'].cpu().tolist(),
        canonical_PPO_mask=ppo['adv_mask'].cpu().tolist(),
        source_global_raw_reward_std=float(ppo['global_std']),
        safety_global_raw_credit_std=float(rollout['normalization_std']),
        Safety_advantages=rollout['advantages'].cpu().tolist(),
        Safety_mask=rollout['adv_mask'].cpu().tolist(),
        components_NC_DAC_TTC_EP=parts['components'].cpu().tolist(),
        **{k:parts[k].cpu().reshape(-1).tolist() for k in ('feasible','cost','A_safe','A_EP','A_raw')})


def _device_tree(value, device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {k:_device_tree(v,device) for k,v in value.items()}
    if isinstance(value, list):
        return [_device_tree(v,device) for v in value]
    if isinstance(value, tuple):
        return tuple(_device_tree(v,device) for v in value)
    return value


def run_credit_replay(runtime, actor, cfg, args, native, seal, policy_sha, host):
    """Called only by the optional existing P0 entry branch, under torchrun8."""
    source, output = Path(args.credit_replay), Path(args.output)
    rank, world = dist.get_rank(), dist.get_world_size()
    protocol = json.loads((source/'PROTOCOL.json').read_text())
    summary = json.loads((source/'SUMMARY.json').read_text())
    if (summary.get('status') != 'COMPLETE' or protocol.get('world') != 8 or world != 8
            or protocol.get('native_updates') != seal['updates']
            or protocol.get('native_files_sha256') != seal['files_sha256']
            or protocol.get('policy_sha256') != policy_sha
            or Path(protocol['checkpoint']).resolve() != Path(args.rl_checkpoint).resolve()
            or protocol.get('panel_sha256') != file_sha(args.panel)
            or len(protocol.get('tokens',[])) != 24 or summary.get('candidates') != 192):
        raise ValueError('Credit replay must bind the completed original 3x8 P0/native identity')
    train_tokens = set(runtime.conditions.data.tokens)
    if not set(protocol['tokens']) <= train_tokens:
        raise ValueError('Credit replay only accepts original legal TRAIN tokens')
    binding = dict(schema=SCHEMA,host=host,world=world,status='RUNNING',
        source_probe=str(source),source_protocol_sha256=file_sha(source/'PROTOCOL.json'),
        source_summary_sha256=file_sha(source/'SUMMARY.json'),checkpoint=str(args.rl_checkpoint),
        native_updates=seal['updates'],native_files_sha256=seal['files_sha256'],policy_sha256=policy_sha,
        panel_sha256=file_sha(args.panel),groups=24,candidates=192,optimizer_steps=0,
        newly_generated_candidates=0,new_reward_scores=0,reference='original complete IL70',
        training_basis='counterfactual credit at Safety theta28k on its original scored G8; not a trained PPO policy comparison',
        PPO_normalization='source FP64 offset group center / population std of all64 RAW canonical rewards +1e-8; source tie mask',
        Safety_split='total normalized A cap5 and total-A PPO branch shared; zero-total cap gain0; EP objective exact complement',
        gradient_units='8rank equal-mean raw objective parameter gradients before Adam/preconditioning/global clipping',
        limitations=['fixed24 TRAIN groups, not population estimates or causal final-score evidence',
            'source transition ratio is clamped dimension-mean proxy, not exact importance sampling',
            'separate BF16 component backward sums can differ from fused native backward (~0.34% observed elsewhere); no bitexact gradient claim',
            'no optimizer or stochastic sampling/scoring service is constructed; five-component sum is not an update'])
    if rank == 0:
        output.mkdir(parents=True,exist_ok=False)
        atomic_json(output/'PROTOCOL.json',binding)
    checkpoint.check_errors(None)
    actor._loss = CreditReplayLoss(cfg)
    reports=[]; started=time.monotonic()
    for case in range(3):
        runtime.require_dispatch()
        path=source/f'case{case}_rank{rank}.pt'
        rollout=torch.load(path,map_location='cpu',weights_only=False)
        if (rollout['token'] != protocol['tokens'][case*8+rank]
                or rollout['log'] != runtime.conditions.data.logs[rollout['token']]
                or rollout['behavior_version'] != dict(run=native['sha256'], update=seal['updates'], adapter_sha256=policy_sha)
                or rollout['chain'].shape != (8,11,8,3)
                or rollout['stoch_idx'].tolist() != [7,8,9]):
            raise ValueError('Saved rollout identity/layout disagrees with original W3 P0')
        # Existing official readouts are checked, not rescored.
        official=canonical_components(rollout['records'])
        if (not torch.equal(official['canonical_rewards'],rollout['rewards'])
                or not torch.equal(official['components'],rollout['structured_advantage']['components'])):
            raise ValueError('Saved canonical scoring payload changed')
        rollout=_device_tree(rollout,actor.device)
        expected=safety_efficiency_advantages(rollout['rewards'],rollout['structured_advantage']['components'])
        if (not torch.equal(expected['advantages'],rollout['advantages'])
                or not torch.equal(expected['adv_mask'],rollout['adv_mask'])
                or not torch.allclose(expected['normalization_std'],rollout['normalization_std'],rtol=1e-12,atol=1e-12)
                or any(not torch.allclose(expected[k],rollout['structured_advantage'][k],rtol=1e-12,atol=1e-12)
                       for k in ('A_safe','A_EP','A_raw','feasible','cost'))):
            raise ValueError('Saved Safety advantages differ from original global64 finalization')
        ppo=source_advantages(rollout['rewards'])
        rollout['credit_replay_ppo']=ppo
        actor.zero_grad(set_to_none=True)
        total,metrics=actor(rollout)
        parts=actor._loss.parts
        report=candidate_update_geometry(measure_components(actor,parts))
        report.update(schema=SCHEMA,case=case,native_updates=seal['updates'],
            input_rollouts_by_rank=checkpoint.gather(dict(path=str(path),sha256=file_sha(path))),
            groups_by_rank=checkpoint.gather(group_credit_material(rollout,ppo)),
            forward_diagnostics_by_rank=checkpoint.gather(actor._loss.diagnostics),
            actual_source_Safety_loss_by_rank=checkpoint.gather(float(total.detach())))
        if rank == 0:
            atomic_json(output/f'case{case}.json',report)
        reports.append(report)
        actor._loss.parts=None
        del total,parts,metrics,rollout
    if rank == 0:
        atomic_json(output/'SUMMARY.json',dict(**{**binding,'status':'COMPLETE'},
            seconds=time.monotonic()-started,cases=[dict(case=r['case'],scopes=r['scopes']) for r in reports]))

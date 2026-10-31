"""Pure CPU counterfactual for one explicit all-failed DAC cost-tie subtype.

No runtime, sampler, scoring, trainer hook or registered algorithm is provided.
Only existing official feedback's 41 Boolean time samples are accepted. Raw
progress and uncomputed geometric margins are deliberately absent from the API.
"""
from __future__ import annotations

import math
from collections.abc import Mapping

import torch

from .safety_efficiency import safety_efficiency_parts

SCHEMA = 'world_DAC_only_allfail_cost_tie_raw_counterfactual_v1'
PROTOCOL = dict(
    schema=SCHEMA,
    eligibility='all8 NC==1,TTC==1,DAC==0; allfail; coarse deficit cost tie; original raw SafetyA exactly0',
    field='records[candidate].non_drivable_body_by_time',
    shape='8 candidates x41 Boolean samples, t=0,.1,...,4.0seconds',
    dense_cost='mean Boolean non-drivable over indices1:41; fraction0..1, not distance/risk probability',
    replacement='A_tie=mean(dense_cost)-dense_cost, offset-centered in FP64',
    other_groups='original raw advantages retained bit-for-bit; missing fields never filled with zero',
    t0='reported separately, not silently added as an eligibility change; review initially-invalid scenes before any future deployment',
    interpretation='a relative reduction of failed DAC occupancy, not canonical success or a safety certificate',
    normalization='raw replacement only; any new global64 std changes other normalized advantages by a shared scale',
    scope='CPU-only mathematical prototype; no actual training update or new scoring',
)


def raw_tie_counterfactual(components, raw_advantages, *, offroad_by_group: Mapping):
    """Return a detached raw-A copy plus explicit per-group availability.

    components is [groups,8,NC/DAC/TTC/EP]. raw_advantages is the actual saved
    native FP64 [groups,8], checked against the source formula to its FP64
    reduction tolerance. offroad_by_group maps only already-read groups to
    real [8,41] Boolean fields; no observation or privileged future label is
    passed to an actor by this module.
    """
    source=safety_efficiency_parts(components)
    raw=raw_advantages.detach()
    if (raw.dtype != torch.float64 or raw.shape != components.shape[:2]
            or not torch.isfinite(raw).all()
            or not torch.allclose(raw,source['A_raw'].to(raw),rtol=1e-12,atol=1e-12)):
        raise ValueError('Saved FP64 source raw SafetyA is required')
    x=components.detach().double()
    subtype=x[:,:,0].eq(1).all(1)&x[:,:,1].eq(0).all(1)&x[:,:,2].eq(1).all(1)
    eligible=subtype & ~source['feasible'].any(1) & source['cost'].eq(source['cost'][:,:1]).all(1) & raw.eq(0).all(1)
    result=raw.clone(); rows=[]
    for g in range(len(raw)):
        row=dict(group=g,eligible=bool(eligible[g]),status='UNCHANGED_NOT_ELIGIBLE')
        if not eligible[g]:
            rows.append(row);continue
        value=offroad_by_group.get(g)
        if value is None:
            row['status']='UNAVAILABLE_REAL_41STEP_FIELD';rows.append(row);continue
        flags=torch.as_tensor(value,device=raw.device)
        if (flags.shape != (8,41) or not torch.isfinite(flags).all()
                or not ((flags==0)|(flags==1)).all()):
            raise ValueError('Real binary [8,41] non_drivable_body_by_time required')
        dense=flags[:,1:].double().mean(1)
        offsets=dense-dense[:1]
        a=offsets.mean()-offsets
        result[g]=a
        row.update(status='APPLIED_RAW_COUNTERFACTUAL',dense_cost=dense.cpu().tolist(),
            A_tie=a.cpu().tolist(),t0_all_drivable=not bool(flags[:,0].bool().any()),
            changed=bool(a.ne(raw[g]).any()),has_nonzero_raw_credit=bool(a.ne(0).any()))
        rows.append(row)
    if not torch.equal(result[~eligible],raw[~eligible]):
        raise AssertionError('Unrelated raw Safety advantages changed')
    return dict(schema=SCHEMA,raw_advantages=result,eligible=eligible.detach(),groups=rows,
                unchanged_ineligible_raw_bit_exact=True)


def global_normalization_effect(original, counterfactual, *, eps=1e-8):
    """Actual supplied all64 group matrix; never assume missing groups are0."""
    if (original.shape != (8,8) or counterfactual.shape != original.shape
            or original.dtype != torch.float64 or counterfactual.dtype != torch.float64
            or eps != 1e-8 or not torch.isfinite(original).all()
            or not torch.isfinite(counterfactual).all()):
        raise ValueError('Complete finite eight-rank64-candidate raw matrices required')
    old,new=original.std(unbiased=False),counterfactual.std(unbiased=False)
    a0=(original/(old+eps)).float();a1=(counterfactual/(new+eps)).float()
    return dict(original_std=float(old),counterfactual_std=float(new),
        unchanged_raw_normalized_scale=float((old+eps)/(new+eps)),
        original_advantages=a0,counterfactual_advantages=a1,
        original_cap5_hits=int((a0.abs()>5).sum()),counterfactual_cap5_hits=int((a1.abs()>5).sum()))


def centered_zero_group_std_effect(original_std, A_tie, *, population=64, eps=1e-8):
    """Analytic variance increment when replacing one zero, centered G8 group.

    Source group credit has mean0. With the other56 raw advantages unchanged,
    std_new²=std_old²+sum(A_tie²)/64 in exact centered arithmetic. This is an
    analytic report when other56 are unavailable, not a fabricated all64 replay.
    """
    a=torch.as_tensor(A_tie,dtype=torch.float64)
    old=float(original_std)
    if (a.shape != (8,) or not torch.isfinite(a).all() or abs(float(a.sum()))>1e-12
            or not math.isfinite(old) or old<0 or population!=64 or eps!=1e-8):
        raise ValueError('Centered finite8 replacement and original64 std required')
    increment=float(a.square().sum())/population
    new=math.sqrt(old*old+increment)
    return dict(method='centered-source analytical variance increment; other56 not imputed',
                original_std=old,counterfactual_std=new,variance_increment=increment,
                unchanged_raw_normalized_scale=(old+eps)/(new+eps))

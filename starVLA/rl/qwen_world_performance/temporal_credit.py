"""User-defined indicator + real physical-time safety cost.

All-feasible groups retain original SafetyEP. Mixed groups change only unsafe
credit with strict feasible precedence. All-failed groups center the complete
user cost, including unequal indicator costs. Unknown time terms are omitted
and explicitly audited; they never become invented horizon-end events.
"""
from __future__ import annotations

import math
import torch

SCHEMA = 'world_performance_user_temporal_safety_cost_v2'
CONFIG = dict(schema=SCHEMA, indicator_weights=[.40, .35, .25],
              collision_time_weight=.15, ttc_time_weight=.10, DAC_duration_weight=.10,
              physical_horizon_seconds=4., drivable_samples=41, mixed_max_multiplier=1.)
PROTOCOL = dict(CONFIG,
    fields=['non_drivable_body_by_time', 'official_first_at_fault_collision_seconds',
            'official_first_ttc_infraction_seconds'],
    indicator_cost='.40*(1-NC)+.35*(1-DAC)+.25*(1-TTC)',
    complete_cost='c_ind+.15*(1-t_collision/4) if actual event+.10*(1-t_TTC/4) if actual event+.10*mean(actual DACflags[1:41])',
    all_feasible='original SafetyEP raw credit bit-exact; no temporal modification',
    mixed='only unsafe get centered full-cost correction; multiplier <=1 bounded by half actual feasible-credit gap and half negative slack',
    all_failed='A_safe=mean(full_c)-full_c; not restricted to old coarse ties; A_EP=0',
    missing='retain indicator; omit only the unavailable event/duration term with explicit audit; null is not a 4s event',
    initial_state='actual t=0 event contributes its stated term; DAC t0 flag reported, post-t0 duration still used',
    canonical='original scores/records and actor inputs unchanged; parent finalizes fresh global raw-A std',
    interpretation='weighted failed-trajectory cost, not accident severity, canonical success or a safety guarantee')


def safetyep_parts(components):
    """Original offset-centered SafetyEP, extended only in B/G shape."""
    if (components.ndim != 3 or components.shape[-1] != 4 or components.shape[1] not in (8, 16, 32)
            or not bool(torch.isfinite(components).all()) or not bool(((components >= 0) & (components <= 1)).all())):
        raise ValueError('Finite [scene,G,NC/DAC/TTC/EP] components required')
    x = components.detach().double()
    feasible = x[:, :, :3].eq(1).all(-1)
    f = feasible.double()
    count = f.sum(1, keepdim=True)
    cost = (1-x[:, :, :3]).mean(-1)
    dc = cost-cost[:, :1]
    a_safe = torch.where(count > 0, f-f.mean(1, keepdim=True), dc.mean(1, keepdim=True)-dc)
    ep = x[:, :, 3]
    first = feasible.long().argmax(1, keepdim=True)
    offsets = (ep-ep.gather(1, first))*f
    a_ep = (offsets-offsets.sum(1, keepdim=True)/count.clamp_min(1))*f
    a_ep = torch.where(count >= 2, a_ep, torch.zeros_like(a_ep))
    return dict(feasible=feasible, cost=cost, A_safe=a_safe, A_EP=a_ep, A_raw=a_safe+a_ep)


def _event(record, key, indicator_failed):
    if key not in record:
        return None, 'MISSING_REAL_EVENT_FIELD'
    if record[key] is None:
        return None, 'FAILED_INDICATOR_EVENT_TIME_UNKNOWN' if indicator_failed else 'NO_EVENT_REPORTED'
    if torch.is_tensor(record[key]) and record[key].requires_grad:
        raise ValueError('Official event times must be detached scorer feedback')
    value = float(record[key])
    if not math.isfinite(value) or not 0 <= value <= 4:
        raise ValueError('Official event time must be finite and within physical [0,4]')
    return value, 'ACTUAL_EVENT_TIME'


def real_temporal_cost(record, component):
    """Literal full user cost; unavailable terms stay None in the evidence."""
    if record.get('status') != 'OK':
        raise ValueError('Only successful canonical scorer records may define credit')
    names = ('no_at_fault_collisions', 'drivable_area_compliance', 'time_to_collision_within_bound', 'ego_progress')
    for name, value in zip(names, component.tolist()):
        if float(record['metrics'][name]) != float(value):
            raise ValueError('Actual feedback and canonical candidate components differ')
    nc, dac, ttc, _ = (float(v) for v in component)
    indicator = .40*(1-nc)+.35*(1-dac)+.25*(1-ttc)
    collision, nc_status = _event(record, 'official_first_at_fault_collision_seconds', nc < 1)
    ttc_time, ttc_status = _event(record, 'official_first_ttc_infraction_seconds', ttc < 1)
    nc_term = .15*(1-collision/4) if collision is not None else None
    ttc_term = .10*(1-ttc_time/4) if ttc_time is not None else None
    flags = record.get('non_drivable_body_by_time')
    duration = None;initial = None;first = None
    if flags is not None:
        flags = torch.as_tensor(flags)
        if (flags.requires_grad or flags.shape != (41,) or not bool(torch.isfinite(flags).all())
                or not bool(((flags == 0) | (flags == 1)).all())):
            raise ValueError('Actual detached 41-step binary DAC flags required')
        flags = flags.detach().cpu().bool()
        duration = float(flags[1:41].double().mean())
        initial = bool(flags[0])
        first = int(flags.long().argmax())*.1 if bool(flags.any()) else None
    dac_term = .10*duration if duration is not None else None
    missing = []
    if nc_status not in ('ACTUAL_EVENT_TIME', 'NO_EVENT_REPORTED'):missing.append('collision_time')
    if ttc_status not in ('ACTUAL_EVENT_TIME', 'NO_EVENT_REPORTED'):missing.append('TTC_time')
    if flags is None:missing.append('DAC_duration')
    cost = indicator+sum(term for term in (nc_term, ttc_term, dac_term) if term is not None)
    return cost, dict(status='PARTIAL_REAL_FEEDBACK_INDICATOR_RETAINED' if missing else 'ACTUAL_AVAILABLE_TERMS',
        indicator_cost=indicator, collision_time_term=nc_term, TTC_time_term=ttc_term, DAC_duration_term=dac_term,
        first_at_fault_collision_seconds=collision, first_ttc_infraction_seconds=ttc_time,
        collision_time_status=nc_status, TTC_time_status=ttc_status,
        non_drivable_fraction_after_t0=duration, initially_non_drivable=initial,
        first_non_drivable_sample_seconds=first, omitted_unknown_terms=missing,
        no_event_time_fabricated=True)


@torch.no_grad()
def temporal_credit_parts(components, feedback_records, *, base_parts=None, refinement_weight=1.):
    if refinement_weight != 1.:
        raise ValueError('Register a separate mixed temporal multiplier')
    base = safetyep_parts(components)
    if base_parts is not None:
        for key in ('A_safe', 'A_EP', 'A_raw'):
            if base_parts[key].shape != base[key].shape or not torch.allclose(base_parts[key].to(base[key]), base[key], atol=1e-12, rtol=1e-12):
                raise ValueError('Temporal refinement must start from original SafetyEP')
            base[key] = base_parts[key].detach().to(base[key]).clone()
    b, g = components.shape[:2]
    records = [feedback_records] if b == 1 and len(feedback_records) == g and isinstance(feedback_records[0], dict) else feedback_records
    if len(records) != b or any(len(group) != g for group in records):
        raise ValueError('Real scorer records must match every scene/candidate')
    raw = base['A_raw'].clone()
    weights = components.new_tensor([.40, .35, .25], dtype=torch.float64)
    full_costs = ((1-components[:, :, :3].double())*weights).sum(-1)
    rows = []
    for scene in range(b):
        feasible = base['feasible'][scene]
        info = dict(group=scene, applied=False, status='UNCHANGED_ALL_FEASIBLE', candidate_feedback=[])
        if bool(feasible.all()):
            rows.append(info);continue
        unsafe = ~feasible
        costs = full_costs[scene].clone()
        for candidate in unsafe.nonzero().flatten().tolist():
            cost, evidence = real_temporal_cost(records[scene][candidate], components[scene, candidate])
            costs[candidate] = cost
            info['candidate_feedback'].append(dict(candidate=candidate, **evidence))
        full_costs[scene] = costs
        if bool(feasible.any()):
            minimum_good = base['A_raw'][scene, feasible].min()
            maximum_failed = base['A_raw'][scene, unsafe].max()
            gap = minimum_good-maximum_failed
            if not bool(gap > 0) or not bool(maximum_failed < 0):
                raise AssertionError('Original SafetyEP strict feasible precedence missing')
            offset = costs[unsafe]-costs[unsafe][0]
            correction = offset.mean()-offset
            maximum = float(correction.abs().max())
            amplitude = min(1., float(gap)/(2*maximum), float(-maximum_failed)/(2*maximum)) if maximum else 1.
            raw[scene, unsafe] += amplitude*correction
            info['status'] = 'MIXED_UNSAFE_ONLY_FULL_USER_COST'
        else:
            offset = costs-costs[:1]
            raw[scene] = offset.mean()-offset
            amplitude = 1.
            info['status'] = 'ALL_FAILED_FULL_USER_COST_CENTERED'
        delta = raw[scene]-base['A_raw'][scene]
        info.update(applied=bool(delta.ne(0).any()), complete_cost=costs.cpu().tolist(),
                    raw_delta=delta.cpu().tolist(), mixed_multiplier=amplitude)
        if bool(feasible.any()):
            if not torch.equal(raw[scene, feasible], base['A_raw'][scene, feasible]):
                raise AssertionError('Mixed feasible credit changed')
            if not bool(raw[scene, unsafe].lt(0).all()) or not bool(raw[scene, feasible].min() > raw[scene, unsafe].max()):
                raise AssertionError('Temporal credit violated strict feasible precedence')
        if abs(float(delta.sum())) > 1e-12:
            raise AssertionError('Temporal refinement must preserve group raw-A sum')
        rows.append(info)
    delta = raw-base['A_raw']
    new_safe = base['A_safe'].clone()
    new_safe[~base['feasible']] = raw[~base['feasible']]  # A_EP is exactly zero there; avoid cancellation noise.
    return dict(base, cost=full_costs, original_coarse_cost=base['cost'], A_safe=new_safe,
                A_raw=raw, temporal_delta=delta, temporal_audit=dict(schema=SCHEMA, groups=rows,
                    applied_groups=sum(int(row['applied']) for row in rows),
                    canonical_feedback_modified=False, missing_times_fabricated=False))

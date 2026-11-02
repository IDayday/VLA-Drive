"""S6 cold-start-safe, rank-local detached auxiliary gradient calibration.

The original S1 calibration remains unchanged. This protocol measures the same
per-teacher/microbatch components, never mutates .grad, and never adds a loss.
"""
import copy
import math

from starVLA.rl.qwen_world_performance.nominal_improvement import _gradient_norm

SCHEMA = 'world_verified_quality_PG_floor_aux_calibration_v2'
PROTOCOL = dict(schema=SCHEMA,
    target='max(norm(gIL), .04 * EMA(norm(gPG)))', PG_floor_ratio=.04,
    PG_EMA_decay=.9, PG_EMA_initialization='first strictly positive actual PG norm',
    ratio='target/max(norm(gcomponent),1e-12)', minimum_scale=1e-4, maximum_scale=1e4,
    scale_EMA_decay=.9, calibration_calls=100, measurement_calls=500,
    clock='eligible per-rank teacher-slot loss calls, including inner reuse; not Adam updates',
    cold_start='PG and PG EMA both zero: minimum scales, probe again next call, no eligible/calibration/measurement budget consumed',
    PG_zero_after_initialization='update PG EMA with zero and use its positive historical value',
    gradients='all current trainable parameters; local norms, no cross-rank collective; no .grad mutation',
    freeze='after100 eligible calls freeze scales; diagnose to500 eligible calls, then hold last measurements and PG EMA')


def new_calibration_state():
    return dict(schema=SCHEMA, protocol=copy.deepcopy(PROTOCOL), effective_updates=0,
        eligible_updates=0, cold_start_skips=0, calibration_updates=0, measured_updates=0,
        gradient_probe_calls=0, PG_EMA_updates=0, PG_gradient_EMA=None,
        frozen=False, scale_cfm=None, scale_ode=None, last_norms=None,
        last_target_norm=None, scale_clip_hits_cfm=0, scale_clip_hits_ode=0)


def validate_calibration_state(state):
    if set(state) != set(new_calibration_state()) or state.get('schema') != SCHEMA or state.get('protocol') != PROTOCOL:
        raise ValueError('S6 PG-floor calibration protocol/state fields changed')
    counters = ('effective_updates', 'eligible_updates', 'cold_start_skips', 'calibration_updates',
                'measured_updates', 'gradient_probe_calls', 'PG_EMA_updates', 'scale_clip_hits_cfm', 'scale_clip_hits_ode')
    if any(type(state[k]) is not int or state[k] < 0 for k in counters):
        raise ValueError('S6 calibration counters must be nonnegative Python integers')
    n, eligible, skipped = (state[k] for k in ('effective_updates', 'eligible_updates', 'cold_start_skips'))
    if (eligible + skipped != n or state['calibration_updates'] != min(100, eligible)
            or state['measured_updates'] != min(500, eligible)
            or state['PG_EMA_updates'] != min(500, eligible)
            or state['gradient_probe_calls'] != min(500, eligible) + skipped
            or type(state['frozen']) is not bool or state['frozen'] != (eligible >= 100)):
        raise ValueError('S6 calibration eligible/cold-start/freeze counters changed')
    ema = state['PG_gradient_EMA']
    if (ema is None) != (eligible == 0) or (ema is not None and (not math.isfinite(ema) or ema <= 0)):
        raise ValueError('S6 actual PG EMA cannot reset or be fabricated at cold start')
    for name in ('cfm', 'ode'):
        value = state['scale_' + name]
        if ((value is None) != (n == 0) or
                value is not None and (not math.isfinite(value) or not 1e-4 <= value <= 1e4) or
                not 0 <= state['scale_clip_hits_' + name] <= state['calibration_updates']):
            raise ValueError('Invalid S6 frozen/EMA auxiliary scale')
        if n and eligible == 0 and value != 1e-4:
            raise ValueError('Uninitialized S6 PG must retain minimum scale')
    norms, target = state['last_norms'], state['last_target_norm']
    if n == 0:
        if norms is not None or target is not None:
            raise ValueError('Fresh S6 calibration must have no measured target')
    else:
        if (not isinstance(norms, dict) or set(norms) != {'pg', 'IL', 'cfm', 'ode'}
                or any(not math.isfinite(v) or v < 0 for v in norms.values())
                or not isinstance(target, (int, float)) or not math.isfinite(target) or target < 0
                or target != max(norms['IL'], .04 * (ema or 0.))):
            raise ValueError('S6 measured target/PG EMA/component norms changed')
    return state


def calibrate_components(model, pg, IL, cfm, ode, state):
    validate_calibration_state(state)
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise ValueError('No actual trainable scope for S6 calibration')
    measured = state['eligible_updates'] < 500
    cold_start = False
    if measured:
        norms = {k: _gradient_norm(v, params) for k, v in [('pg', pg), ('IL', IL), ('cfm', cfm), ('ode', ode)]}
        state['last_norms'] = norms
        state['gradient_probe_calls'] += 1
        previous = state['PG_gradient_EMA']
        ema = norms['pg'] if previous is None and norms['pg'] > 0 else previous
        if previous is not None:
            ema = .9 * previous + .1 * norms['pg']
        state['PG_gradient_EMA'] = ema
        target = max(norms['IL'], .04 * (ema or 0.))
        state['last_target_norm'] = target
        cold_start = ema is None
        if cold_start:
            state['cold_start_skips'] += 1
            state['scale_cfm'] = state['scale_ode'] = 1e-4
        else:
            state['measured_updates'] += 1
            state['PG_EMA_updates'] += 1
            if not state['frozen']:
                for name in ('cfm', 'ode'):
                    raw = target / max(norms[name], 1e-12)
                    clipped = max(1e-4, min(1e4, raw))
                    key = 'scale_' + name
                    state['scale_clip_hits_' + name] += int(raw != clipped)
                    state[key] = clipped if state['calibration_updates'] == 0 else max(1e-4, min(1e4, .9 * state[key] + .1 * clipped))
                state['calibration_updates'] += 1
    state['effective_updates'] += 1
    state['eligible_updates'] += int(not cold_start)
    state['frozen'] = state['eligible_updates'] >= 100
    validate_calibration_state(state)
    out = dict(cfm=cfm * state['scale_cfm'], ode=ode * state['scale_ode'])
    out['rescue'] = .75 * out['cfm'] + .25 * out['ode']
    z = cfm.detach().new_zeros(())
    metrics = {f'nominal_gradient_norm_{k}_last_measured': z + v for k, v in state['last_norms'].items()}
    metrics.update(nominal_gradient_norm_measured=z + int(measured),
        nominal_calibration_effective_updates=z + state['effective_updates'],
        nominal_calibration_eligible_updates=z + state['eligible_updates'],
        nominal_calibration_cold_start_skips=z + state['cold_start_skips'],
        nominal_calibration_target_norm=z + state['last_target_norm'],
        nominal_PG_gradient_EMA=z + (state['PG_gradient_EMA'] or 0.),
        nominal_calibration_frozen=z + int(state['frozen']),
        nominal_scale_cfm=z + state['scale_cfm'], nominal_scale_ode=z + state['scale_ode'],
        nominal_scale_cfm_clip_hits=z + state['scale_clip_hits_cfm'],
        nominal_scale_ode_clip_hits=z + state['scale_clip_hits_ode'],
        nominal_scaled_cfm_gradient_norm_last_measured=z + state['scale_cfm'] * state['last_norms']['cfm'],
        nominal_scaled_ode_gradient_norm_last_measured=z + state['scale_ode'] * state['last_norms']['ode'],
        nominal_normed_cfm=out['cfm'].detach(), nominal_normed_ode=out['ode'].detach())
    return out, metrics

"""Evidence-bound, safety-first guidance. GT deltas are explicitly privileged."""
import numpy as np
from common import poses_valid, wrap

METRICS = ('PDMS', 'NC', 'DAC', 'TTC', 'EP', 'CF', 'DDC')


def validate_score(score):
    m = score['metrics']
    if set(m) != set(METRICS) or any(not np.isfinite(m[k]) or not 0 <= m[k] <= 1 for k in METRICS):
        raise ValueError('Missing/invalid official score component')
    return m


def classify(base, target, config):
    b, g = validate_score(base), validate_score(target)
    failure = b['PDMS'] < config['failure']['pdms_below'] or any(b[k] < 1 for k in config['failure']['any_safety_below_one'])
    safety_pass = all(g[k] == 1 for k in ('NC', 'DAC', 'TTC', 'DDC'))
    gain = g['PDMS'] - b['PDMS']
    accepted = failure
    reason = 'accepted_all_gt' if accepted else 'base_not_failure'
    return dict(base_failure=failure, gt_retained=True, gt_safety_pass=safety_pass,
                gt_comfort_pass=g['CF'] == 1, gt_pdms_improves=gain > 0,
                gt_verified_repair=safety_pass and gain > 0,
                pdms_gain=gain, accepted=accepted, reason=reason,
                target_policy='all_gt_without_score_safety_or_gain_filter')


def alignment(pred, gt):
    pred, gt = poses_valid(pred), poses_valid(gt)
    delta = gt - pred
    delta[:, 2] = wrap(delta[:, 2])
    # Initial-ego axes and instantaneous predicted-heading axes are distinct.
    c, s = np.cos(pred[:, 2]), np.sin(pred[:, 2])
    longitudinal = c * delta[:, 0] + s * delta[:, 1]
    lateral = -s * delta[:, 0] + c * delta[:, 1]
    rows = [dict(waypoint=i+1, time_s=(i+1)*.5, delta_x_m=float(d[0]), delta_y_m=float(d[1]),
                 delta_heading_rad=float(d[2]), delta_heading_deg=float(np.degrees(d[2])),
                 path_longitudinal_m=float(longitudinal[i]), path_lateral_m=float(lateral[i]))
            for i, d in enumerate(delta)]
    return dict(privilege='oracle_gt_alignment', sign='GT minus Base; x forward, y left, positive heading counterclockwise',
                heading_is_not_steering_angle=True, waypoints=rows,
                mean_delta_x_m=float(delta[:, 0].mean()), mean_delta_y_m=float(delta[:, 1].mean()),
                final_delta_x_m=float(delta[-1, 0]), final_delta_y_m=float(delta[-1, 1]),
                final_delta_heading_deg=float(np.degrees(delta[-1, 2])),
                lateral_sign_changes=bool((delta[:, 1] > .2).any() and (delta[:, 1] < -.2).any()))


def _event_text(event):
    location = event['ego_rear_axle_local']
    anchor = 'at the initial state' if event.get('at_initial_state') else f"near future waypoint {event['nearest_waypoint_1based']}"
    prefix = f"At t={event['time_s']:.2f}s, {anchor}, the simulated ego rear axle is at x={location[0]:.2f}m, y={location[1]:.2f}m"
    if event['kind'] == 'NC':
        return prefix + f"; the scorer records an at-fault {event['collision_type']} with {event['object_type']} (track {event['object_token']}). This is the ego location, not an estimated contact point."
    if event['kind'] == 'TTC':
        return prefix + f"; the official TTC check finds a projected conflict with {event['object_type']} at look-ahead {event['projection_horizon_s']:.2f}s. This horizon is a tested projection, not an exact continuous TTC."
    return prefix + '; the scorer marks the vehicle footprint as outside the drivable area.'


def build_guidance(pred, gt, base_score, gt_score, navigation, config):
    b, g = validate_score(base_score), validate_score(gt_score)
    eligibility = classify(base_score, gt_score, config)
    a = alignment(pred, gt)
    diagnoses = []
    names = {'NC':'at_fault_collision', 'DAC':'drivable_area_violation', 'TTC':'insufficient_collision_margin',
             'DDC':'driving_direction_violation', 'EP':'insufficient_progress', 'CF':'comfort_violation'}
    for metric in config['guidance']['safety_priority']:
        if b[metric] >= 1:
            continue
        events = [e for e in base_score['events'] if e['kind'] == metric]
        if metric == 'EP' and (b['NC'] == 0 or b['DAC'] == 0):
            diagnoses.append(dict(metric=metric, diagnosis='progress_zeroed_by_safety_gate', priority='secondary',
                evidence=f"EP={b['EP']:.3f} is safety-gated; raw centerline progress={base_score['raw_progress_m']:.2f}m. EP alone does not establish insufficient motion.", event=None))
            continue
        if metric == 'EP' and b['NC'] < 1:
            evidence = f"EP={b['EP']:.3f} is attenuated by NC; raw centerline progress={base_score['raw_progress_m']:.2f}m. Do not infer a need to accelerate from EP alone."
        elif events:
            evidence = _event_text(events[0])
        elif metric == 'EP':
            evidence = f"Official normalized progress={b['EP']:.3f}; raw centerline progress={base_score['raw_progress_m']:.2f}m, GT={gt_score['raw_progress_m']:.2f}m. This is relative to the scorer reference, not simply final x."
        else:
            evidence = f'{metric}={b[metric]:.3f}; detailed event evidence is unavailable.'
        diagnoses.append(dict(metric=metric, diagnosis=names[metric], priority='safety' if metric in ('NC','DAC','TTC','DDC') else 'secondary', evidence=evidence, event=events[0] if events else None))
    summary = ', '.join(f'{k}={b[k]:.3f}' for k in METRICS)
    verdict = 'The initial trajectory fails the collection criteria and should be corrected before use.' if eligibility['base_failure'] else 'The initial trajectory does not meet the failure collection criteria.'
    alignment_text = (f"In the initial ego frame, the mean GT-minus-initial displacement is dx={a['mean_delta_x_m']:+.2f}m (forward), dy={a['mean_delta_y_m']:+.2f}m (left); "
                      f"the final displacement is dx={a['final_delta_x_m']:+.2f}m, dy={a['final_delta_y_m']:+.2f}m and wrapped heading change={a['final_delta_heading_deg']:+.2f} degrees. "
                      f"Navigation command: {navigation}. Heading is vehicle yaw, not steering-wheel angle.")
    if a['lateral_sign_changes']:
        alignment_text += ' Lateral correction changes sign across the horizon; do not apply one uniform left/right shift.'
    per_waypoint = '; '.join(f"w{r['waypoint']}@{r['time_s']:.1f}s: dx={r['delta_x_m']:+.2f}m, dy={r['delta_y_m']:+.2f}m, dyaw={r['delta_heading_deg']:+.2f}deg" for r in a['waypoints'])
    directives = []
    if b['NC'] < 1: directives.append('First avoid the recorded collision region and establish clearance; use the scored GT trajectory as a time-aligned reference, not an unverified uniform translation.')
    if b['DAC'] < 1: directives.append('Keep the entire vehicle footprint inside the drivable area before optimizing progress.')
    if b['TTC'] < 1: directives.append('Increase temporal/spatial clearance at the projected conflict. Reduce closing speed only where the reference timing and verified geometry support it; do not assume braking alone is sufficient.')
    if b['DDC'] < 1: directives.append('Restore route-consistent direction while respecting collision and drivable-area constraints.')
    directives.append('Apply the signed waypoint displacement and wrapped heading corrections toward GT smoothly; address longitudinal progress after hard safety constraints and preserve the navigation intent. Re-score every refined output: partial interpolation toward a safe GT is not guaranteed safe.')
    if not eligibility['gt_safety_pass']:
        directives.append('The recorded GT also has safety violations. It is retained as a privileged reference, not a verified safe repair; matching it alone does not establish safety.')
    if not eligibility['gt_pdms_improves']:
        directives.append('The recorded GT does not improve PDMS over the initial trajectory. It is retained by the all-GT supervision policy; do not claim a measured repair gain.')
    focus = next((d['diagnosis'] for d in diagnoses if d['priority']=='safety'), diagnoses[0]['diagnosis'] if diagnoses else 'none')
    text = ('Guidance:\n1. Trajectory Verdict: ' + verdict + ' Metric summary: ' + summary + '.\n'
            '2. Planning Diagnosis: ' + ' '.join(d['evidence'] for d in diagnoses) + '\n'
            '3. Expert Alignment: ' + alignment_text + ' Time-aligned corrections: ' + per_waypoint + '\n'
            '4. Refinement Directive: ' + ' '.join(directives) + f' Primary focus: {focus}.')
    return dict(schema='safety_first_all_gt_guidance_v3', privilege='posthoc_scores_events_and_gt_alignment',
                eligibility=eligibility, verdict=verdict, metric_summary=b, gt_metric_summary=g,
                planning_diagnosis=diagnoses, expert_alignment=a, refinement_directive=directives,
                primary_focus=focus, navigation=navigation, guidance_text=text)

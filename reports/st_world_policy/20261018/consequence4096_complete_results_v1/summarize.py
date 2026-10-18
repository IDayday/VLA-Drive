"""Bounded read-only aggregation of completed official evaluations. No bootstrap rerun."""
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import resource
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
ROOT = Path('/mnt/project/st-world-policy-20261016/formal/evaluation')
BUDGET = 16 * 1024**2
RECONNAISSANCE_RESERVE = 8 * 1024**2
resource.setrlimit(resource.RLIMIT_AS, (512 * 1024**2, 512 * 1024**2))
resource.setrlimit(resource.RLIMIT_CPU, (30, 30))
pins = []
bytes_read = 0
start = time.monotonic()


def read(path):
    global bytes_read
    size = path.stat().st_size
    assert RECONNAISSANCE_RESERVE + bytes_read + size <= BUDGET, 'read budget exceeded'
    raw = path.read_bytes()
    assert len(raw) == size
    bytes_read += len(raw)
    pin = {'path': str(path), 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
    pins.append(pin)
    return raw, pin


def read_json(path):
    raw, pin = read(path)
    return json.loads(raw), pin


def close(a, b):
    assert math.isclose(a, b, abs_tol=1e-12, rel_tol=1e-12), (a, b)


def mean(values):
    return math.fsum(values) / len(values)


audit, audit_pin = read_json(REPO / 'reports/st_world_policy/20261018/consequence4096_training_signal_audit_v1/TRAINING_SIGNAL_AUDIT.json')
assert audit['matched_sample_order_and_lr_across_all_arms'] is True
assert len({x['sample_order_and_lr_sha256'] for x in audit['arms'].values()}) == 1
assert len({x['frozen_encoder_hash_recorded'] for x in audit['arms'].values()}) == 1
analyses = {}
expected = {'C1': 'ae8092042b5d85c8ff760f3280c1022774b67a05a35c49ae8c6258222f17557e',
            'C4': '482620fce5fba0b4151edf54cf107cf379e574b826f6327b3b8287f80f936a11'}
for arm in ('C1', 'C2', 'C3', 'C4'):
    analysis, pin = read_json(ROOT / f'analyses/{arm}_minus_C0_COMMON_ENDPOINT_dev1696_v1/ANALYSIS.json')
    if arm in expected:
        assert pin['sha256'] == expected[arm]
    assert analysis['status'] == 'COMPLETE' and analysis['stage'] == 'COMMON_ENDPOINT'
    assert analysis['validity'] == {'invalid': 0, 'paired_valid': 1696, 'requested': 1696, 'unresolved': 0}
    assert analysis['treatment']['checkpoint']['completed'] == analysis['control']['checkpoint']['completed'] == 4096
    analyses[arm] = analysis
assert len({json.dumps(a['statistics_definition'], sort_keys=True) for a in analyses.values()}) == 1
assert len({a['population_sha256'] for a in analyses.values()}) == 1
assert len({a['treatment']['checkpoint']['parent_checkpoint_sha256'] for a in analyses.values()}
           | {a['control']['checkpoint']['parent_checkpoint_sha256'] for a in analyses.values()}) == 1
assert len({json.dumps(a['treatment']['checkpoint']['inference_contract'], sort_keys=True) for a in analyses.values()}
           | {json.dumps(a['control']['checkpoint']['inference_contract'], sort_keys=True) for a in analyses.values()}) == 1
definitions = analyses['C1']['statistics_definition']
assert definitions['bootstrap_replicates'] == 2000 and definitions['bootstrap_seed'] == 20261017
assert definitions['bootstrap_unit'] == 'paired_log_cluster'

complete, scenes = {}, {}
metric_names = None
for arm in ('C0', 'C1', 'C2', 'C3', 'C4'):
    directory = ROOT / f'jobs/{arm}_step004096_dev1696_seed42_v1/official_score'
    c, _ = read_json(directory / 'COMPLETE.json')
    assert c['status'] == 'COMPLETE' and c['requested_scenes'] == c['valid_scored'] == 1696
    assert c['invalid'] == c['unresolved'] == 0 and c['score_units'] == '0..1'
    assert c['official_single_candidate_reference'] and c['deployment_candidates_per_scene'] == 1
    assert c['checkpoint_scope'] == 'FORMAL' and c['diagnostic_weights'] is False
    raw, pin = read(directory / 'SCENES.csv')
    assert pin['sha256'] == c['scenes_csv_sha256']
    rows = list(csv.DictReader(io.StringIO(raw.decode())))
    assert len(rows) == 1696 and len({r['scene_id'] for r in rows}) == 1696
    metric_names = list(c['aggregate']['metrics'])
    for row in rows:
        for k in metric_names:
            row[k] = float(row[k])
            assert math.isfinite(row[k]) and 0 <= row[k] <= 1
    by_id = {r['scene_id']: r for r in rows}
    assert len({r['source_log'] for r in rows}) == 16
    for k in metric_names:
        close(mean([r[k] for r in rows]), c['aggregate']['metrics'][k])
    assert sum(r['score'] <= definitions['zero_epsilon'] for r in rows) == c['aggregate']['zero_count']
    complete[arm], scenes[arm] = c, by_id
    if arm != 'C0':
        a = analyses[arm]
        assert pin['sha256'] == a['treatment']['scenes_csv_sha256']
        assert c['identity'] == a['treatment']['score_identity']
        assert set(by_id) == set(scenes['C0'])
        assert all(r['source_log'] == scenes['C0'][sid]['source_log'] for sid, r in by_id.items())
        assert a['control']['scenes_csv_sha256'] == complete['C0']['scenes_csv_sha256']
        for k in metric_names:
            close(mean([r[k] for r in rows]), a['statistics']['metrics'][k]['treatment']['mean'])
            close(mean([by_id[sid][k] - scenes['C0'][sid][k] for sid in by_id]), a['statistics']['metrics'][k]['delta']['mean'])

labels = {'C0': 'BC continuation', 'C1': 'BC + total PDMS feedback', 'C2': 'BC + road penalty',
          'C3': 'BC + no-at-fault collision penalty', 'C4': 'BC + TTC penalty'}
arms = {}
for arm in complete:
    c, train = complete[arm], audit['arms'][arm]
    assert train['result_completed'] == 4096
    assert train['unique_scene_ids_in_scalar_log'] == 101592
    assert train['frozen_encoder_unchanged_recorded'] is True
    window = train['windows']['all_1_4096']
    assert window['updates'] == 4096 and window['first_step'] == 1 and window['last_step'] == 4096
    assert window['counts']['original_GT_scenes'] == 131064
    assert window['counts']['BC_repeat8_presentations'] == 1048512
    result = {'label': labels[arm], 'complete': c,
        'checkpoint': analyses['C1']['control']['checkpoint'] if arm == 'C0' else analyses[arm]['treatment']['checkpoint'],
        'training': {
        'unique_scene_ids_recorded': train['unique_scene_ids_in_scalar_log'],
        'previously_verified_steps_receipt_not_reread': train['steps'],
        'recipe': train['recipe'], 'sample_order_and_lr_sha256': train['sample_order_and_lr_sha256'],
        'feedback_protocol_hash': train['feedback_protocol_hash'],
        'frozen_encoder_hash_recorded': train['frozen_encoder_hash_recorded'],
        'frozen_encoder_unchanged_recorded': train['frozen_encoder_unchanged_recorded'],
        'full_window': window}, 'paired_analysis': None, 'csv_cross_checks': None}
    if arm != 'C0':
        a = analyses[arm]
        result['paired_analysis'] = {'statistics': a['statistics'], 'checkpoint': a['treatment']['checkpoint'],
            'analysis_verification_boundary': a['verification_boundary']}
        base, treatment = scenes['C0'], scenes[arm]
        eps = definitions['zero_epsilon']
        diffs = [treatment[s]["score"] - base[s]["score"] for s in base]
        transition = {}
        for k in ('no_at_fault_collisions', 'drivable_area_compliance', 'time_to_collision_within_bound', 'comfort', 'driving_direction_compliance'):
            transition[k] = {
                'control_below_full': sum(base[s][k] < 1 - eps for s in base),
                'treatment_below_full': sum(treatment[s][k] < 1 - eps for s in base),
                'new_below_full': sum(base[s][k] >= 1 - eps and treatment[s][k] < 1 - eps for s in base),
                'restored_full': sum(base[s][k] < 1 - eps and treatment[s][k] >= 1 - eps for s in base),
                'improved': sum(treatment[s][k] > base[s][k] + eps for s in base),
                'worsened': sum(treatment[s][k] < base[s][k] - eps for s in base)}
        flags = {'new_zero': 0, 'zero_repaired': 0, 'high_to_zero': 0, 'high_large_drop': 0, 'high_any_drop': 0}
        for s in base:
            b, t = base[s]['score'], treatment[s]['score']
            flags['new_zero'] += b > eps and t <= eps
            flags['zero_repaired'] += b <= eps and t > eps
            flags['high_to_zero'] += b >= definitions['high_score_threshold'] and t <= eps
            flags['high_large_drop'] += b >= definitions['high_score_threshold'] and t < b - definitions['high_drop_threshold']
            flags['high_any_drop'] += b >= definitions['high_score_threshold'] and t < b - eps
        for k, count in flags.items():
            assert count == a['statistics']['events'][k]['count']
        result['csv_cross_checks'] = {'denominator': 1696, 'score_improved': sum(x > eps for x in diffs),
            'score_worsened': sum(x < -eps for x in diffs), 'score_tied_within_1e_minus8': sum(abs(x) <= eps for x in diffs),
            'score_delta_min': min(diffs), 'score_delta_max': max(diffs), 'events': flags, 'component_transitions': transition}
    arms[arm] = result

out = {'schema': 'st_consequence4096_complete_results_v1', 'status': 'FORMAL_COMPLETE_FOR_REGISTERED_C0_C4_4096_MATRIX',
    'analysis_base_sha': 'cf702132c37482a737246cd570b629284b3bc341',
    'training_source_sha': audit['training_source_sha'], 'evaluation_split': 'fixed DEV; not Navtest',
    'population_sha256': analyses['C1']['population_sha256'],
    'analysis_plan_sha256_by_comparison': {arm: a['plan_sha256'] for arm, a in analyses.items()},
    'score_units': '0..1 in JSON; percent/percentage-points in CSV and prose where explicitly labelled',
    'statistics_definition': definitions, 'arms': arms, 'sources': pins,
    'scientific_status': {
        'CANONICAL_TRUE_FEEDBACK_TRAINING': 'FORMAL_COMPLETE_4096_COMMON_ENDPOINT',
        'PRACTICAL_HELDOUT_GAIN': 'NOT_ESTABLISHED',
        'C1_PDMS_GAIN': 'INCONCLUSIVE_CI_CROSSES_ZERO',
        'C2_ROAD_GAIN': 'NO_GAIN_SMALL_NEGATIVE_DIFFERENCE',
        'C3_COLLISION_GAIN': 'INCONCLUSIVE_ONE_DAC_ZERO_REPAIR_NOT_NC_GAIN',
        'C4_TTC_GAIN': 'NO_GAIN_SMALL_NEGATIVE_DIFFERENCE',
        'RISK_GUARANTEE': 'NOT_ESTABLISHED_SINGLE_SEED_16_LOGS',
        'POLICY_GRADIENT_RELATIVE_TO_BC': 'UNMEASURED',
        'PROCESS_TIME_CREDIT_ASSIGNMENT': 'NOT_USED_BY_THESE_TERMINAL_REWARDS',
        'G4_G8_INFORMATION_DIAGNOSTIC': 'NOT_RUN_BY_THIS_REPORT',
        'ROAD_PROXY_REPAIR_EFFICACY': 'NOT_ESTABLISHED',
        'FEATURE_DEVELOPMENT': 'PAUSED_BY_USER',
        'NEW_TRAINING_DISPATCHED': False, 'BOOTSTRAP_RECOMPUTED': False},
    'next_actions_without_launch_authorization': [
        'Preserve all C0-C4 endpoints; do not extend or tune rewards from this DEV ranking.',
        'Pre-register a bounded TRAIN-only paired G4 subset G8 diagnostic under original plan section 9.2, with the same policy/noise/context and full scoring cost.',
        'Measure policy-only and BC-only gradients and their cosine/update scale on a fixed TRAIN diagnostic before attributing C1 weakness to dominance, scale, or estimator insensitivity.',
        'Continue separately reviewed command-space road-proxy direction validation under section 10; an all-DAC-full sample cannot demonstrate failure repair.',
        'Only evidence-supported proposals proceed to independent correctness/resource review; this report approves neither C5 nor a new formal trainer.'],
    'limitations': audit['limitations'] + [
        'Existing paired bootstrap is reused, not recomputed. Unadjusted exploratory intervals, one training seed and 16 logs do not establish a familywise or general safety claim.',
        'The narrow negative C2/C4 intervals concern tiny effects and do not establish a practically important deficit.',
        'This aggregation rechecks summary/CSV hashes and row statistics, not model bytes, simulator runs, trajectories or source metadata.',
        'Step wall is measured descriptive execution time; setup/checkpoint/cleanup, retry cost and integrated GPU energy/utilization are not measured here.',
        'First-event times were recorded by the feedback pipeline but any-event reward does not use them for process credit assignment.',
        'Zero new severe failures in 1696 scenes is an observation, not a guarantee on new scenes.',
        'No feature-side work or unrelated paused experiment is resumed.'],
    'aggregation_budget': {'limit_bytes': BUDGET, 'reconnaissance_reserved_bytes': RECONNAISSANCE_RESERVE,
        'actual_bytes_read_by_script': bytes_read, 'cumulative_upper_bound_bytes': bytes_read + RECONNAISSANCE_RESERVE,
        'max_address_space_bytes': 512 * 1024**2, 'cpu_limit_seconds': 30, 'wall_seconds': time.monotonic() - start,
        'peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, 'gpu_used': False, 'official_score_calls': 0}}

(HERE / 'RESULTS.json').write_text(json.dumps(out, indent=2, ensure_ascii=False) + '\n')
fields = ['arm', 'label', 'scenes', 'logs', 'PDMS_percent', 'delta_pp_vs_C0', 'delta_ci95_low_pp', 'delta_ci95_high_pp',
          'NC_percent', 'DAC_percent', 'TTC_percent', 'EP_percent', 'comfort_percent', 'DDC_percent', 'zeros', 'new_zeros',
          'repaired_zeros', 'high90_to_zero', 'high90_drop_over5pp', 'new_stop_proxy', 'nonzero_advantage_scene_presentations',
          'scene_presentations', 'rollout_score_presentations', 'normal_update_mean_seconds', 'normal_update_p50_seconds',
          'normal_update_p95_seconds', 'summed_normal_update_wall_hours']
with (HERE / 'RESULTS.csv').open('w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fields, lineterminator='\n')
    writer.writeheader()
    for arm, r in arms.items():
        c = r['complete']; m = c['aggregate']['metrics']; w = r['training']['full_window']
        a = r['paired_analysis']; d = a['statistics']['metrics']['score']['delta'] if a else None
        row = {'arm': arm, 'label': r['label'], 'scenes': 1696, 'logs': 16, 'PDMS_percent': 100 * m['score'],
            'delta_pp_vs_C0': 100 * d['mean'] if d else 0,
            'delta_ci95_low_pp': 100 * d['ci95'][0] if d else '', 'delta_ci95_high_pp': 100 * d['ci95'][1] if d else '',
            'zeros': c['aggregate']['zero_count'], 'scene_presentations': w['counts']['original_GT_scenes'],
            'rollout_score_presentations': w['counts']['actual_rollouts'],
            'nonzero_advantage_scene_presentations': w['feedback'].get('nonzero_advantage_scene_presentations', ''),
            'normal_update_mean_seconds': w['slowest_rank_update_seconds']['mean'],
            'normal_update_p50_seconds': w['slowest_rank_update_seconds']['p50'],
            'normal_update_p95_seconds': w['slowest_rank_update_seconds']['p95'],
            'summed_normal_update_wall_hours': w['summed_update_seconds_excluding_checkpoint_setup_cleanup'] / 3600}
        for short, key in [('NC','no_at_fault_collisions'), ('DAC','drivable_area_compliance'), ('TTC','time_to_collision_within_bound'), ('EP','ego_progress'), ('comfort','comfort'), ('DDC','driving_direction_compliance')]:
            row[short + '_percent'] = 100 * m[key]
        if a:
            events = a['statistics']['events']
            row.update({'new_zeros': events['new_zero']['count'], 'repaired_zeros': events['zero_repaired']['count'],
                'high90_to_zero': events['high_to_zero']['count'], 'high90_drop_over5pp': events['high_large_drop']['count'],
                'new_stop_proxy': a['statistics']['stopping_proxy']['new_stop_count']})
        writer.writerow(row)
print(json.dumps({'status': 'PASS_SUMMARY_CSV_AND_IDENTITY_CHECKS', 'bytes_read': bytes_read,
                  'cumulative_upper_bound_bytes': bytes_read + RECONNAISSANCE_RESERVE,
                  'wall_seconds': time.monotonic()-start, 'peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}))

"""Apply the fixed shortlist to existing metadata; no bootstrap, models or scoring."""
import hashlib
import json
from pathlib import Path

TREE = Path('/mnt/project/VLA-Drive-st-world-policy-20261016')
ROOT = TREE / 'reports/st_world_policy/20261016/evaluation'
OUT = Path(__file__).resolve().parent
FILES = [('FIXED_PAIRS_C3_512_SP1_5000_v1', 'SP1_minus_SP0'),
         ('FIXED_PAIRS_C1_1024_SP2_5000_v2', 'SP2_minus_SP1'),
         ('FIXED_PAIRS_C4_512_SP3_5000_v1', 'SP3_minus_SP2')]
read_bytes = 0
pins = {}
def read(path, sha=None, decode=True):
    global read_bytes
    path = Path(path)
    assert path.stat().st_size < 4 << 20
    raw = path.read_bytes()
    read_bytes += len(raw)
    assert read_bytes < 32 << 20
    digest = hashlib.sha256(raw).hexdigest()
    if sha:
        assert digest == sha, str(path)
    pins[str(path)] = digest
    return json.loads(raw) if decode else raw

plan = read(ROOT / 'PREREGISTRATION_EXECUTION_v4.json')
contract = TREE / 'docs/research/WMVLA_ST_CONSEQUENCE_MULTI_AGENT_PLAN_20261017.md'
read(contract, decode=False)
assert plan['shortlist_rule']['maximum_primary_contrasts'] == 1
assert plan['shortlist_rule']['maximum_20k_arms'] == 2
rows = []
arms = {}
for directory, pair in FILES:
    d = read(ROOT / directory / 'RESULTS.json')
    read(ROOT / directory / 'REPORT.md', decode=False)
    assert (d['requested'], d['paired_valid'], d['invalid'], d['unresolved'], d['logs']) == (1696, 1696, 0, 0, 16)
    x, = [x for x in d['results'] if x['comparison'] == pair]
    s = x['statistics']['metrics']['score']
    assert x['stage'] == 'SCREENING'
    assert abs(s['delta']['mean'] - (s['treatment']['mean'] - s['control']['mean'])) < 1e-12
    for side in ('control', 'treatment'):
        a = x[side]
        arm = a['job_id'].split('_')[0]
        if arm in arms:
            assert arms[arm]['score_receipt'] == a['score_complete']
            continue
        receipt = read(**{'path': a['score_complete']['path'], 'sha': a['score_complete']['sha256']})
        assert receipt['status'] == 'COMPLETE'
        assert (receipt['requested_scenes'], receipt['valid_scored'], receipt['invalid'], receipt['unresolved']) == (1696, 1696, 0, 0)
        assert receipt['official_single_candidate_reference'] and receipt['deployment_candidates_per_scene'] == 1
        assert receipt['identity'] == a['score_identity']
        assert receipt['scenes_csv_sha256'] == a['scenes_csv']['sha256']
        read(a['scenes_csv']['path'], a['scenes_csv']['sha256'], decode=False)
        assert abs(receipt['aggregate']['metrics']['score'] - s[side]['mean']) < 1e-12
        checkpoint = a['checkpoint']
        assert checkpoint['completed'] == 5000
        native = Path(checkpoint['run']) / 'checkpoints' / checkpoint['tag'] / 'COMPLETE.json'
        nc = read(native, checkpoint['native_receipt_sha256'])
        assert nc['state']['completed'] == 5000 and nc['world_size'] == 4
        assert nc['receipt_identity'] == checkpoint['receipt_identity']
        assert nc['state']['identity'] == checkpoint['run_identity']
        sizes = {}
        for name, record in nc['files'].items():
            p = native.parent / name
            assert p.is_file() and p.stat().st_size == record['bytes']
            sizes[name] = record['bytes']
        reg = read(a['formal_registration']['path'], a['formal_registration']['sha256'])
        assert reg['execution']['max_updates'] == 20000
        cfgpin = reg['artifacts']['config']
        cfg = read(cfgpin['path'], cfgpin['sha256'])
        arms[arm] = {'score_points': 100*s[side]['mean'], 'zero_count': receipt['aggregate']['zero_count'],
                     'score_receipt': a['score_complete'], 'native_receipt': {'path': str(native), 'sha256': checkpoint['native_receipt_sha256']},
                     'native_state': nc['state'], 'file_sizes_verified_not_hashed': sizes,
                     'registration': a['formal_registration'], 'execution': reg['execution'], 'config': cfgpin}
    rows.append({'comparison': pair, 'control': pair.split('_minus_')[1], 'treatment': pair.split('_minus_')[0],
                 'control_points': 100*s['control']['mean'], 'treatment_points': 100*s['treatment']['mean'],
                 'delta_points': 100*s['delta']['mean'], 'ci95_points_existing': [100*v for v in s['delta']['ci95']],
                 'equal_log_delta_points': 100*s['equal_log_delta_sensitivity'],
                 'events': {k:v['count'] for k,v in x['statistics']['events'].items() if 'count' in v},
                 'safety_metric_improved_degraded_not_failure_threshold_counts': x['safety_transitions'],
                 'full_population': 1696, 'logs': 16, 'source_result': str(ROOT/directory/'RESULTS.json'),
                 'verification_boundary': d['status']})
ranking = sorted(rows, key=lambda x: (-x['delta_points'], [p for _, p in FILES].index(x['comparison'])))
assert ranking[0]['comparison'] == 'SP3_minus_SP2'
report = {'schema': 'st_spatial_shortlist_application_v1', 'reviewer': '/root/temporal_narrow_review',
          'status': 'LOCKED_RULE_APPLIED_TO_EXISTING_COMPLETE_RESULTS', 'rule': plan['shortlist_rule'],
          'selected_contrast': ranking[0]['comparison'], 'retained_arms': ['SP2','SP3'],
          'maximum_additional_updates_per_arm': 15000, 'common_endpoint': 20000,
          'excluded': ['PCA256_minus_384 (new exploratory comparison)', 'SP2_minus_SP0 (descriptive only)', 'all consequence contrasts'],
          'ranked_contrasts': ranking, 'all_four_arms': arms, 'evidence_pins': pins,
          'score_protocol_sha256_inherited': plan['score_protocol_sha256'],
          'metadata_bytes_read': read_bytes, 'native_payloads_read': 0, 'new_bootstrap_or_scoring_or_model_calls': 0,
          'launch_approval': False, 'temporal_recipe_selected_by_this_review': False,
          'lead_provisional_temporal_decision': {
              'source': 'Lead explicit message received during this review, 2026-10-18 UTC; pending formal registration.',
              'recipe': 'Original SP2: teacher384x288, native grid width24/height18, W48 per view, PCA384, K96.',
              'fixed_across': ['T0', 'T1', 'T2', 'T3', 'T4'],
              'loss_scales': {'current': 1.0, 'future_when_enabled': 0.9954445849209375, 'delta_T4_only': 1.6396291934956382},
              'reason': 'Preserve registered temporal composition and existing real4 calibration; do not silently replace PCA384 with new dev-selected256, or W48 with96 based on a screening contrast.',
              'claim': 'Provisional shared controlled recipe, not independently proved optimal spatial representation.',
              'remaining_execution_dependencies': ['full history/provider bindings', 'actual formal registration', 'matching topology and fresh resource admission'],
              'PCA256': 'Separate exploratory mechanism study; not an original shortlist candidate.'},
          'temporal_dependency_interpretation': {
              'literal_requirement': 'Plan 6.2: 在空间筛选固定一套目标/W后运行。',
              'common_endpoint_requirement': 'Plan12.2 and locked shortlist retain both selected spatial arms to the common20k endpoint.',
              'explicit_wait_for_20k_before_any_temporal_work': False,
              'boundary': 'Neither clause explicitly makes all temporal preparation or an explicitly frozen provisional common recipe wait for spatial20k. However this shortlist is a contrast selection, not a scientifically established W96 winner or permission to silently change the shared temporal recipe. Lead must record any common target/W decision; reuse T0 only with strictly matching recipe. No new gate is invented.'},
          'continuation': 'Use each same native run/checkpoint, FP32 masters, optimizer/moments, scheduler20k clock, rank RNG and consumed cursor; preserve global32/world4/micro4, data and target identities. Do not warm-start from exported inference weights or reset time. Actual native load and fresh resource/lease/policy checks remain Lead execution prerequisites, not performed by this metadata review.',
          'risk': 'Single seed42,16logs,dev selection and three exploratory unadjusted contrasts. All intervals cross0. SP3 has improved mean/zero count but NC/TTC declines and new zeros. Rank order is not significance, convergence, no-regression or a final temporal recipe verdict.'}
(OUT/'REVIEW.json').write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n')
print(json.dumps({'selected': report['selected_contrast'], 'arms': report['retained_arms'], 'bytes_read': read_bytes, 'review_sha256': hashlib.sha256((OUT/'REVIEW.json').read_bytes()).hexdigest()}))

"""Bounded metadata/CSV audit; never imports a model or loads checkpoint payloads."""
import csv
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

ROOT = Path('/mnt/project/st-world-policy-20261016')
LEAD = Path('/mnt/project/VLA-Drive-st-world-policy-20261016')
RUNTIME = Path('/mnt/project/VLA-Drive-st-official-runtime-v2-20261016')
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(RUNTIME))
from tools.st_world_policy.check_formal_approval import verify_registration, file_hash, digest

def read(path):
    return json.loads(Path(path).read_text())

def pin(path):
    p = Path(path).resolve()
    return {'path': str(p), 'sha256': file_hash(p)}

def checked(ref):
    assert file_hash(ref['path']) == ref['sha256'], ref['path']
    return read(ref['path'])

def recipe(cfg):
    c = deepcopy(cfg)
    c.pop('continuation', None)
    c['execution'].pop('stop_after')
    c['runtime'].pop('output_root')
    return c

short_path = LEAD/'reports/st_world_policy/20261018/spatial_shortlist_application_v1/REVIEW.json'
short = read(short_path)
for path, expected in short['evidence_pins'].items():
    assert file_hash(path) == expected, path
prereg_path = LEAD/'reports/st_world_policy/20261016/evaluation/PREREGISTRATION_EXECUTION_v4.json'
prereg = read(prereg_path)
assert prereg['shortlist_rule'] == short['rule']
assert short['selected_contrast'] == 'SP3_minus_SP2'
assert prereg['shortlist_rule']['maximum_20k_arms'] == 2
assert prereg['endpoints']['spatial'] == {'SCREENING': 5000, 'COMMON_ENDPOINT': 20000}
parent_prereg = checked(prereg['execution_binding_revision']['parent_preregistration'])
assert parent_prereg['shortlist_rule'] == prereg['shortlist_rule']
rows_by_arm, original_results = {}, {}
for arm in ('SP0', 'SP1', 'SP2', 'SP3'):
    folder = ROOT/f'formal/evaluation/jobs/{arm}_step005000_dev1696_seed42_v1/official_score'
    receipt = checked(short['all_four_arms'][arm]['score_receipt'])
    assert receipt['status'] == 'COMPLETE'
    assert receipt['valid_scored'] == receipt['requested_scenes'] == 1696
    assert receipt['invalid'] == receipt['unresolved'] == 0
    assert receipt['score_units'] == '0..1'
    assert receipt['deployment_candidates_per_scene'] == 1
    assert receipt['official_single_candidate_reference'] is True
    assert receipt['checkpoint_scope'] == 'FORMAL' and not receipt['diagnostic_weights']
    assert file_hash(folder/'SCENES.csv') == receipt['scenes_csv_sha256']
    rows = list(csv.DictReader((folder/'SCENES.csv').open()))
    pairs = {(r['scene_id'], r['source_log']) for r in rows}
    assert len(pairs) == len(rows) == 1696
    assert len({r['source_log'] for r in rows}) == 16
    for row in rows:
        assert all(math.isfinite(float(v)) and 0 <= float(v) <= 1 for k, v in row.items() if k not in ('scene_id', 'source_log'))
    rows_by_arm[arm] = {(r['scene_id'], r['source_log']): float(r['score']) for r in rows}
    mean = math.fsum(r['score'] if isinstance(r['score'], float) else float(r['score']) for r in rows)/len(rows)
    assert abs(mean-receipt['aggregate']['metrics']['score']) < 1e-12
    original_results[arm] = {'receipt': pin(folder/'COMPLETE.json'), 'csv': pin(folder/'SCENES.csv'), 'mean_score': mean, 'zero_count': sum(float(r['score']) <= 1e-8 for r in rows), 'scenes':len(rows), 'logs':16}
assert all(set(x) == set(rows_by_arm['SP0']) for x in rows_by_arm.values())
contrasts = []
for control, treatment in [('SP0','SP1'), ('SP1','SP2'), ('SP2','SP3')]:
    delta = math.fsum(rows_by_arm[treatment][k]-rows_by_arm[control][k] for k in rows_by_arm[control])/1696
    name = treatment+'_minus_'+control
    old = next(r for r in short['ranked_contrasts'] if r['comparison'] == name)
    assert abs(delta*100-old['delta_points']) < 1e-10
    contrasts.append({'comparison':name,'delta_points':delta*100, 'ci95_points_inherited_not_recomputed':old['ci95_points_existing']})
assert max(contrasts,key=lambda c:c['delta_points'])['comparison'] == short['selected_contrast']

arms = {}
for arm in ('SP2', 'SP3'):
    folder = ROOT/f'formal/spatial/registrations/{arm}_fulltrain_seed42_official_20k_v1'
    reg = read(folder/'REGISTRATION.json')
    actual = verify_registration(folder/'REGISTRATION.json', repo=RUNTIME)
    assert actual == read(folder/'FINGERPRINT.json')
    assert '/root/spatial20k_evidence' not in reg['implementer_agents']
    cfg = checked(reg['artifacts']['config'])
    parent_reg = checked(cfg['continuation']['parent_registration'])
    pcfg = checked(parent_reg['artifacts']['config'])
    assert recipe(cfg) == recipe(pcfg)
    assert cfg['execution']['max_updates'] == pcfg['execution']['max_updates'] == 20000
    assert cfg['execution']['stop_after'] == 20000 and pcfg['execution']['stop_after'] == 5000
    for key in ('base_binding','data_manifest','environment','target_contract','teacher_manifest'):
        assert reg['artifacts'][key] == parent_reg['artifacts'][key], key
    assert reg['source_sha'] == parent_reg['source_sha'] == 'fdf25f99b702754787904215bf1c2a2b7515ea75'
    assert reg['source_fingerprint'] == parent_reg['source_fingerprint']
    assert reg['weights_usable_for_formal_init'] and parent_reg['weights_usable_for_formal_init']
    assert reg['run_id'] != parent_reg['run_id']
    parent_run = Path(pcfg['runtime']['output_root'])/parent_reg['run_id']
    checkpoint = parent_run/'checkpoints'/cfg['continuation']['parent_tag']
    receipt = checked({'path':str(checkpoint/'COMPLETE.json'),'sha256':cfg['continuation']['parent_receipt_sha256']})
    assert receipt['state']['completed'] == 5000 and receipt['world_size'] == 4
    assert receipt['state']['termination_reason'] == 'RUNNING_CHECKPOINT'
    assert receipt['state']['counts']['ego_scenes'] == receipt['state']['exposure'] == 159992
    assert receipt['state']['counts']['current'] == 159992*3*96*384
    for name, metadata in receipt['files'].items():
        assert (checkpoint/name).stat().st_size == metadata['bytes'], name
    parent_review_path = Path(cfg['continuation']['parent_registration']['path']).parent/'reviews/EVIDENCE.json'
    parent_review = read(parent_review_path)
    assert parent_review['role'] == 'evidence' and parent_review['status'] == 'PASS'
    assert parent_review['fingerprint']['registration_sha256'] == cfg['continuation']['parent_registration']['sha256']
    assert parent_review['fingerprint'] != actual
    arms[arm] = {'registration':pin(folder/'REGISTRATION.json'), 'fingerprint':actual, 'parent_registration':cfg['continuation']['parent_registration'], 'parent_config':parent_reg['artifacts']['config'], 'parent_checkpoint_receipt':pin(checkpoint/'COMPLETE.json'), 'parent_state':receipt['state'], 'parent_evidence_review_inherited':pin(parent_review_path), 'native_checkpoint_payload_verification':'NOT_RERUN_IN_REVIEW; native loader must validate at launch', 'recipe_exact_ignoring_only_continuation_stop_output':True, 'new_optimizer_updates_maximum':15000}

result = {'schema':'st_spatial20k_actual_evidence_checks_v1', 'created_utc':datetime.now(timezone.utc).isoformat(), 'reviewer_agent':'/root/spatial20k_evidence','review_base_sha':'55cda2d2726919e9760507552def760a4a205eca','status':'PASS', 'preregistered_rule':pin(prereg_path), 'parent_preregistered_rule':prereg['execution_binding_revision']['parent_preregistration'], 'preregistered_utc':prereg['execution_binding_revision']['recorded_utc'], 'shortlist_application':pin(short_path), 'selected_contrast':'SP3_minus_SP2', 'contrasts':contrasts, 'original_results':original_results, 'arms':arms, 'model_calls':0, 'gpu_jobs':0, 'checkpoint_payloads_loaded':0, 'new_scoring_calls':0, 'new_bootstrap_calls':0, 'methods_changed':False, 'native_source':str(RUNTIME), 'native_source_sha':subprocess.check_output(['git','rev-parse','HEAD'],cwd=RUNTIME,text=True).strip(), 'scope':'Registration/method continuity and locked selection, not launch dispatch, resource availability, new payload validation, or scientific success.'}
(OUT/'ACTUAL_CHECKS.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
print(json.dumps({'status':'PASS','selected':result['selected_contrast'],'arms':list(arms),'checks_sha256':file_hash(OUT/'ACTUAL_CHECKS.json')}))

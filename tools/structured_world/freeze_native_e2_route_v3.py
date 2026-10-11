"""New G1-only route after the preserved v2 lexical image residency failure.

CPU-only. Reuse byte-verified shared images through their qualified alias. Never
edit old phase/code/metadata, retry old output, dispatch GPU or relax criteria.
"""
import argparse
import copy
from datetime import datetime, timezone
from pathlib import Path

import execution_e2_provenance as facts
import freeze_native_e2_phase as original
import prepare_native_e2_comparisons as observer
import prepare_execution_e2_envelope as pins

ROOT = Path('/mnt/project/structured-world-fgtr-execution-artifacts/20261011/native_zt3_recollected_p0_v3')


def replace_image_route(argv, stage, alias):
    observer.residency_check(stage['image_root'], stage, alias)
    expected = str(Path(alias['alias'])/'images')
    pins.require(expected == stage['image_root'] and alias['original_stage_bytes_unchanged'] is True
                 and alias['shared_resolved'] == '/mnt/project/structured-world-fgtr-execution-assets/20261011/navsim_current_train_v1', 'Actual shared alias does not match registered stage')
    out = list(argv); facts.argv_value(out, '--image-root')
    out[out.index('--image-root')+1] = stage['image_root']
    return out


def failed_cost(reader, controller, old_run):
    result = reader.read(reader.file(Path(controller)/'RESULTS.json'))
    pins.require(result['status'] == 'FAILED' and len(result['jobs']) == 0, 'Only preserved zero-capture failed attempt can reroute')
    reader.read(reader.file(Path(controller)/'FAILURE.json'))
    process = reader.read(reader.file(Path(old_run)/'PROCESS.json'))
    exited = reader.read(reader.file(Path(old_run)/'PROCESS_EXIT.json'))
    pins.require(exited['returncode'] == 1 and exited['pid'] == process['pid']
                 and exited.get('monitor_error') is None, 'Unexpected original worker failure; retain actual classification')
    _, gate = pins.dependencies()
    wall = gate.scalar(exited['wall_seconds'], 'failed PROCESS wall'); pins.require(wall > 0, 'Unknown failed cost cannot be zero')
    metrics = Path(old_run)/'metrics.jsonl'
    pins.require(not metrics.exists() or not metrics.read_text().strip(), 'Old attempt has updates; do not treat it as startup-only')
    return {'PROCESS': reader.file(Path(old_run)/'PROCESS.json'), 'PROCESS_EXIT': reader.file(Path(old_run)/'PROCESS_EXIT.json'),
            'PID': process['pid'], 'wall_seconds': wall, 'GPU_hours': wall*8/3600, 'updates': 0,
            'classification': 'Preserved infrastructure routing/residency failure before any native training update; not a numerical failure'}


def freeze(prior_phase, failed_controller, output):
    out = Path(output).resolve(); pins.require(not out.exists(), 'NEW prospective phase directory required')
    reader = facts.Reader(); prior_ref = reader.file(prior_phase); prior = reader.read(prior_ref)
    pins.require(prior['schema'] == 'execution_E2_phase_enrollment_v2' and prior['kind'] == 'native', 'Actual prior native phase required')
    prior_manifest = reader.read(prior['manifest']); old_manifest_sha = prior['manifest']['sha256']
    old_A = next(j for j in prior['jobs'] if j['key'] == ['G1_mid', 'A'])
    failed = failed_cost(reader, failed_controller, old_A['run'])
    adapter, gate = pins.dependencies()
    source = prior['training_source']
    pins.require(source['root'] == str(pins.MODEL) and source['git_sha'] == pins.PINS['model'][1], 'Model source changed')
    facts.exact(source['files_sha256'], adapter.source_inventory(pins.MODEL), 'Actual full427 native source changed')
    q = reader.read(prior['qualification'])
    original.qualification(reader, prior['qualification'], prior['qualification_actual_metadata']['qualification_run'],
                           source, prior_manifest['gpu_uuids'])
    pins.require(q['endbank_all17_payload_SHA256_exact'] is True and q['model_source_tree_sha256'] == facts.metadata_hash(source['files_sha256']), 'Actual qualified all17/source proof missing')
    stage_ref = reader.file('/mnt/project/structured-world-fgtr-execution-assets/20261011/navsim_current_train_v1/student_train_v1/local_stage.json')
    stage = reader.read(stage_ref); alias = reader.read(q['shared_alias'])
    manifest = copy.deepcopy(prior_manifest)
    manifest['recommended_selected_cases'] = 'G1_mid'
    manifest['dispatch_ready'] = True
    manifest['route_v3_provenance'] = {'prior_phase': prior_ref, 'failed_cost': failed,
        'prior_phase_bytes_unchanged': True, 'reason': 'Frozen dataset literal local_stage image-root equality; qualified alias resolves to unchanged shared bytes',
        'stage': stage_ref, 'actual_shared_alias': q['shared_alias'], 'qualified_images': q['shared_images'], 'qualified_metadata': q['shared_metadata'],
        'worker_argument_changes_only': ['--image-root lexical alias, physicalsharedpixelsunchanged', '--output NEWrun', '--capture-storage-budget NEWownedpaths samecaps'],
        'model_optimizer_seed_schedule_loss_or_capture_changes': False}
    case = next(c for c in manifest['cases'] if c['name'] == 'G1_mid')
    prior_storage = reader.read(case['storage_budget']); storage = copy.deepcopy(prior_storage)
    case['output_root'] = str(ROOT/'G1_mid')
    pins.require(all(not (ROOT/'G1_mid'/role).exists() for role in facts.ROLES), 'New G1 role output required; never retry v2')
    for row, role in zip(storage['owned_runs'], facts.ROLES):
        pins.require(Path(row['path']).name == role, 'Owned role order changed')
        row['path'] = str(ROOT/'G1_mid'/role)
    storage['scope'] = 'NEW shared v3 G1-only triplet: five45GiB banks +1GiB metadata perrole, same145GiB reserve; no previous payload deletion'
    storage['previous_storage_budget'] = case['storage_budget']
    case['train_args'] = replace_image_route(case['train_args'], stage, alias)
    budget = manifest['budget']
    budget.update(phase_limit=15., already_spent_external=failed['GPU_hours'], reserved_elsewhere=0.,
                  accounting_scope='Fresh G1-only v3 plus actual failed v2 PROCESS cost; other history is disclosed separately, not zero')
    budget['accounting_sources'] = [failed['PROCESS_EXIT']]
    budget['G3_external_counter_rule'] = 'G3 remains unselected/pending; later fresh phase binds all actual G1 A/AA/B and v2 failure costs before its A'
    pins.require(budget['total_internal_limit'] == 30. and budget['user_GPU_hours_limit'] is None, 'Do not silently change internal estimate/user-null')
    out.mkdir(parents=True)
    pins.write_new(out/'OWNED_STORAGE_BUDGET.json', storage)
    case['storage_budget'] = pins.ref(out/'OWNED_STORAGE_BUDGET.json')
    pins.write_new(out/'MANIFEST.json', manifest)
    native = pins.module(pins.COLD/'tools/structured_world/run_native_execution_continuation.py',
                         'a8e69113a3c1a09a2e892e28b244f3d8953427a9aa13d9067b8e18aa0710c572', '_native_route_v3_cold68')
    controller = ROOT/'control/G1/controller'
    plan = native.build_plan(manifest, controller, 'G1_mid')
    registered = native.validate_gpu_budget(manifest, plan['jobs'])
    admission = native.gpu_admission(registered, plan['jobs'], native.actual_process_charges(plan['jobs']), whole_plan=True)
    pins.require(len(plan['jobs']) == 3 and plan['expected_full_native_banks'] == 15
                 and all(j['steps'] == 500 for j in plan['jobs']), 'Original triplet500/fivecapture plan changed')
    for job in plan['jobs']:
        pins.require(facts.argv_value(job['command'], '--image-root') == stage['image_root']
                     and job['retained_global_updates'] == [25001,25032,25100,25250,25500], 'Alias/fiveboundary mismatch')
    reader.recheck()
    pins.require(pins.digest(prior['manifest']['path']) == old_manifest_sha and pins.digest(prior_phase) == prior_ref['sha256'], 'Prior frozen bytes changed')
    pins.write_new(out/'PLAN.json', plan)
    enrollment = copy.deepcopy(prior)
    enrollment.update(registered_UTC=datetime.now(timezone.utc).isoformat(), selected_cases=['G1_mid'],
        manifest=pins.ref(out/'MANIFEST.json'), plan=pins.ref(out/'PLAN.json'),
        jobs=[{'key':[j['case'],j['role']], 'command':j['command'], 'run':j['run'], 'origin':j['origin'], 'mode':j['mode']} for j in plan['jobs']],
        CPU_budget_admission=admission, budget_snapshot={'explicit':registered,'actual_failed_cost':failed},
        G3='Unselected pending fresh G1all5 CPUcomparisons and authorized intermediate retirement; no simultaneous30bank buildup',
        prior_phase_preserved=prior_ref, route_v3_provenance=manifest['route_v3_provenance'],
        execution_environment=q['scoped_FusedAdam_execution_environment'])
    pins.write_new(out/'PHASE_ENROLLMENT.json', enrollment)
    command = [manifest['python'],'-B',str(pins.COLD/'tools/structured_world/run_native_execution_continuation.py'),
               '--manifest',str(out/'MANIFEST.json'),'--cases','G1_mid','--output',str(controller),'--execute']
    pins.write_new(out/'HANDOFF.json', {'phase_enrollment':pins.ref(out/'PHASE_ENROLLMENT.json'), 'manifest':pins.ref(out/'MANIFEST.json'),
        'command_for_parent_ONLY':command, 'required_environment':q['scoped_FusedAdam_execution_environment'],
        'actual_failed_v2_cost':failed, 'input_refs':reader.inputs, 'freezer':pins.ref(__file__),
        'GPU_jobs_started':0,'passed':False,'formal_authorized':False,'thresholds_changed':False})
    return {'phase_enrollment':pins.ref(out/'PHASE_ENROLLMENT.json'),'manifest':pins.ref(out/'MANIFEST.json'),
            'selected_jobs':3,'banks':15,'GPU_jobs_started':0,'passed':False}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prior-phase',required=True,type=Path)
    parser.add_argument('--failed-controller',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args(); print(__import__('json').dumps(freeze(args.prior_phase,args.failed_controller,args.output)))


if __name__ == '__main__':main()

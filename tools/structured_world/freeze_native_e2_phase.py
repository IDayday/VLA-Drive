"""Freeze a NEW shared native mid phase from an actual one-update qualification.

CPU metadata only; no GPU launch or payload tensor load/hash sweep. Qualification
payload audit belongs to its independently completed receipt. Numeric acceptance
is always pending here. Old819/44/869 and all failed attempts remain unchanged.
"""
import argparse
import copy
from datetime import datetime, timezone
import math
from pathlib import Path

import prepare_execution_e2_envelope as pins
import execution_e2_provenance as facts


def qualification(reader, reference, run, source, expected_uuids):
    q = reader.read(reference)
    pins.require(q['passed'] is True and q['host'] == 'training-vlawm-zt3'
                 and set(q['physical_gpu_UUIDs']) == set(expected_uuids)
                 and len(q['physical_gpu_UUIDs']) == 8, 'Fresh actual full8 qualification has not passed')
    run = Path(run).resolve()
    identity = reader.read(reader.file(run/'identity.json'))
    pins.require(identity['training_source_sha'] == source['git_sha']
                 and identity['training_source_tree_sha256'] == facts.metadata_hash(source['files_sha256'])
                 and identity['execution_mode'] == 'io_preserving_v1', 'Actual qualification source/mode changed')
    parent = identity['engineering_continue_origin']
    pins.require(parent['completed'] == 25000, 'Qualification must load the original G1_25k')
    complete = reader.read(reader.file(run/'checkpoints/milestone_0025001/COMPLETE.json'))
    inventory = {'mp_rank_00_model_states.pt'} | {f'zero_pp_rank_{r}_mp_rank_00_optim_states.pt' for r in range(8)} | {f'rng_rank{r}.pt' for r in range(8)}
    pins.require(complete['identity'] == identity['identity'] and complete['completed'] == 25001
                 and complete['ranks'] == 8 and complete['status'] == 'COMPLETE'
                 and set(complete['file_sha256']) == inventory, 'Actual qualification bank incomplete')
    boundary = reader.read(reader.file(run/'RESUME_BOUNDARY_VERIFIED.json'))
    pins.require(boundary['passed'] is True and boundary['identity'] == identity['identity']
                 and boundary['completed'] == 25000 and boundary['all_task_RNG'] == 'exact all ranks', 'Original load boundary did not pass')
    process = reader.read(reader.file(run/'PROCESS.json'))
    exited = reader.read(reader.file(run/'PROCESS_EXIT.json'))
    pins.require(exited['returncode'] == 0 and exited['pid'] == process['pid']
                 and exited.get('monitor_error') is None, 'Qualification PROCESS did not exit0')
    pins.require(type(exited['wall_seconds']) in (int, float) and math.isfinite(exited['wall_seconds'])
                 and exited['wall_seconds'] > 0, 'Qualification actual cost missing')
    argv = process['command']
    pins.require(str(pins.MODEL/'tools/structured_world/train_vla.py') in argv
                 and facts.argv_value(argv, '--execution-mode') == 'io_preserving_v1'
                 and facts.argv_value(argv, '--nproc_per_node') == '8', 'Qualification actual command differs')
    for name, sha in complete['file_sha256'].items():
        p = run/'checkpoints/milestone_0025001'/name
        pins.require(Path(name).name == name and p.is_file() and p.stat().st_size > 0
                     and len(sha) == 64, 'Qualification payload missing or unsafe')
    return {'actual_qualification': reference, 'qualification_run': str(run),
            'actual_source_from_identity': identity['training_source_sha'], 'actual_PID': process['pid'],
            'actual_exit': exited['returncode'], 'actual_completed': complete['completed'],
            'actual_boundary_completed': boundary['completed'], 'actual_PROCESS_GPU_hours': exited['wall_seconds']*8/3600,
            'qualified_checkpoint_is_NOT_500_initialization': True,
            'payload_verification': 'Independent qualification receipt; freeze reader checks small metadata and all17 file existence, not tensor values'}


def freeze(template_path, qualification_path, qualification_sha, qualification_run, budget_path, output,
           selected_cases='G1_mid,G3_mid'):
    out = Path(output).resolve(); pins.require(not out.exists(), 'NEW phase directory required')
    adapter, gate = pins.dependencies()
    reader = facts.Reader()
    template_ref = reader.file(template_path); template = reader.read(template_ref)
    source = {'root': str(pins.MODEL), 'git_sha': pins.PINS['model'][1], 'files_sha256': adapter.source_inventory(pins.MODEL)}
    pins.require(template['source_root'] == source['root'] and template['source_sha'] == source['git_sha']
                 and template['host'] == 'training-vlawm-zt3' and template['candidate_mode'] == 'engineering_fast', 'Foreign native phase template')
    selected = selected_cases.split(',')
    pins.require(selected and len(selected) == len(set(selected)) and set(selected) <= {'G1_mid', 'G3_mid'}, 'Only registered mid phases supported')
    qref = reader.file(qualification_path, qualification_sha)
    proof = qualification(reader, qref, qualification_run, source, template['gpu_uuids'])
    q = reader.read(qref)
    pins.require(q['environment']['packages'] and 'flash_attn' in q['environment']['packages'], 'Actual qualified runtime versions incl flash_attn required')
    old = reader.read(reader.file(pins.HISTORY, pins.HISTORY_SHA))
    for k, value in q['environment']['packages'].items():
        if k in old['environment']:
            facts.exact(value, old['environment'][k], 'Actual qualified package differs from E2 recipe: '+k)
    manifest = copy.deepcopy(template)
    manifest['qualification'] = qref
    manifest['budget'] = reader.read(reader.file(budget_path))
    # Remove planning readiness labels; no resource reservation/idle guarantee.
    manifest.update(dispatch_ready=True, recommended_selected_cases=','.join(selected),
                    qualification_and_ownership='Actual qualified native1 update bound; live full8 resource/UUID/locks rechecked by cold68 at each job',
                    preparation_scope='Prospective native metadata/route only; numerical whole E2 remains pending or rejected')
    for case in manifest['cases']:
        if case['name'] not in selected:
            continue
        root = Path(case['output_root'])
        expected = Path('/mnt/project/structured-world-fgtr-execution-artifacts/20261011/native_zt3_recollected_p0_v2')/case['name']
        pins.require(root == expected and all(not (root/r).exists() for r in facts.ROLES), 'New shared native output required before all A/AA/B')
        case['storage_pool'] = 'SHARED_NEW_NATIVE_RECOLLECTED_P0_V2'
        case['origin_readiness'] = 'Actual immutable original native metadata re-read by cold68; full payload hashes verified by trainer before restore and later be5 CPU comparator'
        storage = reader.read(case['storage_budget'])
        pins.require(storage['approved_for_pilot'] is True, 'Owned shared storage budget not approved')
    native = pins.module(pins.COLD/'tools/structured_world/run_native_execution_continuation.py',
                         'a8e69113a3c1a09a2e892e28b244f3d8953427a9aa13d9067b8e18aa0710c572', '_native_phase_cold68')
    plan = native.build_plan(manifest, out/'controller', ','.join(selected))
    budget = native.validate_gpu_budget(manifest, plan['jobs'])
    # Unknown costs are not silently zero: the supplied manifest snapshots them.
    admission = native.gpu_admission(budget, plan['jobs'], native.actual_process_charges(plan['jobs']), whole_plan=True)
    pins.require(len(plan['jobs']) == 3*len(selected) and plan['expected_full_native_banks'] == 15*len(selected), 'Full five-point triplet plan changed')
    for job in plan['jobs']:
        pins.require(job['mode'] == ('engineering_fast' if job['role'] == 'B' else 'io_preserving_v1'), 'Native role changed')
        pins.require(len(job['retained_global_updates']) == 5, 'Missing registered native capture')
    reader.recheck()
    now = datetime.now(timezone.utc).isoformat()
    out.mkdir(parents=True)
    pins.write_new(out/'MANIFEST.json', manifest)
    pins.write_new(out/'PLAN.json', plan)
    enrollment = {'schema': 'execution_E2_phase_enrollment_v2', 'registered_UTC': now, 'kind': 'native',
        'registration': pins.ref(pins.REG, pins.REG_SHA), 'history819_science_expectations_only': reader.file(pins.HISTORY, pins.HISTORY_SHA),
        'fullgrad869_separate_context': reader.file(pins.GRAD_BINDING, pins.GRAD_BINDING_SHA),
        'training_source': source, 'qualification': qref, 'qualification_actual_metadata': proof,
        'manifest': pins.ref(out/'MANIFEST.json'), 'plan': pins.ref(out/'PLAN.json'),
        'candidate_mode': 'engineering_fast', 'role_modes': {'A': 'io_preserving_v1', 'AA': 'io_preserving_v1', 'B': 'engineering_fast'},
        'jobs': [{'key': [j['case'], j['role']], 'command': j['command'], 'run': j['run'],
                  'origin': j['origin'], 'mode': j['mode']} for j in plan['jobs']],
        'selected_cases': selected, 'outputs_absent_at_enrollment': True, 'formal_authorized': False,
        'whole_E2_passed': False, 'GPU_jobs_started': 0, 'thresholds_changed': False,
        'CPU_budget_admission': admission, 'budget_snapshot': reader.file(budget_path),
        'actual_500_exposure': 'PENDING real all8 count records; epoch tails prevent assuming1500*32',
        'pending': ['Actual500 banks and full native1/32/100/250/500 comparator', 'Actual same-mode recovery and full canonical dev',
                    'Original fullgrad norm_relative/strict failures preserved; no early formal pass']}
    pins.write_new(out/'PHASE_ENROLLMENT.json', enrollment)
    command = [template['python'], '-B', str(pins.COLD/'tools/structured_world/run_native_execution_continuation.py'),
               '--manifest', str(out/'MANIFEST.json'), '--cases', ','.join(selected), '--output', str(out/'controller'), '--execute']
    pins.write_new(out/'HANDOFF.json', {'schema': 'native_E2_phase_freeze_handoff_v1', 'phase_enrollment': pins.ref(out/'PHASE_ENROLLMENT.json'),
        'command_for_parent_ONLY': command, 'no_GPU_dispatched_by_freezer': True, 'passed': False,
        'source_refs': {'freezer': pins.ref(__file__), 'provenance': pins.ref(facts.__file__), 'pins': pins.ref(pins.__file__)},
        'input_refs': reader.inputs, 'immutable_model': source['git_sha'], 'immutable_driver': pins.PINS['driver'][1]})
    return {'status': 'prospective_native_phase_frozen_acceptance_pending', 'phase_enrollment': pins.ref(out/'PHASE_ENROLLMENT.json'),
            'jobs': len(plan['jobs']), 'banks': plan['expected_full_native_banks'], 'GPU_jobs_started': 0, 'passed': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest-template', 'qualification', 'qualification-run', 'gpu-budget', 'output'):
        parser.add_argument('--'+name, required=True, type=Path)
    parser.add_argument('--qualification-sha256', required=True)
    parser.add_argument('--cases', default='G1_mid,G3_mid')
    args = parser.parse_args()
    print(__import__('json').dumps(freeze(args.manifest_template, args.qualification, args.qualification_sha256,
          args.qualification_run, args.gpu_budget, args.output, args.cases)))


if __name__ == '__main__':
    main()

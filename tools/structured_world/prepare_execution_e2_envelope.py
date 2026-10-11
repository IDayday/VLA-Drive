"""CPU-only prospective P0 envelope; no launch, acceptance or old-file mutation.

prepare writes a NEW draft. --freeze additionally requires actual, complete
environment receipts. enroll records an external pre-dispatch binding without
changing cold producers. check-entry delegates to the unchanged V1 assembler.
"""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import socket
import subprocess

BASE = Path('/mnt/project')
MODEL = BASE/'VLA-Drive-structured-world-fgtr-perf-native-v4-frozen-20261011'
COLD = BASE/'VLA-Drive-structured-world-fgtr-perf-validation-v2-frozen-20261011'
FULL = BASE/'VLA-Drive-structured-world-fgtr-perf-fullgrad-v1-frozen-20261011'
FULL_DRIVER = BASE/'VLA-Drive-structured-world-fgtr-perf-fullgrad-driver-v1-frozen-20261011'
FULL_DRIVER_GIT = 'a520f162dbd04b5adc2610ad4ad98efe58f08a65'
FULL_DRIVER_SHA = 'f370dcf4e8a5db560e90e5417a833335a93c21ab76f35b39e67ad91ec64361a0'
ORIGINAL = BASE/'VLA-Drive-structured-world-fgtr-perf-v3-frozen-20261011'
ART = BASE/'structured-world-fgtr-round1-artifacts/20261008/training_efficiency_20261011T0546'
SUPPLEMENT = BASE/'structured-world-fgtr-execution-artifacts/20261011'
PINS = {
    'model': (MODEL, 'be5e72248a46afd43e1bf279c21134c391c5edad'),
    'driver': (COLD, '68f6924bf39e6ab3f98ecfb7f8efc9ec86144318'),
    'fullgrad': (FULL, '74651848e5d9a0f3522f7abdd66c9557b83d1b2b'),
}
REG = MODEL/'reports/structured_world/execution_20261011/INDEPENDENT_ACCEPTANCE_PREREGISTRATION_V1.json'
REG_SHA = '8bc0cc1dec82c88cc73e8cde4e5e9eac1d4d88633339c6fcd6670a1d984a8007'
HISTORY = ART/'P0_V1_PROSPECTIVE_BINDINGS/BINDINGS.json'
HISTORY_SHA = '819d5c268eb2bc14c9f47043fd46abe1f7c2e8f1cd700867d884b7745fc4819e'
GRAD_BINDING = SUPPLEMENT/'p0_fullgrad_v1_prospective/SOURCE_BINDING.json'
GRAD_BINDING_SHA = '8691e3390cce77fcd55a38c90336fff732180b4a9e967b1d09743b9f72af0598'
FORWARD_BINDING_SHA = '44c78ea7aa563a4d6b863081ffd8db7af92d0fdb549c957c334a66a4fd608799'
FULL_SHA = '75f51d573110c775229ca9363daed4a5c7317ed8f24a9ccb3a60f743220b5f55'
COMPARATOR_SHA = '9dede8df91b9def8623c608e9581cb5246647c4870c8e52cdca4f6a8c76cb703'
ASSEMBLER_SHA = 'de25acd42c0db2a40d46ebee9ee8ea07b27adc212ca13f97c144556fbf2fd0f3'
REVIEWER_SHA = 'a4effc9b1e56ea6787de9e37bb57cffb2c83b87f7543e78c04569e01b2e34c55'
PRODUCERS = {
    'forward': (FULL/'tools/structured_world/validate_p0_fullgrad_numerics.py', FULL_SHA),
    'native': (MODEL/'tools/structured_world/compare_execution_checkpoints.py', COMPARATOR_SHA),
    'recovery': (MODEL/'tools/structured_world/compare_execution_checkpoints.py', COMPARATOR_SHA),
    'dev': (COLD/'tools/structured_world/run_execution_dev.py', 'd272cfeffa893a6faf2f9baa04d0bde5936552c5b88ae9b74a277a36cc1e3117'),
}


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text(), parse_constant=lambda v: (_ for _ in ()).throw(ValueError('Nonfinite JSON: '+v)))


def ref(path, expected=None):
    path = Path(path).resolve(); actual = digest(path)
    require(expected is None or actual == expected, 'Pinned bytes changed: '+str(path))
    return {'path': str(path), 'sha256': actual}


def document(reference):
    ref(reference['path'], reference['sha256'])
    return load(reference['path'])


def write_new(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False); stream.write('\n')


def module(path, expected, name):
    ref(path, expected)
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value)
    return value


def dependencies():
    for root, sha in PINS.values():
        require(subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip() == sha, 'Cold commit changed')
        require(not subprocess.check_output(['git', 'status', '--porcelain'], cwd=root, text=True).strip(), 'Cold tree dirty')
    a = module(COLD/'tools/structured_world/assemble_execution_evidence.py', ASSEMBLER_SHA, '_prospective_e2_assembler')
    g = module(COLD/'tools/structured_world/evaluate_execution_acceptance.py', REVIEWER_SHA, '_prospective_e2_reviewer')
    return a, g


def prepare(spec_path, output, freeze=False):
    out = Path(output); require(not out.exists(), 'NEW envelope directory required')
    spec = load(spec_path)
    require(spec['schema'] == 'prospective_execution_E2_envelope_spec_v1', 'Foreign envelope spec')
    adapter, gate = dependencies()
    refs = {'registration': ref(REG, REG_SHA), 'original819': ref(HISTORY, HISTORY_SHA),
            'fullgrad869': ref(GRAD_BINDING, GRAD_BINDING_SHA), 'spec': ref(spec_path)}
    old, supplement, policy = load(HISTORY), load(GRAD_BINDING), load(REG)
    gate.validate_bindings(policy, old, REG_SHA)
    require(supplement['original_acceptance_bindings'] == refs['original819']
            and supplement['forward_source_binding']['sha256'] == FORWARD_BINDING_SHA
            and supplement['producer']['git_sha'] == PINS['fullgrad'][1]
            and supplement['producer']['files_sha256'] == {'tools/structured_world/validate_p0_fullgrad_numerics.py': FULL_SHA}
            and supplement['cases'] == old['cases'], 'Supplement/origins differ from preserved context')
    ref(supplement['forward_source_binding']['path'], FORWARD_BINDING_SHA)
    files = adapter.source_inventory(MODEL)
    require(all(s['git_sha'] == PINS['model'][1] and s['files_sha256'] == files for s in old['sources'].values()), 'Actual native source inventory changed')
    binding = copy.deepcopy(old)
    binding.update(adopted_for_pilot=False, frozen_utc=datetime.now(timezone.utc).isoformat())
    producer_refs = {k: ref(*v) for k, v in PRODUCERS.items()}
    binding['producer_sha256'] = {k: v['sha256'] for k, v in producer_refs.items()}
    # Old environment describes the old container. Never substitute it for the
    # actual requalified host, even when package expectations look identical.
    binding['environment'] = {k: None for k in policy['E1']['required_environment_fields']}
    pending = []
    if spec.get('physical_binding_paused', True):
        pending.append('Physical binding paused by latest user pending shared storage plan')
    if spec.get('storage_plan'):
        document(spec['storage_plan']); refs['storage_plan'] = spec['storage_plan']
    else:
        pending.append('User-requested shared storage plan is not yet bound')
    phase_environments = {}
    for kind in ('forward', 'native', 'recovery', 'dev'):
        phase = spec['phases'][kind]
        if not phase.get('environment_receipt'):
            pending.append(kind+': actual environment receipt absent'); continue
        receipt = document(phase['environment_receipt'])
        require(receipt['host'] == phase['host'] and receipt.get('observed_utc')
                and len(receipt['gpu_uuids']) == len(set(receipt['gpu_uuids'])) == 8
                and all(u.startswith('GPU-') for u in receipt['gpu_uuids']), 'Actual full8 environment/route required')
        gate.time_value(receipt['observed_utc'])
        env = receipt['environment']; require(gate.populated(env), 'Incomplete actual environment')
        require(set(policy['E1']['required_environment_fields']) <= set(env), 'Environment fields missing')
        phase_environments[kind] = env
        refs['environment_'+kind] = ref(**{'path': phase['environment_receipt']['path'], 'expected': phase['environment_receipt']['sha256']})
    if len(phase_environments) == 4:
        first = phase_environments['forward']
        if all(env == first for env in phase_environments.values()):
            binding['environment'] = first
        else:
            pending.append('V1 single-environment binding cannot represent differing phase environments')
    if freeze:
        require(not pending, 'Freeze requires actual complete compatible phase environments: '+repr(pending))
        binding['adopted_for_pilot'] = True
        gate.validate_bindings(policy, binding, REG_SHA)
    protected = [{**r, 'path': str(ORIGINAL/r['path'])} for r in (*policy['protected_files'], *policy['background_only'])]
    for r in protected:
        document_ref = ref(r['path'], r['sha256']); refs[document_ref['path']] = document_ref
    required = [list(k) for k in sorted(gate.required_keys(policy))]
    require(len(required) == 28, 'Original complete four-case matrix changed')
    out.mkdir(parents=True)
    write_new(out/'BINDINGS.json', binding)
    envelope = {
        'schema': 'prospective_execution_E2_envelope_v1', 'status': 'frozen_prospective_pilot_only' if freeze else 'draft_pending_actual_environment',
        'created_utc': binding['frozen_utc'], 'bindings': ref(out/'BINDINGS.json'), 'registration': refs['registration'],
        'source_pins': {k: {'root': str(root), 'git_sha': sha} for k, (root, sha) in PINS.items()},
        'actual_training_source': PINS['model'][1], 'locked_scientific_origin_source': policy['scientific_training_source'],
        'source_inventory_sha256': adapter.metadata_hash(files), 'source_file_count': len(files),
        'producers': producer_refs, 'assembler': ref(COLD/'tools/structured_world/assemble_execution_evidence.py', ASSEMBLER_SHA),
        'envelope_builder': ref(__file__), 'physical_binding_paused': spec.get('physical_binding_paused', True),
        'storage_plan': spec.get('storage_plan'),
        'reviewer': ref(COLD/'tools/structured_world/evaluate_execution_acceptance.py', REVIEWER_SHA),
        'fullgrad_source_binding': refs['fullgrad869'], 'history819': refs['original819'],
        'history44': supplement['forward_source_binding'], 'routing_d8': spec.get('routing_d8'),
        'phases': spec['phases'], 'input_refs': refs, 'required_receipt_keys': required,
        'pending': pending, 'passed': False, 'accepted_mode': None, 'formal_authorized': False,
        'trainer_consumable': False, 'GPU_jobs_started': 0, 'thresholds_changed': False,
        'legacy_reuse': {
            'original819_context_only': True, 'payload_recovery_verified': False,
            'requirements': ['Actual all-rank payload hashes, COMPLETE/identity/tags/500/PROCESS_EXIT and original ancestry',
                             'Actual full CPU comparisons at 1/32/100/250/500, exact counts and old strict results retained'],
            'new_envelope_eligibility': 'Old candidate B that predates this freeze cannot be relabelled with this binding SHA',
            'current_parent_observation': spec.get('legacy_observation'),
        },
        'hookup': {
            'pre_dispatch': 'enroll records this binding and exact future command/output BEFORE that trial; cold producers remain unchanged',
            'forward': 'P0_FULL_GRADIENT_COMPARISON.json + P0_FORWARD_COMPARISON.json + untouched strict/COMPLETE.json; direct all8 A_A/A_B/AA_B, actual E1 header required',
            'native': 'Unchanged comparator with --origin; strict failure exit2 is preserved, not retried',
            'recovery': 'Comparator produces post-resume statistics; cold68 recovery orchestrator and be5 boundary are separate input refs',
            'dev': 'cold68 run_execution_dev receipt; canonical0c54 FP32 full1696/16 both q0/q_final from matching +500 banks',
        },
        'known_assembler_gaps': [
            'Split16 load boundary is not the +32 boundary required by the unchanged assembler; do not relabel it',
            'Same-mode fast recovery controls differ from whole-binding IO A/AA assignments; truthful phase facts cannot be substituted',
            'Assembler accepts one bindings SHA; mixed historical819/new envelope receipts cannot be declared one complete V1 trial',
            'New prospective source binding does not itself supply any observed E1 check, gradient pass, payload comparison or canonical dev result',
        ],
    }
    if spec.get('routing_d8'):
        document(spec['routing_d8'])
    write_new(out/'ENVELOPE.json', envelope)
    write_new(out/'CRITERIA_UNCHANGED.json', {'registration': refs['registration'], 'E1': policy['E1'],
        'numeric': policy['numeric'], 'E2': policy['tiers']['E2'], 'trial_plan': policy['trial_plan'], 'dev': policy['dev'], 'passed': False})
    write_new(out/'ASSEMBLY_MANIFEST_PENDING.json', {'schema': 'execution_evidence_assembly_manifest_v1', 'entries': []})
    review = gate.evaluate(REG, out/'BINDINGS.json', (), source_root=ORIGINAL)
    write_new(out/'ADVISORY_PENDING.json', review)
    return envelope


def enroll(envelope_dir, trial_spec_path, output):
    envelope = load(Path(envelope_dir)/'ENVELOPE.json')
    binding = document(envelope['bindings']); require(binding['adopted_for_pilot'] is True, 'Draft cannot enroll a candidate')
    trial = load(trial_spec_path)
    require(trial['schema'] == 'prospective_execution_E2_trial_v1', 'Actual trial plan schema required')
    require(trial['key'] in envelope['required_receipt_keys'], 'Unregistered trial key')
    kind = trial['key'][0]
    require(trial['host'] == envelope['phases'][kind]['host'], 'Wrong phase host')
    require(socket.gethostname() in (trial['host'], trial['host']+'-worker-0'), 'Enroll on the actual worker; local existence checks cannot prove a remote path is fresh')
    require(isinstance(trial['command'], list) and trial['command'] and all(isinstance(v, str) for v in trial['command']), 'Exact argv required')
    script = {'forward': PRODUCERS['forward'][0], 'native': MODEL/'tools/structured_world/train_vla.py',
              'recovery': COLD/'tools/structured_world/run_native_execution_recovery.py',
              'dev': PRODUCERS['dev'][0]}[kind]
    require(str(script) in trial['command'], 'Pinned actual entrypoint missing from argv')
    for destination in trial['future_artifact_paths']:
        require(Path(destination).is_absolute() and not Path(destination).exists(), 'Existing output cannot be retrospectively enrolled')
    require(trial['future_artifact_paths'], 'Future observation paths required')
    now = datetime.now(timezone.utc).isoformat()
    require(now >= binding['frozen_utc'], 'Binding is in the future')
    receipt = {'schema': 'prospective_execution_E2_external_enrollment_v1', 'created_utc': now,
        'envelope': ref(Path(envelope_dir)/'ENVELOPE.json'), 'bindings': envelope['bindings'],
        'trial': trial, 'trial_spec': ref(trial_spec_path), 'passed': False, 'formal_authorized': False, 'GPU_jobs_started': 0}
    write_new(output, receipt); return receipt


def check_entry(envelope_dir, entry_path, output):
    adapter, gate = dependencies()
    envelope = load(Path(envelope_dir)/'ENVELOPE.json'); b = document(envelope['bindings'])
    require(b['adopted_for_pilot'] is True, 'Draft cannot qualify observations')
    gate.validate_bindings(load(REG), b, REG_SHA); b['_sha256'] = envelope['bindings']['sha256']
    result = {'status': 'pending', 'passed': False, 'formal_authorized': False, 'trainer_consumable': False, 'GPU_jobs_started': 0}
    try:
        entry = load(entry_path); enrollment = document(entry['external_enrollment'])
        header = document(entry['observation'])
        key = [header[k] for k in ('kind', 'group', 'stage', 'repeat', 'updates')]
        require(enrollment['schema'] == 'prospective_execution_E2_external_enrollment_v1'
                and enrollment['bindings'] == envelope['bindings'] and enrollment['trial']['key'] == key,
                'External enrollment belongs to different binding/trial')
        require(gate.time_value(header['started_utc']) >= gate.time_value(enrollment['created_utc']), 'Candidate predates external enrollment')
        if key[0] == 'forward':
            comparison = document(entry['comparison'])
            require(comparison['producer_source_binding'] == envelope['fullgrad_source_binding']
                    and comparison['producer_sha256'] == FULL_SHA
                    and comparison['AA_B_directly_measured'] is True
                    and comparison['full_parameter_gradient_statistics_captured'] is True,
                    'Foreign/partial gradient producer')
        key, receipt = adapter.build_receipt(entry, b, load(REG), Path(entry_path).parent, {}, gate)
        result.update(status='structural_receipt_only_numeric_review_pending', key=list(key), receipt=receipt)
    except (ValueError, KeyError, OSError, TypeError) as exc:
        result.update(status='not_supported', error=str(exc))
    write_new(output, result); return result


def freeze_forward(runtime_receipt, output_root, output, coordinator_pin=None):
    """Freeze only the imminent12 forward slots; other phases remain pending.

    Live CPU metadata binds versions/hardware. Behavior fields are explicitly
    locked source/recipe expectations, not assertions of a GPU numerical pass.
    """
    require(not Path(output).exists(), 'NEW forward binding directory required')
    adapter, gate = dependencies(); runtime = load(runtime_receipt)
    require(runtime['host'] in ('training-vlawm-zt4', 'training-vlawm-zt4-worker-0')
            and runtime['CUDA_initialized'] is False, 'Actual zt4 CPU runtime receipt required')
    gate.time_value(runtime['observed_utc'])
    require(Path(output_root).is_absolute(), 'Exact future output root required')
    pin = load(coordinator_pin) if coordinator_pin else {'root': str(FULL_DRIVER), 'git_sha': FULL_DRIVER_GIT, 'script_sha256': FULL_DRIVER_SHA}
    driver_root = Path(pin['root']).resolve()
    require(driver_root.parent == BASE and driver_root.name.startswith('VLA-Drive-structured-world-fgtr-perf-fullgrad-driver-')
            and driver_root.name.endswith('-frozen-20261011'), 'Separate cold fullgrad-driver root required')
    require(subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=driver_root, text=True).strip() == pin['git_sha']
            and not subprocess.check_output(['git', 'status', '--porcelain'], cwd=driver_root, text=True).strip(), 'New fullgrad coordinator must be cold')
    relative = 'tools/structured_world/run_p0_fullgrad_probes.py'
    coordinator = ref(driver_root/relative, pin['script_sha256'])
    committed = subprocess.check_output(['git', 'show', pin['git_sha']+':'+relative], cwd=driver_root)
    require(hashlib.sha256(committed).hexdigest() == coordinator['sha256'], 'Coordinator bytes are not committed')
    original = document(ref(HISTORY, HISTORY_SHA)); supplement = document(ref(GRAD_BINDING, GRAD_BINDING_SHA))
    require(supplement['cases'] == original['cases'] and supplement['original_acceptance_bindings'] == ref(HISTORY, HISTORY_SHA), 'Original source context changed')
    files = adapter.source_inventory(MODEL)
    require(all(s['files_sha256'] == files for s in original['sources'].values()), 'Model source inventory changed')
    bound_model = supplement['model']
    require(bound_model['root'] == str(MODEL) and bound_model['git_sha'] == PINS['model'][1]
            and all(bound_model['files_sha256'].get(k) == v for k, v in files.items())
            and all(digest(MODEL/k) == v for k, v in bound_model['files_sha256'].items()), 'Full869 model inventory differs from actual source')
    gpus = [[v.strip() for v in row.split(',')] for row in runtime['gpus']]
    require(len(gpus) == 8 and len({v[0] for v in gpus}) == 8 and len({v[1] for v in gpus}) == len({v[2] for v in gpus}) == 1, 'Full8 unambiguous actual metadata required')
    env = copy.deepcopy(original['environment'])
    env.update(python=runtime['python'], cuda_runtime=runtime['cuda_runtime'], cudnn=runtime['cudnn'],
               nccl=runtime['nccl'], cuda_driver=gpus[0][2], gpu_model=gpus[0][1], **runtime['packages'])
    value = {'schema': 'prospective_execution_E2_forward_binding_v1', 'frozen_utc': datetime.now(timezone.utc).isoformat(),
        'registration': ref(REG, REG_SHA), 'criteria_changed': False,
        'actual_runtime_receipt': ref(runtime_receipt), 'environment': env,
        'recipe_environment_fields_NOT_live_GPU_observations': ['TF32', 'determinism', 'sync_BN', 'activation_checkpoint_policy', 'model_dtype', 'master_dtype', 'optimizer_dtype', 'world_size'],
        'model': copy.deepcopy(supplement['model']), 'sources': original['sources'],
        'cold68_operator': {'root': str(COLD), 'git_sha': PINS['driver'][1]},
        'coordinator': coordinator, 'coordinator_git_sha': pin['git_sha'],
        'coordinator_pin_receipt': ref(coordinator_pin) if coordinator_pin else None,
        'producer': ref(*PRODUCERS['forward']), 'producer_git_sha': PINS['fullgrad'][1],
        'producer_source_binding': ref(GRAD_BINDING, GRAD_BINDING_SHA), 'historical819': ref(HISTORY, HISTORY_SHA),
        'historical44': supplement['forward_source_binding'], 'cases': original['cases'], 'gpu_uuids': [v[0] for v in gpus],
        'output_root': str(Path(output_root)), 'expected_slots': [[g, s, r] for g in ('G1', 'G3') for s in ('initial', 'mid') for r in range(3)],
        'whole_E2_adopted': False, 'native_physical_binding': 'PENDING_SHARED_STORAGE_AND_REQUALIFICATION',
        'recovery': 'PENDING_ACTUAL_SAME_MODE_BOUNDARY_AND_INTERFACE', 'dev': 'PENDING_ACTUAL_FULL_CANONICAL',
        'passed': False, 'formal_authorized': False, 'trainer_consumable': False, 'GPU_jobs_started': 0,
        'scope': 'Prospective supplementary forward-only source/criteria context; no retroactive replacement of819/44/869 or whole28receipt acceptance',
        'start_rule': 'Each actual candidate B start must be >= this freeze; earlier job observations remain in their original context',
    }
    out = Path(output); out.mkdir(parents=True)
    write_new(out/'FORWARD_BINDING.json', value)
    return {**value, 'binding': ref(out/'FORWARD_BINDING.json'), 'status': 'forward_only_frozen_whole_pending'}


def bridge_forward(binding_path, probe_path, output):
    """Parse real frozen-coordinator reports, then apply unchanged forward limits.

    Does not manufacture a complete E1 header or V1 whole-binding observation.
    The old process exit/strict failure are preserved by the cold classifier.
    """
    binding = load(binding_path); require(binding['schema'] == 'prospective_execution_E2_forward_binding_v1', 'Forward-only binding required')
    require(binding['registration'] == ref(REG, REG_SHA) and binding['producer_source_binding'] == ref(GRAD_BINDING, GRAD_BINDING_SHA), 'Forward binding context changed')
    require(binding['producer'] == ref(*PRODUCERS['forward']), 'Forward producer changed')
    document_ref = binding['coordinator']; ref(document_ref['path'], document_ref['sha256'])
    driver_root = Path(document_ref['path']).parent.parent.parent
    require(subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=driver_root, text=True).strip() == binding['coordinator_git_sha']
            and not subprocess.check_output(['git', 'status', '--porcelain'], cwd=driver_root, text=True).strip(), 'Actual cold coordinator changed')
    require(not Path(output).exists(), 'NEW bridge output required')
    adapter, gate = dependencies()
    proto = load(probe_path); job = proto['job']
    require(Path(job['folder']).resolve().is_relative_to(Path(binding['output_root']).resolve()), 'Foreign output cannot enter prospective forward bank')
    require([*job['case'].split('/'), job['repeat']] in binding['expected_slots'], 'Foreign forward slot')
    coordinator = module(document_ref['path'], document_ref['sha256'], '_prospective_actual_fullgrad_coordinator')
    observed = coordinator.classify(job)
    report = document(observed['P0_FULL_GRADIENT_COMPARISON'])
    forward = document(report['forward_observation'])
    frozen = gate.time_value(binding['frozen_utc'])
    require(all(gate.time_value(row['roles']['B']['started_UTC']) >= frozen for row in forward['per_rank']), 'Actual candidate B predates new forward-only freeze')
    expected_model = document(binding['producer_source_binding'])['model']
    require(binding['model'] == expected_model and report['model_source'] == expected_model, 'Actual full869 model inventory differs')
    require(report['producer_source_binding'] == binding['producer_source_binding'], 'Actual gradient source differs')
    policy = load(REG)
    failures, stats = gate.numerical_checks(policy, 'E2', report, native=False)
    require(all(forward['forward_exact'].get(k) is True for k in policy['tiers']['E2']['same_input_forward_exact']), 'Actual exact RNG/BN/inventories missing')
    failures += gate.forward_checks(policy, 'engineering_fast', {'forward_exact': forward['forward_exact']}, Path(probe_path).parent, binding['cases'][job['case']])
    value = {'schema': 'prospective_execution_E2_forward_bridge_v1', 'status': 'not_supported' if failures else 'forward_only_reviewed_whole_pending',
        'binding': ref(binding_path), 'probe': ref(probe_path), 'actual_cold_observation': observed,
        'original_strict': report['original_strict'], 'failures': failures, 'numerical': stats,
        'passed': False, 'formal_authorized': False, 'trainer_consumable': False, 'GPU_jobs_started': 0,
        'whole_receipt_generated': False, 'limits': ['No E1 fact populated from expectations; whole gate remains pending native/recovery/dev', 'Old strict exit/fail remains unchanged; no retry'],
    }
    write_new(output, value); return value


def main():
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest='action', required=True)
    p = sub.add_parser('prepare'); p.add_argument('--spec', required=True, type=Path); p.add_argument('--output', required=True, type=Path); p.add_argument('--freeze', action='store_true')
    e = sub.add_parser('enroll'); e.add_argument('--envelope-dir', required=True, type=Path); e.add_argument('--trial-spec', required=True, type=Path); e.add_argument('--output', required=True, type=Path)
    c = sub.add_parser('check-entry'); c.add_argument('--envelope-dir', required=True, type=Path); c.add_argument('--entry', required=True, type=Path); c.add_argument('--output', required=True, type=Path)
    f = sub.add_parser('freeze-forward'); f.add_argument('--runtime-receipt', required=True, type=Path); f.add_argument('--output-root', required=True, type=Path); f.add_argument('--output', required=True, type=Path); f.add_argument('--coordinator-pin', required=True, type=Path)
    b = sub.add_parser('bridge-forward'); b.add_argument('--binding', required=True, type=Path); b.add_argument('--probe', required=True, type=Path); b.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.action == 'prepare': result = prepare(args.spec, args.output, args.freeze)
    elif args.action == 'enroll': result = enroll(args.envelope_dir, args.trial_spec, args.output)
    elif args.action == 'check-entry': result = check_entry(args.envelope_dir, args.entry, args.output)
    elif args.action == 'freeze-forward': result = freeze_forward(args.runtime_receipt, args.output_root, args.output, args.coordinator_pin)
    else: result = bridge_forward(args.binding, args.probe, args.output)
    print(json.dumps({k: result[k] for k in ('status', 'passed', 'GPU_jobs_started') if k in result}))


if __name__ == '__main__':
    main()

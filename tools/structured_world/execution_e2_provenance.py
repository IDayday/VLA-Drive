"""CPU-only readers for new phase provenance; expectations are never observations.

No torch import, payload loading, dispatch, signals or legacy receipt rewriting.
Native tensor integrity/numerics remain the responsibility of the pinned be5
comparator. A phase enrollment must precede its actual PROCESS, independently
of when the final collection manifest is assembled.
"""
import hashlib
import json
import math
from pathlib import Path

import prepare_execution_e2_envelope as pins

ROLES = ('A', 'AA', 'B')
PAIRS = ('A_A', 'A_B', 'AA_B')


def require(value, message):
    pins.require(value, message)


def canonical(value):
    return json.dumps(value, sort_keys=True, allow_nan=False)


def metadata_hash(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def exact(actual, expected, message):
    require(canonical(actual) == canonical(expected), message)


class Reader:
    def __init__(self):
        self.inputs = {}

    def read(self, reference):
        path = Path(reference['path']).resolve()
        require(reference['sha256'] == pins.digest(path), 'Evidence hash differs: '+str(path))
        self.inputs[str(path)] = reference['sha256']
        return pins.load(path)

    def file(self, path, expected=None):
        reference = pins.ref(path, expected)
        self.inputs[reference['path']] = reference['sha256']
        return reference

    def recheck(self):
        for path, sha in self.inputs.items():
            require(pins.digest(path) == sha, 'Evidence changed during review: '+path)


def argv_value(argv, name):
    require(isinstance(argv, list) and all(isinstance(v, str) for v in argv), 'Actual argv list required')
    values = [argv[i+1] for i, arg in enumerate(argv[:-1]) if arg == name]
    values += [arg.split('=', 1)[1] for arg in argv if arg.startswith(name+'=')]
    require(len(values) == 1, 'Missing/duplicate argv option: '+name)
    return values[0]


def process_facts(reader, process, exit_record, command, enrolled_utc, gate, *, strict_forward=False):
    observed, exited = reader.read(process), reader.read(exit_record)
    exact(observed['command'], command, 'Actual PROCESS argv differs from enrolled plan')
    require(type(observed['pid']) is int and observed['pid'] > 0 and exited['pid'] == observed['pid'], 'PROCESS/EXIT PID mismatch')
    unix = gate.scalar(observed['started_unix'], 'actual PROCESS start')
    require(unix >= gate.time_value(enrolled_utc).timestamp(), 'Actual process predates phase enrollment')
    require(exited['returncode'] in ((0, 1) if strict_forward else (0,)), 'Actual worker failed; do not retry/relabel')
    require(exited.get('monitor_error') is None, 'Infrastructure failure is not a complete observation')
    seconds = gate.scalar(exited['wall_seconds'], 'PROCESS wall')
    require(seconds >= 0, 'Negative process wall')
    exact(argv_value(command, '--nproc_per_node'), '8', 'Actual worker was not full8')
    exact(argv_value(command, '--max_restarts'), '0', 'Worker retry violates trial')
    return {'process': process, 'exit': exit_record, 'pid': observed['pid'], 'started_unix': unix,
            'returncode': exited['returncode'], 'wall_seconds': seconds, 'GPU_hours': seconds*8/3600,
            'actual_command': command}


def phase_enrollment(reader, phase, kind, gate):
    enrollment = reader.read(phase['enrollment'])
    require(enrollment['schema'] == 'execution_E2_phase_enrollment_v2', 'New phase enrollment required; no old819 alias')
    require(enrollment['kind'] == kind and enrollment['registration'] == pins.ref(pins.REG, pins.REG_SHA), 'Wrong phase/criteria enrollment')
    require(enrollment['candidate_mode'] == 'engineering_fast' and enrollment['formal_authorized'] is False,
            'Only prospectively registered P0 is supported')
    gate.time_value(enrollment['registered_UTC'])
    require(enrollment['outputs_absent_at_enrollment'] is True, 'Fresh-output pre-dispatch evidence required')
    require(enrollment['jobs'] and len({canonical(j['key']) for j in enrollment['jobs']}) == len(enrollment['jobs']),
            'Duplicate/empty enrolled jobs')
    source = enrollment['training_source']
    require(source['root'] == str(pins.MODEL) and source['git_sha'] == pins.PINS['model'][1], 'Actual model source must remain be5')
    adapter, _ = pins.dependencies()
    actual = adapter.source_inventory(pins.MODEL)
    exact(source['files_sha256'], actual, 'Native/science source inventory must be complete427, not root/git projection')
    for relative, sha in actual.items():
        reader.file(pins.MODEL/relative, sha)
    allowed = {'A': 'io_preserving_v1', 'AA': 'io_preserving_v1', 'B': 'engineering_fast'}
    if kind == 'recovery':
        allowed = dict.fromkeys(ROLES, 'engineering_fast')
    exact(enrollment['role_modes'], allowed, 'Phase role modes are not truthful registered modes')
    return enrollment


def runtime_facts(reader, reference, expected_environment):
    """Runtime metadata only. TF32/BN/checkpoint/dtypes are not CPU observations."""
    actual = reader.read(reference)
    require(actual.get('CUDA_initialized') is False and actual.get('observed_utc'), 'Actual CPU runtime metadata receipt required')
    rows = [[part.strip() for part in row.split(',')] for row in actual['gpus']]
    require(len(rows) == 8 and all(len(row) == 3 for row in rows)
            and len({r[0] for r in rows}) == 8 and len({r[1] for r in rows}) == len({r[2] for r in rows}) == 1,
            'Actual unambiguous full8 GPU metadata required')
    fields = {'python': actual['python'], 'cuda_runtime': actual['cuda_runtime'], 'cudnn': actual['cudnn'],
              'nccl': actual['nccl'], 'gpu_model': rows[0][1], 'cuda_driver': rows[0][2],
              **{k: actual['packages'][k] for k in ('torch', 'deepspeed', 'transformers', 'flash_attn', 'numpy')}}
    exact(fields, {k: expected_environment[k] for k in fields}, 'Observed runtime differs from locked phase environment')
    return {'observed_runtime': fields, 'gpu_uuids': [r[0] for r in rows], 'host': actual['host'],
            'observed_utc': actual['observed_utc'], 'receipt': reference,
            'not_live_GPU_observations': ['TF32', 'sync_BN', 'activation_checkpoint_policy', 'model_dtype', 'master_dtype', 'optimizer_dtype']}


def science_facts(reader, identity, origin_completed, train_index, source_root, scientific_source):
    """Use the original assembler's exact hash recipes on actual small metadata."""
    require(source_root == str(pins.MODEL), 'Unsupported actual training tree')
    root = Path(source_root)
    index = reader.file(train_index['path'], train_index['sha256'])
    losses = {n: reader.file(root/n)['sha256'] for n in (
        'starVLA/model/framework/vla_structured_fgtr.py', 'starVLA/model/modules/structured_world/losses.py',
        'starVLA/model/modules/structured_world/queries.py', 'starVLA/model/modules/structured_world/grid.py')}
    calibration = {'grid': identity['config']['structured_world']['grid'],
        'geometry_image_size': identity['config']['structured_world']['geometry_image_size'],
        'current_and_GT_identity': identity['current_and_GT_identity'], 'shared_geometry_identity': identity['shared_geometry_identity']}
    order = {'epoch_batches_source_sha256': reader.file(root/'tools/ddpolicy_vehicle/training_state.py')['sha256'],
        'seed': identity['training_seed'], 'world_size': identity['world_size'], 'global_batch': identity['effective_global_batch'],
        'microbatch': identity['microbatch'], 'train_index_sha256': index['sha256'], 'origin_completed': origin_completed}
    schedule = {k: identity['config']['trainer'][k] for k in ('lr_scheduler_type', 'scheduler_specific_kwargs', 'learning_rate')}
    schedule.update(horizon=identity['schedule_horizon'], warmup=identity['schedule_warmup'])
    return {'training_source_sha': scientific_source, 'config_sha256': metadata_hash(identity['config']),
        'training_population_sha256': index['sha256'], 'current_and_GT_identity': identity['current_and_GT_identity'],
        'DINO_identity': identity['DINO_identity'], 'calibration_axis_sha256': metadata_hash(calibration),
        'batch_order_sha256': metadata_hash(order), 'training_seed': identity['training_seed'],
        'global_batch': identity['effective_global_batch'], 'FM_repeat': identity['FM_repeat'],
        'schedule_sha256': metadata_hash(schedule), 'schedule_horizon': identity['schedule_horizon'], 'warmup': identity['schedule_warmup'],
        'loss_definition_sha256': metadata_hash({'source_files': losses, 'structured_world_config': identity['config']['structured_world']}),
        'physical_conversion_sha256': metadata_hash({'grid_source_sha256': losses['starVLA/model/modules/structured_world/grid.py'],
             'wrapper_source_sha256': losses['starVLA/model/framework/vla_structured_fgtr.py'], 'grid': calibration['grid']})}


def native_facts(reader, comparison, source, role_modes, case, train_index, policy, gate):
    require(comparison['schema'] == 'native_execution_checkpoint_comparison_v1'
            and comparison['tool_sha256'] == pins.COMPARATOR_SHA, 'Wrong native comparator')
    require(all(comparison.get(k) is True for k in ('integrity_verified', 'valid_native_inputs', 'comparison_performed'))
            and comparison['coverage']['all_rank_slice_membership_complete'] is True, 'Full actual native payload coverage required')
    require(all(comparison['E1'][p]['passed'] is True and comparison['E1'][p]['failure_count'] == 0 for p in PAIRS), 'Native exact E1 failed')
    origin = comparison['provenance']['origin']
    require(origin['payload_hashes_verified'] is True, 'Actual origin payload verification missing')
    for key in ('identity', 'completed', 'COMPLETE_sha256'):
        exact(origin[key], case['origin'][key], 'Actual origin differs: '+key)
    facts = {}
    for role in ROLES:
        p = comparison['provenance'][role]
        exact(p['source'], source['git_sha'], 'Actual native source differs')
        exact(p['execution_mode'], role_modes[role], 'Actual native role mode differs')
        require(p['payload_hashes_verified'] is True and p['ranks'] == 8, 'All17 actual payloads must be verified')
        folder = Path(p['path']); run = folder.parent.parent
        identity = reader.read({'path': str(run/'identity.json'), 'sha256': p['identity_sha256']})
        exact(identity['identity'], p['identity'], 'Native identity contradiction')
        exact(identity['training_source_sha'], source['git_sha'], 'Observed identity source differs')
        exact(identity['training_source_tree_sha256'], metadata_hash(source['files_sha256']), 'Observed identity full source inventory differs')
        science = science_facts(reader, identity, origin['completed'], train_index, source['root'], policy['scientific_training_source'])
        exact(science, case['science'], 'Actual config/data/calibration/loss/schedule/order differs')
        gate.native_identity_checks(p, source, science, case['origin'], Path('/'))
        groups = reader.file(run/'optimizer_groups.json', p['optimizer_groups_sha256'])
        exact(groups['sha256'], case['optimizer_groups_sha256'], 'Actual optimizer groups differ')
        policy_ref = reader.file(run/'execution_policy.json')
        execution = reader.read(policy_ref)
        exact(execution['mode'], role_modes[role], 'Actual execution policy differs')
        require(execution['TF32'] is False and execution['time_vectorization'] is False
                and execution['geometry_convolution_BF16_autocast'] is False
                and execution['Qwen_language_activation_checkpointing'] is True
                and execution['DiT_activation_checkpointing'] is True, 'P0 recipe flags differ; no no-recompute/precision/timevec admission')
        exact(execution['future_shared_current_spatial_mix_and_action_MLN'], role_modes[role] == 'engineering_fast', 'Actual P0 reuse flag differs')
        exact(identity['effective_training_precision'], identity['base_precision'], 'E3 compute is not E2')
        facts[role] = {'source_from_identity': {'git_sha': identity['training_source_sha'], 'tree_sha256': identity['training_source_tree_sha256']},
            'source_root_from_verified_PROCESS_required_separately': source['root'], 'execution_mode': p['execution_mode'],
            'identity': p['identity'], 'completed': p['completed'], 'science_from_metadata_and_actual_file_hashes': science,
            'origin_from_identity': identity['engineering_continue_origin'], 'optimizer_groups': groups,
            'execution_policy': policy_ref, 'determinism_from_identity': identity['determinism'],
            'FP32_dtype_and_BN_discrete_checks': 'Actual pinned comparator E1; native floating BN separately receives unchanged E2 numeric limits'}
    return facts


def restore_boundary(reader, reference, candidate, origin_completed, split_offset=16):
    actual = reader.read(reference)
    require(type(split_offset) is int and split_offset == 16, 'Registered same-mode split is16; endpoint32 is different')
    require(actual['completed'] == origin_completed+split_offset and actual['identity'] == candidate['identity'],
            'Actual restore boundary is not the candidate split16')
    expected = {'passed': True, 'FP32_model_and_buffers': 'bitwise',
        'FP32_masters': 'bitwise all ranks, actual parameters', 'Adam_moments_steps_and_LR': 'bitwise all ranks',
        'all_task_RNG': 'exact all ranks', 'timing': 'before next forward/backward/update'}
    for k, value in expected.items():
        exact(actual[k], value, 'Actual restore state check differs: '+k)
    return {'actual_restore_completed': actual['completed'], 'endpoint_completed': origin_completed+32,
        'load_boundary_source': reference, 'exact_model_master_moments_BN_RNG_discrete': True,
        'not_relabelled_as_endpoint32': True}


"""Static exact-proposal wiring audit; no imports of builder/model or ledger IO."""
import argparse
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import re
import resource
import stat
import time
from types import SimpleNamespace, ModuleType

HERE = Path(__file__).resolve().parent
PROPOSAL = Path('/mnt/project/VLA-Drive-st-history-scale-independent-v2-20261017/reports/st_world_policy/20261017/history_packed64_registration_proposal_v1')
RUNTIME = Path('/mnt/project/VLA-Drive-st-history-packed-runtime-v1-20261017')
INPUTS, CHECKS, CACHE = [], [], {}
started = time.monotonic()


def io():
    return {k: int(v) for k, v in (line.split(':') for line in Path('/proc/self/io').read_text().splitlines())}


before = io()


def read(path, expected=None):
    path = Path(path)
    if str(path) in CACHE:
        data, digest = CACHE[str(path)]
    else:
        assert path.suffix in ('.json', '.py')
        st = path.lstat()
        assert stat.S_ISREG(st.st_mode) and st.st_size < 4 << 20
        assert sum(x['bytes'] for x in INPUTS) + st.st_size < 32 << 20
        raw = path.read_bytes()
        assert st.st_size == len(raw)
        digest = hashlib.sha256(raw).hexdigest()
        data = json.loads(raw) if path.suffix == '.json' else raw.decode()
        INPUTS.append({'path': str(path), 'sha256': digest, 'bytes': len(raw)})
        CACHE[str(path)] = (data, digest)
    if expected:
        assert digest == expected, str(path)
    return data


def check(name, value):
    assert value, name
    CHECKS.append(name)
    print('PASS', name, flush=True)


def pinned(pin):
    return read(pin['path'], pin['sha256'])


p = read(PROPOSAL / 'PROPOSAL.json')
reg, budget = pinned(p['new_registration']), pinned(p['builder_budget'])
template, allocation = pinned(p['admission_template']), pinned(p['control_reservation_proposal'])
old, new = pinned(p['old_policy']), pinned(p['new_policy'])
selection, cp = pinned(p['fixed_selection']), pinned(p['checkpoint_receipt'])
inventory, addendum = pinned(p['pilot_inventory']), pinned(allocation['separate_existing_monitor_addendum'])
for pin in [*p['controller_sources'], *p['source_pins'], *p['additional_runtime_dependencies'].values(), p['proposal_helper']]:
    pinned(pin)
source_binding = read(RUNTIME / 'reports/st_world_policy/20261017/history_metadata_scale_independent_v2/SOURCE_BINDING.json')
correctness = read(RUNTIME / 'reports/st_world_policy/20261017/history_metadata_scale_independent_v2/REVIEW.json')
resources = read(RUNTIME / 'reports/st_world_policy/20261017/history_packing_resource_independent_v1/REVIEW.json')
for relative, digest in source_binding['source_sha256'].items():
    read(RUNTIME / relative, digest)
for pin in resources['source_pins']:
    read(RUNTIME / pin['path'], pin['sha256'])
check('frozen runtime and prior narrow reviews are bound', p['runtime'] == str(RUNTIME)
      and p['runtime_commit'] == 'b847f5a99dd44ee5256362d380d9d96efb47d35b'
      and correctness['tests']['passed'] == 44 and correctness['tests']['synthetic_only'] is True
      and resources['tests']['passed'] == 14 and resources['status'] == 'PASS_CODE_RESOURCE_ONLY')
check('original fixed64 identity is unchanged', p['fixed_selection']['sha256'] == 'dee94e75957fd8e9734f1669b47552b5cf6c39f719aa9d3298bc87f94aac1c01'
      and selection['split'] == 'train' and len(selection['scenes']) == 64
      and len({row['token'] for row in selection['scenes']}) == 64
      and len({row['log'] for row in selection['scenes']}) == 64)

# Compile ONLY the real argparse factory definitions, not their heavy modules.
def parser_function(text, namespace):
    node = next(n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == 'parser')
    module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    exec(compile(module, '<reviewed-real-parser-definition>', 'exec'), namespace)
    return namespace['parser']

original_parser = parser_function(read(RUNTIME / 'tools/st_world_policy/prepare_history.py'),
                                  {'argparse': argparse, '__doc__': 'Read-only parser audit'})
builder_source = read(RUNTIME / 'tools/st_world_policy/prepare_causal_history.py')
builder_parser = parser_function(builder_source, {'original': SimpleNamespace(parser=original_parser)})
command = reg['command']
args = builder_parser().parse_args(command[3:])
check('registered actual argv reaches the existing build parser',
      command[:3] == ['/root/miniconda3/envs/ddp/bin/python', '-m', 'tools.st_world_policy.prepare_causal_history']
      and args.action == 'build' and args.target_times == [0.] and args.split == 'train'
      and args.record_layout == reg['record_layout'] == 'packed_log_v1'
      and args.mode == 'registered' and args.max_scenes == reg['max_scenes'] == 64
      and args.scene_list == p['fixed_selection']['path'] and args.output == reg['output_root']
      and args.registration == p['builder_budget']['path'] and not args.resume and not args.pin_selection)
check('existing history input and target math retain reviewed source',
      all(source_binding['unchanged_history_semantic_functions'].values())
      and args.tolerance == .05 and args.seed == 20261017
      and budget['target_times_s'] == p['target_times'] == [0.]
      and budget['workers'] == 1 and budget['max_scenes'] == 64
      and reg['inventory_strategy'] == 'immutable_history_shards_v1'
      and 'result=original.run(args)' in builder_source)
check('registered callback sources cover declared builder dependencies', reg['source_pins'] == p['source_pins']
      and reg['callback_protocol'] == 'st_metadata_storage_check_v1'
      and len(reg['source_pins']) == 8)

# Loading the scoped launcher only defines stdlib helpers; never call controls/main.
controller_path = RUNTIME / 'reports/st_world_policy/20261017/full_causal_history_preparation_v1/run_pilot_once.py'
controller_source = read(controller_path)
launcher = ModuleType('history_wiring_launch_source')
launcher.__file__ = str(controller_path)
exec(compile(controller_source, str(controller_path), 'exec'), launcher.__dict__)
controls_ast = next(n for n in ast.parse(controller_source).body if isinstance(n, ast.FunctionDef) and n.name == 'controls')
expected_node = next(n.value for n in controls_ast.body if isinstance(n, ast.Assign)
                     and any(isinstance(t, ast.Name) and t.id == 'expected' for t in n.targets))
expected = eval(compile(ast.Expression(expected_node), '<controller-expected-argv>', 'eval'),
                {'BASE': launcher.BASE, 'ROOT': launcher.ROOT, 'str': str})
check('controller exact parsed-argument checks match registration',
      all(getattr(args, key) == value for key, value in expected.items())
      and template['controller_sources'] == p['controller_sources']
      and template['limits'] == launcher.LIMITS and template['checkpoint_receipt'] == launcher.CP
      and template['runtime'] == str(launcher.RUNTIME)
      and template['output_root'] == str(launcher.ROOT)
      and template['control_root'] == str(launcher.CONTROL_ROOT))
try:
    launcher.validate(template, 0)
except ValueError as exc:
    template_error = str(exc)
else:
    raise AssertionError('Unsigned proposal must not admit execution')
check('template is rejected before actual launch', 'finite admission' in template_error
      and template['status'] == 'PENDING_LEAD_AND_INDEPENDENT_ADMISSION'
      and template['reviewer'] is None and template['issued_at_unix'] is None)
check('same guarded runner executes exact registration arguments',
      'result=m.run_registered_metadata(**guard)' in controller_source
      and 'args = module.parser().parse_args(command[3:])' in read(RUNTIME / 'tools/st_world_policy/metadata_storage.py')
      and 'module.run(args)' in read(RUNTIME / 'tools/st_world_policy/metadata_storage.py'))

check('artifact31 plus separate control1 stays within combined32MiB',
      sum(reg['storage_requirements'].values()) == reg['limits']['max_disk_bytes'] == allocation['artifact_bytes'] == 31 << 20
      and allocation['reserved_bytes'] == template['control_reserved_bytes'] == launcher.CONTROL_CAP == 1 << 20
      and allocation['combined_artifact_control_bytes'] == launcher.LIMITS['max_disk_bytes'] == 32 << 20
      and allocation['root'] == str(launcher.CONTROL_ROOT))
check('packing count and IO budgets are explicit', reg['max_history_shards'] == p['shard_limits']['max_shards_for_64_scenes'] == 64
      and p['shard_limits']['bytes'] == 4 << 20 and p['shard_limits']['records'] == 128
      and reg['max_source_read_bytes'] == launcher.LIMITS['read_bytes'] == 16 << 30
      and reg['read_mib_per_second'] == 32 and reg['cpu_affinity'] == [6, 7]
      and reg['limits']['max_seconds'] == budget['max_seconds'] == launcher.LIMITS['wall_seconds'] == 900
      and reg['limits']['max_rss_bytes'] == launcher.LIMITS['rss_bytes'] == 4 << 30
      and launcher.LIMITS['cleanup_seconds'] == 40 and launcher.LIMITS['max_invocations'] == 1)
required_names = ['METADATA_ADMISSION.json', 'BUILD.lock', 'PREPARE.lock', 'resolved_config.json',
    'SELECTED_SCENES.json', 'HISTORY_MANIFEST.json', 'TEACHER_REQUESTS.json', 'RESOURCE_COST.json',
    'COMPLETE.json', 'CONTROL_HISTORY_STARTED.json', 'CONTROL_HISTORY_RESULT.json',
    'history_shards/000000_0123456789abcdef.jsonseq', 'history_shards/000000_0123456789abcdef.jsonseq.0123.partial']
check('registration permits actual known artifact names and retained partials',
      all(any(re.fullmatch(pattern, name) for pattern in reg['file_patterns']) for name in required_names)
      and not any(re.fullmatch(pattern, 'model.pt') for pattern in reg['file_patterns']))
check('old policy remains intact with one additive declaration', old['claims'] == new['claims'][:-1]
      and len(old['claims']) == p['old_claim_count'] == 10 and len(new['claims']) == 11
      and {k:v for k,v in old.items() if k != 'claims'} == {k:v for k,v in new.items() if k != 'claims'}
      and new['claims'][-1] == p['ledger_increment']['add_declared_claims'][0]
      and new['claims'][-1]['registration_sha256'] == p['new_registration']['sha256'])
check('separate old4MiB addendum and global envelopes are preserved',
      allocation['separate_existing_monitor_addendum']['sha256'] == '2946083d7ed65aa931bcdf12bee333b340d9e0c9ec742509314ebfed9de61ddf'
      and allocation['existing_monitor_addendum_bytes'] == 4 << 20
      and allocation['existing_admin_bytes_preserved'] == 128 << 20
      and new['cache_cap_bytes'] == 768 << 30 and new['metadata_package']['maximum_bytes'] == 8 << 30
      and new['free_reserve_bytes'] == 1 << 40
      and allocation['new_planned_metadata_bytes'] == allocation['prior_planned_metadata_bytes'] + (32 << 20)
      and allocation['new_planned_metadata_bytes'] == 5652 << 20
      and allocation['metadata_envelope_bytes'] == 8192 << 20)
check('no fake current ledger pin or new adoption claim',
      p['ledger_increment']['expected_old_ledger_sha256'] is None
      and p['ledger_increment']['quiescence_receipt'] is None
      and p['ledger_increment']['must_wait_for_active_and_queued_writer_quiescence'] is True
      and allocation['actual_adoption_receipt'] is None and allocation['not_a_ledger_claim'] is True
      and p['actual_ledger_read'] is False and p['actual_ledger_written'] is False
      and p['actual_claim_created'] is False and p['actual_run'] is False and p['automatic_retry'] is False)
check('checkpoint reservation stays pinned but time validity remains a Lead check',
      cp['remaining_bytes'] == p['checkpoint_remaining_bytes'] == 2303 << 30
      and cp['free_floor_bytes'] == p['free_floor_bytes'] == 1 << 40
      and cp['valid_until_unix'] == p['checkpoint_valid_until_unix'])

after = io()
cost = {'elapsed_seconds': time.monotonic()-started,
    'peak_RSS_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
    'logical_read_delta': after['rchar']-before['rchar'], 'explicit_pin_read_bytes': sum(x['bytes'] for x in INPUTS),
    'cpu_affinity': sorted(os.sched_getaffinity(0)), 'GPU_calls':0, 'ledger_reads_or_writes':0, 'payload_reads':0}
assert cost['elapsed_seconds'] < 60 and cost['logical_read_delta'] < 32 << 20 and cost['cpu_affinity'] == [6,7]
report = {'schema':'st_history_packed64_wiring_independent_v1','status':'PASS_NARROW_STATIC_WIRING_ONLY',
    'reviewer':'/root/st_calibration_independent_correctness_v2',
    'proposal_commit':'bfdfd61cfdf6b96a503e36713c8f6c7e328f7741','runtime_commit':p['runtime_commit'],
    'proposal_path':str(PROPOSAL), 'runtime':str(RUNTIME),'checks':CHECKS,'input_artifacts':INPUTS,
    'parsed_actual_command':vars(args),'existing_test_evidence':{'correctness':44,'resource':14,'rerun':False},
    'test_type':'exact static pins and real parser-only execution; unsigned controller validation rejection',
    'cost':cost, 'launch_approved':False, 'actual_history_build':'NOT_RUN_BY_REVIEWER',
    'ledger_or_runtime_source_mutation':False,
    'review_side_incidental_bytecode':'ATTEMPT1 created one small pyc; exact file removed; this run compiles already read source in memory',
    'lead_live_requirements':['Wait for active and queued writers to finish; obtain actual quiescence evidence',
        'Re-read live policy and ledger hash; if policy advanced append without restoring stale snapshot',
        'Adopt separate1MiB control allocation; preserve existing4MiB addendum and128MiB admin scope',
        'Verify actual shared free space and pinned checkpoint reservation/absolute expiry covers900+40s',
        'Verify prospective output/control roots and unexpected files without deleting evidence',
        'Perform authorized same-ledger additive migration with fresh inventory evidence',
        'Issue separate exact-source independent real admission with current host and lease',
        'After one supervised run require COMPLETE plus accounting and no live owned descendants'],
    'limits':['Packing preserves reviewed semantics; no real64 metadata cost/coverage measured here',
        '64 distinct logs may require64 shards;128-record shard cap does not imply one shard',
        '31MiB is a hard allocation cap, not a measured guarantee all64 will fit',
        'Existing44/14 tests are inherited from byte-matched source and were not rerun',
        'No ledger/current-writer scan, raw log, image, tensor, teacher or GPU use',
        'No formal training, full history build, future target generation or automatic retry permission']}
path=HERE/'REVIEW.json';assert not path.exists()
path.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'status':report['status'],'checks':len(CHECKS),'cost':cost}))

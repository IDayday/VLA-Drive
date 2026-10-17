"""One bounded metadata observation; never invokes a launcher or payload reader."""
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import time

START = time.monotonic()
os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:2])
resource.setrlimit(resource.RLIMIT_AS, (2 * 2**30, 2 * 2**30))
ROOT = Path('/mnt/project/st-world-policy-20261016')
OUT = Path(__file__).resolve().parent
SOURCE = '6714b1e9fb2e0c8059b90b8b4d48cd40796fe345'
JOB = 'C2_step001024_dev1696_seed42_v1'
TAG = 'step_001024_1792224443976811073'
PREP = Path('/mnt/project/VLA-Drive-st-evaluation-agent-20261016/reports/st_world_policy/20261016/evaluation/full_dev_execution_v1')
REPO = Path('/mnt/project/VLA-Drive-st-flow-context-runtime-v3-20261017')
BASE_PINS = {'POLICY.json': '5373e3eb57dc4bbb0e042f4f08df4569bc0a5cbb5c5fedcbd45bfbb0a29ed17f', 'COMMANDS.json': 'b86ded00ad852034fedfdc4f21b44a2ef6480332d321f71f4bf0059c65e9f421', 'INTENTS.json': 'e749a4969803c2a9f0a6eb67cbd19e7544de33013e946517fb67709a6a1df1b6'}
HELPER_HASHES = {'launch_formal_dev_export_v5.py': 'ad0ee24987bdba9ee3fbc46a5efbdb1445b0a6a69d4d588fc32e960738b876bf', 'launch_completed_dev_score_v5.py': '2e62c84a2eecde3e9af585d0a6fb77f9ff33d27a2ec76cd0fa2b594b3a8f15c3'}
CACHE = {}
READ_BYTES = 0
ORIGINAL_READ_TEXT = Path.read_text


def bounded_bytes(path):
    global READ_BYTES
    p = Path(path).resolve()
    assert p.suffix in ('.json', '.py'), 'Only metadata/source reads'
    assert p.stat().st_size <= 32 * 2**20
    if p not in CACHE:
        assert READ_BYTES + p.stat().st_size <= 128 * 2**20
        CACHE[p] = p.read_bytes()
        READ_BYTES += len(CACHE[p])
    assert time.monotonic() - START < 180
    return CACHE[p]


def sha(path):
    return hashlib.sha256(bounded_bytes(path)).hexdigest()


def load(path):
    return json.loads(bounded_bytes(path))


def pin(path):
    return {'path': str(path), 'sha256': sha(path)}


def identity_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def emit(name, value):
    p = OUT / name
    assert not p.exists(), 'Versioned observation is immutable; do not overwrite'
    p.write_text(json.dumps(value, sort_keys=True, indent=2) + '\n')


# Execute only definitions from the already reviewed helper, never top-level main.
Path.read_text = lambda self, *a, **k: bounded_bytes(self).decode('utf-8')
helper_refs = {name: pin(ROOT / 'resources' / name) for name in HELPER_HASHES}
assert {k: v['sha256'] for k, v in helper_refs.items()} == HELPER_HASHES
tree = ast.parse(bounded_bytes(helper_refs['launch_formal_dev_export_v5.py']['path']))
selected = []
for node in tree.body:
    if isinstance(node, ast.FunctionDef):
        selected.append(node)
    elif isinstance(node, ast.Assign) and all(isinstance(t, ast.Name) and t.id.isupper() for t in node.targets):
        selected.append(node)
ns = {'Path': Path, 'json': json, 'hashlib': hashlib, 'identity_hash': identity_hash}
exec(compile(ast.Module(body=selected, type_ignores=[]), '<reviewed-helper-definitions-only>', 'exec'), ns)
ns['sha'] = sha
base = {name: load(PREP / name) for name in BASE_PINS}
assert {name: sha(PREP / name) for name in BASE_PINS} == BASE_PINS
fixed = ns['load_fixed_additive_v3'](JOB, base)
job, command, pins = fixed['job'], fixed['command'], fixed['pins']
assert job['expected_completed'] == 1024 and job['stage'] == 'DIAGNOSTIC_1024'
assert job['control_job_id'] == 'C0_step001024_dev1696_seed42_v1'
assert command['launch_authorized'] is False
plan = load(pins['EXECUTION_PLAN.json']['path'])
policy = load(pins['POLICY.json']['path'])
train_reg = load(job['training_binding']['registration']['path'])
assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip() == SOURCE
assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=REPO).strip()
for ref in base['POLICY.json']['execution_runtime']['source_files']:
    assert sha(ref['path']) == ref['sha256']

run = ROOT / 'formal/consequence/runs/C2_fulltrain_seed42_official_4096_v1'
receipt_path = run / 'checkpoints' / TAG / 'COMPLETE.json'
receipt = load(receipt_path)
assert receipt['schema'] == 'st_native_checkpoint_v1'
assert identity_hash({k: v for k, v in receipt.items() if k != 'receipt_identity'}) == receipt['receipt_identity']
assert receipt['receipt_identity'] == 'eebb3c66f17fab266c103457d9608b8f0756fe8855a0d36921809864979f4569'
state = receipt['state']
assert state['completed'] == state['policy_version'] == state['offset'] == 1024
assert state['exposure'] == 32768 and state['epoch'] == 0
assert state['identity'] == job['runtime_identity'] and receipt['world_size'] == 4
assert state['pending_feedback'] is None and state['pending_old_policy'] is None
assert state['termination_reason'] == 'RUNNING_CHECKPOINT'
expected_files = {f'bf16_zero_pp_rank_{r}_mp_rank_00_optim_states.pt' for r in range(4)} | {f'rng_rank{r}.pt' for r in range(4)} | {'mp_rank_00_model_states.pt'}
assert set(receipt['files']) == expected_files == {p.name for p in receipt_path.parent.glob('*.pt')}
for name, entry in receipt['files'].items():
    assert (receipt_path.parent / name).stat().st_size == entry['bytes']
receipt_ref = pin(receipt_path)

manifest_ref = policy['population_manifest']
assert sha(manifest_ref['path']) == manifest_ref['sha256']
manifest = load(manifest_ref['path'])
assert manifest['split'] == 'dev'
metric_path = Path(command['score']['argv'][command['score']['argv'].index('--metric-index') + 1])
assert sha(metric_path) == '60fc773ab05c442db33ab1f2237c9ee8cf57a0de7464a2b76232444325b65266'
metric = load(metric_path)
pairs = [(x['token'], x['log']) for x in manifest['scenes']]
assert len(pairs) == len(set(pairs)) == len(metric) == 1696
assert len({x[1] for x in pairs}) == 16
assert set(pairs) == {(x['token'], x['log']) for x in metric}
assert len({x[0] for x in pairs}) == 1696

# Reuse the C1@1024 schema, with no stale GPU/UUID/admission-time reuse.
template_path = ROOT / 'formal/evaluation/registrations/C1_step001024_dev1696_seed42_v1/ROUTE.json'
route = copy.deepcopy(load(template_path))
regroot = Path(job['evaluation_registration_path']).parent
route.update(job_id=JOB, checkpoint_tag=TAG, complete_receipt=receipt_ref,
             receipt_identity=receipt['receipt_identity'], runtime_identity=job['runtime_identity'],
             training_registration=job['training_binding']['registration'], evaluation_policy=job['evaluation_policy'],
             environment=train_reg['artifacts']['environment'], gpu=None, gpu_uuid=None,
             valid_from_unix=None, valid_until_unix=None)
assert route['host'] == 'training-vla-zt' and route['hostname'] == 'training-vla-zt-worker-0'
assert route['inference'] == ns['ROUTE_INFERENCE_V3'] and route['source_sha'] == SOURCE
assert sha(route['environment']['path']) == route['environment']['sha256']
emit('ROUTE.DRAFT.json', route)
previous_reg = load(ROOT / 'formal/evaluation/registrations/C2_step000512_dev1696_seed42_v1/REGISTRATION.json')
draft = copy.deepcopy(previous_reg)
draft.update(job=job, checkpoint_tag=TAG, complete_receipt=receipt_ref,
             receipt_identity=receipt['receipt_identity'], native_files=receipt['files'], output_root=job['output_root'],
             route={'path': str(regroot / 'ROUTE.json'), 'sha256': None},
             formal_evaluation_authorized_by='PENDING_LEAD_FRESH_ADMISSION; DRAFT_ONLY_NOT_ISSUED')
assert draft['additive_preparation'] == pins and draft['policy_files_sha256'] == BASE_PINS
assert sha(draft['evaluation_correctness']['path']) == draft['evaluation_correctness']['sha256']
emit('REGISTRATION.DRAFT.json', draft)

commands = {'schema': 'st_fixed_endpoint_ready_commands_v1', 'job_id': JOB, 'launch_authorized': False,
    'source_sha': SOURCE, 'cwd': str(REPO), 'checkpoint_tag': TAG, 'complete_receipt': receipt_ref,
    'helper_refs': helper_refs, 'export_helper_argv_template': ['/root/miniconda3/envs/ddp/bin/python', helper_refs['launch_formal_dev_export_v5.py']['path'], '--job-id', JOB, '--checkpoint-tag', TAG, '--gpu', '__LEAD_FRESH_ROUTE_GPU_INDEX__', '--route-sha256', '__ACTUAL_IMMUTABLE_ROUTE_SHA256__'],
    'score_helper_argv_template': ['/root/miniconda3/envs/ddp/bin/python', helper_refs['launch_completed_dev_score_v5.py']['path'], '--job-id', JOB, '--route-sha256', '__SAME_ACTUAL_IMMUTABLE_ROUTE_SHA256__'],
    'fixed_child_command_provenance': command, 'do_not_bypass_reviewed_helpers': True,
    'score_gate': 'All1696 COMPLETE + RESULT + export exit; same route/host; no scoring before these gates.'}
emit('COMMANDS.json', commands)

# One observation only; no waiting for C3 and no checkpoint fallback.
c3run = ROOT / 'formal/consequence/runs/C3_fulltrain_seed42_official_4096_v1'
c3status = load(c3run / 'status.json')
c3receipts = list((c3run / 'checkpoints').glob('step_001024_*/COMPLETE.json'))
c3 = {'schema': 'st_endpoint_presence_observation_v1', 'observed_unix': time.time(), 'arm': 'C3', 'step': 1024,
      'status': 'WAITING_CHECKPOINT' if not c3receipts else 'RECEIPT_PRESENT_NOT_REVIEWED',
      'training_status': {'path': str(c3run / 'status.json'), 'sha256': sha(c3run / 'status.json'), 'snapshot': c3status},
      'complete_receipt_paths': [str(p) for p in c3receipts], 'launch_authorized': False, 'polling': False}
emit('C3_1024_STATUS.json', c3)
ready = {'schema': 'st_registered_endpoint_queue_observation_v1', 'reviewer': '/root/st_evaluation', 'observed_unix': time.time(),
    'job_id': JOB, 'arm': 'C2', 'expected_completed': 1024, 'stage': job['stage'],
    'checkpoint_status': 'READY', 'status': 'WAITING_RESOURCE', 'launch_allowed_now': False,
    'checkpoint_tag': TAG, 'complete_receipt': receipt_ref, 'receipt_identity': receipt['receipt_identity'],
    'runtime_identity': job['runtime_identity'], 'training_binding': job['training_binding'], 'training_config': job['training_config'],
    'native_metadata': {'state': state, 'file_count': len(receipt['files']), 'total_bytes': sum(x['bytes'] for x in receipt['files'].values()), 'receipt_identity_recomputed': True, 'exact_inventory': True, 'file_sizes_verified': True, 'payload_hashes_recomputed': False, 'payload_reads': 0, 'payload_verification_deferred_to_existing_frozen_exporter': True},
    'fixed_preparation_pins': pins, 'parent_policy_pins': BASE_PINS, 'helper_refs': helper_refs,
    'fixed_helper_loader_used_without_launcher_main': True, 'source_sha': SOURCE, 'runtime_path': str(REPO), 'runtime_clean': True,
    'population_manifest': manifest_ref, 'metric_index': pin(metric_path),
    'population_alignment': {'scenes': 1696, 'logs': 16, 'contexts': 1696, 'token_log_exact': True, 'payloads_reread': False},
    'inference': route['inference'], 'score': policy['score'], 'dedup_resume': policy['dedup_resume'],
    'statistics_unchanged': plan['statistics'], 'endpoints_unchanged': plan['endpoints'],
    'control_job_id': job['control_job_id'], 'comparison': job['comparison'], 'early_elimination': False,
    'route_draft_only': True, 'requested_host': route['host'], 'unresolved_route_fields': ['gpu', 'gpu_uuid', 'valid_from_unix', 'valid_until_unix'],
    'route_schema_template': pin(template_path), 'existing_evaluation_registration': Path(job['evaluation_registration_path']).exists(),
    'existing_export_root': Path(job['export_root']).exists(), 'existing_score_root': Path(job['score_root']).exists(),
    'required_before_dispatch': ['Lead fresh environment, shared mount/locks, CPU quota, RAM, disk, GPU UUID/memory/util and process inventory', 'Lead seal immutable exact job ROUTE with a fresh GPU and sufficient validity', 'Use reviewed v5 helper; it generates final REGISTRATION after admission and enforces campaign slots4/export and2/score, GPU exclusion, two invocations/endpoint'],
    'scientific_results_created': False, 'navtest': False, 'tasks_launched': 0, 'signals_sent': False,
    'cost': {'metadata_source_bytes_read': READ_BYTES, 'native_payload_bytes_read': 0, 'cpu_affinity_count': len(os.sched_getaffinity(0)), 'rss_peak_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024, 'wall_seconds': time.monotonic() - START, 'read_cap_bytes': 128 * 2**20}}
emit('READY.json', ready)
Path.read_text = ORIGINAL_READ_TEXT
print(json.dumps({'status': ready['status'], 'receipt_sha256': receipt_ref['sha256'], 'C3': c3['status'], 'cost': ready['cost']}))

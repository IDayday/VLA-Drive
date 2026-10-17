"""Read only bounded JSON/source metadata; never import/execute a dispatcher."""
import ast
import csv
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import time
from datetime import datetime, timezone

signal.alarm(120)
os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:2])
resource.setrlimit(resource.RLIMIT_AS, (1024**3, 1024**3))
ROOT = Path('/mnt/project/st-world-policy-20261016')
OUT = Path(__file__).resolve().parent
pins, total = {}, 0


def read_bytes(path):
    global total
    path = Path(path)
    assert path.suffix in ('.json', '.py')
    assert path.stat().st_size <= 2 * 1024**2
    data = path.read_bytes()
    total += len(data)
    assert total <= 64 * 1024**2
    pins[str(path)] = {'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}
    return data


def sha(path):
    return hashlib.sha256(read_bytes(path)).hexdigest()


def load(path):
    return json.loads(read_bytes(path))


def ident(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def pinned(ref):
    value = load(ref['path'])
    assert pins[ref['path']]['sha256'] == ref['sha256'], ref['path']
    return value


helper = ROOT / 'resources/launch_formal_dev_export_v5.py'
source = read_bytes(helper)
assert pins[str(helper)]['sha256'] == 'ad0ee24987bdba9ee3fbc46a5efbdb1445b0a6a69d4d588fc32e960738b876bf'
assert sha(ROOT / 'resources/launch_completed_dev_score_v5.py') == '2e62c84a2eecde3e9af585d0a6fb77f9ff33d27a2ec76cd0fa2b594b3a8f15c3'
# Copy definitions into an isolated namespace; no top-level argparse/launch/import.
names = {'ROOT', 'PREP', 'REPO', 'SOURCE', 'ADDITIVE_V2_PREP', 'ADDITIVE_V2_PINS',
         'C3_V3_PREP', 'C3_V3_PINS', 'C4_V5_PREP', 'C4_V5_PINS'}
nodes = [n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) or
         isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in names for t in n.targets)]
class BoundTextReads(ast.NodeTransformer):
    def visit_Call(self, node):
        node = self.generic_visit(node)
        if isinstance(node.func, ast.Attribute) and node.func.attr == 'read_text' and not node.args and not node.keywords:
            return ast.copy_location(ast.Call(func=ast.Name(id='bounded_read_text', ctx=ast.Load()),
                                             args=[node.func.value], keywords=[]), node)
        return node

namespace = {'Path': Path, 'json': json, 'hashlib': hashlib, 'identity_hash': ident,
             'bounded_read_text': lambda p: read_bytes(p).decode()}
tree = ast.fix_missing_locations(BoundTextReads().visit(ast.Module(body=nodes, type_ignores=[])))
exec(compile(tree, str(helper), 'exec'), namespace)
namespace['sha'] = sha
prep = namespace['PREP']
base_pins = {'POLICY.json': '5373e3eb57dc4bbb0e042f4f08df4569bc0a5cbb5c5fedcbd45bfbb0a29ed17f',
             'COMMANDS.json': 'b86ded00ad852034fedfdc4f21b44a2ef6480332d321f71f4bf0059c65e9f421',
             'INTENTS.json': 'e749a4969803c2a9f0a6eb67cbd19e7544de33013e946517fb67709a6a1df1b6'}
docs = {n: pinned({'path': str(prep/n), 'sha256': h}) for n, h in base_pins.items()}
policy = docs['POLICY.json']
for ref in policy['execution_runtime']['source_files']:
    assert sha(ref['path']) == ref['sha256']
manifest = pinned(policy['population_manifest'])
assert len(manifest['scenes']) == len({s['token'] for s in manifest['scenes']}) == 1696
assert len({s['log'] for s in manifest['scenes']}) == 16
pinned(policy['scientific_plan'])
# Metric index and underlying metric caches intentionally not opened.
rows, commands = [], {}
for arm in ('C1', 'C2', 'C3', 'C4'):
    job_id = f'{arm}_step004096_dev1696_seed42_v1'
    additive = namespace['load_fixed_additive_v3'](job_id, docs)
    job = additive['job'] if additive else next(j for j in docs['INTENTS.json']['jobs'] if j['job_id'] == job_id)
    command = additive['command'] if additive else next(j for j in docs['COMMANDS.json']['checkpoint_jobs'] if j['job_id'] == job_id)
    assert job['expected_completed'] == 4096 and job['stage'] == 'COMMON_ENDPOINT'
    binding = job['training_binding']
    registration, approval, launch, identity = [pinned(binding[k]) for k in ('registration', 'approval', 'launch', 'run_identity_file')]
    config = pinned(job['training_config'])
    assert identity['identity'] == job['runtime_identity'] == ident({k: v for k, v in identity.items() if k != 'identity'})
    assert registration['source_sha'] == launch['source_sha'] == namespace['SOURCE']
    assert config['execution']['max_updates'] == config['execution']['stop_after'] == 4096
    assert config['execution']['world_size'] == 4
    run = ROOT / 'formal/consequence/runs' / registration['run_id']
    status = load(run/'status.json')
    previous_result = load(run/'RESULT.json')
    assert status['identity'] == job['runtime_identity'] and status['source_sha'] == namespace['SOURCE']
    receipt_path = run / 'checkpoints' / status['tag'] / 'COMPLETE.json'
    receipt = load(receipt_path)
    assert receipt['receipt_identity'] == ident({k: v for k, v in receipt.items() if k != 'receipt_identity'})
    assert receipt['state']['identity'] == job['runtime_identity'] and receipt['world_size'] == 4
    # Stat inventory only; no weight/optimizer/RNG payload bytes are read.
    assert all((receipt_path.parent/name).stat().st_size == meta['bytes'] for name, meta in receipt['files'].items())
    invocations = Path(binding['registration']['path']).parent/'invocations'
    resumes = sorted(invocations.glob('*/LAUNCH.json'))
    newest = load(resumes[-1])
    assert newest['registration_sha256'] == binding['registration']['sha256']
    assert newest['approval_sha256'] == binding['approval']['sha256'] and newest['source_sha'] == namespace['SOURCE']
    final_receipts = list((run/'checkpoints').glob('step_004096_*/COMPLETE.json'))
    eval_reg = Path(job['evaluation_registration_path'])
    row = dict(arm=arm, job_id=job_id, host=status['host'], shared_status=status['status'],
               completed=status['completed'], remaining=4096-status['completed'],
               updated_utc=datetime.fromtimestamp(status['updated_unix'], timezone.utc).isoformat(),
               status_age_seconds=round(time.time()-status['updated_unix'], 2),
               runtime_identity=job['runtime_identity'], latest_tag=status['tag'],
               latest_saved_step=receipt['state']['completed'], latest_receipt=str(receipt_path),
               latest_receipt_sha256=pins[str(receipt_path)]['sha256'],
               last_resume_launch=str(resumes[-1]), last_resume=bool(newest.get('resume')),
               previous_result_step=previous_result['completed'], previous_result_status=previous_result['status'],
               previous_result_is_stale=previous_result['completed']<status['completed'],
               final4096_complete_receipts=[str(p) for p in final_receipts],
               final_evaluation_registration_exists=eval_reg.exists(),
               final_route_exists=(eval_reg.parent/'ROUTE.json').exists(),
               final_export_complete=(Path(job['export_root'])/'COMPLETE_rank0.json').exists(),
               final_score_complete=(Path(job['score_root'])/'COMPLETE.json').exists(),
               static_source_and_dependency_pins='PASS',
               dispatch_state='WAITING_EXACT_NATIVE4096_AND_FRESH_LEAD_ROUTE',
               process_liveness='NOT_CHECKED_NO_SSH')
    rows.append(row)
    commands[arm] = {'fixed_job': job, 'existing_command': command,
                     'helper_export': str(helper),
                     'helper_score': str(ROOT/'resources/launch_completed_dev_score_v5.py')}
control = ROOT/'formal/evaluation/jobs/C0_step004096_dev1696_seed42_v1/official_score/COMPLETE.json'
load(control)
snapshot = {'observed_utc': datetime.now(timezone.utc).isoformat(), 'scope': 'READ_ONLY_SHARED_METADATA',
            'model_tensor_bytes_read': 0, 'metric_cache_bytes_read': 0, 'ssh_calls': 0,
            'jobs_launched': 0, 'gpu_queries_or_loaders': 0, 'metadata_bytes_read': total,
            'rows': rows, 'evidence_pins': pins, 'control_complete': str(control),
            'population_manifest_pin': policy['population_manifest'],
            'population_scenes': 1696, 'population_logs': 16,
            'score_protocol': 'db383e17b887e08beb358b282fbeab223c80c558164bead81d52a45b1b798143',
            'runtime': str(namespace['REPO']), 'source': namespace['SOURCE'],
            'fresh_resource_availability': 'NOT_CHECKED_LEAD_ONLY',
            'resource_contract': {'export_slots': 4, 'score_slots': 2, 'score_workers': 2,
                                  'score_admission_limit': 1728, 'retry_attempts': 1},
            'protected': 'Lead local GPU0/1 teacher64 PROFILE and all active training'}
(OUT/'SNAPSHOT.json').write_text(json.dumps(snapshot, indent=2, sort_keys=True)+'\n')
(OUT/'EXISTING_COMMANDS.json').write_text(json.dumps(commands, indent=2, sort_keys=True)+'\n')
with (OUT/'STATUS.csv').open('w') as f:
    fields = ['arm','shared_status','completed','remaining','latest_saved_step','latest_tag','host','updated_utc','previous_result_step','previous_result_is_stale','dispatch_state']
    writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore');writer.writeheader();writer.writerows(rows)
print(json.dumps({'observed_utc': snapshot['observed_utc'], 'metadata_bytes_read': total,
                  'rows': [{k:r[k] for k in ('arm','completed','latest_saved_step','final4096_complete_receipts','static_source_and_dependency_pins')} for r in rows]}))

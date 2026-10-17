"""Read-only production receipt audit. Never opens images, tensors or SQLite."""
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import stat
import subprocess
import sys
import time

ROOT = Path('/mnt/project/VLA-Drive-st-indexed-qualified-runtime-v1-20261017')
DATA = Path('/mnt/project/st-world-policy-20261016')
SOURCE = '31c3dca2fbc697b419bf343ce05d10be0a4fe13f'
assert Path.cwd() == ROOT
assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip() == SOURCE
assert not subprocess.check_output(['git', 'status', '--porcelain=v1'], text=True).strip()
assert os.environ['CUDA_VISIBLE_DEVICES'] == '' and len(os.sched_getaffinity(0)) <= 2
sys.path.insert(0, str(ROOT))
from tools.st_world_policy import qualify_indexed_provider as q

started = time.monotonic()
opened = {}
def audit(event, args):
    if event != 'open' or not isinstance(args[0], (str, bytes, os.PathLike)):
        return
    path = Path(os.fsdecode(args[0])).absolute()
    if not str(path).startswith('/mnt/project/'):
        return
    if args[2] & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC):
        raise RuntimeError('Review must not write project/runtime artifacts')
    if path.suffix not in {'.json', '.py', '.pyc', '.yaml', '.yml', '.log'}:
        raise RuntimeError('Forbidden production asset read: ' + str(path))
    size = path.stat().st_size
    if size > 32 << 20:
        raise RuntimeError('Refuse oversized metadata')
    opened[str(path)] = size
sys.addaudithook(audit)

pins = {}
def read(path):
    raw = path.read_bytes()
    pins[str(path)] = {'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}
    return json.loads(raw)

run = DATA/'audits/indexed_provider_composition_run_v1'
plan_root = DATA/'audits/indexed_provider_composition_plan_v1'
plan = read(run/'RESOLVED_PLAN.json')
original_plan = read(plan_root/'PLAN.json')
receipt = read(run/'RESULT.json')
budget = read(DATA/'resources/QUALIFICATION_REPORT_BUDGET_v1.json')
launch = read(plan_root/'QUALIFICATION_LAUNCH.json')
progress = read(run/'PROGRESS.json')
assert plan == original_plan
assert pins[str(plan_root/'PLAN.json')]['sha256'] == budget['plan_sha256']
assert budget['runtime_commit'] == SOURCE and budget['claim_in_cache_ledger'] is False
assert budget['total_report_reservation_bytes'] == sum(budget['root_bounds'].values()) == 64 << 20
assert budget['effective_metadata_reserved_bytes'] <= budget['unchanged_metadata_limit_bytes'] == 4 << 30

# Descriptor view is assembled solely for this receipt consumer. No provider is
# constructed or admitted, and no production descriptor is created or rewritten.
descriptor = {key: plan[key] for key in ['requests', 'request_index', 'teacher_config', 'execution', 'limits', 'provider']}
descriptor['sources'] = [{'index': plan['source_index'], 'cache_identity': plan['current_identity']}]
q.validate_receipt(plan, receipt, descriptor)

elements = math.prod(plan['contract']['output_grid_hw']) * plan['contract']['channels']
case_counts = {}
all_diff = []
for name, rows in receipt['cases'].items():
    diffs = [c for row in rows for c in row['comparisons']]
    case_counts[name] = {
        'scenes': len(rows), 'comparisons': len(diffs),
        'elements': sum(c['difference']['elements'] for c in diffs),
        'online_elements': sum(c['online_difference']['elements'] for c in diffs),
        'partial_ram_hit_and_compute_scenes': sum(r['counts']['ram_hits'] > 0 and r['counts']['computed_images'] > 0 for r in rows),
        'mixed_immutable_hit_and_compute_scenes': sum(r['counts']['immutable_hits'] > 0 and r['counts']['computed_images'] > 0 for r in rows),
        'counts': {k: sum(r['counts'][k] for r in rows) for k in rows[0]['counts']},
    }
    all_diff.extend(c['difference'] for c in diffs)
    all_diff.extend(c['online_difference'] for c in diffs)
all_diff += [c['difference'] for c in receipt['legacy_cached_reference'] + receipt['reference_reordered4']]
assert all(d['elements'] == elements and d['bitwise_equal'] and d['max_abs'] == 0 for d in all_diff)

def census(root):
    total = {'apparent_bytes': 0, 'allocated_bytes': 0, 'files': 0, 'directories': 0}
    for path in [root, *root.rglob('*')]:
        s = path.lstat()
        assert not stat.S_ISLNK(s.st_mode)
        assert stat.S_ISDIR(s.st_mode) or (stat.S_ISREG(s.st_mode) and s.st_nlink == 1)
        total['apparent_bytes'] += s.st_size
        total['allocated_bytes'] += s.st_blocks * 512
        total['directories' if stat.S_ISDIR(s.st_mode) else 'files'] += 1
    return total

roots = {path: census(Path(path)) for path in budget['root_bounds']}
for path, size in roots.items():
    assert max(size['apparent_bytes'], size['allocated_bytes']) < budget['root_bounds'][path]
assert sum(v['allocated_bytes'] for v in roots.values()) < 64 << 20
source_hashes = {p['path']: hashlib.sha256(Path(p['path']).read_bytes()).hexdigest() for p in plan['source_files']}
assert all(source_hashes[p['path']] == p['sha256'] for p in plan['source_files'])
assert source_hashes[str(ROOT/'tools/st_world_policy/qualify_indexed_provider.py')] == 'd072f46ac3b1da4fffb42dec2439f35382e793b4a9de1cd4317eb60872a46bbb'
assert source_hashes[str(ROOT/'tools/st_world_policy/teacher_provider.py')] == '83410555d7e879407b09039f6a094fa6daac1259d0df7379cd6a220c2954b6bf'
assert not q.torch.cuda.is_initialized()
assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 < 2 << 30
assert time.monotonic() - started < 180
result = {
    'status': 'PASS_EXACT_32_SCENE_REAL_RECEIPT_CONSUMER', 'tested_runtime_commit': SOURCE,
    'reviewed_consumer_fix': '42854f40d0dab551463d63abae80a1045b64722f',
    'consumer': 'tools.st_world_policy.qualify_indexed_provider.validate_receipt',
    'descriptor_scope': 'in-memory exact plan-derived view; not factory or training admission',
    'plan_identity': plan['identity'], 'receipt_identity': receipt['identity'], 'pins': pins,
    'scenes': plan['scenes'], 'slots': plan['slots'], 'missing_slots': plan['missing_slots'],
    'unique_images': plan['unique_images'], 'cached_unique': len(plan['hit_keys']), 'miss_unique': len(plan['miss_keys']),
    'elements_per_full_feature': elements, 'case_counts': case_counts,
    'all_checked_comparisons': len(all_diff), 'all_checked_elements': sum(x['elements'] for x in all_diff),
    'fixed_policy_summary': receipt['fixed_policy_summary'], 'legacy_reference_summary': receipt['legacy_reference_summary'],
    'compute_stats': receipt['compute_stats'], 'provider_memory': receipt['provider_memory'],
    'numerical_and_state_gates': {k: receipt[k] for k in ['construction_rng_unchanged', 'completed_all_cases', 'fixed_policy_repeatability', 'legacy_reference_compatibility', 'all_masks_equal', 'sampler_equal', 'partial_lru_exercised', 'mixed_cache_compute_exercised', 'rng_forward_checks']},
    'measured_resources': {k: receipt[k] for k in ['elapsed_seconds', 'peak_rss_bytes', 'cuda_peak_allocated_bytes', 'cuda_peak_reserved_bytes', 'process_rchar_bytes', 'tensor_mmap_logical_bytes', 'tensor_disk_bytes']},
    'logical_input_bytes': receipt['process_rchar_bytes'] + receipt['tensor_mmap_logical_bytes'],
    'budget': plan['budget'], 'metadata_root_census': roots,
    'metadata_reserved_administratively_not_cache_claim': True,
    'progress_file_is_last_periodic_running_snapshot_not_terminal': progress['status'],
    'launch': launch, 'source_files': source_hashes, 'opened_project_files': opened,
    'review_cpu_affinity': sorted(os.sched_getaffinity(0)),
    'review_peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
    'review_elapsed_seconds_after_import': time.monotonic() - started,
    'review_cuda_initialized': False, 'formal_approved': False, 'full_population_approved': False,
    'training_throughput_approved': False, 'scientific_gain': 'UNTESTED',
}
print(json.dumps(result, indent=2))

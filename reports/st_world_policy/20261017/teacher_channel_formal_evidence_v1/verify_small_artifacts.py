"""Independent, bounded evidence audit. No tensor libraries or runtime dispatch."""
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess

resource.setrlimit(resource.RLIMIT_AS, (2 * 1024**3, 2 * 1024**3))
resource.setrlimit(resource.RLIMIT_CPU, (30, 30))
os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:2])
ROOT = Path(__file__).resolve().parents[4]
RUNTIME = Path('/mnt/project/VLA-Drive-st-official-runtime-v2-20261016')
OUT = ROOT / 'reports/st_world_policy/20261017/teacher_channel_formal_evidence_v1'
REG = Path('/mnt/project/st-world-policy-20261016/formal/spatial/registrations/SP2_PCA256_fulltrain_seed42_official_5k_v1')
OLD = REG.parent / 'SP2_fulltrain_seed42_official_5k_v1'
CONTROL = ROOT / 'reports/st_world_policy/20261017/teacher_channel_control_v1'
PCA = Path('/mnt/project/st-world-policy-20261016/audits/history_data/pca_fixed384_full_current_v1')
PUBLIC = Path('/mnt/project/st-world-policy-20261016/audits/evidence/public_sources/dino_foresight')
REVIEWER = '/root/st_teacher_channel_evidence_v2'
cache, records = {}, {}


def read(path):
    path = Path(path).resolve()
    if str(path) not in cache:
        size = path.stat().st_size
        assert size <= 1024**2, ('Small-document limit exceeded', str(path), size)
        assert sum(len(x) for x in cache.values()) + size <= 64 * 1024**2
        value = path.read_bytes()
        cache[str(path)] = value
        records[str(path)] = {'path': str(path), 'bytes': len(value),
                              'sha256': hashlib.sha256(value).hexdigest()}
    return cache[str(path)]


def js(path):
    return json.loads(read(path))


def sha(path):
    read(path)
    return records[str(Path(path).resolve())]['sha256']


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def pin(record):
    assert sha(record['path']) == record['sha256'], record['path']
    if 'bytes' in record:
        assert len(read(record['path'])) == record['bytes']


def differences(a, b, prefix=''):
    if isinstance(a, dict) and isinstance(b, dict):
        assert set(a) == set(b), prefix
        return [x for k in sorted(a) for x in differences(a[k], b[k], prefix + '/' + k)]
    return [] if a == b else [{'path': prefix, 'new': a, 'reference': b}]


registration = js(REG / 'REGISTRATION.json')
reference = js(OLD / 'REGISTRATION.json')
fingerprint = js(REG / 'FINGERPRINT.json')
assert sha(REG / 'REGISTRATION.json') == 'a644ad60477b5dd857b44069568e9b93fde68994b4862ffcbcaaae9b57e29ca6'
assert registration['schema'] == 'st_world_run_v1'
assert REVIEWER not in registration['implementer_agents']
expected = {'registration_sha256': sha(REG / 'REGISTRATION.json'),
            'source_fingerprint': registration['source_fingerprint'],
            'artifacts': {k: v['sha256'] for k, v in registration['artifacts'].items()},
            'execution': registration['execution'], 'run_id': registration['run_id']}
assert fingerprint == expected
for name, record in registration['artifacts'].items():
    if name not in ('teacher_manifest', 'data_manifest'):
        pin(record)
    else:
        assert record == reference['artifacts'][name]
assert registration['execution'] == reference['execution']
for name in ('base_binding', 'data_manifest', 'teacher_manifest'):
    assert registration['artifacts'][name] == reference['artifacts'][name]
assert registration['artifacts']['environment']['sha256'] == reference['artifacts']['environment']['sha256']
assert registration['source_sha'] == reference['source_sha'] == 'fdf25f99b702754787904215bf1c2a2b7515ea75'
assert registration['source_fingerprint'] == reference['source_fingerprint']
old_approval = js(OLD / 'RUN_APPROVAL.json')
assert old_approval['fingerprint']['registration_sha256'] == sha(OLD / 'REGISTRATION.json')
for role, ref in old_approval['reviews'].items():
    pin(ref)
    r = js(ref['path'])
    assert r['status'] == 'PASS' and r['role'] == role
    assert r['fingerprint'] == old_approval['fingerprint']

cfg = js(REG / 'CONFIG.json')
old_cfg = js(OLD / 'CONFIG.json')
diff = differences(cfg, old_cfg)
assert {x['path'] for x in diff} == {'/space_time/teacher/channels', '/space_time/teacher/transform_identity', '/runtime/output_root'}
assert cfg['execution'] == registration['execution']
assert cfg['space_time']['teacher']['channels'] == 256
assert old_cfg['space_time']['teacher']['channels'] == 384
assert cfg['space_time']['teacher']['image_hw'] == [288, 384]
assert cfg['space_time']['teacher']['native_grid_hw'] == cfg['space_time']['teacher']['output_grid_hw'] == [18, 24]
assert cfg['space_time']['world_queries_per_view'] == 48 and cfg['space_time']['sampled_positions'] == 96
assert cfg['space_time']['cell'] == 'S-P2' and cfg['data']['history_root'] is None
assert cfg['execution']['max_updates'] == 20000 and cfg['execution']['stop_after'] == 5000
assert cfg['execution']['seed'] == 42
assert cfg['execution']['global_batch'] == 32 and cfg['execution']['micro_batch'] == cfg['execution']['world_size'] == 4
assert cfg['optimizer']['warmup'] == 1000
assert cfg['space_time']['teacher'] == js(REG / 'TARGET_CONTRACT.json')
pins = js(CONTROL / 'INPUT_PINS.json')
for item in pins.values():
    pin(item)
binding = js(CONTROL / 'STUDY_BINDING.json')
for width in ('256', '384'):
    for key in ('config', 'target_contract'):
        item = binding['arms'][width][key]
        assert sha(ROOT / item['repo_path']) == item['sha256']
assert read(REG / 'CONFIG.json') == read(ROOT / binding['arms']['256']['config']['repo_path'])
assert read(OLD / 'CONFIG.json') == read(ROOT / binding['arms']['384']['config']['repo_path'])
for name, record in binding['common_artifacts'].items():
    assert registration['artifacts'][name] == record

pca = js(PCA / 'PCA.json')
scope = js(PCA / 'CHANNEL_SCOPE.json')
assert pca['identity'] == digest({k: v for k, v in pca.items() if k != 'identity'})
assert pca['identity'] == binding['pca_identity'] == scope['pca_identity']
assert pca['fit_role'] == 'train' and pca['dimensions'] == [256, 384]
assert pca['fit_patches'] == 32768 and pca['standardize_channels'] and not pca['whiten']
assert scope['identity'] == digest({k: v for k, v in scope.items() if k != 'identity'})
allowed = next(x for x in scope['allowed_caches'] if x['cache_root'] == cfg['data']['teacher_root'])
assert allowed['identity_sha256'] == registration['artifacts']['teacher_manifest']['sha256']
assert digest(allowed['contract']) == allowed['teacher_contract_identity']
for width, config in ((256, cfg), (384, old_cfg)):
    transform = {'schema': 'st_projected_teacher_transform_v1',
                 'raw_teacher_contract_identity': allowed['teacher_contract_identity'],
                 'pca_identity': pca['identity'], 'dimension': width,
                 'standardize_channels': True, 'whiten': False}
    assert digest(transform) == config['space_time']['teacher']['transform_identity']
selection = js(PCA / 'DIMENSION_SELECTION.json')
assert selection['selected_channels'] == 384 and selection['search_closed']
receipt = js(pins['reference_checkpoint_receipt']['path'])
assert receipt['state']['completed'] == 5000 and receipt['state']['exposure'] == 159992
assert receipt['state']['counts']['current'] == 159992 * 3 * 96 * 384
assert receipt['state']['identity'] == binding['reference384']['state_identity']
initialization = js(pins['reference_initialization']['path'])
assert initialization['base'] == binding['reference384']['init_base_identity']
assert initialization['master_initialization']['status'] == 'RESTORED_EXACT_FP32_SOURCE'
boundary = js(REG / 'DATA_AND_STORAGE_BOUNDARY.json')
for key in ('PCA_namespace_receipt', 'teacher_stage_receipt'):
    pin(boundary[key])
assert boundary['PCA_identity'] == pca['identity'] and boundary['original_dimension_selection_unchanged']

git_head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=RUNTIME, text=True).strip()
assert git_head == registration['source_sha']
assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=RUNTIME).strip()
tree = subprocess.check_output(['git', 'ls-tree', '-r', '-z', 'HEAD'], cwd=RUNTIME)
included = []
for row in tree.split(b'\0'):
    if not row:
        continue
    meta, name = row.split(b'\t', 1)
    name = name.decode()
    if not name.startswith(('reports/', 'docs/')) and (name.endswith(('.py', '.sh', '.toml', '.yaml', '.yml', '.lock', '.json', '.cpp', '.cu', '.h', '.c', '.ini', '.cfg')) or Path(name).name.startswith('requirements')):
        included.append([name, meta.decode()])
assert digest(included) == registration['source_fingerprint']
manifest = js(ROOT / 'reports/st_world_policy/20261017/teacher_alignment_evidence_v1/SOURCE_MANIFEST.json')
dino = next(x for x in manifest['sources'] if x['method'] == 'DINO-Foresight')
assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=PUBLIC, text=True).strip() == dino['commit']
for item in dino['files']:
    assert sha(PUBLIC / item['path']) == item['sha256']
for path in ('tools/st_world_policy/teacher_pca.py', 'starVLA/dataloader/st_world_dataset.py',
             'starVLA/model/modules/st_world/losses.py', 'starVLA/model/modules/st_world/config.py',
             'starVLA/model/modules/st_world/sparse_readout.py', 'starVLA/model/framework/st_world_model.py',
             'tools/st_world_policy/train_world.py', 'tools/st_world_policy/check_formal_approval.py'):
    read(RUNTIME / path)
probe_path = Path('/mnt/project/st-world-policy-20261016/audits/history_data/spatial_probes_train64_dev16_v1/PROBE_RESULTS.json')
probes = js(probe_path)
assert sha(probe_path) == next(x['sha256'] for x in selection['evidence'] if x['path'] == str(probe_path))
probe_summary = {task: {width: {k: v for k, v in arm.items() if k != 'train_losses'}
                        for width, arm in arms.items()}
                 for task, arms in probes['results'].items()}
for task, arms in probe_summary.items():
    assert arms['256']['parameters'] == 257 and arms['384']['parameters'] == 385
    for key in ('updates', 'patch_presentations', 'sample_order_sha256'):
        assert arms['256'][key] == arms['384'][key]
result = {'status': 'PASS_SMALL_ARTIFACT_EVIDENCE_AUDIT', 'reviewer_agent': REVIEWER,
          'fingerprint': fingerprint, 'config_differences': diff,
          'pca_identity': pca['identity'], 'source_head': git_head,
          'official_source_commit': dino['commit'], 'probe_summary': probe_summary,
          'old_endpoint': receipt['state'], 'small_file_reads_bytes': sum(len(x) for x in cache.values()),
          'git_tree_output_bytes': len(tree), 'cpu_affinity': sorted(os.sched_getaffinity(0)),
          'full_gate_executed_by_this_reviewer': False,
          'inherited_large_hashes': {k: registration['artifacts'][k] for k in ('data_manifest', 'teacher_manifest')},
          'inheritance_basis': 'Identical historical approved paths/hashes; Lead separately reported complete verify_registration hashing for this exact registration.',
          'tensor_libraries_imported': False, 'gpu_used': False, 'ssh_used': False,
          'artifact_reads': list(records.values())}
(OUT / 'SMALL_ARTIFACT_AUDIT.json').write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
print(json.dumps({k: result[k] for k in ('status', 'small_file_reads_bytes', 'git_tree_output_bytes', 'source_head')}))

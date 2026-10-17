"""Bounded W-launch metadata checks; never execute launcher main."""
import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
ROOT = Path('/mnt/project/st-world-policy-20261016')
PREP = HERE.parent / 'full_dev_execution_v1'
CHECKS = []
READS = []
STARTED = time.monotonic()


def digest(path):
    path = Path(path)
    assert path.stat().st_size < 2 * 1024 * 1024, 'Only small controls in this validation'
    return hashlib.sha256(path.read_bytes()).hexdigest()


def identity_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def check(name, condition):
    assert condition, name
    CHECKS.append({'name': name, 'status': 'PASS'})


def rejects(name, call):
    try:
        call()
    except (AssertionError, KeyError, FileNotFoundError, ValueError) as error:
        CHECKS.append({'name': name, 'status': 'REJECTED_AS_REQUIRED',
                       'exception': type(error).__name__, 'message': str(error)})
        return
    raise AssertionError('Accepted negative case: ' + name)


def git_only(args, **kwargs):
    assert args in (['git', 'rev-parse', 'HEAD'], ['git', 'status', '--porcelain'])
    assert kwargs['cwd'] == '/mnt/project/VLA-Drive-st-official-runtime-v2-20261016'
    return subprocess.check_output(args, **kwargs)


def namespace(path):
    tree = ast.parse(path.read_text())
    ns = {'Path': Path, 'json': json, 'hashlib': hashlib,
          'identity_hash': identity_hash,
          'subprocess': SimpleNamespace(check_output=git_only)}
    assignments = {'ROOT', 'PREP', 'REPO', 'SOURCE', 'C3_V3_PREP', 'C3_V3_PINS',
                   'ADDITIVE_V2_PREP', 'ADDITIVE_V2_PINS', 'W_ARMS_V4',
                   'W_TRAINING_SOURCE_V4', 'W_TRAINING_RUNTIME_V4', 'W_LAUNCH_SUPPLEMENT_V4'}
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in assignments for t in node.targets):
            exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), ns)
        if isinstance(node, ast.FunctionDef) and node.name != 'main':
            exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), ns)
    ns['sha'] = digest
    return ns, tree


def static_registration(job, ns):
    result = {'job': copy.deepcopy(job), 'schema': 'st_formal_endpoint_evaluation_binding_v1',
              'evaluation_source_sha': ns['SOURCE'], 'requested_scenes': 1696, 'seed': 42,
              'export_world_size': 1, 'navtest': False, 'output_root': job['output_root']}
    if job['arm'] in ('C2', 'C3'):
        result['additive_preparation'] = ns['additive_pin_refs_v2']() if job['arm'] == 'C2' else ns['c3_pin_refs_v3']()
    if job['arm'] in ns['W_ARMS_V4']:
        result['w_launch_supplement'] = ns['W_LAUNCH_SUPPLEMENT_V4']
    return result


base = {name: json.loads((PREP / name).read_text()) for name in ('POLICY.json', 'COMMANDS.json', 'INTENTS.json')}
document = json.loads((HERE / 'W_LAUNCH_BINDINGS.json').read_text())
for arm in ('SP0', 'SP2'):
    job = next(x for x in base['INTENTS.json']['jobs'] if x['arm'] == arm)
    rejects('original ' + arm + ' missing launch reproduction', lambda job=job: job['training_binding']['launch'])

for stem in ('launch_formal_dev_export', 'launch_completed_dev_score'):
    path = HERE / (stem + '_v4.py')
    compile(path.read_text(), str(path), 'exec')
    ns, tree = namespace(path)
    old, old_tree = namespace(ROOT / 'resources' / (stem + '_v3.py'))
    old_functions = {n.name: ast.dump(n, include_attributes=False) for n in old_tree.body if isinstance(n, ast.FunctionDef)}
    new_functions = {n.name: ast.dump(n, include_attributes=False) for n in tree.body if isinstance(n, ast.FunctionDef)}
    unchanged = set(old_functions) - {'main', 'validate_static_job'}
    check(stem + ': existing route/native/budget/C-loader definitions unchanged',
          all(old_functions[name] == new_functions[name] for name in unchanged))
    reader = ns['read_w_launch_control_v4']
    def recorded_reader(pin):
        READS.append(pin['path'])
        assert not pin['path'].endswith('/identity.json') and not pin['path'].endswith('.pt')
        return reader(pin)
    ns['read_w_launch_control_v4'] = recorded_reader
    for arm in ('SP0', 'SP1', 'SP2', 'SP3'):
        job = next(x for x in base['INTENTS.json']['jobs'] if x['arm'] == arm)
        before = copy.deepcopy(job)
        normalized, pin = ns['supplement_w_launch_v4'](job)
        expected = copy.deepcopy(job)
        expected['training_binding']['launch'] = document['arms'][arm]['launch']
        check(stem + ': actual ' + arm + ' exact launch/source/approval fingerprint',
              normalized == expected and job == before and pin == ns['W_LAUNCH_SUPPLEMENT_V4'])
        if 'validate_static_job' in ns:
            registration = static_registration(normalized, ns)
            ns['validate_static_job'](registration, normalized)
            check(stem + ': score accepts sealed ' + arm + ' supplement', True)
            bad = copy.deepcopy(registration); bad.pop('w_launch_supplement')
            rejects(stem + ': missing score supplement ' + arm,
                    lambda bad=bad, normalized=normalized: ns['validate_static_job'](bad, normalized))
    job = next(x for x in base['INTENTS.json']['jobs'] if x['arm'] == 'SP0')
    binding = document['arms']['SP0']
    values = [job, binding] + [json.loads(Path(binding[k]['path']).read_text())
                              for k in ('registration', 'config', 'approval', 'launch')]
    mutations = [
        ('missing launch pin', 1, ('launch',), None),
        ('wrong launch path', 1, ('launch', 'path'), document['arms']['SP1']['launch']['path']),
        ('wrong arm', 0, ('arm',), 'SP1'),
        ('wrong endpoint', 0, ('expected_completed',), 4096),
        ('changed runtime identity', 0, ('runtime_identity',), '0' * 64),
        ('training source conflated with evaluation', 2, ('source_sha',), ns['SOURCE']),
        ('launch source conflated with evaluation', 5, ('source_sha',), ns['SOURCE']),
        ('wrong training cwd', 5, ('cwd',), str(ns['REPO'])),
        ('wrong launch registration', 5, ('registration_sha256',), '0' * 64),
        ('wrong launch approval', 5, ('approval_sha256',), '0' * 64),
        ('wrong approval fingerprint', 4, ('fingerprint', 'source_fingerprint'), '0' * 64),
        ('wrong launch gate fingerprint', 5, ('gate', 'identity'), '0' * 64),
        ('wrong input config arm', 3, ('space_time', 'cell'), 'S-P1'),
    ]
    for name, index, keys, value in mutations:
        bad = copy.deepcopy(values)
        target = bad[index]
        for key in keys[:-1]:
            target = target[key]
        if value is None:
            target.pop(keys[-1])
        else:
            target[keys[-1]] = value
        rejects(stem + ': ' + name, lambda bad=bad: ns['validate_w_launch_v4'](*bad))
    prior_pin = ns['W_LAUNCH_SUPPLEMENT_V4']
    ns['W_LAUNCH_SUPPLEMENT_V4'] = {**prior_pin, 'sha256': '0' * 64}
    rejects(stem + ': changed supplement bytes/pin', lambda: ns['supplement_w_launch_v4'](job))
    ns['W_LAUNCH_SUPPLEMENT_V4'] = prior_pin
    original_subprocess = ns['subprocess']
    ns['subprocess'] = SimpleNamespace(check_output=lambda *a, **k: ns['SOURCE'] + '\n')
    rejects(stem + ': wrong training runtime HEAD', lambda: ns['supplement_w_launch_v4'](job))
    ns['subprocess'] = original_subprocess
    c_jobs = [x for x in base['INTENTS.json']['jobs'] if x['arm'] in ('C0', 'C1')]
    for arm in ('C2', 'C3'):
        for step in (512, 1024, 4096):
            c_jobs.append(ns['load_fixed_additive_v3'](f'{arm}_step{step:06d}_dev1696_seed42_v1', base)['job'])
    def forbidden_control_read(pin):
        raise AssertionError('C path attempted W control read')
    ns['read_w_launch_control_v4'] = forbidden_control_read
    for c_job in c_jobs:
        normalized, pin = ns['supplement_w_launch_v4'](c_job)
        assert normalized is c_job and pin is None
        if 'validate_static_job' in ns:
            registration = static_registration(c_job, ns)
            ns['validate_static_job'](registration, c_job)
            old['validate_static_job'](registration, c_job)
    check(stem + ': all12 existing C jobs unchanged; no W reads', len(c_jobs) == 12)
    ns['read_w_launch_control_v4'] = recorded_reader

check('only small W controls opened; no model/native/large identity', bool(READS) and
      all(not p.endswith(('.pt', '/identity.json')) for p in READS))
expected_baselines = {
    'launch_formal_dev_export_v3.py': 'e58f58e3d0e3fb05e05dcd28c5d04142c4f3a4ba2002e5cc8ea9b3a5bf95217f',
    'launch_completed_dev_score_v3.py': '9217c5c0b5362453a86843bfe8dd9c6ad1609b7440c5441be4bd1af6511e8d7c',
}
check('deployed v3 helpers untouched', all(digest(ROOT / 'resources' / name) == sha
                                        for name, sha in expected_baselines.items()))
result = {'schema': 'st_W_launch_candidate_validation_v4', 'status': 'IMPLEMENTER_CHECKS_ONLY_NEEDS_INDEPENDENT_REVIEW',
          'checks_passed': len(CHECKS), 'checks': CHECKS, 'elapsed_seconds': time.monotonic() - STARTED,
          'read_control_paths': sorted(set(READS)), 'read_large_W_identity': False,
          'no_launcher_main': True, 'no_GPU_queries_contexts_models_weights_or_dispatch': True,
          'candidate_files': [{'path': str(HERE / name), 'sha256': digest(HERE / name)}
                              for name in ('launch_formal_dev_export_v4.py', 'launch_completed_dev_score_v4.py')],
          'scope': 'Actual small source/launch/approval metadata; CPU rejection tests. No new resource, math or performance matrix.'}
parser = argparse.ArgumentParser()
parser.add_argument('--output', type=Path, default=HERE / 'CANDIDATE_VALIDATION.json')
args = parser.parse_args()
assert not args.output.exists(), 'Preserve existing validation evidence; choose an independent output'
args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
print(json.dumps({'checks_passed': len(CHECKS), 'elapsed_seconds': result['elapsed_seconds'], 'output': str(args.output)}))

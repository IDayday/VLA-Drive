"""Read-only real metadata and synthetic mutations; never run launcher main."""
import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

RUNTIME = Path('/mnt/project/VLA-Drive-st-W-launch-review-runtime-c0e4523f')
CANDIDATE = RUNTIME/'reports/st_world_policy/20261016/evaluation/W_LAUNCH_HELPER_v4'
TRAINING = 'fdf25f99b702754787904215bf1c2a2b7515ea75'
EVALUATION = '6714b1e9fb2e0c8059b90b8b4d48cd40796fe345'
HERE = Path(__file__).resolve().parent
READS = []


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def small(path, expected=None):
    path = Path(path)
    assert path.suffix != '.pt' and path.name != 'identity.json'
    with path.open('rb') as stream:
        data = stream.read((1 << 20) + 1)
    assert len(data) <= 1 << 20
    digest = hashlib.sha256(data).hexdigest()
    if expected is not None: assert digest == expected
    READS.append({'path': str(path), 'bytes': len(data), 'sha256': digest})
    return json.loads(data)


def git_only(args, **kwargs):
    assert args in (['git', 'rev-parse', 'HEAD'], ['git', 'status', '--porcelain'])
    assert kwargs['cwd'] == '/mnt/project/VLA-Drive-st-official-runtime-v2-20261016'
    return subprocess.check_output(args, **kwargs)


def namespace(kind):
    path = CANDIDATE/f'launch_{kind}_v4.py'
    ns = {'Path': Path, 'hashlib': hashlib, 'json': json, 'identity_hash': identity,
          'subprocess': SimpleNamespace(check_output=git_only)}
    names = {'ROOT', 'PREP', 'REPO', 'SOURCE', 'W_ARMS_V4', 'W_TRAINING_SOURCE_V4',
             'W_TRAINING_RUNTIME_V4', 'W_LAUNCH_SUPPLEMENT_V4'}
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in names for t in node.targets):
            exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), ns)
        if isinstance(node, ast.FunctionDef) and node.name != 'main':
            exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), ns)
    state_path = ns['REPO']/'tools/st_world_policy/training_state.py'
    for node in ast.parse(state_path.read_text()).body:
        if (isinstance(node, ast.FunctionDef) and node.name == 'require_native_inventory' or
            isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'TERMINATIONS' for t in node.targets)):
            exec(compile(ast.Module(body=[node], type_ignores=[]), str(state_path), 'exec'), ns)
    return ns


@pytest.fixture(scope='module')
def real():
    ns = namespace('completed_dev_score')
    supplement = small(CANDIDATE/'W_LAUNCH_BINDINGS.json', ns['W_LAUNCH_SUPPLEMENT_V4']['sha256'])
    parent = supplement['parent_intents']
    intents = small(parent['path'], parent['sha256'])
    controls = {arm: {name: small(pin['path'], pin['sha256'])
        for name, pin in binding.items() if name in ('registration','config','approval','launch')}
        for arm, binding in supplement['arms'].items()}
    return ns, supplement, {j['arm']: j for j in intents['jobs'] if j['arm'].startswith('SP')}, controls


@pytest.mark.parametrize('kind', ['formal_dev_export', 'completed_dev_score'])
@pytest.mark.parametrize('arm', ['SP0', 'SP2'])
def test_actual_missing_launch_supplement_preserves_old_intent_and_two_sources(real, kind, arm):
    _, doc, jobs, _ = real
    ns = namespace(kind); original = deepcopy(jobs[arm])
    assert 'launch' not in original['training_binding']
    normalized, pin = ns['supplement_w_launch_v4'](original)
    expected = deepcopy(jobs[arm]); expected['training_binding']['launch'] = doc['arms'][arm]['launch']
    assert normalized == expected and original == jobs[arm]
    assert pin == ns['W_LAUNCH_SUPPLEMENT_V4']
    assert normalized['training_source_sha'] == TRAINING and ns['SOURCE'] == EVALUATION


@pytest.mark.parametrize('kind', ['formal_dev_export', 'completed_dev_score'])
def test_all_C_arms_return_same_object_without_W_reads(kind):
    ns = namespace(kind)
    def forbidden(*args): raise AssertionError('C path read W metadata')
    ns['read_w_launch_control_v4'] = forbidden
    for arm in ('C0','C1','C2','C3'):
        for step in (512,1024,4096):
            job = {'arm':arm, 'expected_completed':step}
            got, pin = ns['supplement_w_launch_v4'](job)
            assert got is job and pin is None


@pytest.mark.parametrize('kind', ['formal_dev_export', 'completed_dev_score'])
def test_control_reader_checks_bytes_and_bounded_size(tmp_path, kind):
    read = namespace(kind)['read_w_launch_control_v4']
    path = tmp_path/'control.json'; path.write_bytes(b'{}')
    assert read({'path':str(path), 'sha256':hashlib.sha256(b'{}').hexdigest()}) == {}
    with pytest.raises(AssertionError): read({'path':str(path), 'sha256':'0'*64})
    path.write_bytes(b' ' * ((1 << 20)+1))
    with pytest.raises(AssertionError, match='metadata bound'):
        read({'path':str(path), 'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})


@pytest.mark.parametrize('mutation', ['other_arm_receipt','argv','output','diagnostic'])
def test_wrong_actual_launch_content_rejected(real, mutation):
    ns, doc, jobs, controls = real
    values = [deepcopy(jobs['SP0']), deepcopy(doc['arms']['SP0'])] + [deepcopy(controls['SP0'][key])
        for key in ('registration','config','approval','launch')]
    if mutation == 'other_arm_receipt': values[-1] = deepcopy(controls['SP2']['launch'])
    elif mutation == 'argv': values[-1]['argv'].append('--unexpected-resume')
    elif mutation == 'output': values[-1]['output'] = controls['SP2']['launch']['output']
    else: values[-1]['diagnostic_weights_used'] = True
    with pytest.raises(AssertionError): ns['validate_w_launch_v4'](*values)


def actual_endpoint(real):
    ns, _, jobs, _ = real
    observation = small(CANDIDATE/'SP0_READY_ENDPOINT.json')
    native = small(observation['receipt']['path'], observation['receipt']['sha256'])
    result = small(observation['result']['path'], observation['result']['sha256'])
    job, supplement = ns['supplement_w_launch_v4'](jobs['SP0'])
    registration = {'job':job, 'schema':'st_formal_endpoint_evaluation_binding_v1',
        'evaluation_source_sha':EVALUATION, 'requested_scenes':1696, 'seed':42,
        'export_world_size':1, 'navtest':False, 'output_root':job['output_root'],
        'w_launch_supplement':supplement, 'complete_receipt':observation['receipt'],
        'receipt_identity':native['receipt_identity'], 'native_files':deepcopy(native['files']),
        'checkpoint_tag':observation['checkpoint_tag']}
    checkpoint = {'receipt_identity':native['receipt_identity'], 'run_identity':native['state']['identity'],
        'completed':5000, 'tag':observation['checkpoint_tag'],
        'native_receipt_sha256':observation['receipt']['sha256'],
        'native_checkpoint_sha256':identity({k:v['sha256'] for k,v in native['files'].items()}),
        'scope':'FORMAL', 'weights_usable_for_formal_init':True, 'training_source_sha':TRAINING}
    checkpoint['sha256'] = identity(checkpoint)
    return ns, observation, native, result, job, registration, checkpoint


def test_actual_SP0_5000_metadata_and_inventory_only(real):
    ns, observation, native, result, job, registration, checkpoint = actual_endpoint(real)
    ns['validate_static_job'](registration, job)
    ns['validate_native_reference'](registration, job, checkpoint, native)
    folder = Path(observation['receipt']['path']).parent
    assert {p.name for p in folder.glob('*.pt')} == set(native['files'])
    assert all((folder/name).stat().st_size == info['bytes'] for name,info in native['files'].items())
    assert sum(v['bytes'] for v in native['files'].values()) == observation['native_total_bytes']
    assert result['completed'] == native['state']['completed'] == 5000
    assert result['identity'] == native['state']['identity'] == job['runtime_identity']
    assert result['source_sha'] == TRAINING and result['status'] == 'PAUSED'
    assert result['pause_reason'] == 'BUDGET_PAUSED' and result['tag'] == checkpoint['tag']
    report = {'status':'PASS_REAL_SP0_METADATA_ONLY', 'receipt':observation['receipt'],
        'result':observation['result'], 'checkpoint_tag':checkpoint['tag'],
        'runtime_identity':job['runtime_identity'], 'training_source_sha':TRAINING,
        'evaluation_source_sha':EVALUATION, 'completed':5000, 'status_at_read':result['status'],
        'exposure':result['exposure'], 'pause_reason':result['pause_reason'],
        'native_file_count':len(native['files']), 'native_total_bytes':observation['native_total_bytes'],
        'payload_reads_or_rehashes':False, 'weight_loading':False, 'dispatch':False,
        'synthetic_validation_envelope_not_published':True, 'read_metadata':READS}
    output = HERE/'SP0_REAL_METADATA_REVIEW.json'
    with output.open('x') as stream: json.dump(report,stream,indent=2); stream.write('\n')


@pytest.mark.parametrize('mutation', ['receipt_identity','runtime_identity','tag','inventory','training_source'])
def test_actual_native_receipt_mutations_rejected(real, mutation):
    ns, _, native, _, job, registration, checkpoint = actual_endpoint(real)
    if mutation == 'receipt_identity': native['receipt_identity'] = '0'*64
    elif mutation == 'training_source': checkpoint['training_source_sha'] = EVALUATION; checkpoint['sha256'] = identity({k:v for k,v in checkpoint.items() if k != 'sha256'})
    else:
        if mutation == 'runtime_identity': native['state']['identity'] = '0'*64
        elif mutation == 'tag': native['state']['tag'] = 'step_005000_999'
        else: native['files'].pop('rng_rank3.pt')
        native['receipt_identity'] = identity({k:v for k,v in native.items() if k != 'receipt_identity'})
        registration['receipt_identity'] = checkpoint['receipt_identity'] = native['receipt_identity']
        registration['native_files'] = deepcopy(native['files'])
    with pytest.raises((AssertionError,ValueError)):
        ns['validate_native_reference'](registration,job,checkpoint,native)

"""Finite owned CPU subprocesses and tmp_path only; never invoke real builder."""
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import pytest

SOURCE = Path(__file__).with_name('run_once.py')


@pytest.fixture
def runner(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location('independent_index_controller', SOURCE)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    root = tmp_path / 'output'
    guard_lock = threading.Lock()
    state = {'atomic_calls': 0, 'processes': [], 'cap': 32 << 20}
    def current_bytes():
        return sum(max(p.stat().st_size, p.stat().st_blocks * 512) for p in root.iterdir()) if root.exists() else 0
    class Context:
        limits = {'max_disk_bytes': 32 << 20}
        claim = {'output_bytes': 14 << 20, 'staging_bytes': 14 << 20, 'metadata_bytes': 4 << 20}
        budget = SimpleNamespace(_total=lambda claim: sum(claim.values()))
        def check(self, **kwargs):
            amount = current_bytes()
            if amount > state['cap']: raise ValueError('test guard observed over-cap output')
            return {'charged_disk_bytes': amount}
    @contextmanager
    def fake_guard(**kwargs):
        with guard_lock:
            root.mkdir(exist_ok=True)
            yield Context()
    monkeypatch.setattr(module.m, 'metadata_writing', fake_guard)
    original_atomic = module.b._atomic
    def atomic(path, value):
        state['atomic_calls'] += 1
        return original_atomic(path, value)
    monkeypatch.setattr(module.b, '_atomic', atomic)
    prep = {'policy': str(tmp_path / 'policy'), 'policy_sha256': '1'*64,
            'stages': {'request_index': {'registration': str(tmp_path / 'registration'),
                'registration_sha256': '2'*64, 'output_root': str(root)}}}
    review = {'status': 'PASS_BOUNDED_INDEX_FOLLOWUP', 'owner_role': 'IndependentSystems',
              'prepared_sha256': '3'*64, 'controller_sha256': '4'*64,
              'checkpoint_receipt': str(tmp_path/'checkpoint.json'), 'checkpoint_sha256': '6'*64,
              'valid_until_unix': time.time()+600}
    monkeypatch.setattr(module, 'load', lambda path: prep if Path(path) == module.PREP else
                        {'allowed_hosts': ['training-vla-zt']})
    original_sha = module.sha
    monkeypatch.setattr(module, 'sha', lambda path: '3'*64 if Path(path)==module.PREP else
                        ('4'*64 if Path(path)==SOURCE else original_sha(path)))
    monkeypatch.setattr(module.socket, 'gethostname', lambda: 'training-vla-zt-worker-0')
    monkeypatch.setattr(module.subprocess, 'check_output', lambda cmd, **kw:
                        '1162b955\n' if 'rev-parse' in cmd else b'')
    monkeypatch.setattr(module.m, 'pinned_json', lambda path, digest:
                        (review, b'') if str(path).endswith('/review.json') else ({'output_root': str(root)}, b''))
    monkeypatch.setattr(module.m, '_checkpoint_reserve', lambda *a, **kw: 12345)
    argv = ['controller', '--stage', 'request_index', '--review', str(tmp_path/'review.json'),
            '--review-sha256', '5'*64, '--checkpoint-receipt', str(tmp_path/'checkpoint.json'),
            '--checkpoint-sha256', '6'*64]
    monkeypatch.setattr(sys, 'argv', argv)
    original_popen = subprocess.Popen
    def child(code):
        def launch(command, **kwargs):
            assert command[:3] == ['taskset', '-c', '0,1']
            assert kwargs['start_new_session'] and kwargs['env']['CUDA_VISIBLE_DEVICES'] == ''
            result = original_popen([sys.executable, '-c', code], **{**kwargs, 'cwd': tmp_path})
            state['processes'].append(result)
            return result
        monkeypatch.setattr(module.subprocess, 'Popen', launch)
    return SimpleNamespace(module=module, root=root, state=state, child=child, review=review,
                           current_bytes=current_bytes, argv=argv)


def good_code():
    return "import json;print(json.dumps({'status':'GUARDED_METADATA_RETURN','artifact':{'image_files_opened':0,'tensor_files_opened':0,'tensor_disk_bytes':0}}))"


def test_positive_one_shot_and_same_lock_reentry(runner):
    runner.child(good_code())
    assert runner.module.main() == 0
    with pytest.raises(ValueError, match='one-shot'): runner.module.main()
    assert len(runner.state['processes']) == 1
    record = json.loads((runner.root/'CONTROL_BUILD_RESULT.json').read_text())
    assert record['status'] == 'PASS' and record['max_serialized_control_write_bytes'] == 1 << 20


def test_simultaneous_start_only_one_child(runner):
    runner.child(good_code())
    def invoke():
        try: return runner.module.main()
        except ValueError as error: return str(error)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: invoke(), range(2)))
    assert 0 in results and any(isinstance(x,str) and 'one-shot' in x for x in results)
    assert len(runner.state['processes']) == 1


@pytest.mark.parametrize('field',['checkpoint_receipt','checkpoint_sha256','prepared_sha256','controller_sha256','valid_until_unix'])
def test_bad_review_pin_or_expiry_never_launches_child(runner,field):
    runner.child(good_code())
    runner.review[field]=0 if field=='valid_until_unix' else 'mismatch'
    with pytest.raises(ValueError,match='independent exact unexpired review'):
        runner.module.main()
    assert not runner.state['processes'] and not runner.root.exists()


@pytest.mark.parametrize('artifact', [None, [], {}, {'image_files_opened':False,'tensor_files_opened':0,'tensor_disk_bytes':0},
                                     {'image_files_opened':1,'tensor_files_opened':0,'tensor_disk_bytes':0}])
def test_malformed_or_nonzero_artifact_is_failed_receipt(runner, artifact):
    body={'status':'GUARDED_METADATA_RETURN'}
    if artifact is not None: body['artifact']=artifact
    runner.child('print('+repr(json.dumps(body))+')')
    assert runner.module.main() == 2
    record=json.loads((runner.root/'CONTROL_BUILD_RESULT.json').read_text())
    assert record['status']=='FAILED' and record['failure']=='malformed or nonzero metadata-only result'


def test_control_serialization_rejects_before_any_atomic_file(runner):
    # Original 1MiB/stream allowance could expand to about12MiB as JSON escapes.
    data={'stdout_unparsed':'\ufffd'*(1<<20),'stderr':'\ufffd'*(1<<20)}
    assert runner.module.b.json_write_bound(data) > 4 << 20
    with pytest.raises(ValueError,match='serialized control'):
        runner.module.publish({},runner.root/'CONTROL_BUILD_RESULT.json',data)
    assert runner.state['atomic_calls']==0 and not runner.root.exists()


def test_control_atomic_peak_checked_before_publication(runner):
    runner.root.mkdir()
    payload=runner.root/'metadata.sqlite'
    with payload.open('wb') as stream: stream.truncate((32<<20)-4096)
    before=runner.current_bytes()
    with pytest.raises(ValueError,match='remaining reservation'):
        runner.module.publish({},runner.root/'CONTROL_BUILD_RESULT.json',{'error':'bounded'})
    assert runner.current_bytes()==before and runner.state['atomic_calls']==0


def test_invalid_utf8_streams_remain_inside_control_allowance(runner):
    runner.root.mkdir()
    with (runner.root/'metadata.sqlite').open('wb') as stream: stream.truncate(28<<20)
    runner.child("import os;os.write(1,b'\\xff'*65536);os.write(2,b'\\xff'*65536)")
    assert runner.module.main()==2
    assert runner.current_bytes() < 32 << 20
    assert (runner.root/'CONTROL_BUILD_RESULT.json').stat().st_size < 1 << 20


def test_nonfinite_result_writes_failure(runner):
    runner.child("print('{\"status\":\"GUARDED_METADATA_RETURN\",\"artifact\":NaN}')")
    assert runner.module.main()==2
    record=json.loads((runner.root/'CONTROL_BUILD_RESULT.json').read_text())
    assert record['status']=='FAILED' and record['stdout_json'] is None


def test_pipe_overflow_stops_owned_child(runner):
    runner.child("import os,time;os.write(1,b'x'*131072);time.sleep(30)")
    assert runner.module.main()==2
    child=runner.state['processes'][0]
    assert child.poll() is not None
    record=json.loads((runner.root/'CONTROL_BUILD_RESULT.json').read_text())
    assert record['failure']=='bounded output exceeded'


@pytest.mark.parametrize('cause',['rss','time'])
def test_resource_failure_stops_only_owned_group(runner,monkeypatch,cause):
    runner.child('import time;time.sleep(30)')
    signals=[];original=os.killpg
    def capture(pid,sig):
        assert pid==runner.state['processes'][0].pid
        signals.append((pid,sig));return original(pid,sig)
    monkeypatch.setattr(runner.module.os,'killpg',capture)
    if cause=='rss':monkeypatch.setattr(runner.module.m,'process_tree_rss',lambda _:3<<30)
    else:
        monkeypatch.setattr(runner.module,'time',SimpleNamespace(time=time.time,monotonic=lambda:time.monotonic()*2000))
    assert runner.module.main()==2
    assert signals and runner.state['processes'][0].poll() is not None

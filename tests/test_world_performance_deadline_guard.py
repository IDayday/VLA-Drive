"""Time/identity injection only: these tests never signal a real process."""
import copy
import json
import os
import signal
from pathlib import Path

from tools.qwen_world_performance import deadline_guard as guard


def process(pid=100, parent=1, module='tools.qwen_world_safety_exploration.train', output=None):
    args = ['python', '-m', module, '--output', output or '/mnt/project/qwen-world-safety-exploration-20261031/runs/test']
    return dict(pid=pid, ppid=parent, start_ticks=1234+pid, state='S', args=args,
                cmdline_sha256=guard.sha(json.dumps(args, separators=(',', ':')).encode()), usr1_handler=True)


def test_exact_time_boundaries_and_native_first_no_evaluation_default_USR1():
    deadline = 10000.
    assert guard.phase_for(deadline-901, deadline) == 'monitor'
    assert guard.phase_for(deadline-900, deadline) == 'cooperative'
    assert guard.phase_for(deadline-91, deadline) == 'cooperative'
    assert guard.phase_for(deadline-90, deadline) == 'term'
    assert guard.phase_for(deadline-31, deadline) == 'term'
    assert guard.phase_for(deadline-30, deadline) == 'kill'
    native = dict(process(), role='native', receipt_root=None)
    assert guard.actions_for(native, 'cooperative') == [('signal', int(signal.SIGUSR1))]
    evaluation = dict(native, role='evaluation', usr1_handler=False)
    assert guard.actions_for(evaluation, 'cooperative') == []
    assert guard.actions_for(evaluation, 'term') == [('signal', int(signal.SIGTERM))]


def test_known_module_and_output_ancestry_both_required_unknown_untouched():
    task = 'qwen_world_safety_exploration_20261031'
    native = process()
    unknown = process(200, module='unrelated.train')
    wrong = process(300, output='/mnt/project/qwen-world-safety-exploration-20261031_LOOKALIKE/runs/test')
    child = process(101, parent=100, module='multiprocessing.spawn')
    rows = guard.discover_owned({r['pid']:r for r in [native, unknown, wrong, child]}, [task], 'host', [])
    assert {r['pid'] for r in rows} == {100, 101}
    assert next(r for r in rows if r['pid'] == 101)['ownership'] == 'verified_live_parent_chain'
    reused = dict(child, ppid=1, start_ticks=child['start_ticks']+1)
    assert guard.discover_owned({101:reused}, [task], 'host', rows) == []


def test_pidfd_revalidates_cmdline_and_start_and_inspection_never_signals(monkeypatch):
    row = process()
    called = []
    monkeypatch.setattr(signal, 'pidfd_send_signal', lambda *a:called.append(a))
    result = guard.verified_pidfd_signal(row, signal.SIGKILL, inspect_only=True)
    assert result['status'] == 'INSPECT_ONLY' and called == []
    changed = dict(row, start_ticks=row['start_ticks']+1)
    values = iter([row, changed])
    monkeypatch.setattr(guard, 'read_process', lambda _:next(values))
    monkeypatch.setattr(os, 'pidfd_open', lambda *a:os.open('/dev/null', os.O_RDONLY), raising=False)
    result = guard.verified_pidfd_signal(row, signal.SIGKILL, inspect_only=False)
    assert result['status'] == 'REFUSED_IDENTITY_CHANGED_AFTER_PIDFD' and called == []
    assert not guard.identity_matches(row, dict(row, cmdline_sha256='different'))


def test_admission_worker_requires_its_exact_receipt_and_renewal_supersedes(tmp_path, monkeypatch):
    task = 'qwen_world_safety_exploration_20261031';host = 'training-test'
    monkeypatch.setattr(guard, 'TASK_ROOTS', {str(tmp_path):task})
    output = str(tmp_path/'run');job = tmp_path/'jobs'/'attempt';job.mkdir(parents=True)
    native = process(output=output)
    row = process(500, module='tools.qwen_world_safety_exploration.job_worker', output=output)
    row['args'] = ['python', '-m', 'tools.qwen_world_safety_exploration.job_worker', '--receipt-root', str(job), '--command-json', json.dumps(native['args'])]
    receipt = dict(host=host, worker_pid=row['pid'], worker_start_ticks=row['start_ticks'])
    (job/'RUNNING.json').write_text(json.dumps(receipt))
    assert guard.classify_direct(row, [task], host)['role'] == 'worker'
    receipt['worker_start_ticks'] += 1;(job/'RUNNING.json').write_text(json.dumps(receipt))
    assert guard.classify_direct(row, [task], host) is None
    grant = dict(permanent_hosts=[], temporary_hosts=[host], unbounded_temporary_hosts=[], expires_unix=10000, record=str(tmp_path/'grant.json'))
    (tmp_path/'grant.json').write_text(json.dumps(grant));policy = tmp_path/'policy.json'
    policy.write_text(json.dumps({'task_authorizations':{task:grant}}));monkeypatch.setattr(guard, 'POLICY', str(policy))
    assert guard.current_task_scope(host, 10000) == [task]
    renewed = copy.deepcopy(grant);renewed['unbounded_temporary_hosts'] = [host];renewed['temporary_hosts'] = []
    policy.write_text(json.dumps({'task_authorizations':{task:renewed}}))
    assert guard.current_task_scope(host, 10000) == []

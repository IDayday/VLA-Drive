"""Independent narrow CPU lifecycle counterexample; no real model or scoring.

The completion receipts here are explicitly test fixtures. Real native/source
and admission evidence is checked separately in the source-bound review.
"""
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest


SOURCE = Path('/mnt/project/VLA-Drive-st-rollout-native-resume-completion-v5-20261018/resources/st_rollout_resume_completion_v5.py')


def test_normal_exit_verified_before_owned_orphan_drain(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location('independent_completion_v5', SOURCE)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    v = m.legacy()
    root = tmp_path / 'fixture_only'
    observer = root / 'interrupted/observer'
    observer.mkdir(parents=True)
    reg = root / 'interrupted/REGISTRATION.json'
    reg.write_text('{"fixture":true}\n')
    monkeypatch.setattr(m, 'ROOT', root)
    monkeypatch.setattr(v, 'ROOT', root)
    checked = []
    expected = {'fixture_only_verified_native_boundary': 2}
    monkeypatch.setattr(v, 'validate_result', lambda *a: checked.append(a) or expected)

    # A failure exit must never be reclassified as success or invoke validation.
    with pytest.raises(ValueError, match='exit successfully'):
        m.finish_successful_job({'child': SimpleNamespace(poll=lambda: 1)}, 'initial')
    assert checked == []

    release = tmp_path / 'allow_parent_exit'
    child_code = (
        'import subprocess,sys,time;from pathlib import Path;'
        'p=subprocess.Popen([sys.executable,"-c",'
        '"import os,time;os.setsid();time.sleep(30)"],stdout=subprocess.DEVNULL);'
        'print(p.pid,flush=True);'
        '\nwhile not Path(sys.argv[1]).exists(): time.sleep(.01)\n'
    )
    foreign = subprocess.Popen([sys.executable, '-c', 'import time;time.sleep(30)'], start_new_session=True)
    parent = subprocess.Popen([sys.executable, '-c', child_code, str(release)], stdout=subprocess.PIPE, text=True, start_new_session=True)
    job = {'child': parent, 'owned': {}, 'start_ticks': None}
    orphan = None
    try:
        orphan = int(parent.stdout.readline())
        deadline = time.monotonic() + 5
        while orphan not in v.owned_processes(job) and time.monotonic() < deadline:
            time.sleep(.01)
        assert orphan in job['owned'], 'Must prove child ownership before reparenting'
        assert foreign.pid not in job['owned']
        release.touch()
        assert parent.wait(timeout=5) == 0
        assert v.process_record(orphan)['state'] != 'Z'

        # Launcher exit0 alone is insufficient: missing rank evidence preserves
        # the live orphan and cannot return a completed/drained result.
        with pytest.raises(FileNotFoundError):
            m.finish_successful_job(job, 'initial')
        assert not job.get('cleanup_complete', False)
        assert v.process_record(orphan)['state'] != 'Z'
        for rank in range(4):
            (observer / f'INITIAL_OBSERVER_COMPLETE_rank{rank}.json').write_text(json.dumps({
                'observer_sha256': m.OLD_SHA, 'runtime_sha': v.SOURCE,
                'rank': rank, 'completed': 2, 'registration': v.pin(reg),
            }))
        assert m.finish_successful_job(job, 'initial') == expected
        assert job['cleanup_complete'] is True
        assert foreign.poll() is None, 'An unrelated same-user process must remain alive'
        assert not any(row['state'] != 'Z' for row in v.owned_processes(job).values())
        assert checked == [('interrupted', 2, 2), ('interrupted', 2, 2)]
    finally:
        # Only processes launched and identified by this test are cleaned.
        if parent.poll() is None:
            parent.terminate()
            parent.wait(timeout=5)
        if orphan is not None:
            try:
                row = v.process_record(orphan)
                if row['start_ticks'] == job['owned'].get(orphan, {}).get('start_ticks') and row['state'] != 'Z':
                    os.kill(orphan, signal.SIGKILL)
            except ProcessLookupError:
                pass
        if foreign.poll() is None:
            foreign.terminate()
        foreign.wait(timeout=5)
        parent.stdout.close()

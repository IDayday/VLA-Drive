"""Exercise the actual process-session deadline mechanism without a GPU job."""
import json
import os
from pathlib import Path
import sys
import time

import psutil
import pytest

from scripts.analysis.constrained_flow_noise import temporary_guard as guard


def test_native_deadline_and_owned_orphan_cleanup(tmp_path, monkeypatch):
    auth = tmp_path/'auth.json'
    auth.write_text('{}')
    receipt = tmp_path/'receipt.json'
    checkpoint = tmp_path/'checkpoint.json'
    orphan = tmp_path/'child.pid'
    worker = tmp_path/'worker.py'
    worker.write_text('''import json, os, signal, subprocess, sys, time
from pathlib import Path
child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(90)'])
Path(sys.argv[2]).write_text(str(child.pid))
def stop(*args):
    destination=Path(sys.argv[1]); temporary=destination.with_suffix('.incomplete')
    temporary.write_text(json.dumps({'complete_boundary':True}))
    os.replace(temporary,destination)
    raise SystemExit(0)
signal.signal(signal.SIGUSR1,stop)
while True: time.sleep(.05)
''')
    lease = dict(expires_unix=time.time()+8, graceful_stop_before_expiry_seconds=6,
                 hard_exit_before_expiry_seconds=1)
    monkeypatch.setattr(guard, 'authorize', lambda *a, **k: (lease, 'test-host'))
    monkeypatch.setattr(sys, 'argv', ['guard', '--authorization', str(auth), '--receipt', str(receipt),
                                    '--', sys.executable, str(worker), str(checkpoint), str(orphan)])
    with pytest.raises(SystemExit) as result:
        guard.main()
    assert result.value.code == 0
    assert json.loads(checkpoint.read_text())['complete_boundary']
    saved = json.loads(receipt.read_text())
    assert saved['status'] == 'EXITED_OWNED_PROCESSES_CLEANED'
    assert not saved.get('deadline_fallback_used', False)
    pid = int(orphan.read_text())
    assert not psutil.pid_exists(pid) or psutil.Process(pid).status() == psutil.STATUS_ZOMBIE
    assert pid in saved['orphan_cleanup_pids']

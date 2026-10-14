import os
from pathlib import Path
import subprocess
import sys
import pytest
from scripts.iqe.specialist_study_completion import AdoptedProcess, FinishedProcess


def test_adoption_observes_same_process_without_launch_or_signal():
    command=[sys.executable,'-c','import time;time.sleep(0.5)']
    child=subprocess.Popen(command)
    try:
        record={'pid':child.pid,'command':command,'cwd':str(Path.cwd()),
                'start_ticks':Path(f'/proc/{child.pid}/stat').read_text().split()[21]}
        adopted=AdoptedProcess(record,lambda:True)
        assert adopted.pid==child.pid and adopted.poll() is None
        with pytest.raises(Exception,match='PID was reused'):
            AdoptedProcess({**record,'start_ticks':'0'},lambda:True)
        with pytest.raises(Exception,match='command changed'):
            AdoptedProcess({**record,'command':['wrong']},lambda:True)
        child.wait(timeout=5)
        assert adopted.poll()==0
        assert AdoptedProcess(record,lambda:False).poll()==1
    finally:
        child.wait(timeout=5)


def test_finished_handle_is_pollable_without_pid():
    value=FinishedProcess()
    assert value.pid==0 and value.poll()==value.returncode==0

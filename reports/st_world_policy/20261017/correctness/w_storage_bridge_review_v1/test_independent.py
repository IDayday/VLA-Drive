"""CPU-only negative file-type check; no campaign controls/ledger are opened."""
import os
from pathlib import Path
import subprocess
import sys

import pytest

BRIDGE = Path(__file__).resolve().parents[2] / 'w_storage_bridge_v1'


def test_fifo_input_rejects_before_waiting_for_writer(tmp_path):
    fifo = tmp_path / 'pinned-control.json'
    os.mkfifo(fifo)
    code = f'''
import sys
sys.path.insert(0, {str(BRIDGE)!r})
import storage_bridge
try:
    storage_bridge.Reads().raw({str(fifo)!r})
except ValueError as exc:
    assert 'regular' in str(exc)
    print('NONREGULAR_REJECTED')
else:
    raise AssertionError('FIFO was accepted')
'''
    try:
        result = subprocess.run([sys.executable, '-c', code], text=True,
                                capture_output=True, timeout=1,
                                env={**os.environ, 'CUDA_VISIBLE_DEVICES': '',
                                     'PYTHONDONTWRITEBYTECODE': '1'})
    except subprocess.TimeoutExpired:
        pytest.fail('Reads.raw blocked opening FIFO before its regular-file check; bounded control admission did not return')
    assert result.returncode == 0, result.stderr
    assert 'NONREGULAR_REJECTED' in result.stdout

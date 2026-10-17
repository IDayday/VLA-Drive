"""Independent bounded CPU fixtures for H64-1/2/3; no campaign writes."""
import ast
from contextlib import contextmanager
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch

import pytest

HERE = Path(__file__).resolve().parent
PILOT = HERE.parent / 'full_causal_history_preparation_v1/run_pilot_once.py'

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value)
    return value

p = load('independent_history_v2', PILOT)
core = load('independent_history_monitor_v2', HERE.parent / 'full_future_metadata_preparation_v1/run_build_once.py')
helper = HERE.parent / 'spatial_probe_actual_preparation_v1'
sys.path.insert(0, str(helper))
import supervise_once as supervisor
import observed_cli as observed
from tools.st_world_policy import cache_storage_budget as budget, metadata_storage


def admission():
    return dict(schema='st_history64_owned_admission_v1',
        status='APPROVED_HISTORY64_PILOT_ONLY', limits=p.LIMITS.copy(),
        reviewer='/independent', hostname=p.socket.gethostname(),
        checkpoint_receipt=p.CP.copy(), output_root=str(p.ROOT), runtime=str(p.RUNTIME),
        control_root=str(p.CONTROL_ROOT), control_reserved_bytes=p.CONTROL_CAP,
        issued_at_unix=0, latest_start_unix=100, valid_until_unix=1000)


@pytest.mark.parametrize('key,value', [('control_root','/tmp/other'),
    ('control_reserved_bytes', None), ('control_reserved_bytes', 2 << 20)])
def test_control_admission_changed_or_missing_rejected(key,value):
    a=admission(); a[key]=value
    with pytest.raises(ValueError): p.validate(a,50)


def test_combined_claim_bound_and_early_io_baseline_are_wired():
    tree=ast.parse(PILOT.read_text())
    controls=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='controls')
    source=ast.unparse(controls)
    assert "LIMITS['max_disk_bytes'] - CONTROL_CAP" in source
    assert 'core.BOOT_IO = dict(ENTRY_IO)' in source
    assert 'core.START = ENTRY_MONOTONIC' in source
    assert p.CONTROL_CAP + (31 << 20) == p.LIMITS['max_disk_bytes']


@pytest.mark.parametrize('kind',['foreign','symlink','hardlink','oversized'])
def test_control_root_rejects_unadmitted_content(tmp_path,kind):
    if kind=='foreign': (tmp_path/'unregistered').write_text('x')
    elif kind=='symlink': (tmp_path/'RESULT.json').symlink_to('/dev/null')
    elif kind=='hardlink':
        (tmp_path/'RESULT.json').write_text('{}')
        os.link(tmp_path/'RESULT.json',tmp_path/'RESULT.json.tmp')
    else:(tmp_path/'RESULT.json.tmp').write_bytes(b'x'*(p.CONTROL_CAP+1))
    with pytest.raises(ValueError):p.control_census(tmp_path)


def test_real_original_ledger_start_contention_gets_failure_receipt(tmp_path):
    artifact=tmp_path/'artifact'; artifact.mkdir()
    evidence=tmp_path/'control'; evidence.mkdir()
    lock=tmp_path/'CLAIMS.lock'; policy=tmp_path/'policy.json'; policy.write_text('{}')
    fixture_budget=SimpleNamespace(policy_path=policy,state_dir=tmp_path,
        state_path=tmp_path/'CLAIMS.json',lock_path=lock,core={},
        _read_policy=lambda: ({},{},{}),_check_cap=lambda _:None)
    @contextmanager
    def writing(**kwargs):
        with budget.StorageBudget._locked(fixture_budget):
            yield SimpleNamespace(limits={'max_disk_bytes':31<<20},
                check=lambda **kw:{'charged_disk_bytes':0})
    fixture_module=SimpleNamespace(metadata_writing=writing)
    with lock.open('a+') as held:
        fcntl.flock(held,fcntl.LOCK_EX)
        before=time.monotonic()
        with patch.object(p,'ROOT',artifact),patch.object(p,'CONTROL_ROOT',evidence), \
             patch.object(core,'monitor',side_effect=AssertionError('must not spawn')):
            assert p.run_parent(SimpleNamespace(admission='/fixture',admission_sha256='a'*64),
                {'valid_until_unix':time.time()+1000},{},{},None,observed,
                fixture_module,budget,core,lambda:None,
                work_deadline=time.monotonic()+10)==2
        assert time.monotonic()-before<1
    result=json.loads((evidence/'RESULT.json').read_text())
    assert result['phase']=='START_PUBLICATION'
    assert result['cleanup']=={'status':'NOT_STARTED'}
    assert result['charged_rchar_bytes']==16<<30
    assert not (artifact/'CONTROL_HISTORY_STARTED.json').exists()


def test_nonblocking_patch_restores_original_flock_on_exception():
    original=budget.fcntl.flock
    with pytest.raises(RuntimeError):
        with p.nonblocking_original_ledger(budget):raise RuntimeError('fixture')
    assert budget.fcntl.flock is original


def test_real_child_timeout_and_cleanup_are_not_fabricated(tmp_path):
    # Review uses CPU8/9; production CPU6/7 routing retained from v1 audit.
    core.BOOT_IO=dict(observed.proc_io())
    with (tmp_path/'fd').open('w') as f:
        result=core.monitor([sys.executable,'-c','import time; time.sleep(10)'],
            runtime=str(tmp_path),env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),
            fd=f.fileno(),root=tmp_path,s=supervisor,m=metadata_storage,obs=observed,
            budget={'wall_seconds':.1,'read_bytes':64<<20,'rss_bytes':4<<30},
            cpu_affinity=(8,9),lease_valid_until=time.time()+60)
    assert result['returncode']!=0 and result['failure']=='WALL_LIMIT'
    assert result['cleanup']['live_owned_remaining']==0
    assert result['supervised_child_seconds']<5


def test_entry_baseline_count_includes_control_read(tmp_path):
    # New wrapper baseline is earlier than all source/control reads. Test the
    # actual repaired wait formula without loading any production payload.
    source=tmp_path/'one_meg_control';source.write_bytes(b'x'*(1<<20))
    baseline=dict(p.ENTRY_IO);source.read_bytes();before=observed.proc_io()
    value=core.terminal_read_accounting(baseline,before,
        {**before,'rchar':before['rchar']+100},100,1<<30,
        final_child_available=True)
    assert value['status']=='COMPLETE'
    assert value['charged_parent_child_rchar_bytes']>=1<<20


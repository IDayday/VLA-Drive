"""Independent CPU process fixtures; no model/checkpoint/dispatch assets."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
import pytest

ROOT=Path(__file__).parents[1]
sys.path.insert(0,str(ROOT/'spatial_probe_actual_preparation_v1'))
import supervise_once as legacy

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    value=importlib.util.module_from_spec(spec); spec.loader.exec_module(value); return value

helper=load('independent_owned_helper',ROOT/'torchrun_descendant_scope_v1/owned_descendants.py')
adapter=load('independent_adapter',ROOT/'official_T1_native4_execution_v1/run_once.py')


def test_unwitnessed_cross_sid_orphan_is_not_claimed_or_signalled(tmp_path,monkeypatch):
    script=tmp_path/'orphan.py'
    script.write_text('''import json,os,subprocess,sys,time\nfrom pathlib import Path\np=Path(sys.argv[1])\nif sys.argv[2]=='root':\n while not (p/'go').exists(): time.sleep(.01)\n subprocess.Popen([sys.executable,__file__,str(p),'orphan'],start_new_session=True)\nelse:\n (p/'orphan.json').write_text(json.dumps({'pid':os.getpid(),'sid':os.getsid(0)}))\n time.sleep(2)\n''')
    with legacy.lifetime_scope():
        facade=helper.ancestry_supervisor(legacy)
        child=subprocess.Popen([sys.executable,str(script),str(tmp_path),'root'],start_new_session=True)
        owner=facade.OwnedSession(child)
        try:
            (tmp_path/'go').touch()
            deadline=time.monotonic()+4
            while not owner.leader_done() or not (tmp_path/'orphan.json').exists():
                assert time.monotonic()<deadline; time.sleep(.01)
            orphan=json.loads((tmp_path/'orphan.json').read_text())
            assert orphan['sid']==orphan['pid'] and orphan['sid']!=child.pid
            owner.discover()
            assert orphan['pid'] not in owner.members
            assert orphan['pid'] in {r['pid'] for r in owner.unknown_adopted}
            sent=[]
            original=legacy.pidfd_signal
            def record(fd,sig): sent.append(fd); return original(fd,sig)
            monkeypatch.setattr(legacy,'pidfd_signal',record)
            with pytest.raises(RuntimeError,match='Unwitnessed'):
                owner.finish(terminate=True,deadline=deadline)
            assert not sent  # dead owned root, live unknown orphan deliberately untouched
            # This test-only child exits itself; reap it as the fixture's subreaper.
            assert os.waitpid(orphan['pid'],0)[0]==orphan['pid']
            assert not Path('/proc/'+str(orphan['pid'])).exists()
        finally:
            if child.returncode is None: child.wait(timeout=4)
            owner.close()


def test_child_identity_reuse_during_pidfd_open_rejected(monkeypatch):
    facade=helper.ancestry_supervisor(legacy)
    owner=object.__new__(facade.OwnedSession); owner.members={}; owner.verified=0; owner.evidence=[]
    parent={'pid':10001,'start_ticks':9,'sid':10001}
    child={'pid':10002,'start_ticks':11,'sid':10002,'ppid':10001}
    calls=[]
    def identity(pid):
        if pid==10001: return parent
        calls.append(pid)
        return child if len(calls)==1 else {**child,'start_ticks':12}
    fd=os.open('/dev/null',os.O_RDONLY)
    monkeypatch.setattr(legacy,'proc_identity',identity)
    monkeypatch.setattr(legacy,'pidfd_open',lambda pid: fd)
    assert not owner._admit_child(child,parent)
    assert not owner.members and not owner.evidence
    with pytest.raises(OSError): os.fstat(fd)


def test_new_helper_implementation_author_cannot_self_review(monkeypatch):
    a=dict(schema='st_T1_native4_execution_admission_v1',status='APPROVED_BOUNDED_T1_ONLY',
      limits=adapter.LIMITS,hostname=adapter.socket.gethostname(),
      runtime={'path':str(adapter.RUNTIME),'commit':adapter.SOURCE},
      adapter={'path':str(Path(adapter.__file__).absolute()),'sha256':'a'*64},
      issued_at_unix=0,latest_start_unix=1e20,valid_until_unix=1e20,
      ownership_helper={'path':str(adapter.OWNERSHIP_HELPER.absolute()),'sha256':adapter.OWNERSHIP_HELPER_SHA},
      checkpoint_commitments={},reviews={'correctness':{'role':'correctness'},'resource':{'role':'resource'}})
    def pinned(pin,decode=True):
        if not decode: return b'fixture'
        if 'role' not in pin: return a
        return dict(status='PASS_BOUNDED_T1_ADAPTER_ONLY',adapter_sha256='a'*64,
          ownership_helper=a['ownership_helper'],reviewer='/root/st_torchrun_process_scope',
          new_checkpoint_reservation_bytes=adapter.LIMITS['new_checkpoint_bytes'],
          checkpoint_root=str(adapter.ROOT/'runs'),uses_old_profile_reservation=False,valid_until_unix=1e20)
    monkeypatch.setattr(adapter,'pin_read',pinned)
    monkeypatch.setattr(adapter,'shared_storage_guard',lambda *args: {})
    monkeypatch.setattr(adapter.os,'sched_getaffinity',lambda pid: set(range(2,6)))
    with pytest.raises(ValueError,match='Independent adapter reviews required'):
        adapter.controls(SimpleNamespace(admission='/fixture',admission_sha256='a'*64))

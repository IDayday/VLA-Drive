"""Original deterministic F1 test body, against the newly fixed core only."""
from contextlib import contextmanager
import ast
import importlib.util
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace
import pytest
ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'full_future_metadata_preparation_v1/run_build_once.py'
spec=importlib.util.spec_from_file_location('independent_fixed_exit_core',SOURCE)
core=importlib.util.module_from_spec(spec);spec.loader.exec_module(core)
m=SimpleNamespace(core=core)

@pytest.mark.parametrize('real_descendant',[False,True])
def test_exit_between_discover_and_done_is_not_a_descendant(tmp_path,monkeypatch,real_descendant):
    envelope={'schema':'st_full_future_child_result_v2','counters':{
        'proc_self_io_baseline':{'rchar':0},'proc_self_io_final':{'rchar':50},'proc_self_io_delta':{'rchar':50}}}
    streams=[]
    for payload in (json.dumps(envelope).encode(),b''):
        r,w=os.pipe();os.write(w,payload);os.close(w);streams.append(os.fdopen(r,'rb'))
    child=SimpleNamespace(pid=123,stdout=streams[0],stderr=streams[1])
    anchor={'pid':123,'sid':123,'start_ticks':1}
    class Owner:
        verified=1
        def __init__(self):self.calls=0;self.signals=[];self.finishes=0;self.finished=False
        def discover(self):
            self.calls+=1
            if real_descendant and not self.signals:return [({'pid':124,'sid':123,'start_ticks':2},-1)]
            # A coherent earlier snapshot saw only the live leader. It exits
            # immediately before leader_done; the next identity scan is empty.
            return [(anchor,-1)] if self.calls==1 else []
        def leader_done(self):return True
        def send(self,signal):self.signals.append(signal)
        def finish(self,**kwargs):self.finishes+=1;self.finished=True;return 0
        def close(self):pass
    owner=Owner()
    @contextmanager
    def lifetime():yield {'signal':None}
    s=SimpleNamespace(lifetime_scope=lifetime,pidfd_open=lambda _:os.open('/dev/null',os.O_RDONLY),
        OwnedSession=lambda _:owner,abort_spawn_setup=lambda *_:pytest.fail('unexpected spawn abort'))
    monkeypatch.setattr(m.core.subprocess,'Popen',lambda *a,**kw:child)
    monkeypatch.setattr(m.core,'owned_proc_io',lambda *a:{'rchar':50})
    monkeypatch.setattr(m.core,'BOOT_IO',{'rchar':0})
    result=m.core.monitor(['fixture'],runtime=str(tmp_path),env={},fd=0,root=tmp_path,s=s,
        m=SimpleNamespace(process_tree_rss=lambda _:0),
        obs=SimpleNamespace(proc_io=lambda:{'rchar':150 if owner.finished else 100}),
        budget={'read_bytes':1000,'rss_bytes':1<<20,'wall_seconds':1000})
    assert owner.finishes==1 and result['cleanup']['live_owned_remaining']==0
    assert result['read_accounting']['charged_parent_child_rchar_bytes']==150
    if real_descendant:
        assert result['failure']=='OWNED_DESCENDANTS_AFTER_LEADER_EXIT'
        assert owner.signals
    else:
        assert result['failure'] is None,'stale leader-only snapshot was misclassified as a surviving descendant'
        assert owner.signals==[]


def test_only_monitor_lifecycle_changed_and_wait_math_unchanged():
    old=subprocess.check_output(['git','show','4f28ee04^:reports/st_world_policy/20261017/full_future_metadata_preparation_v1/run_build_once.py'],text=True)
    funcs=lambda txt:{n.name:ast.dump(n,include_attributes=False) for n in ast.parse(txt).body if isinstance(n,ast.FunctionDef)}
    before,after=funcs(old),funcs(SOURCE.read_text())
    assert set(before)==set(after)
    for name in before:
        if name!='monitor':assert before[name]==after[name],name

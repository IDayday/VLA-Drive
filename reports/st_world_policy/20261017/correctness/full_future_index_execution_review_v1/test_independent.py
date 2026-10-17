"""Deterministic exit-race and additive-policy fixtures; no real index/ledger."""
from contextlib import contextmanager
import builtins
import copy
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace
import pytest

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('review_index',ROOT/'full_future_index_execution_v1/run_index_once.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
spec2=importlib.util.spec_from_file_location('review_owner_fixture',ROOT/'full_future_index_execution_v1/test_run_index_once.py')
t=importlib.util.module_from_spec(spec2);spec2.loader.exec_module(t)

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

@pytest.mark.parametrize('change',['duplicate_old','duplicate_new','legacy','state_dir'])
def test_additive_index_declarations_cannot_rewrite_or_duplicate_existing_allocations(change):
    p,_,_=t.preparation()
    old={'metadata_package':{'schema':'v2','maximum_bytes':8<<30},'state_dir':'/fixture/ledger',
         'claims':[{'root':'/fixture/old','exact':123}],'legacy_reservations':[{'claim_id':'old','reserved_bytes':99}]}
    new=copy.deepcopy(old)
    for role in m.COUNTS:
        pin=p['registrations'][role];root=str(m.BASE/('caches/full_future_request_index_'+role+'_v1'));size=(56 if role=='dev' else 1008)*m.MiB
        new['claims'].append({'root':root,'claim_id':'full-future-'+role+'-index-v1','identity_kind':'st_metadata_build_registration_v1',
            'cache_identity':pin['sha256'],'registration_path':pin['path'],'registration_sha256':pin['sha256'],
            'output_bytes':size,'staging_bytes':size,'metadata_bytes':(16 if role=='dev' else 32)*m.MiB})
    if change=='duplicate_old':new['claims'].append(copy.deepcopy(new['claims'][0]))
    elif change=='duplicate_new':new['claims'].append(copy.deepcopy(new['claims'][1]))
    elif change=='legacy':new['legacy_reservations'][0]['reserved_bytes']=1
    else:new['state_dir']='/fixture/other'
    with pytest.raises(ValueError):m.validate_additive_policy(p,old,new)

def test_complete_growth_is_bounded_before_rejection(tmp_path,monkeypatch):
    path=tmp_path/'COMPLETE.json';path.write_bytes(b'{}')
    original=Path.open;original_lstat=Path.lstat;read_sizes=[];injected=False
    class Reader:
        def __init__(self,stream):self.stream=stream
        def __enter__(self):return self
        def __exit__(self,*args):self.stream.close()
        def read(self,n=-1):
            raw=self.stream.read(n);read_sizes.append(len(raw));return raw
    def racing_open(self,*args,**kwargs):
        if self==path and (args[0] if args else kwargs.get('mode'))=='rb':
            return Reader(original(self,*args,**kwargs))
        return original(self,*args,**kwargs)
    def racing_lstat(self,*args,**kwargs):
        nonlocal injected
        info=original_lstat(self,*args,**kwargs)
        if self==path and not injected:
            injected=True
            # Grow immediately after taking the caller's initial 2-byte stat.
            with builtins.open(path,'wb') as stream:stream.write(b'x'*(2*m.MiB))
        return info
    monkeypatch.setattr(Path,'open',racing_open)
    monkeypatch.setattr(Path,'lstat',racing_lstat)
    with pytest.raises(ValueError):m.read_small_actual(path)
    assert sum(read_sizes)<=m.MiB+1,'COMPLETE size is rejected only after an unbounded read_bytes has already read the grown file'

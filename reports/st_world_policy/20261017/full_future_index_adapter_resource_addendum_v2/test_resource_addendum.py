"""Independent small resource fixtures; never dispatch, index, or mutate a real ledger."""
import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from contextlib import contextmanager
import pytest
OWNER=Path('/mnt/project/VLA-Drive-st-index-bounded-read-20261017/reports/st_world_policy/20261017/full_future_index_execution_v1')
spec=importlib.util.spec_from_file_location('owner_index_control_fixture',OWNER/'test_run_index_once.py')
o=importlib.util.module_from_spec(spec);spec.loader.exec_module(o);m=o.m

@pytest.mark.parametrize('role',['dev','train'])
@pytest.mark.parametrize('key',['read_bytes','rss_bytes','root_bytes','cleanup_seconds','wall_seconds'])
def test_every_role_limit_must_match_exactly(role,key,monkeypatch):
    a=o.admission();a.update(role=role,status='PASS_BOUNDED_'+role.upper()+'_INDEX_ONLY',limits=m.limits(role))
    a['limits'][key]+=1
    monkeypatch.setattr(m.os,'sched_getaffinity',lambda _: {0,1})
    with pytest.raises(ValueError):m.validate_admission(a,10)

@pytest.mark.parametrize('key',['state_dir','cache_cap_bytes','free_reserve_bytes','allowed_roots','legacy_reservations'])
def test_index_declaration_cannot_change_same_ledger_or_global_reserves(key):
    p,_,_=o.preparation()
    old={'state_dir':'/fixture/same_ledger','cache_cap_bytes':768<<30,'free_reserve_bytes':1<<40,
         'allowed_roots':['/fixture/cache'],'legacy_reservations':[{'unchanged':1}], 'claims':[]}
    new=copy.deepcopy(old);new[key]='changed'
    with pytest.raises(ValueError,match='core/reservations'):m.validate_additive_policy(p,old,new)

@pytest.mark.parametrize('case',['headroom','already_started'])
def test_atomic_publication_rejects_before_any_write(tmp_path,case):
    target=tmp_path/'CONTROL_INDEX_STARTED.json'
    if case=='already_started':target.write_text('preserve')
    writes=[]
    ctx=SimpleNamespace(limits={'max_disk_bytes':128<<20},check=lambda **kw:{'charged_disk_bytes':(128<<20)-4096})
    @contextmanager
    def writing(**kwargs):yield ctx
    storage=SimpleNamespace(metadata_writing=writing)
    budget=SimpleNamespace(json_write_bound=lambda _:8192,_atomic=lambda *args:writes.append(args))
    with pytest.raises(ValueError):m.core.publish_guarded(storage,budget,{},target,{'fixture':True})
    assert not writes
    if case=='already_started':assert target.read_text()=='preserve'
    else:assert not target.exists()


def test_complete_hardlink_and_atomic_replace_race_fail_closed(tmp_path,monkeypatch):
    import os
    p=tmp_path/'COMPLETE.json';p.write_text('{}');os.link(p,tmp_path/'alias')
    with pytest.raises(ValueError):m.read_small_actual(p)
    (tmp_path/'alias').unlink();original=m.os.read;done=[]
    def replace(fd,size):
        raw=original(fd,size)
        if not done:
            done.append(True);temp=p.with_suffix('.new');temp.write_text('{"changed":true}');os.replace(temp,p)
        return raw
    monkeypatch.setattr(m.os,'read',replace)
    with pytest.raises(ValueError,match='changed or exceeded bound'):m.read_small_actual(p)


def test_exact_pair_and_unchanged_resource_functions():
    import hashlib, ast
    old=Path('/mnt/project/VLA-Drive-st-full-future-index-control-20261017/reports/st_world_policy/20261017/full_future_index_execution_v1/run_index_once.py')
    new=OWNER/'run_index_once.py'
    assert hashlib.sha256(new.read_bytes()).hexdigest()=='eb9713666a84e8ffba7a55c75f656bb6133218dc626d772200189adad6d41a9a'
    assert hashlib.sha256(m.CORE_PATH.read_bytes()).hexdigest()==m.CORE_SHA=='8674aea1f2a3df894bb3c451287e6a14b5c20d59b01efe4883158f73a3690644'
    funcs=lambda p:{n.name:ast.dump(n,include_attributes=False) for n in ast.parse(p.read_bytes()).body if isinstance(n,ast.FunctionDef)}
    before,after=funcs(old),funcs(new)
    for name in before:
        if name!='read_small_actual':assert before[name]==after[name],name
    oldcore=old.parent.parent/'full_future_metadata_preparation_v1/run_build_once.py'
    before,after=funcs(oldcore),funcs(m.CORE_PATH)
    for name in before:
        if name!='monitor':assert before[name]==after[name],name


def test_bounded_reader_closes_descriptor_on_growth_error(tmp_path,monkeypatch):
    import os
    p=tmp_path/'COMPLETE.json';p.write_text('{}');opened=[];reads=[]
    real_open,real_read=os.open,os.read
    def op(path,*args,**kwargs):
        fd=real_open(path,*args,**kwargs);opened.append(fd);return fd
    def grow(fd,n):
        if not reads:
            with p.open('ab') as stream:stream.write(b' '*(2*m.MiB))
        data=real_read(fd,n);reads.append(len(data));return data
    monkeypatch.setattr(m.os,'open',op);monkeypatch.setattr(m.os,'read',grow)
    with pytest.raises(ValueError):m.read_small_actual(p)
    assert sum(reads)==m.MiB+1 and len(opened)==1
    with pytest.raises(OSError):os.fstat(opened[0])


def test_shared_storage_observable_fd_timestamp_change_is_rejected(monkeypatch):
    import os
    p=Path(__file__).parent/'shared_observable_mtime_fixture.json'
    with p.open('x') as stream:stream.write('{}')
    original=m.os.read;changes=[]
    def change(fd,size):
        data=original(fd,size)
        if not changes:
            before=os.fstat(fd)
            os.utime(p,ns=(before.st_atime_ns,before.st_mtime_ns+2000000000))
            after=os.fstat(fd);changes.append(after.st_mtime_ns!=before.st_mtime_ns)
        return data
    monkeypatch.setattr(m.os,'read',change)
    with pytest.raises(ValueError,match='changed or exceeded bound'):m.read_small_actual(p)
    assert changes==[True]

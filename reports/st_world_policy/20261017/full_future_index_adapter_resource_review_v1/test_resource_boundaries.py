"""Independent small resource fixtures; never dispatch, index, or mutate a real ledger."""
import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from contextlib import contextmanager
import pytest
OWNER=Path('/mnt/project/VLA-Drive-st-full-future-index-control-20261017/reports/st_world_policy/20261017/full_future_index_execution_v1')
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
    (tmp_path/'alias').unlink();original=Path.read_bytes
    def replace(path):
        raw=original(path);temp=path.with_suffix('.new');temp.write_text('{"changed":true}');os.replace(temp,path);return raw
    monkeypatch.setattr(Path,'read_bytes',replace)
    with pytest.raises(ValueError,match='changed during read'):m.read_small_actual(p)

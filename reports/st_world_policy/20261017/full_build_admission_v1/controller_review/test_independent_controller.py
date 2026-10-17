"""Independent small CPU fixtures: no cache builder, ledger, model or GPU."""
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import pytest

OWNER=Path('/mnt/project/VLA-Drive-st-full-future-metadata-preparation-20261017/reports/st_world_policy/20261017/full_future_metadata_preparation_v1/run_build_once.py')
assert hashlib.sha256(OWNER.read_bytes()).hexdigest()=='866c35bce29b2215def6f1fea215dd75d7a62dc650a39d15c6e861f170ec7216'
spec=importlib.util.spec_from_file_location('independent_full_build',OWNER)
c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c)
HELPERS=Path('/mnt/project/VLA-Drive-st-spatial-probe-runtime-v2-20261017/reports/st_world_policy/20261017/spatial_probe_actual_preparation_v1')
for name,sha in c.MONITORS.items(): assert hashlib.sha256((HELPERS/name).read_bytes()).hexdigest()==sha
sys.path.insert(0,str(HELPERS))
import supervise_once as s
import observed_cli as obs

@pytest.mark.parametrize('case',['success','rss','source_read','wall','stale','bad_counter','pipe'])
def test_real_owned_cpu_child_limits(tmp_path,monkeypatch,case):
    # Only this fixture's fresh process is launched/signalled, never a real job.
    (tmp_path/'CONTROL_BUILD_COUNTER_LOCK.json').write_text('')
    p=tmp_path/'CONTROL_BUILD_COUNTERS.json'
    p.write_text(json.dumps({'limit_exceeded':None,'proc_self_io_delta':{'rchar':0}}))
    if case=='bad_counter':p.write_text('{bad')
    if case=='stale':os.utime(p,(time.time()-60,time.time()-60))
    budget={'wall_seconds':10,'rss_bytes':1<<30,'read_bytes':64<<20}
    if case=='source_read':budget['read_bytes']=1
    if case=='wall':budget['wall_seconds']=0
    source="import time;time.sleep(.6);print('{}')"
    if case=='pipe':source="import sys,time;sys.stdout.write('x'*((1<<20)+2));sys.stdout.flush();time.sleep(.6)"
    monkeypatch.setattr(c,'START',time.monotonic())
    stub=SimpleNamespace(process_tree_rss=lambda pid: (2<<30) if case=='rss' else 32<<20)
    with (tmp_path/'owned-fd').open('wb') as fd:
        if case=='bad_counter':
            # Final corrupt read also refuses to publish a successful resource receipt.
            with pytest.raises(json.JSONDecodeError):
                c.monitor([sys.executable,'-c',source],runtime=str(tmp_path),env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),fd=fd.fileno(),root=tmp_path,s=s,m=stub,obs=obs,budget=budget)
            return
        r=c.monitor([sys.executable,'-c',source],runtime=str(tmp_path),env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),fd=fd.fileno(),root=tmp_path,s=s,m=stub,obs=obs,budget=budget)
    assert r['cleanup']['live_owned_remaining']==0
    assert len(r['stdout'].encode())<=1<<20 and len(r['stderr'].encode())<=1<<20
    expected={'success':None,'rss':'RSS_LIMIT','source_read':'PARENT_CHILD_READ_LIMIT','wall':'WALL_LIMIT','stale':'STALE_COUNTER','pipe':'STDOUT_LIMIT'}[case]
    assert r['failure']==expected
    if case=='success':assert r['returncode']==0
    else:assert r['returncode']!=0


def test_control_symlink_and_modified_bytes_reject(tmp_path):
    p=tmp_path/'source';p.write_text('{}');link=tmp_path/'link';link.symlink_to(p)
    h=hashlib.sha256(p.read_bytes()).hexdigest()
    with pytest.raises(ValueError,match='non-symlink'):c.pinned({'path':str(link),'sha256':h})
    p.write_text('{"different":1}')
    with pytest.raises(ValueError,match='changed'):c.pinned({'path':str(p),'sha256':h})


def test_control_atime_only_is_not_content_change(tmp_path):
    p=tmp_path/'source';p.write_text('{}');h=hashlib.sha256(p.read_bytes()).hexdigest()
    os.utime(p,ns=(1,p.stat().st_mtime_ns))
    assert c.pinned({'path':str(p),'sha256':h})=={}


def test_escaped_receipt_is_precharged_before_write(tmp_path):
    # Actual budget helper accounts JSON escaping, unlike len(stdout+stderr).
    source=Path('/mnt/project/VLA-Drive-st-metadata-build-dev-v1-20261017/tools/st_world_policy/cache_storage_budget.py')
    sp=importlib.util.spec_from_file_location('independent_cache_budget',source)
    b=importlib.util.module_from_spec(sp);sp.loader.exec_module(b)
    value={'stdout':'\x00'*(1<<20),'stderr':'\x00'*(1<<20)}
    bound=b.json_write_bound(value)
    assert bound>12<<20
    written=[]
    @contextmanager
    def context(**kw):
        yield SimpleNamespace(limits={'max_disk_bytes':8<<20},check=lambda **kw:{'charged_disk_bytes':0})
    stub=SimpleNamespace(_atomic=lambda *args:written.append(args),json_write_bound=b.json_write_bound)
    with pytest.raises(ValueError,match='headroom'):
        c.publish_guarded(SimpleNamespace(metadata_writing=context),stub,{},tmp_path/'RESULT.json',value)
    assert not written

"""CPU-only wait-boundary accounting; <=12MiB total fixture payload, no real builder."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import pytest

spec=importlib.util.spec_from_file_location('future_final_io',Path(__file__).with_name('run_build_once.py'))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def row(value):return {'rchar':value}


def test_parent_after_wait_absorbs_child_once_and_keeps_parent_tail():
    a=m.terminal_read_accounting(row(100),row(250),row(1260),1000,10000,final_child_available=True)
    assert a['status']=='COMPLETE'
    assert a['parent_pre_reap_delta_rchar_bytes']==150
    assert a['parent_activity_during_wait_rchar_bytes']==10
    assert a['charged_parent_child_rchar_bytes']==1160
    assert m.charge_after_reap(a,row(1280))==1180
    assert m.charge_after_reap(a,row(1280))!=2180  # historical duplicated-child formula


@pytest.mark.parametrize('case',['missing_before','missing_after','negative','bool','fractional','regressed','not_absorbed','missing_child'])
def test_incomplete_terminal_evidence_is_not_discounted(case):
    before,after,child,available=row(250),row(1260),1000,True
    if case=='missing_before':before=None
    elif case=='missing_after':after=None
    elif case=='negative':before=row(-1)
    elif case=='bool':after=row(True)
    elif case=='fractional':child=1000.0
    elif case=='regressed':before=row(50)
    elif case=='not_absorbed':after=row(1249)
    else:available=False
    a=m.terminal_read_accounting(row(100),before,after,child,10000,final_child_available=available)
    assert a['status']=='INCOMPLETE_CONSERVATIVE'
    assert a['charged_parent_child_rchar_bytes']==10000


@pytest.mark.parametrize('current',[None,{},row(True),row(-1),row(1259)])
def test_bad_post_wait_parent_tail_is_rejected(current):
    a=m.terminal_read_accounting(row(100),row(250),row(1260),1000,10000,final_child_available=True)
    with pytest.raises(ValueError):m.charge_after_reap(a,current)


def test_historical_dev_double_addend_is_not_reused():
    # Diagnostic arithmetic from independent c14f0aa2; old receipt is not changed.
    child=4273867611;parent_after=4322097186
    a=m.terminal_read_accounting(row(0),row(48229000),row(parent_after),child,16<<30,final_child_available=True)
    assert a['charged_parent_child_rchar_bytes']==4322097186
    assert a['charged_parent_child_rchar_bytes']+child==8595964797


@pytest.mark.parametrize('exitcode',[0,3])
def test_real_owned_read_child_is_counted_once_across_wait(tmp_path,monkeypatch,exitcode):
    helpers=Path('/mnt/project/VLA-Drive-st-spatial-probe-runtime-v2-20261017/reports/st_world_policy/20261017/spatial_probe_actual_preparation_v1')
    for name,digest in m.MONITORS.items():assert hashlib.sha256((helpers/name).read_bytes()).hexdigest()==digest
    monkeypatch.syspath_prepend(str(helpers))
    import supervise_once as supervisor
    import observed_cli as observer
    child_file=tmp_path/'child_payload';parent_file=tmp_path/'parent_payload'
    child_file.write_bytes(b'c'*(4<<20));parent_file.write_bytes(b'p'*(2<<20))
    code="""import json,sys,time
read=lambda:{k:int(v) for k,v in (line.split(':') for line in open('/proc/self/io'))}
before=read()
with open(sys.argv[1],'rb') as f:
    while f.read(1<<20):pass
time.sleep(.3)
after=read()
print(json.dumps({'schema':'st_full_future_child_result_v2','cli_result':{'status':'GUARDED_METADATA_RETURN'},'error':None if int(sys.argv[2])==0 else {'type':'FixtureExit','message':'bounded nonzero fixture'},'counters':{'proc_self_io_baseline':before,'proc_self_io_final':after,'proc_self_io_delta':{k:after[k]-before[k] for k in after}}}))
raise SystemExit(int(sys.argv[2]))
"""
    parent_read=[]
    def rss(_):
        if not parent_read:
            with parent_file.open('rb') as f:
                while f.read(1<<20):pass
            parent_read.append(True)
        return 32<<20
    monkeypatch.setattr(m,'BOOT_IO',observer.proc_io())
    monkeypatch.setattr(m,'START',time.monotonic())
    budget=m.limits('dev');budget.update(wall_seconds=10,read_bytes=64<<20,rss_bytes=1<<30)
    with (tmp_path/'owned-fd').open('wb') as fd:
        r=m.monitor([sys.executable,'-c',code,str(child_file),str(exitcode)],runtime=str(tmp_path),
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),fd=fd.fileno(),root=tmp_path,s=supervisor,
            m=SimpleNamespace(process_tree_rss=rss),obs=observer,budget=budget)
    a=r['read_accounting'];child=r['charged_owned_child_rchar_bytes']
    assert r['returncode']==exitcode and r['failure'] is None
    assert r['cleanup']['live_owned_remaining']==0 and a['status']=='COMPLETE'
    assert child>=4<<20 and a['parent_pre_reap_delta_rchar_bytes']>=2<<20
    assert a['parent_post_reap']['rchar']>=a['parent_pre_reap']['rchar']+child
    assert a['charged_parent_child_rchar_bytes']==a['parent_post_reap']['rchar']-m.BOOT_IO['rchar']
    assert a['charged_parent_child_rchar_bytes']>=child+(2<<20)
    # Further parent reads are added exactly once after the child has been reaped.
    prior=observer.proc_io()
    with parent_file.open('rb') as f:
        while f.read(1<<20):pass
    current=observer.proc_io()
    total=m.charge_after_reap(a,current)
    assert total==current['rchar']-m.BOOT_IO['rchar']
    assert total-m.charge_after_reap(a,prior)>=2<<20
    assert not list(tmp_path.glob('CONTROL*'))

"""CPU metadata only; all host probes/launches mocked, no live files touched."""
import copy,json
import pytest
from pathlib import Path
from tools.d46_priority_rl import owner as o

H=('training-vlawm-zt','training-vlawm-tsinghua-zt2')

def fixture(monkeypatch,tmp_path):
    monkeypatch.setattr(o,'ROOT',tmp_path)
    control={'capacity_resume':dict(enabled=True,arms=['D3_AWM','D5','D1'],hosts=list(H),minimum_free_MiB=79000)}
    runs={a:dict(arm=a,output=str(tmp_path/a),qualification=str(tmp_path/(a+'_q')),jobs=[],qualification_jobs=[],status='HELD')for a in('D3_AWM','D5','D1','D0')}
    c={'runs':runs};calls=[]
    monkeypatch.setattr(o,'allowed_hosts',lambda **kw:set(H))
    monkeypatch.setattr(o,'capacity',lambda h,m:(calls.append((h,m))or list(range(8))))
    return c,control,calls


def test_capacity_default_failure_closed_and_reserved(monkeypatch,tmp_path):
    c,p,calls=fixture(monkeypatch,tmp_path);r=c['runs']['D3_AWM']
    assert o.capacity_resume(c,{},r,'resource hold',{})is None and not calls
    assert o.capacity_resume(c,p,c['runs']['D0'],'gang hold',{})is None
    wrong=copy.deepcopy(p);wrong['capacity_resume']['hosts']=['training-vla-zt']
    assert o.capacity_resume(c,wrong,r,'resource hold',{})is None
    for values in([],list(range(7))):
        monkeypatch.setattr(o,'capacity',lambda h,m:values)
        assert o.capacity_resume(c,p,r,'resource hold',{})is None
    monkeypatch.setattr(o,'capacity',lambda *a:(_ for _ in()).throw(OSError('unknown memory/unreachable')))
    assert o.capacity_resume(c,p,r,'resource hold',{})is None
    monkeypatch.setattr(o,'capacity',lambda h,m:list(range(8)))
    c['world16_reserved_hosts']=list(H)
    assert o.capacity_resume(c,p,r,'resource hold',{})is None
    c.pop('world16_reserved_hosts');monkeypatch.setattr(o,'allowed_hosts',lambda **kw:[])
    assert o.capacity_resume(c,p,r,'resource hold',{})is None
    monkeypatch.setattr(o,'allowed_hosts',lambda **kw:H)
    job=tmp_path/'existing_launch';job.mkdir();r['jobs']=[dict(job=str(job),host=H[0],at=0)]
    c['world16_reserved_hosts']=[H[1]]
    assert o.capacity_resume(c,p,c['runs']['D5'],'resource hold',{})is None
    assert not(job/'STOP_REQUESTED').exists()


def test_two_hosts_cached_reserved_active_release_and_changed_hold(monkeypatch,tmp_path):
    c,p,calls=fixture(monkeypatch,tmp_path);cache={}
    for a,h in zip(('D3_AWM','D5'),H):
        assert o.capacity_resume(c,p,c['runs'][a],'resource hold',cache)==h
    assert o.capacity_resume(c,p,c['runs']['D1'],'resource hold',cache)is None
    assert calls==[(h,79000)for h in H]
    r=c['runs']['D3_AWM'];saved=copy.deepcopy(r['capacity_resume_release'])
    job=tmp_path/'active';job.mkdir();r['jobs']=[dict(job=str(job),host=H[0],at=0)]
    monkeypatch.setattr(o,'capacity',lambda *a:(_ for _ in()).throw(AssertionError('must not probe active release')))
    assert o.capacity_resume(c,p,r,'resource hold',{})==H[0]
    assert o.capacity_resume(c,p,r,'new resource hold',{})is None
    assert r['capacity_resume_release']==saved and not(job/'STOP_REQUESTED').exists()
    (job/'RESULT.json').write_text(json.dumps({'returncode':0}))
    assert o.capacity_resume(c,p,r,'new resource hold',{})is None
    assert saved['hold_text']=='resource hold'and len(saved['hold_sha256'])==64
    assert json.loads(json.dumps(r))==r


def test_tick_holds_only_override_explicit_receipt_and_dispatch_worker_guards(monkeypatch,tmp_path):
    from tools.d44_native_rl import campaign as base
    from tools.d46_priority_rl import world16_owner
    c,p,calls=fixture(monkeypatch,tmp_path)
    p['execution_holds']={a:'resource hold'for a in c['runs']}
    policy=tmp_path/'operations/EXECUTION_POLICY.json';policy.parent.mkdir();policy.write_text(json.dumps(p))
    monkeypatch.setattr(o,'ORDER',tuple(c['runs']))
    monkeypatch.setattr(world16_owner,'ingest',lambda *a:None)
    monkeypatch.setattr(o,'qualification_pass',lambda r:True)
    monkeypatch.setattr(o,'inspect_latest',lambda r:(None,None))
    monkeypatch.setattr(base,'pending_evaluation',lambda *a:False)
    monkeypatch.setattr(base,'evaluation_tick',lambda *a:None)
    monkeypatch.setattr(base,'selection_index',lambda *a:{})
    monkeypatch.setattr(o,'status',lambda *a:None)
    dispatched=[]
    def dispatch(r,q,blocked,capacity_host=None):
        if r['jobs']and not(Path(r['jobs'][-1]['job'])/'RESULT.json').exists():return 'RUNNING_OR_ADMITTING'
        dispatched.append((r['arm'],capacity_host))
        job=tmp_path/'jobs'/r['arm'];job.mkdir(parents=True)
        r['jobs'].append(dict(job=str(job),host=capacity_host,at=0))
        return 'RUNNING_OR_ADMITTING'
    original=o.dispatch;monkeypatch.setattr(o,'dispatch',dispatch)
    o.tick(c)
    assert dispatched==[('D3_AWM',H[0]),('D5',H[1])]
    assert c['runs']['D0']['status']=='SAVED_AND_HELD_FOR_REGISTERED_STAGE'
    o.tick(c)
    assert not(tmp_path/'jobs/D3_AWM/STOP_REQUESTED').exists()
    p['execution_holds']['D3_AWM']='new hold';policy.write_text(json.dumps(p));o.tick(c)
    assert(tmp_path/'jobs/D3_AWM/STOP_REQUESTED').exists()
    # Actual dispatch still invokes unchanged worker with physical 8 GPUs and
    # the stricter memory threshold. No independent probe/lease bypass.
    r=c['runs']['D1'];cfg=tmp_path/'cfg.yaml';cfg.write_text('arm: D1\n')
    r.update(config=str(cfg),config_sha256=o.file_sha(cfg),source='immutable_science')
    seen=[]
    monkeypatch.setattr(o,'startup_control',lambda *a:None)
    monkeypatch.setattr(o,'capacity',lambda *a:(_ for _ in()).throw(AssertionError('no duplicate probe')))
    monkeypatch.setattr(o,'launch',lambda *a,**kw:(seen.append((a,kw))or dict(host=a[0],job=str(a[2]),at=0)))
    assert original(r,capacity_host=H[0])=='RUNNING_OR_ADMITTING'
    assert seen[0][0][-1]==list(range(8))
    assert seen[0][1]['share_min_free_mib']==79000
    assert seen[0][1]['worker_source']==o.SOURCE and seen[0][1]['task_root']==tmp_path

    failure=Path(r['jobs'][-1]['job']);failure.mkdir(parents=True,exist_ok=True)
    (failure/'RESULT.json').write_text(json.dumps({'returncode':1}))
    with pytest.raises(RuntimeError,match='Actual child failure retained'):
        original(r,capacity_host=H[0])
    assert len(seen)==1

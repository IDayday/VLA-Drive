"""Bounded runner controls and mathematical gate fixtures, no real scoring."""
from copy import deepcopy
import hashlib
import json
import lzma
from pathlib import Path
import pickle
import socket
import time
import numpy as np
import pytest
import torch
from tools.st_world_policy import run_road_direction as r
from starVLA.dataloader.foresight_dataset import decode_ego


def fixture_raw():
    g=torch.Generator().manual_seed(901)
    raw=torch.randn((8,4),generator=g);raw[:,3]+=2
    return raw


def test_xy_analytical_envelope_and_yaw_tolerance():
    raw=fixture_raw();saved=decode_ego(raw).numpy().astype(np.float64)
    gate=r.decode_discrepancy_gate(raw,saved)
    assert np.max(np.abs(gate['observed_offsets']))>0
    changed=saved.copy();changed[:,0]+=.0001
    with pytest.raises(ValueError,match='XY_DECODE'):r.decode_discrepancy_gate(raw,changed)
    changed=saved.copy();changed[:,2]+=2e-6
    with pytest.raises(ValueError,match='YAW_DECODE'):r.decode_discrepancy_gate(raw,changed)
    with pytest.raises(ValueError,match='float32'):r.decode_discrepancy_gate(raw.double(),saved)


def test_xy_envelope_covers_many_finite_fp32_commands():
    for multiplier in (.00001,.01,1,100,10000):
        raw=fixture_raw();raw[:,:2]*=multiplier
        r.decode_discrepancy_gate(raw,decode_ego(raw).numpy().astype(np.float64))


def test_yaw_gate_periodic_seam():
    raw=fixture_raw();raw[:,2]=0;raw[:,3]=-1
    saved=decode_ego(raw).double().numpy();saved[:,2]-=2*np.pi
    r.decode_discrepancy_gate(raw,saved)


def test_bounded_decompression_refuses_bomb_and_trailing():
    value={'fixed':'fixture'};data=lzma.compress(pickle.dumps(value))
    assert r.bounded_unpack(data,1024)==value
    with pytest.raises(r.LimitError):r.bounded_unpack(lzma.compress(pickle.dumps('x'*10000)),1024)
    with pytest.raises(r.LimitError):r.bounded_unpack(data+b'trailing',1024)


def test_budget_hash_change_and_read_cap(tmp_path):
    p=tmp_path/'source';p.write_bytes(b'123')
    limits={**r.LIMITS,'asset_read_bytes':5}
    budget=r.Budget(tmp_path/'out',limits)
    assert budget.read(p,r.sha(b'123'))==b'123'
    with pytest.raises(r.LimitError,match='ASSET_READ'):budget.read(p,r.sha(b'123'))
    with pytest.raises(ValueError,match='ASSET_CHANGED'):r.Budget(tmp_path/'out').read(p,'0'*64)


def test_immutable_result_and_escape_refusal(tmp_path):
    root=tmp_path/'out';root.mkdir();budget=r.Budget(root)
    budget.write(root/'a.json',{'v':1})
    with pytest.raises(ValueError):budget.write(root/'a.json',{'v':2})
    with pytest.raises(ValueError,match='ESCAPE'):budget.write(tmp_path/'a.json',{'v':1})


def test_prepare_only_no_execution_permission(tmp_path):
    reg=tmp_path/'reg.json'
    assert r.main(['--prepare','--registration',str(reg),'--output-root',str(tmp_path/'out')])==0
    value=json.loads(reg.read_text());assert not value['dispatch_authorized']
    assert value['limits']['max_official_calls']==128
    assert value['decode_gate']==r.DECODE_GATE
    assert r.main(['--execute','--registration',str(reg)])==2
    assert not (tmp_path/'out').exists()


def _admission_fixture(tmp_path,monkeypatch):
    hashes={'code':'a'*64};monkeypatch.setattr(r,'source_hashes',lambda:hashes)
    reg={'schema':r.SCHEMA,'source_commit':r.source_commit(),'source_hashes':hashes,'limits':r.LIMITS,'decode_gate':r.DECODE_GATE}
    rp=tmp_path/'reg.json';rp.write_text(json.dumps(reg));rh=r.sha(rp.read_bytes())
    refs={}
    for kind in ('correctness','resources'):
        p=tmp_path/(kind+'.json');p.write_text(json.dumps({'status':'APPROVE_ROAD_DIRECTION_EXECUTION',
            'review_kind':kind,'registration_sha256':rh,'source_hashes':hashes,'reviewer':kind}))
        refs[kind]={'path':str(p),'sha256':r.sha(p.read_bytes())}
    policy=tmp_path/'policy.json';policy.write_text(json.dumps({'allowed_aliases':[socket.gethostname()]}))
    admission={'schema':'road_direction_lead_admission_v1','status':'APPROVED','issuer':'lead',
        'host':socket.gethostname(),'not_before_unix':time.time()-1,'expires_unix':time.time()+4000,
        'registration_sha256':rh,'source_hashes':hashes,'reviews':refs,
        'dispatch_policy':{'path':str(policy),'sha256':r.sha(policy.read_bytes())}}
    ap=tmp_path/'admission.json';ap.write_text(json.dumps(admission))
    return rp,ap,admission


def test_exact_approval_not_file_existence(tmp_path,monkeypatch):
    rp,ap,a=_admission_fixture(tmp_path,monkeypatch)
    assert r.check_admission(rp,ap,r.sha(ap.read_bytes()))[0]['schema']==r.SCHEMA
    a['reviews']['correctness']['sha256']='f'*64;ap.write_text(json.dumps(a))
    with pytest.raises(ValueError,match='Review checksum'):r.check_admission(rp,ap,r.sha(ap.read_bytes()))


def test_prepare_pass_does_not_approve_real_execution(tmp_path,monkeypatch):
    rp,ap,a=_admission_fixture(tmp_path,monkeypatch)
    ref=a['reviews']['correctness'];p=Path(ref['path']);review=json.loads(p.read_text())
    review['status']='PASS_PREPARATION_ONLY';p.write_text(json.dumps(review))
    ref['sha256']=r.sha(p.read_bytes());ap.write_text(json.dumps(a))
    with pytest.raises(ValueError,match='does not approve'):r.check_admission(rp,ap,r.sha(ap.read_bytes()))


def test_source_change_invalidates_admission(tmp_path,monkeypatch):
    rp,ap,a=_admission_fixture(tmp_path,monkeypatch)
    monkeypatch.setattr(r,'source_hashes',lambda:{'changed':'x'})
    with pytest.raises(ValueError,match='code or constants'):r.check_admission(rp,ap,r.sha(ap.read_bytes()))


def test_all_scene_denominators_preserved(tmp_path):
    evidence={'selected':[{'scene_id':f's{i}'} for i in range(32)],'selection_sha256_before_scores':'x'}
    root=tmp_path/'out';root.mkdir()
    scene=root/'scenes/s0';scene.mkdir(parents=True)
    (scene/'SCENE_RESULT.json').write_text(json.dumps({'scene_id':'s0','status':'UNDEFINED_DIRECTION'}))
    result=r.summarize(root,evidence,{'worker_exit_code':2})
    assert result['requested_scenes']==32 and result['complete_scenes']==0
    assert len(result['scenes'])==32 and result['road_repair_denominator']==0
    assert result['status']=='PARTIAL_OR_UNDEFINED'
    assert 'valid_subset_only_direction_contrasts' not in result


def test_verified_context_changed_between_read_and_bind(tmp_path):
    # Constructor is not invoked: no canonical scorer or map asset is loaded.
    from types import SimpleNamespace
    request=SimpleNamespace(metric_cache_path=str(tmp_path/'root/log/unknown/scene/metric_cache.pkl'),
        source_log='log',scene_id='scene',metric_cache_sha256='a'*64)
    p=Path(request.metric_cache_path);p.parent.mkdir(parents=True);p.write_bytes(b'x')
    engine=object.__new__(r.VerifiedContextEngine)
    engine.spec=SimpleNamespace(cache_roots=[str(tmp_path/'root')])
    with pytest.MonkeyPatch.context() as m:
        m.setattr(r,'validate_cache',lambda c:None)
        with pytest.raises(ValueError,match='changed before binding'):
            engine.bind(request,object(),(0,0,0,0,0))


def test_tracking_diagnostics_explicit_undefined_curvature():
    command=np.zeros((41,3));states=np.zeros((41,11))
    result=r.tracking_diagnostics(command,states)
    assert result['command_curvature_max_abs_per_m_moving_only'] is None
    assert result['command_curvature_undefined_zero_distance_intervals']==40
    assert result['tracking_max_xy_error_m']==0


def test_bounded_read_growth_cannot_exceed_reserved_probe(tmp_path,monkeypatch):
    p=tmp_path/'asset';p.write_bytes(b'abc');original_open=Path.open
    def growing_open(self,*args,**kwargs):
        if self==p and args and args[0]=='rb':
            with original_open(self,'wb') as f:f.write(b'a'*10000)
        return original_open(self,*args,**kwargs)
    monkeypatch.setattr(Path,'open',growing_open)
    budget=r.Budget(tmp_path/'out',{**r.LIMITS,'asset_read_bytes':5})
    with pytest.raises(ValueError,match='ASSET_CHANGED'):budget.read(p,r.sha(b'abc'))
    assert budget.read_bytes==4
    assert budget.reads[0]['bytes']==4


def test_output_scan_ignores_only_vanished_file(tmp_path,monkeypatch):
    (tmp_path/'good').write_bytes(b'abc');vanished=tmp_path/'tmpfile';vanished.write_bytes(b'x')
    real=Path.lstat
    def race(self,*args,**kwargs):
        if self==vanished:raise FileNotFoundError('atomic temporary unlinked')
        return real(self,*args,**kwargs)
    monkeypatch.setattr(Path,'lstat',race)
    assert r.owned_output_bytes(tmp_path)==3
    def denied(self,*args,**kwargs):raise PermissionError('real monitor error')
    monkeypatch.setattr(Path,'lstat',denied)
    with pytest.raises(PermissionError):r.owned_output_bytes(tmp_path)


def _fault_worker(*args):
    import os,time
    if os.environ['ROAD_UNIT_FAULT_MODE']=='native_exit':os._exit(23)
    time.sleep(60)


@pytest.mark.parametrize('mode',['native_exit','monitor_error','sigterm','sigint','timeout'])
def test_real_spawn_cleanup_and_full_denominator(tmp_path,monkeypatch,mode):
    import multiprocessing,os,signal
    owned=[];context=multiprocessing.get_context('spawn')
    class Context:
        def Process(self,*args,**kwargs):
            process=context.Process(*args,**kwargs);owned.append(process);return process
    evidence={'selected':[{'scene_id':f'fixture{i}'} for i in range(32)],'selection_sha256_before_scores':'fixture'}
    ep=tmp_path/'evidence.json';ep.write_text(json.dumps(evidence));rp=tmp_path/'reg.json';rp.write_text('{}')
    out=tmp_path/'out'
    reg={'output_root':str(out),'source_hashes':{'fixture':'only'},
         'evidence':{'path':str(ep),'sha256':r.sha(ep.read_bytes())}}
    monkeypatch.setattr(r,'check_admission',lambda *args:(reg,{'attempt_id':mode}))
    monkeypatch.setattr(r,'_child_entry',_fault_worker)
    monkeypatch.setattr(multiprocessing,'get_context',lambda kind:Context())
    monkeypatch.setenv('ROAD_UNIT_FAULT_MODE',mode)
    if mode=='timeout':monkeypatch.setattr(r,'LIMITS',{**r.LIMITS,'wall_seconds':21})
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):monkeypatch.setenv(key,'1')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    original_handlers={sig:signal.getsignal(sig) for sig in (signal.SIGTERM,signal.SIGINT)}
    original_scan=r.owned_output_bytes
    def scan(root):
        if mode=='monitor_error':raise FileNotFoundError('fixture monitor exception')
        if mode=='sigterm':os.kill(os.getpid(),signal.SIGTERM)
        if mode=='sigint':os.kill(os.getpid(),signal.SIGINT)
        return original_scan(root)
    monkeypatch.setattr(r,'owned_output_bytes',scan)
    try:
        assert r.supervise(rp,tmp_path/'notread','fixture',False)==2
        assert owned and not any(c.is_alive() for c in owned)
        result=json.loads((out/f'RESULT_{mode}.json').read_text())
        assert result['requested_scenes']==32 and len(result['scenes'])==32
        assert result['supervisor']['owned_worker_exit_confirmed']
        assert result['supervisor']['resource_tracker_pid'] is not None
        if mode=='native_exit':assert result['supervisor']['worker_exit_code']==23
        elif mode=='monitor_error':assert result['supervisor']['stop_reason']=='MONITOR_ERROR'
        elif mode=='timeout':assert result['supervisor']['stop_reason']=='WALL_CAP'
        else:assert result['supervisor']['stop_reason']=='SUPERVISOR_INTERRUPTED'
        for sig,handler in original_handlers.items():assert signal.getsignal(sig)==handler
    finally:
        for child in owned:
            if child.is_alive():child.kill();child.join(timeout=3)

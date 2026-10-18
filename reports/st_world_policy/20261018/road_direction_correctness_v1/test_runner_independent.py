"""Independent runner fault/identity fixtures; no real asset or scoring reads."""
import io
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from tools.st_world_policy import run_road_direction as r
from starVLA.dataloader.foresight_dataset import decode_ego


def test_loader_joins_all32_by_identity_and_rejects_mutated_condition(tmp_path,monkeypatch):
    entries=[]
    def put(path,data):
        path=tmp_path/path;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(data);entries.append({'path':str(path),'sha256':r.sha(data),'bytes':len(data)})
        return {'path':str(path),'sha256':r.sha(data)}
    def data(value):
        return json.dumps(value).encode()
    def tensor(value):
        stream=io.BytesIO();torch.save(value,stream);return stream.getvalue()
    policy={'optimizer_step':4095,'fixture':True};selected=[];receipts=[];records=[];requests=[];index=[]
    for rank in range(4):
        raw=torch.zeros((8,4,8,4),dtype=torch.float32);raw[:,:,:,3]=1
        scenes=[f'scene_{rank}_{slot}' for slot in range(8)]
        rollout={'policy':policy,'flow_steps':10,'raw_representation':'normalized_xy_sin_cos',
                 'raw_actions':raw,'scene_ids':scenes,'observation_hashes':['a'*64]*8,
                 'condition_hashes':['b'*64]*8}
        payload=put(Path(f'rollout/rank_{rank}.pt'),tensor({'identity':{'rank':rank},'rollout':rollout}))
        receipt=put(Path(f'rollout/rank_{rank}_COMPLETE.json'),data({'identity':{'rank':rank},'payload':payload}))
        receipts.append({'path':receipt['path'],'payload_sha256':payload['sha256']})
        for slot,scene in enumerate(scenes):
            physical=decode_ego(raw[slot,0]).double().numpy()
            request=r.ConsequenceRequest('campaign',scene,'log',f'candidate_{rank}_{slot}','4095',
                'c'*64,'d'*64,r.array_sha256(physical),'e'*64,'f'*64,
                f'/fixture/log/unknown/{scene}/metric_cache.pkl',physical)
            identity=request.identity();requests.append(identity)
            records.append({'identity':identity,'requested_trajectory':physical,'score_units':'0..1',
                            'metrics':{k:1. for k in r.METRICS},'metrics_valid':{k:True for k in r.METRICS}})
            item={'rank':rank,'slot':slot,'candidate':0,'raw_sha256':r.sha(raw[slot,0].numpy().tobytes()),
                  'observation_hash':'a'*64,'condition_hash':'b'*64,**identity}
            selected.append(item);index.append({'token':scene,'log':'log'})
    put(Path('feedback/REQUESTS.json'),data({'policy_version':'4095','requests':requests}))
    admitted=put(Path('feedback/attempt_00/ADMITTED.json'),data({'fixture':True}))
    saved=put(Path('feedback/attempt_00/RESULTS.pt'),tensor(records))
    complete={'admission_sha256':admitted['sha256'],'result':saved}
    put(Path('feedback/attempt_00/COMPLETE.json'),data(complete))
    put(Path('student_train_v1/index.json'),data(index))
    evidence={'budget':{'entries':entries},'policy':policy,'rollout_receipts':receipts,
              'stored_official_result':complete,'train_index_identity':{'index_sha256':r.fingerprint(index)},
              'selected':selected}
    # This fixture isolates the loader joins; existing native reference checks
    # are deliberately mocked, not claimed to have scored a synthetic scene.
    monkeypatch.setattr(r,'validated_reference_input',lambda record,request:{})
    loaded=r.load_inputs(evidence,r.Budget(tmp_path/'out'))
    assert len(loaded)==32 and [row[0]['scene_id'] for row in loaded]==[x['scene_id'] for x in selected]
    selected[13]['condition_hash']='9'*64
    with pytest.raises(ValueError,match='Selected scene/raw/provenance mismatch'):
        r.load_inputs(evidence,r.Budget(tmp_path/'out'))


def fixture_worker_controls(tmp_path,monkeypatch):
    monkeypatch.setattr(torch,'set_num_threads',lambda n:None)
    monkeypatch.setattr(torch,'set_num_interop_threads',lambda n:None)
    monkeypatch.setattr(r.signal,'signal',lambda *args:None)
    monkeypatch.setattr(r.signal,'alarm',lambda *args:None)
    monkeypatch.setattr(r.resource,'setrlimit',lambda *args:None)
    evidence=tmp_path/'evidence.json';evidence.write_text(json.dumps({'sources':[]}))
    root=tmp_path/'out';root.mkdir()
    reg={'output_root':str(root),'evidence':{'path':str(evidence),'sha256':r.sha(evidence.read_bytes())}}
    return reg,root


def test_worker_input_fault_persists_terminal_receipt(tmp_path,monkeypatch):
    reg,root=fixture_worker_controls(tmp_path,monkeypatch)
    def fault(*args):raise RuntimeError('INJECTED_BEFORE_ANY_CONTEXT')
    monkeypatch.setattr(r,'load_inputs',fault)
    assert r.run_worker(reg,{'attempt_id':'fixture'})==2
    result=json.loads((root/'WORKER_RESULT_fixture.json').read_text())
    assert result['status']=='PARTIAL' and result['error']=='INJECTED_BEFORE_ANY_CONTEXT'
    assert result['cuda_initialized'] is False
    assert not (root/'calls').exists()


def test_started_scene_is_not_replayed_after_interruption(tmp_path,monkeypatch):
    reg,root=fixture_worker_controls(tmp_path,monkeypatch)
    scene=root/'scenes/fixed_scene';scene.mkdir(parents=True)
    (scene/'STARTED.json').write_text('{}')
    evidence=Path(reg['evidence']['path']);evidence.write_text(json.dumps({'sources':[],'source_runtime':'/fixture'}))
    reg['evidence']['sha256']=r.sha(evidence.read_bytes())
    request=SimpleNamespace(campaign_id='fixture',score_protocol_hash='f'*64)
    monkeypatch.setattr(r,'load_inputs',lambda *args:[({'scene_id':'fixed_scene',
        'metric_cache_path':'/fixture/log/unknown/fixed_scene/metric_cache.pkl'},None,request,{'metrics':{'score':1.}})])
    class EngineMustNotLoadContext:
        def __init__(self,spec):self.spec=spec
        def bind(self,*args):raise AssertionError('Interrupted scene was replayed')
    monkeypatch.setattr(r,'VerifiedContextEngine',EngineMustNotLoadContext)
    assert r.run_worker(reg,{'attempt_id':'fixture','maps_root':str(tmp_path)})==0
    result=json.loads((scene/'SCENE_RESULT.json').read_text())
    assert result['status']=='UNCERTAIN_INTERRUPTED_NOT_RETRIED'
    assert not (root/'calls').exists()


def test_gate_does_not_modify_saved_inputs():
    raw=torch.ones((8,4),dtype=torch.float32);physical=decode_ego(raw).double().numpy()
    old_raw=raw.clone();old_physical=physical.copy()
    r.decode_discrepancy_gate(raw,physical)
    assert torch.equal(raw,old_raw) and np.array_equal(physical,old_physical)


def test_cuda_not_initialized():
    assert not torch.cuda.is_initialized()

from dataclasses import asdict
import fcntl
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from iqe.io import atomic_json,atomic_torch,file_hash,read_json
from iqe.evaluation.retention import module_hash
from iqe.geometric_readout import bundle,evaluation
from test_geometric_readout import scorer_example


def test_geometric_bundle_roundtrip_rejects_same_shapes_wrong_implementation():
    m,c,t,contract=scorer_example();m.requires_grad_(False)
    payload=dict(kind='iqe_geometric_scorer_research_v1',scorer=m.state_dict(),
        readout_contract=bundle.readout_contract(),contract=dict(trajectory=asdict(contract)),
        initialization=dict(external_scorer_weights_loaded=False,loaded_scorer_weights=[]))
    restored=bundle.scorer_from_payload(payload)
    with torch.no_grad():assert torch.equal(m(c,t)['values'],restored(c,t)['values'])
    with pytest.raises(ValueError,match='implementation changed'):
        bundle.scorer_from_payload(payload|{'readout_contract':dict(kind='global_spatial_same_parameter_shapes')})
    with pytest.raises(ValueError,match='external'):
        bundle.scorer_from_payload(payload|{'initialization':dict(external_scorer_weights_loaded=True,loaded_scorer_weights=['foreign'])})
    with pytest.raises(RuntimeError):
        bundle.scorer_from_payload(payload|{'scorer':{k:v for k,v in m.state_dict().items() if k!='value_head.weight'}})


def test_export_reuses_identical_endpoint_and_recovers_missing_receipt(tmp_path,monkeypatch):
    config=SimpleNamespace(root=tmp_path)
    model=torch.nn.Linear(2,1);atomic_torch(tmp_path/'scorer/best.pt',model.state_dict())
    choice=dict(state_hash=module_hash(model),checkpoint_hash=file_hash(tmp_path/'scorer/best.pt'))
    atomic_json(tmp_path/'MODEL_CHOICE.json',choice)
    atomic_json(tmp_path/'SELECTOR_MARGIN.json',dict(always_Base=True))
    atomic_json(tmp_path/'scorer/SCORER_INITIALIZATION.json',dict(external_scorer_weights_loaded=False))
    monkeypatch.setattr(evaluation,'selected',lambda _: (model,{}))
    monkeypatch.setattr(bundle,'payload_for',lambda *_:dict(scorer=model.state_dict(),
        pool={'hash':'test_pool'},Base_binding={'checkpoint':'test_Base'},readout_contract=bundle.readout_contract()))
    path=bundle.export(config);checksum=file_hash(path)
    assert bundle.export(config)==path and file_hash(path)==checksum
    receipt=tmp_path/'research_bundle/BUNDLE.json';receipt.unlink()  # explicit fault injection fixture
    assert bundle.export(config)==path and read_json(receipt)['checksum']==checksum
    atomic_json(tmp_path/'SELECTOR_MARGIN.json',dict(always_Base=False))
    with pytest.raises(ValueError,match='stale research bundle'):bundle.export(config)
    atomic_json(tmp_path/'MODEL_CHOICE.json',choice|{'state_hash':'changed'})
    with pytest.raises(ValueError,match='endpoint changed'):bundle.export(config)


def test_finalizer_requires_complete_checkpoint_and_released_run_lock(tmp_path):
    from scripts.iqe.finish_geometric_scorer import ready
    config=SimpleNamespace(root=tmp_path);folder=tmp_path/'scorer'
    assert not ready(config)
    checkpoint=folder/'step_006000.pt';atomic_torch(checkpoint,{'test_fixture':True})
    atomic_torch(folder/'best.pt',{'test_fixture':True})
    atomic_json(str(checkpoint)+'.COMPLETE.json',dict(optimizer_step=6000,checksum=file_hash(checkpoint)))
    result=dict(status='COMPLETE',optimizer_steps=6000,termination={'terminal':True},
        checkpoint=str(checkpoint),best_checkpoint_hash=file_hash(folder/'best.pt'))
    atomic_json(folder/'result.json',result)
    assert ready(config)
    with open(folder/'RUN.lock','a') as guard:
        fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not ready(config)
    atomic_json(folder/'result.json',result|{'optimizer_steps':1000,'status':'STOPPED_AT_BOUNDARY'})
    assert not ready(config)
    atomic_json(folder/'result.json',result);checkpoint.write_bytes(b'corrupt fixture')
    with pytest.raises(ValueError,match='corrupt'):ready(config)


def test_no_test_access_and_no_silent_invalid_prediction_filter():
    with pytest.raises(ValueError,match='no test access'):
        evaluation.predictions(None,'final_test','cpu')
    from iqe.specialist_study.metrics import select
    with pytest.raises(ValueError,match='nonfinite'):
        select(np.array([[.9,np.nan]]),expert_ids=['expert_0','expert_1'])


def test_finalizer_skips_invisible_occupancy(monkeypatch):
    from scripts.iqe import finish_geometric_scorer as f
    monkeypatch.setattr(f,'basic_available_gpu',lambda _:(True,{'unoccupied':True}))
    monkeypatch.setattr(f.subprocess,'check_output',lambda *a,**k:'634\n')
    assert not f.available_gpu(0)[0]
    monkeypatch.setattr(f.subprocess,'check_output',lambda *a,**k:'0\n')
    assert f.available_gpu(0)[0]

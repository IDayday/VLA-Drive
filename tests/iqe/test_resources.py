import json
import pytest
from iqe import resources
from iqe.io import atomic_json, BlockedError


def test_task_authorization_does_not_inherit_other_task_hosts(tmp_path,monkeypatch):
    policy=tmp_path/'policy.json'
    atomic_json(policy,{'hosts':{},'task_authorizations':{
        'other_task':{'allowed_hosts':['training-vlawm-zt4']},
        'iqe_v1':{'allowed_hosts':['training-vlawm-zt2']}}})
    monkeypatch.setattr(resources.socket,'gethostname',lambda:'training-vlawm-zt2-worker-0')
    assert resources.qualify('cpu',policy)['canonical_host']=='training-vlawm-zt2'
    monkeypatch.setattr(resources.socket,'gethostname',lambda:'training-vlawm-zt4-worker-0')
    with pytest.raises(BlockedError,match='authorization'):resources.qualify('cpu',policy)


def test_occupied_device_refused_even_on_authorized_host(tmp_path,monkeypatch):
    policy=tmp_path/'policy.json'
    atomic_json(policy,{'task_authorizations':{'iqe_v1':{'allowed_hosts':['training-vlawm-zt3']}}})
    monkeypatch.setattr(resources.socket,'gethostname',lambda:'training-vlawm-zt3-worker-0')
    monkeypatch.setattr(resources.subprocess,'check_output',lambda *a,**k:'0, 39403, 100\n1, 0, 0\n')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','0')
    with pytest.raises(BlockedError,match='occupied'):resources.qualify('cuda',policy)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','1')
    assert resources.qualify('cuda',policy)['GPUs'][0]['index']=='1'

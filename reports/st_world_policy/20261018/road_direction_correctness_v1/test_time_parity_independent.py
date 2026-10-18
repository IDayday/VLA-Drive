"""Timestamp delta only: synthetic geometry and controls, no real metric caches."""
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from starVLA.model.modules.flow_posttraining import road_margin_proxy as proxy
from tools.st_world_policy import diagnose_road_time_parity as d
from tools.st_world_policy import run_road_direction as r


def test_many_epoch_grids_match_official_TimePoint():
    from nuplan.common.actor_state.state_representation import TimePoint
    rng=np.random.default_rng(61018)
    timestamps=[0,1,1000001]+[int(v) for v in rng.integers(1500000000000000,1800000000000000,size=128)]
    old_diff=0
    for t in timestamps:
        a,b=proxy.command_time_grid(t)
        official_s=TimePoint(t).time_s
        official_a=np.array([t]+[int(v*1e6) for v in np.arange(0.,4.,.5)+.5+official_s])
        official_b=np.clip([int(v*1e6) for v in np.arange(0.,4.+.1,.1)+official_s],official_a[0],official_a[-1])
        np.testing.assert_array_equal(a,official_a);np.testing.assert_array_equal(b,official_b)
        assert len(a)==9 and len(b)==41 and np.all(np.diff(a)>0) and np.all(np.diff(b)>0)
        old_diff+=not np.array_equal(d.legacy_time_grid(t)[1],b)
    assert old_diff>0

@pytest.mark.parametrize('bad',[-1,1.0,True,float('nan')])
def test_time_grid_refuses_wrong_timestamp_contract(bad):
    with pytest.raises(ValueError):proxy.command_time_grid(bad)


def test_timestamp_change_preserves_physical_path_gradient():
    physical=torch.tensor([[.4*i,.01*i*i,.003*i] for i in range(1,9)],dtype=torch.float64,requires_grad=True)
    origin=torch.tensor([1200.,-500.,.4],dtype=torch.float64);t=1630000000000003
    g=torch.Generator().manual_seed(87);direction=torch.randn((8,3),generator=g,dtype=torch.float64)
    def f(x):return proxy.command_poses(x,origin,t)[0].square().mean()
    grad,=torch.autograd.grad(f(physical),physical)
    eps=1e-4
    actual=float((grad*direction).sum());numeric=float((f(physical.detach()+eps*direction)-f(physical.detach()-eps*direction))/(2*eps))
    assert np.isclose(actual,numeric,atol=2e-6,rtol=2e-6)


def test_prepare_fixed_failures_no_cache_open(tmp_path,monkeypatch):
    old=tmp_path/'original';scenes=old/'scenes';scenes.mkdir(parents=True)
    others=tuple(f'fixture_{i:02d}' for i in range(23))
    for scene in (*d.IDS,*others):
        folder=scenes/scene;folder.mkdir()
        failure=scene in d.IDS
        (folder/'SCENE_RESULT.json').write_text(json.dumps({'error':'REAL_OFFICIAL_GEOMETRY_PARITY_FAILURE' if failure else 'other'}))
        if failure:
            trajectory=[[0.,0.,0.]]*8
            request={'scene_id':scene,'trajectory_sha256':r.array_sha256(trajectory),'source_log':'log','metric_cache_path':str(tmp_path/'missing_cache'/'log'/'unknown'/scene/'metric_cache.pkl')}
            (folder/'STARTED.json').write_text(json.dumps({'scene_id':scene,'request':request,'original_physical':trajectory}))
    (old/'EXECUTION_BINDING.json').write_text('{}')
    monkeypatch.setattr(d,'ORIGINAL_ROOT',old);monkeypatch.setattr(d,'code_hashes',lambda:{'synthetic':'test'})
    a=d.prepare(tmp_path/'new')
    assert a['scene_ids']==list(d.IDS) and len(a['rows'])==9 and len(a['receipt_refs'])==42
    assert a['original_denominator']==32 and a['selected_failed_denominator']==9 and a['official_score_calls']==0
    assert not (tmp_path/'missing_cache').exists()
    with pytest.raises(ValueError,match='Do not write'):d.prepare(old/'new')
    (scenes/others[0]/'SCENE_RESULT.json').write_text(json.dumps({'error':'REAL_OFFICIAL_GEOMETRY_PARITY_FAILURE'}))
    with pytest.raises((ValueError,FileNotFoundError)):d.prepare(tmp_path/'changed')


def test_cuda_not_initialized():
    assert not torch.cuda.is_initialized()

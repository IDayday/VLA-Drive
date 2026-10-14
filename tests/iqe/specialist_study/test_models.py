import copy
from types import SimpleNamespace
import pytest
import torch
from torch import nn
from iqe.contracts import TrajectoryContract
from iqe.internal_baseline.scorer import initialization_rng
from iqe.evaluation.retention import _tensor_map_hash
from iqe.specialist_study.models import StudyScorer,EMAExpert,SpatialReader


def contract():return TrajectoryContract(8,.5,'normalized_xy_sincos','ego_relative','rear_axle','radian',(10.,0.,0.,0.),(8.,2.,1.,1.),'fixture')
def context():
    return {'world':torch.randn(2,5,16),'action':torch.randn(2,3,16),'current_ego':torch.randn(2,4),
            'world_valid':torch.ones(2,5,dtype=torch.bool),'action_valid':torch.ones(2,3,dtype=torch.bool)}

def test_independent_candidates_and_common_initialization():
    torch.set_num_threads(1)
    with initialization_rng(1):a=StudyScorer(16,16,contract())
    with initialization_rng(1):b=StudyScorer(16,16,contract(),spatial=True,visual_dim=20)
    common=lambda m:_tensor_map_hash((k,v) for k,v in m.state_dict().items() if not k.startswith('spatial_reader.'))
    assert common(a)==common(b)
    c=context();t=torch.randn(2,3,8,3);a.eval()
    with torch.no_grad():
        p=a(c,t)['values'];q=a(c,t[:,[2,0,1]])['values'];r=a(c,torch.cat((t,t[:,0:1]),1))['values']
    assert torch.allclose(p[:,[2,0,1]],q,atol=1e-7)
    assert torch.allclose(p,r[:,:3],atol=1e-7)
    assert torch.allclose(r[:,0],r[:,3],atol=1e-7)
    c['GT']=torch.zeros(2,8,3)
    with pytest.raises(Exception,match='whitelist'):a(c,t)


def test_spatial_readout_uses_views_masks_and_detaches():
    torch.set_num_threads(1);m=SpatialReader(12,views=3)
    v=torch.randn(2,3,7,12,requires_grad=True);valid=torch.ones(2,3,7,dtype=torch.bool);xy=torch.rand(2,3,7,2)*2-1
    x=m(v,valid,xy);x.square().sum().backward()
    assert v.grad is None
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
    valid[0,1]=False
    with pytest.raises(Exception,match='empty view'):m(v,valid,xy)


def test_ema_private_state_saved_and_not_in_optimizer():
    raw=nn.Sequential(nn.Linear(3,4),nn.Dropout(.2),nn.Linear(4,2));m=EMAExpert(raw)
    m.train();assert m.raw.training and not m.ema.training
    old={k:v.clone() for k,v in m.ema.state_dict().items()};opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=.1)
    m(torch.ones(2,3)).square().sum().backward();opt.step();m.update()
    assert int(m.ema_updates)==1
    assert any(not torch.equal(v,old[k]) for k,v in m.ema.state_dict().items())
    restored=EMAExpert(copy.deepcopy(raw));restored.load_state_dict(m.state_dict(),strict=True)
    assert all(torch.equal(a,b) for a,b in zip(m.state_dict().values(),restored.state_dict().values()))
    assert not any(p.requires_grad for p in restored.ema.parameters())

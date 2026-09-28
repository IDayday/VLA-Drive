import copy
import pytest
import torch
from starVLA.model.modules.trajectory_mae.model import TrajectoryMAE
from starVLA.model.modules.trajectory_mae.tokenizer import TeacherInputs
from starVLA.model.modules.trajectory_mae.masking import TeacherMaskScheduler
from starVLA.model.modules.trajectory_mae.losses import reconstruction_loss


def fixture(batch=2):
    current=torch.randn(batch,4,8);current[...,3:6]=2.;current[:,0,:3]=0.
    active=torch.tensor([[True,True,True,False]]).expand(batch,-1).clone()
    future=torch.randn(batch,4,8,2);visible=active[:,:,None].expand(-1,-1,8).clone();visible[:,0]=False
    return TeacherInputs(current,active,future,visible,torch.zeros(batch,dtype=torch.long),
                         torch.ones(batch,dtype=torch.long),torch.zeros(batch,4))


def model():
    torch.manual_seed(17)
    return TrajectoryMAE(dim=32,layers=2,decoder_layers=2,heads=4).eval()


def test_hidden_gt_changes_only_loss_not_encoding_or_prediction():
    m=model();x=fixture();y=copy.deepcopy(x)
    y.future[:,0]=torch.randn_like(y.future[:,0])*1000
    first=m(x);second=m(y)
    assert torch.equal(first['latent'],second['latent'])
    assert torch.equal(first['xy'],second['xy'])
    valid=torch.ones(2,8,dtype=torch.bool)
    assert reconstruction_loss(first['xy'],x.future[:,0],valid)[0]!=reconstruction_loss(first['xy'],y.future[:,0],valid)[0]
    y.future[:,0]=float('nan')
    assert torch.equal(first['latent'],m(y)['latent'])


def test_visible_gt_is_used_and_masked_before_encoder():
    m=model();x=fixture();first=m(x)['latent'];x.future[:,1]+=10
    assert not torch.allclose(first,m(x)['latent'])
    x.visible[:,0,0]=True
    with pytest.raises(ValueError,match='ENTIRE'):m(x)


def test_invalid_padding_nan_large_and_backward_isolation():
    m=model();a=fixture();b=copy.deepcopy(a)
    b.current[:,3]=float('nan');b.future[:,3]=1e30;b.future[:,0]=float('nan')
    out=m(a);out['xy'].square().mean().backward();grads={n:p.grad.clone() for n,p in m.named_parameters() if p.grad is not None}
    m.zero_grad();other=m(b);other['xy'].square().mean().backward()
    assert torch.equal(out['xy'],other['xy'])
    for n,p in m.named_parameters():
        if p.grad is not None: assert torch.isfinite(p.grad).all() and torch.equal(grads[n],p.grad)
    b.current[:,1,0]=float('nan')
    with pytest.raises(ValueError,match='active current'):m(b)


def test_invalid_navigation_sizes_and_visible_future_rejected():
    m=model()
    x=fixture();x.navigation[0]=4
    with pytest.raises(ValueError):m(x)
    x=fixture();x.current[0,1,3]=-1
    with pytest.raises(ValueError):m(x)
    x=fixture();x.future[0,1,0]=float('inf')
    with pytest.raises(ValueError):m(x)


def test_balanced_batch1_odd_tail_resume_and_partial_points():
    x=fixture(1);valid=x.active[:,:,None].expand(-1,-1,8).clone();valid[:,1,1:]=False
    sampler=TeacherMaskScheduler(42);roles=[]
    for _ in range(7):roles.append(int(sampler.sample(x.active,valid)[0]))
    assert 0 in roles and any(t>0 for t in roles)
    restored=TeacherMaskScheduler();restored.load_state_dict(sampler.state_dict())
    for batch in [3,1,5]:
        active=x.active.expand(batch,-1);v=valid.expand(batch,-1,-1)
        a,b=sampler.sample(active,v),restored.sample(active,v)
        assert torch.equal(a[0],b[0]) and torch.equal(a[1],b[1])
        assert not a[1][torch.arange(batch),a[0]].any()
    active=torch.ones(1,1,dtype=torch.bool);valid=torch.ones(1,1,8,dtype=torch.bool)
    target,visible=sampler.sample(active,valid)
    assert target.item()==0 and not visible.any()


def test_save_load_and_ego_only_teacher():
    m=model();other=model();other.load_state_dict(m.state_dict(),strict=True)
    x=fixture();x.active[:,1:]=False;x.visible[:]=False
    assert torch.equal(m(x)['latent'],other(x)['latent'])
    m(x)['xy'].square().mean().backward()
    assert m.encoder.layers[0].self_attn.in_proj_weight.grad.abs().sum()>0

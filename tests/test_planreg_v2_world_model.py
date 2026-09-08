from types import SimpleNamespace
import copy
import torch
from navsim.agents.EpisodeDrive.planreg_v2.ema import FP32MasterEMA
from navsim.agents.EpisodeDrive.planreg_v2.predictor import ActionCausalPredictor
from navsim.agents.EpisodeDrive.planreg_v2.losses import world_model_loss, global_valid_mean


def test_block_causal_rollout_gradients_and_no_teacher_leak():
    torch.manual_seed(3)
    model = ActionCausalPredictor(dim=32,layers=2,heads=4,ffn=64)
    z0 = torch.randn(2,16,32,requires_grad=True)
    target = torch.randn(2,3,16,32,requires_grad=True)
    actions = torch.randn(2,3,32,requires_grad=True)
    geometry = torch.randn(2,16,5)
    valid = torch.ones(2,16,dtype=torch.bool)
    semantic = torch.randn(2,16,32,requires_grad=True)
    tf,ro,steps = model.branches(z0,target,actions,geometry,valid,semantic)
    torch.testing.assert_close(tf[:,0],ro[:,0],atol=2e-6,rtol=1e-5)
    changed_actions = actions.detach().clone(); changed_actions[:,2] += 10
    tf2,ro2,_ = model.branches(z0,target+5,changed_actions,geometry,valid,semantic)
    torch.testing.assert_close(tf[:,0],tf2[:,0],atol=2e-6,rtol=1e-5)
    _,ro_same,_ = model.branches(z0,torch.randn_like(target),actions,geometry,valid,semantic)
    torch.testing.assert_close(ro,ro_same)
    for step in steps: step.retain_grad()
    (ro[:,-1]-torch.randn_like(ro[:,-1])).abs().mean().backward()
    assert steps[0].grad.norm() > 0 and z0.grad.norm() > 0 and semantic.grad.norm() > 0
    assert target.grad is None


def test_ema_sub_ulp_and_resume_and_dtype():
    vision = torch.nn.Linear(1,1,bias=False)
    with torch.no_grad(): vision.weight.fill_(1.)
    student = SimpleNamespace(model=SimpleNamespace(vision_model=vision),planning_register_adapter=torch.nn.Linear(1,1,bias=False))
    ema = FP32MasterEMA(student,1200,16).bfloat16()
    reference = {n:getattr(ema,'master_%04d'%i).double().clone() for i,n in enumerate(ema.names)}
    with torch.no_grad(): vision.weight.fill_(1.001)
    for i in range(1000):
        m = ema.momentum()
        source = {'vision.weight':vision.weight,'adapter.weight':student.planning_register_adapter.weight}
        for n,r in reference.items(): r.add_(source[n].detach().double()-r,alpha=1-m)
        ema.update(student)
        if i == 400:
            restored = FP32MasterEMA(student,1200,16).bfloat16()
            restored.load_state_dict(copy.deepcopy(ema.state_dict()),strict=True)
        elif i > 400:
            restored.update(student)
    assert ema.master_0000.dtype == torch.float32 and ema.master_0000.item() > 1.
    torch.testing.assert_close(ema.master_0000.double(),reference['vision.weight'],atol=4e-6,rtol=0)
    torch.testing.assert_close(ema.master_0000,restored.master_0000,atol=0,rtol=0)
    assert all(not p.requires_grad for p in ema.parameters())


def test_invalid_tf_prefix_does_not_invalidate_rollout_middle_target():
    tf = torch.randn(2,3,16,32,requires_grad=True); ro = torch.randn_like(tf,requires_grad=True)
    result = world_model_loss(tf,ro,torch.randn_like(tf),torch.ones(2,16,dtype=torch.bool),
                             torch.tensor([[0,1,1],[0,1,1]]).bool(),torch.ones(2,3,dtype=torch.bool))
    assert result['wm_tf_loss'] == 0 and result['wm_ro_loss'] > 0
    result['wm_loss'].backward(); assert ro.grad.norm() > 0
    x = torch.randn(3,requires_grad=True)
    zero = global_valid_mean(x,torch.zeros(3,dtype=torch.bool))
    zero.backward(); assert torch.isfinite(zero) and x.grad.count_nonzero() == 0


def test_ema_compute_copy_eventually_crosses_bfloat16_ulp():
    vision=torch.nn.Linear(1,1,bias=False)
    with torch.no_grad():vision.weight.fill_(1.)
    student=SimpleNamespace(model=SimpleNamespace(vision_model=vision),planning_register_adapter=torch.nn.Linear(1,1,bias=False))
    ema=FP32MasterEMA(student,1200,16).bfloat16()
    with torch.no_grad():vision.weight.fill_(1.01)
    ema.update(student)
    assert ema.vision.weight.item()==1. and ema.master_0000.item()>1.
    for _ in range(999):ema.update(student)
    assert ema.vision.weight.item()>1. and ema.master_0000.dtype==torch.float32

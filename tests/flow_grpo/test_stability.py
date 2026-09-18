"""CPU control/math tests, not BF16 model evidence."""
import copy
import math
import pytest
import torch
from starVLA.rl.flow_grpo.stability import lr_multiplier, ReferenceKLController

SPEC = dict(type="warmup_cosine", total_updates=12912, warmup_updates=388, min_lr_ratio=.1)
KL = dict(target=.02, horizon_scenes=1024, min_coefficient=.01, max_coefficient=1.)


def test_schedule_uses_optimizer_update_and_fixed_horizon():
    assert lr_multiplier(0, SPEC) == 1/388
    assert lr_multiplier(387, SPEC) == 1
    assert lr_multiplier(12911, SPEC) == pytest.approx(.1)
    values = [lr_multiplier(i, SPEC) for i in range(12912)]
    assert all(a <= b for a,b in zip(values[:388], values[1:388]))
    assert all(a >= b for a,b in zip(values[387:], values[388:]))
    assert lr_multiplier(5) == 1
    for change in ({"warmup_updates":0}, {"total_updates":388}, {"min_lr_ratio":float('nan')}):
        with pytest.raises(ValueError):lr_multiplier(0, {**SPEC, **change})


def test_feedback_direction_bounds_and_resume():
    controller = ReferenceKLController(.04, KL)
    assert controller.advance(1, .2, 16) == pytest.approx(.04 * (1+.2*16/1024))
    previous = controller.value
    assert controller.advance(2, .02, 16) == previous
    saved = copy.deepcopy(controller.state_dict())
    restored = ReferenceKLController(.04, KL);restored.load_state_dict(saved)
    assert controller.advance(3, 0., 16) == restored.advance(3, 0., 16) < previous
    for i in range(4,2000):controller.advance(i, 0.,16)
    assert controller.value == .01
    for i in range(2000,6000):controller.advance(i, 1.,16)
    assert controller.value == 1.
    with pytest.raises(ValueError):controller.advance(6001, .02,16)
    with pytest.raises(ValueError):controller.advance(6000,float('nan'),16)
    with pytest.raises(ValueError):ReferenceKLController(.04,{**KL,'target':.01}).load_state_dict(saved)


def test_installed_accelerate_wrapper_steps_once_and_exact_resume():
    from accelerate.scheduler import AcceleratedScheduler
    from accelerate import Accelerator
    accelerator = Accelerator(cpu=True, step_scheduler_with_optimizer=False)
    def make():
        p = torch.nn.Parameter(torch.tensor([1.,2.]))
        opt = torch.optim.AdamW([p], lr=1e-6)
        raw = torch.optim.lr_scheduler.LambdaLR(opt,lambda n:lr_multiplier(n,SPEC))
        wrapped = AcceleratedScheduler(raw,opt,step_with_optimizer=False)
        return p,opt,wrapped
    p,opt,scheduler=make(); rates=[]
    for i in range(6):
        rates.append(opt.param_groups[0]['lr'])
        (p.square().sum()).backward();opt.step();opt.zero_grad();scheduler.step()
        assert scheduler.state_dict()['last_epoch']==i+1
        if i==2:
            saved=copy.deepcopy((p.detach(),opt.state_dict(),scheduler.state_dict()))
    q,other,restored=make();q.data.copy_(saved[0]);other.load_state_dict(saved[1]);restored.load_state_dict(saved[2])
    for _ in range(3):
        q.square().sum().backward();other.step();other.zero_grad();restored.step()
    assert torch.equal(p,q)
    assert restored.state_dict()==scheduler.state_dict()
    assert rates==pytest.approx([1e-6*(i+1)/388 for i in range(6)])


def test_real_actor_uses_explicit_beta_without_mutating_config(monkeypatch):
    from tests.flow_grpo.test_denoising_credit import actor_loss
    from starVLA.rl.flow_grpo.model import FlowGRPOActor
    original_forward = FlowGRPOActor.forward
    coefficient = [.04]
    def forward(actor, mode, **kwargs):
        saved = copy.deepcopy(actor.rl_config)
        result = original_forward(actor, mode, **kwargs, reference_coefficient=coefficient[0])
        assert actor.rl_config == saved
        return result
    monkeypatch.setattr(FlowGRPOActor, 'forward', forward)
    p, first = actor_loss(monkeypatch, 1.)
    coefficient[0] = .08
    q, second = actor_loss(monkeypatch, 1.)
    torch.testing.assert_close(second['loss']-first['loss'], .04*first['reference'])
    assert torch.equal(first['sft'], second['sft'])
    torch.testing.assert_close(torch.autograd.grad(second['loss'],q.weight)[0]
                               -torch.autograd.grad(first['loss'],p.weight)[0],
                               torch.tensor(.04*.7,dtype=torch.float64))

from pathlib import Path
from types import SimpleNamespace
import subprocess
import torch
import pytest
from torch import nn
from starVLA.model.modules.foresight.future_spatiotemporal_head import FutureSpatiotemporalHead
from starVLA.model.modules.foresight.interaction_latent_head import InteractionLatentHead
from starVLA.model.modules.foresight.planner_residual import PlannerResidualW
from starVLA.model.modules.foresight.functional_interaction import frozen_teacher_functional_loss
from starVLA.model.modules.trajectory_mae.model import TrajectoryMAE
from starVLA.model.framework.DDPForesight import DDPForesight
from starVLA.model.modules.foresight.losses import interaction_loss
from starVLA.model.modules.vehicle_joint.initialization import initialization_seed

SPANS=[(.5,1.),(1.5,2.),(2.5,3.),(3.5,4.)]


def head(**kw):
    with initialization_seed(100):
        return FutureSpatiotemporalHead(16,12,SPANS,dim=32,action_condition='gt_ego',**kw)


def test_legacy_head_exact_against_frozen_original_source():
    source=subprocess.check_output(['git','show','1493deda247107efe076b9294863a6753391298a:starVLA/model/modules/foresight/future_spatiotemporal_head.py'],text=True)
    namespace={'__name__':'starVLA.model.modules.foresight._legacy_regression', '__package__':'starVLA.model.modules.foresight'}
    exec(compile(source,'frozen_1493ded_head','exec'),namespace)
    with initialization_seed(100):
        old=namespace['FutureSpatiotemporalHead'](16,12,SPANS,dim=32,action_condition='gt_ego')
    new=head();w=torch.randn(2,6,16);a=torch.randn(2,8,4)
    assert old.state_dict().keys()==new.state_dict().keys()
    for k in old.state_dict():assert torch.equal(old.state_dict()[k],new.state_dict()[k])
    assert torch.equal(old(w,(2,3),gt_action=a),new(w,(2,3),gt_action=a))


def test_zero_query_scale_exact_and_no_rng_or_parameter_change():
    old=head();zero=head(action_injection='memory_and_query',action_query_scale=0.)
    for k in old.state_dict():assert torch.equal(old.state_dict()[k],zero.state_dict()[k])
    w=torch.randn(2,6,16);a=torch.randn(2,8,4)
    assert torch.equal(old(w,(2,3),gt_action=a),zero(w,(2,3),gt_action=a))


def test_physical_time_mapping_averages_encoded_vectors():
    model=head(action_injection='memory_and_query')
    tokens=torch.arange(8.).reshape(1,8,1).expand(1,8,32)
    assert torch.equal(model.aligned_action_tokens(tokens)[0,:,0],torch.tensor([.5,2.5,4.5,6.5]))
    model.time_intervals_s.copy_(torch.tensor([(.6,1.),(1.5,2.),(2.5,3.),(3.5,4.)]))
    with pytest.raises(ValueError,match='endpoints'):model.aligned_action_tokens(tokens)
    single=FutureSpatiotemporalHead(16,12,[(t,t) for t in (.5,1,1.5,2,2.5,3,3.5,4)],dim=32,action_condition='gt_ego')
    assert torch.equal(single.aligned_action_tokens(tokens),tokens)


def test_query_action_changes_auxiliary_not_current_or_execution_gradients():
    model=head(action_injection='memory_and_query');execution=nn.Linear(16,4)
    w=torch.randn(2,6,16,requires_grad=True);a=torch.randn(2,8,4,requires_grad=True)
    before=w.detach().clone();pred=model(w,(2,3),gt_action=a)
    assert not torch.equal(pred,model(w,(2,3),gt_action=a+1))
    assert torch.equal(before,w.detach())
    pred.square().mean().backward()
    assert w.grad.abs().sum()>0 and model.action_encoder.point[0].weight.grad.abs().sum()>0
    assert a.grad is None and all(p.grad is None for p in execution.parameters())


def test_interaction_action_source_uses_time_readout_and_has_no_direct_final_W_edge():
    model=DDPForesight.__new__(DDPForesight);nn.Module.__init__(model)
    model.interaction_head=InteractionLatentHead(16,dim=32)
    model.foresight_config=SimpleNamespace(interaction_readout_source='action')
    w=torch.randn(2,144,16,requires_grad=True);a=torch.randn(2,8,16,requires_grad=True)
    encoded={'W':w,'action_queries':a}
    out=model.predict_interaction(encoded);assert torch.equal(out,model.interaction_head(a))
    loss,_=interaction_loss(out,torch.randn_like(out),torch.ones(2,dtype=torch.bool))
    ga,gw=torch.autograd.grad(loss,(a,w),allow_unused=True)
    assert ga.abs().sum()>0 and gw is None
    model.foresight_config.interaction_readout_source='world'
    assert torch.equal(model.predict_interaction(encoded),model.interaction_head(w))


def test_residual_starts_exact_gate_updates_then_branch_updates_and_is_not_stripped():
    residual=PlannerResidualW(16,dim=32);a=torch.randn(2,8,16);w=torch.randn(2,144,16)
    assert torch.equal(residual(a,w),a)
    residual(a,w).square().mean().backward()
    assert residual.alpha.grad.abs()>0
    assert all(p.grad is None or not p.grad.any() for n,p in residual.named_parameters() if n!='alpha')
    with torch.no_grad():residual.alpha.add_(.1)
    residual.zero_grad();residual(a,w).square().mean().backward()
    assert residual.output.weight.grad.abs().sum()>0
    model=DDPForesight.__new__(DDPForesight);nn.Module.__init__(model)
    model.planner_residual=residual;model.interaction_head=nn.Linear(1,1)
    model.strip_auxiliary_heads();assert hasattr(model,'planner_residual') and not hasattr(model,'interaction_head')


def test_frozen_teacher_reconstruct_keeps_input_gradient():
    teacher=TrajectoryMAE(dim=32,layers=1).eval().requires_grad_(False)
    z=torch.randn(2,8,32,requires_grad=True);truth=torch.randn(2,8,2);valid=torch.ones(2,8,dtype=torch.bool)
    loss=frozen_teacher_functional_loss(teacher,z,torch.zeros(2,2),truth,valid)
    loss.backward();assert z.grad.abs().sum()>0 and all(p.grad is None for p in teacher.parameters())

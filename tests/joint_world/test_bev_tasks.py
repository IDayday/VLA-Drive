import torch
from starVLA.model.modules.joint_world.bev_tasks import (
    TaskBEVEncoder, BEVGraphFusion, InteractionGeometryHead, raster_targets,
    bev_loss_sums, interaction_loss_sum)
from starVLA.model.modules.structured_world.contracts import WorldTargets


def target():
    return WorldTargets(current_boxes=torch.tensor([[10., 0., 0., 4., 2., 1., 0., 1.]]),
                        current_classes=torch.zeros(1,dtype=torch.long), track_ids=('car',),
                        future_xy_in_ego_t0=torch.tensor([[[11.,0.],[12.,0.],[13.,0.]]]),
                        future_valid_mask=torch.tensor([[True,True,False]]),
                        current_supervision_mask=torch.tensor([True]), annotation_valid_mask=torch.tensor(True),
                        box_valid_mask=torch.ones(1,8,dtype=torch.bool),supervision_bounds=torch.tensor([1.,-20.,50.,20.]))


def test_raster_is_geometry_track_and_missing_future_safe():
    t=target();xy=torch.tensor([[10.,0.,0.],[20.,0.,0.],[10.,10.,0.]])
    occupied,covered,motion,valid=raster_targets(t,xy,torch.tensor([True,True,False]),3)
    assert occupied.tolist()==[True,False,False] and covered.tolist()==[True,True,False]
    torch.testing.assert_close(motion[0],torch.tensor([[1.,0.],[2.,0.],[0.,0.]]))
    assert valid[0].tolist()==[True,True,False]
    t.annotation_valid_mask=torch.tensor(False)
    assert not raster_targets(t,xy,torch.ones(3,dtype=torch.bool),3)[1].any()


def test_bev_tasks_empty_support_and_actual_fusion_gradient():
    torch.manual_seed(4);encoder=TaskBEVEncoder(8,16,(2,2),3)
    f=torch.randn(1,4,8);xyz=torch.tensor([[[10.,0.,0.],[20.,0.,0.],[30.,0.,0.],[40.,0.,0.]]]);support=torch.ones(1,4,dtype=torch.bool)
    pred=encoder(f,xyz,support);s,c=bev_loss_sums(pred,[target()],xyz,support)
    sum(s[k]/max(c[k],1) for k in s).backward()
    assert encoder.spatial[0].weight.grad.norm()>0
    fusion=BEVGraphFusion(12,16,4);actors=torch.randn(1,3,12);centres=torch.randn(1,3,2)
    out,h=fusion(actors,centres,pred['memory'].detach(),torch.zeros_like(support))
    torch.testing.assert_close(out,actors,rtol=0,atol=0)
    out.square().sum().backward();assert fusion.gate.grad.abs()>0
    assert torch.isfinite(h).all()


def test_pair_geometry_symmetric_and_missing_pairs_no_nan():
    head=InteractionGeometryHead(8,16);features=torch.randn(1,3,8,requires_grad=True)
    pred=head(features);torch.testing.assert_close(pred,pred.transpose(1,2))
    xy=torch.tensor([[[[0.,0.],[0.,0.]],[[2.,0.],[1.,0.]],[[float('nan'),0.],[float('nan'),0.]]]])
    valid=torch.tensor([[[True,True],[True,True],[False,False]]])
    s,c=interaction_loss_sum(pred,xy,valid);assert c==1 and torch.isfinite(s)
    s.backward();assert features.grad.norm()>0
    s,c=interaction_loss_sum(pred,xy,torch.zeros_like(valid));assert c==0 and float(s)==0

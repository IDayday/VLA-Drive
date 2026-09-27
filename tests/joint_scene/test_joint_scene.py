from dataclasses import replace
import copy
import io
import torch
from starVLA.model.modules.joint_scene.contracts import stack_graphs
from starVLA.model.modules.joint_scene.graphs import GraphConfig,build_supervision_graph,build_inference_graph
from starVLA.model.modules.joint_scene.flow import JointSceneFlow,flow_loss_sums,normalized_loss
from starVLA.model.modules.joint_scene.masks import role_completion_mask


def graph(neighbors=2):
    boxes=torch.tensor([[8.,1.,0.,4.,2.,1.5,0.,1.],[20.,-2.,0.,4.,2.,1.5,0.,1.]])[:neighbors]
    return build_supervision_graph(boxes,torch.zeros(neighbors,dtype=torch.long),torch.ones(neighbors,dtype=torch.bool),torch.ones(neighbors),torch.tensor([4.,4.,0.,0.,1.,0.,0.]),GraphConfig(max_neighbors=2,primary_neighbors=1,max_context=3))


def model():
    torch.manual_seed(4)
    return JointSceneFlow(dim=32,heads=4,layers=2,steps=3,condition_dim=12)


def labels(g):
    y=torch.zeros(1,3,3,4);y[...,:2]=g.boxes[:,:,:2,None].transpose(-1,-2);y[:,0,:,0]=torch.tensor([2.,4.,6.]);y[...,3]=1
    valid=torch.ones_like(y,dtype=torch.bool)&g.active_actor_mask[:,:,None,None];valid[:,1:,:,2:]=False
    return y,valid


def test_graph_selection_uses_current_only_and_preserves_ego_only():
    g=graph();assert g.source_indices.tolist()==[[-2,0,1]]
    assert g.edge_mask[0,1,2] and g.edge_mask[0,2,1]
    empty=graph(0);assert empty.active_actor_mask.tolist()==[[True,False,False]]
    assert empty.origin=='annotated_supervision'


def test_future_label_validity_cannot_change_graph_and_low_support_kept():
    g=graph();before=copy.deepcopy(g)
    y,v=labels(g);y[:,1:]=float('nan');v[:,1:]=False
    assert torch.equal(g.boxes,before.boxes) and torch.equal(g.edge_features,before.edge_features)
    boxes=g.boxes[0,1:];cfg=GraphConfig(max_neighbors=2,primary_neighbors=1,max_context=3)
    low=build_supervision_graph(boxes,g.classes[0,1:],torch.ones(2,dtype=torch.bool),torch.zeros(2),g.ego_state[0],cfg)
    assert low.active_actor_mask.sum()==3 and low.context_mask.sum()==2


def test_balanced_valid_role_targets_and_ego_only_fallback():
    g=stack_graphs([graph(),graph(),graph(0),graph()]);_,v=labels(graph());v=v.repeat(4,1,1,1)&g.active_actor_mask[:,:,None,None]
    v[0,1,:2]=False # partial trajectories remain eligible
    hidden,stats=role_completion_mask(v,g.active_actor_mask,torch.Generator().manual_seed(1),torch.tensor([True,False,True,False]))
    assert stats['chosen_actor'][0]>0 and stats['ego_only_fallback'].tolist()==[False,False,True,False]
    assert hidden.sum()==4 and (stats['valid_hidden_coordinates']>0).all()


def test_hidden_poison_invariant_and_every_integration_step_clamped():
    m=model().eval();g=graph();y,v=labels(g);known=v.clone();known[:,0]=False;known[:,1,1]=False
    noise=torch.randn_like(y);poison=torch.where(known,y,torch.full_like(y,float('nan')))
    first,path=m.sample_conditional(noise,g,known=y,known_mask=known,sampling_steps=3,return_path=True)
    second=m.sample_conditional(noise,g,known=poison,known_mask=known,sampling_steps=3)
    assert torch.equal(first,second) and torch.isfinite(second).all()
    assert all(torch.equal(p[known],y[known]) for p in path)


def test_single_executed_ego_and_teacher_inference_boundary():
    import pytest
    m=model();g=graph();noise=torch.randn(1,3,3,4)
    with pytest.raises(ValueError,match='supervision'):m.predict_action(noise,g,sampling_steps=2)
    predicted=replace(g,origin='predicted_inference')
    result=m.predict_action(noise,predicted,sampling_steps=2)
    assert torch.equal(result['executed_ego_xyyaw'][...,:2],result['joint_trajectories'][:,0,:,:2])
    assert result['ego_xy_sincos'].data_ptr()==result['joint_trajectories'][:,0].data_ptr()
    with pytest.raises(ValueError,match='current'):m.sample(noise,predicted,future_valid=torch.ones_like(noise,dtype=torch.bool))


def test_ego_loss_reaches_actor_interaction_and_visual_conditions():
    m=model();g=graph();y,v=labels(g);v[:,1:]=False
    actor=torch.randn(1,3,12,requires_grad=True);scene=torch.randn(1,4,12,requires_grad=True)
    sums,counts=flow_loss_sums(m,y,v,g.active_actor_mask,torch.randn_like(y),torch.tensor([.4]),g,actor_features=actor,scene_features=scene)
    normalized_loss(sums,counts).backward()
    assert m.blocks[0].actor.in_proj_weight.grad.abs().sum()>0
    assert m.blocks[0].relative_bias[0].weight.grad.abs().sum()>0
    assert m.blocks[0].edge_bias.weight.grad.abs().sum()>0
    assert actor.grad.abs().sum()>0 and scene.grad.abs().sum()>0
    assert m.condition[1].weight.grad.abs().sum()>0


def test_missing_yaw_and_all_invalid_neighbors_no_nan():
    m=model();g=graph(0);y,v=labels(g);y[~v]=float('nan')
    sums,counts=flow_loss_sums(m,y,v,g.active_actor_mask,torch.randn_like(y),torch.tensor([.4]),g)
    loss=normalized_loss(sums,counts);loss.backward()
    assert torch.isfinite(loss) and counts['neighbor_xy']==counts['neighbor_yaw']==0


def test_inference_graph_has_no_assignment_acceptance_gate():
    cfg=GraphConfig(max_neighbors=2,primary_neighbors=1,max_context=3)
    g=graph();prediction={'boxes':g.boxes[0,1:].clone(),'logits':torch.tensor([[5.,0.,0.,0.,0.,0.,0.,-2.],[0.,5.,0.,0.,0.,0.,0.,-2.]])}
    result=build_inference_graph(prediction,torch.ones(2),g.ego_state[0],cfg)
    assert result.origin=='predicted_inference' and result.active_actor_mask.sum()==3


def test_save_load_exact_joint_sample_and_optimizer_state():
    m=model();g=graph();y,v=labels(g);optimizer=torch.optim.AdamW(m.parameters(),lr=1e-3)
    noise=torch.randn_like(y)
    loss=normalized_loss(*flow_loss_sums(m,y,v,g.active_actor_mask,noise,torch.tensor([.4]),g));loss.backward();optimizer.step()
    expected=m.sample(noise,g,sampling_steps=2)
    file=io.BytesIO();torch.save({'model':m.state_dict(),'optimizer':optimizer.state_dict()},file);file.seek(0)
    state=torch.load(file,weights_only=True);restored=model();restored.load_state_dict(state['model'],strict=True)
    opt=torch.optim.AdamW(restored.parameters(),lr=1e-3);opt.load_state_dict(state['optimizer'])
    assert torch.equal(expected,restored.sample(noise,g,sampling_steps=2))
    assert len(opt.state)==len(optimizer.state)>0

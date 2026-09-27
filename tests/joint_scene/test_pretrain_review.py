import copy
from dataclasses import replace
import types
import pytest
import torch
from starVLA.model.modules.joint_scene.contracts import stack_graphs
from starVLA.model.modules.joint_scene.graphs import GraphConfig,build_supervision_graph
from starVLA.model.modules.joint_scene.masks import RoleScheduler,role_completion_mask
from starVLA.model.modules.joint_scene.flow import JointSceneFlow,flow_loss_sums,normalized_loss
from test_joint_scene import graph,model,labels


def test_unmodeled_yaw_noise_and_velocity_cannot_change_xy_or_ego():
    m=model();g=graph();noise=torch.randn(1,3,3,4);first=m.sample(noise,g,sampling_steps=3)
    poison=noise.clone();poison[:,1:,:,2:]=float('nan')
    second=m.sample(poison,g,sampling_steps=3)
    assert torch.equal(first,second) and torch.count_nonzero(second[:,1:,:,2:])==0
    altered=copy.deepcopy(m);original=altered.forward
    def override(self,*a,**kw):
        v=original(*a,**kw).clone();v[:,1:,:,2:]=10000.;return v
    altered.forward=types.MethodType(override,altered)
    assert torch.equal(first,altered.sample(noise,g,sampling_steps=3))
    velocity=m(noise,torch.tensor([.3]),g);assert velocity[:,1:,:,2:].count_nonzero()==0


def test_ego_yaw_modeled_and_partial_neighbor_xy_still_generated():
    m=model();g=graph();y,v=labels(g);v[:,1,1,:2]=False
    sums,counts=flow_loss_sums(m,y,v,g.active_actor_mask,torch.randn_like(y),torch.tensor([.4]),g)
    sums['ego_yaw'].backward();assert m.velocity[-1].weight.grad[2:].abs().sum()>0
    assert g.modeled_state_mask[0,1,:2].all() and not g.modeled_state_mask[0,1,2:].any()
    output=m.sample(torch.randn_like(y),g,sampling_steps=2);assert output[0,1,1,:2].abs().sum()>0
    known=torch.zeros_like(v);known[:,1,:,3]=True
    with pytest.raises(ValueError,match='modeled'):m.sample_conditional(torch.randn_like(y),g,y,known)


@pytest.mark.parametrize('poison',[float('nan'),1e30])
def test_all_padding_is_clean_before_projection_and_backward(poison):
    g=graph(1);m=model();dirty=copy.deepcopy(g)
    dirty.boxes[~dirty.active_actor_mask]=poison;dirty.classes[~dirty.active_actor_mask]=-100
    dirty.geometric_support[~dirty.active_actor_mask]=poison
    dirty.context_boxes[~dirty.context_mask]=poison;dirty.context_classes[~dirty.context_mask]=1000
    dirty.context_geometric_support[~dirty.context_mask]=poison;dirty.edge_features[~dirty.edge_mask]=poison
    noise=torch.randn(1,3,3,4);actor=torch.randn(1,3,12);scene=torch.randn(1,3,12);sm=torch.tensor([[True,False,True]])
    ap=actor.clone();ap[~g.active_actor_mask]=poison;sp=scene.clone();sp[~sm]=poison
    outputs=[];grads=[]
    for gg,aa,ss in [(g,actor,scene),(dirty,ap,sp)]:
        m.zero_grad(set_to_none=True);out=m(noise,torch.tensor([.5]),gg,actor_features=aa,scene_features=ss,scene_mask=sm)
        out.square().sum().backward();outputs.append(out.detach());grads.append({n:p.grad.clone() for n,p in m.named_parameters() if p.grad is not None})
        assert all(torch.isfinite(x).all() for x in grads[-1].values())
    assert torch.equal(outputs[0],outputs[1])
    assert all(torch.equal(grads[0][n],grads[1][n]) for n in grads[0])
    assert torch.isfinite(m.sample(noise,dirty,sampling_steps=2)).all()


@pytest.mark.parametrize('field', ['box','class','size','context','context_class','ego','edge','self_edge','mask_dtype'])
def test_invalid_valid_current_inputs_rejected(field):
    g=graph()
    if field=='box':g.boxes[0,1,0]=float('nan')
    if field=='class':g.classes[0,1]=100
    if field=='size':g.boxes[0,1,3]=-1
    if field=='context':g.context_boxes[0,0,0]=float('nan')
    if field=='context_class':g.context_classes[0,0]=-1
    if field=='ego':g.ego_state[0,0]=float('nan')
    if field=='edge':g.edge_features[0,0,1,0]=float('nan')
    if field=='self_edge':g.edge_mask[0,1,1]=False
    if field=='mask_dtype':g.context_mask=g.context_mask.long()
    with pytest.raises(ValueError):g.validate()


def test_invalid_valid_visual_features_are_not_nan_to_num():
    m=model();g=graph();n=torch.randn(1,3,3,4);scene=torch.randn(1,2,12);scene[0,0,0]=float('nan')
    with pytest.raises(ValueError,match='scene'):m(n,torch.tensor([.1]),g,scene_features=scene)
    with pytest.raises(ValueError,match='width'):m(n,torch.tensor([.1]),g,scene_features=torch.zeros(1,2,13))


def custom_graph(xy,classes=None,support=None,nav=1,**cfg):
    boxes=torch.tensor([[x,y,0.,4.,2.,1.5,0.,1.] for x,y in xy],dtype=torch.float32)
    n=len(boxes);state=torch.tensor([5.,5.,0.,0.,0.,0.,0.]);state[3+nav]=1
    return build_supervision_graph(boxes,torch.tensor(classes or [0]*n),torch.ones(n,dtype=torch.bool),torch.tensor(support or [1.]*n),state,GraphConfig(**cfg))


def test_static_cone_not_role_target_and_zero_support_context():
    g=custom_graph([(2.,0.),(14.,0.),(6.,1.)],classes=[3,0,0],support=[1.,1.,0.],max_neighbors=1,primary_neighbors=1,max_context=3)
    assert g.source_indices[0,1]==1 and g.context_mask.sum()==3
    records=g.selection_metadata[0]['objects']
    assert records[0]['reason']=='static_context_only' and records[2]['reason']=='no_geometric_support'
    assert records[2]['context_retained']


def test_navigation_changes_relevant_candidates_and_far_corridor_can_win():
    xy=[(20.,0.),(10.,16.),(2.,-10.)]
    straight=custom_graph(xy,nav=1,max_neighbors=1,primary_neighbors=1)
    left=custom_graph(xy,nav=0,max_neighbors=1,primary_neighbors=1)
    assert int(straight.source_indices[0,1])==0 and int(left.source_indices[0,1])==1
    nearest=custom_graph(xy,variant='nearest',max_neighbors=1,primary_neighbors=1)
    assert nearest.source_indices[0,1]==2
    # No object velocity is assumed; a parked car in the forward corridor remains eligible.
    parked=custom_graph([(12.,0.)]);assert parked.active_actor_mask.sum()==2


def test_dedup_capacity_and_context_overflow_are_auditable():
    g=custom_graph([(8.,0.),(8.1,.1),(15.,0.),(25.,0.)],max_neighbors=1,primary_neighbors=1,max_context=1)
    meta=g.selection_metadata[0]
    assert sum(r['reason']=='duplicate' for r in meta['objects'])==1
    assert meta['context_overflow']==2 and sum(r['context_overflow'] for r in meta['objects'])==2
    assert meta['selected_neighbors']==1


def test_role_scheduler_batch1_odd_tail_and_exact_sequence_restore():
    g=graph();_,v=labels(g);scheduler=RoleScheduler(12);chosen=[]
    for _ in range(7):chosen.append(int(role_completion_mask(v,g.active_actor_mask,scheduler)[1]['chosen_actor'][0]))
    assert sum(x>0 for x in chosen)==3
    saved=scheduler.state_dict();restored=RoleScheduler(0);restored.load_state_dict(saved)
    for batch in [3,1,5]:
        gg=stack_graphs([g]*batch);vv=v.repeat(batch,1,1,1)
        first=role_completion_mask(vv,gg.active_actor_mask,scheduler)[1];second=role_completion_mask(vv,gg.active_actor_mask,restored)[1]
        assert all(torch.equal(first[k],second[k]) for k in first)
    assert scheduler.position==restored.position==16


def test_partial_xy_coordinate_is_not_a_complete_role_position():
    g=graph();_,v=labels(g);v[:,1:]=False;v[:,1,:,0]=True
    _,stats=role_completion_mask(v,g.active_actor_mask,RoleScheduler(2),torch.tensor([True]))
    assert stats['ego_only_fallback'].all() and stats['chosen_actor'][0]==0
    v[:,2,1,:2]=True
    _,stats=role_completion_mask(v,g.active_actor_mask,RoleScheduler(2),torch.tensor([True]))
    assert stats['chosen_actor'][0]==2 and not stats['ego_only_fallback'].any()


def test_future_valid_cannot_modify_modeled_policy():
    g=graph();g.modeled_state_mask[0,1,0]=False
    with pytest.raises(ValueError,match='role'):g.validate()

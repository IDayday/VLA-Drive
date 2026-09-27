from dataclasses import fields
import pytest
import torch
from starVLA.model.modules.joint_world.flow import JointTrajectoryFlow,training_loss_sums
from starVLA.model.modules.joint_world.local_graph import LocalInteractionGraph
from starVLA.model.modules.joint_world.local_masks import task_masks,stable_noise
from starVLA.model.modules.structured_world.action_adapter import WorldToActionAdapter


def graph(n=4,active_n=3,chain=True):
    source=torch.arange(-1,n-1)[None];active=torch.arange(n)[None]<active_n;source[~active]=-2
    edge=torch.diag_embed(active)
    if chain:
        for i in range(active_n-1):edge[:,i,i+1]=edge[:,i+1,i]=True
    return LocalInteractionGraph(source,active,active.clone(),active.clone(),edge,torch.zeros(1,n,n,8),
        active.float(),active.float(),torch.full((1,4),-2),torch.zeros(1,4,dtype=torch.bool),[[]],[[]],[[]],[[]],[{}]).validate()


def setup(n=4,active_n=3,layers=2):
    torch.manual_seed(11);model=JointTrajectoryFlow(16,dim=16,heads=2,layers=layers,steps=3,edge_feature_dim=8,trajectory_mode='current_residual').eval()
    g=graph(n,active_n)
    current=dict(actor_features=torch.randn(1,n,16),context=torch.randn(1,5,16),
                 current_xy=torch.randn(1,n,2),existence=torch.rand(1,n),local_graph=g)
    noise=stable_noise(['scene'],g.source_slot_ids,3,3)
    return model,g,current,noise


def reindex(g,ix):
    values=g.tensor_state().copy()
    for key in ('source_slot_ids','active_actor_mask','predictable_actor_mask','trajectory_condition_mask','visual_reliability','relevance_scores'):
        values[key]=values[key][:,ix]
    values['edge_mask']=g.edge_mask[:,ix][:,:,ix];values['edge_features']=g.edge_features[:,ix][:,:,ix]
    return LocalInteractionGraph(**values).validate()


@pytest.mark.parametrize('poison',[float('nan'),1e30])
def test_inactive_poison_does_not_affect_graph_or_adapter(poison):
    model,g,current,noise=setup();reference=model.sample(noise,**current,sampling_steps=2)
    changed=dict(current)
    for k in ('actor_features','current_xy','existence'):
        changed[k]=current[k].clone();changed[k][:,-1]=poison
    bad_noise=noise.clone();bad_noise[:,-1]=poison
    prediction,feature=model.sample(bad_noise,**changed,sampling_steps=2)
    assert torch.equal(prediction,reference[0]);assert torch.equal(feature,reference[1])
    adapter=WorldToActionAdapter(16,heads=2);adapter.gate.data.fill_(.4)
    action=torch.randn(1,3,16);baseline=adapter(action,feature,g.active_actor_mask)
    feature=feature.clone();feature[:,-1]=poison
    assert torch.equal(baseline,adapter(action,feature,g.active_actor_mask))
    prediction.sum().backward();assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


def test_edges_block_direct_message_but_allow_two_hops():
    model,g,current,noise=setup(layers=1);other=dict(current);other['actor_features']=current['actor_features'].clone();other['actor_features'][:,2]+=10
    a=model(noise,torch.ones(1),**current)[1];b=model(noise,torch.ones(1),**other)[1]
    torch.testing.assert_close(a[:,0],b[:,0],atol=0,rtol=0)
    model,g,current,noise=setup(layers=2);other=dict(current);other['actor_features']=current['actor_features'].clone();other['actor_features'][:,2]+=torch.arange(16)*10
    a=model(noise,torch.ones(1),**current)[1];b=model(noise,torch.ones(1),**other)[1]
    assert not torch.allclose(a[:,0],b[:,0])
    g.edge_mask[:,1,2]=g.edge_mask[:,2,1]=False
    a=model(noise,torch.ones(1),**current)[1];b=model(noise,torch.ones(1),**other)[1]
    torch.testing.assert_close(a[:,0],b[:,0],atol=0,rtol=0)


def test_joint_permutation_padding_and_noise_identity():
    model,g,current,noise=setup(n=5,active_n=3)
    expected=model.sample(noise,**current,sampling_steps=2)
    ix=torch.tensor([0,2,4,1,3]);gp=reindex(g,ix);cp=dict(current,local_graph=gp)
    for k in ('actor_features','current_xy','existence'):cp[k]=current[k][:,ix]
    pn=stable_noise(['scene'],gp.source_slot_ids,3,3)
    assert torch.equal(noise[:,ix],pn)
    actual=model.sample(pn,**cp,sampling_steps=2)
    for x,y in zip(actual,expected):torch.testing.assert_close(x,y[:,ix],atol=2e-5,rtol=2e-5)
    small=torch.tensor([0,1,2]);cg=dict(current,local_graph=reindex(g,small))
    for k in ('actor_features','current_xy','existence'):cg[k]=current[k][:,small]
    actual=model.sample(noise[:,small],**cg,sampling_steps=2)
    for x,y in zip(actual,expected):torch.testing.assert_close(x,y[:,small],atol=2e-5,rtol=2e-5)
    batch=stable_noise(['other','scene'],g.source_slot_ids.expand(2,-1),3,3)
    assert torch.equal(batch[1],noise[0])


def test_conditional_clamping_hidden_nan_and_allhidden_equivalence():
    model,g,current,noise=setup();known=torch.randn_like(noise);mask=torch.zeros(noise.shape[:-1],dtype=torch.bool);mask[:,1]=True
    xy,feature,path=model.sample_conditional(noise,**current,known_xy=known,known_mask=mask,sampling_steps=4,return_path=True)
    for x in path:assert torch.equal(x[mask],known[mask])
    poisoned=known.clone();poisoned[~mask]=float('nan')
    actual=model.sample_conditional(noise,**current,known_xy=poisoned,known_mask=mask,sampling_steps=4)
    assert torch.equal(actual[0],xy);assert torch.equal(actual[1],feature)
    empty=torch.zeros_like(mask)
    a=model.sample_conditional(noise,**current,known_xy=poisoned,known_mask=empty,sampling_steps=2)
    b=model.sample(noise,**current,sampling_steps=2)
    assert all(torch.equal(x,y) for x,y in zip(a,b))


def test_task_distribution_and_ego_only_fallback():
    g=graph(n=3,active_n=2);generator=torch.Generator().manual_seed(19);counts=torch.zeros(3,dtype=torch.int64)
    for _ in range(4000):
        hidden,task=task_masks(g,generator);counts[task['nominal'][0]]+=1
        assert not hidden[~g.predictable_actor_mask].any()
        if task['actual'][0]==2:assert hidden[0,1] and hidden.sum()==1
    assert torch.all((counts/4000-torch.tensor([.5,.25,.25])).abs()<.035)
    g=graph(n=3,active_n=1);fallbacks=0
    for _ in range(100):
        hidden,task=task_masks(g,generator)
        assert hidden[0,0] and hidden.sum()==1
        if task['nominal'][0]==2:assert task['fallback'][0] and task['actual'][0]==0;fallbacks+=1
    assert fallbacks>0


def test_loss_invalid_nan_and_separate_reductions():
    model,g,current,noise=setup();target=torch.randn_like(noise);valid=torch.ones(noise.shape[:-1],dtype=torch.bool)
    valid[:,2:]=False;target[:,2:]=float('nan');hidden=g.active_actor_mask.clone()
    sums,counts=training_loss_sums(model,target,valid,hidden,noise,torch.tensor([.5]),**current)
    assert counts=={'ego':6,'agents':6};loss=sum(sums[k]/counts[k] for k in sums);loss.backward()
    assert torch.isfinite(loss) and all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    valid.zero_();s,c=training_loss_sums(model,target,valid,hidden,noise,torch.tensor([.5]),**current)
    assert c=={'ego':0,'agents':0} and sum(s.values())==0


def test_gate_zero_and_strict_new_graph_restore():
    model,g,current,noise=setup();adapter=WorldToActionAdapter(16,heads=2);action=torch.randn(1,2,16)
    feature=model.sample(noise,**current,sampling_steps=2)[1]
    assert torch.equal(adapter(action,feature,g.active_actor_mask),action)
    clone,_,_,_=setup();clone.load_state_dict(model.state_dict(),strict=True)
    assert torch.equal(clone.sample(noise,**current,sampling_steps=2)[0],model.sample(noise,**current,sampling_steps=2)[0])
    old=JointTrajectoryFlow(16,dim=16,heads=2,layers=2,steps=3)
    with pytest.raises(RuntimeError):clone.load_state_dict(old.state_dict(),strict=True)

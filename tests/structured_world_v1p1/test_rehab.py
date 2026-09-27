import torch
from starVLA.model.modules.structured_world.rehab import ReferenceAgentHeads,reference_loss_sums,reference_support
from test_metrics import example


def test_decode_and_motion_gradient_separation():
    torch.manual_seed(9);head=ReferenceAgentHeads(16,slots=2,classes=2)
    hidden=torch.randn(1,2,16,requires_grad=True);p=head(hidden)
    torch.testing.assert_close(p['future_xy'],p['boxes'][...,:2,None].transpose(-1,-2)+p['future_displacement'])
    _,target=example();s,c,_=reference_loss_sums(p,[target]);s['motion'].backward()
    assert head.motion.weight.grad.abs().sum()>0
    assert head.box.weight.grad is None or head.box.weight.grad.abs().sum()==0
    assert hidden.grad.abs().sum()>0


def test_no_object_support_independent_of_prediction_escape():
    _,t=example();h=ReferenceAgentHeads(16,slots=2,classes=2)
    p=h(torch.randn(1,2,16));t.current_supervision_mask[:]=False
    a=reference_loss_sums(p,[t])[1]
    p['boxes']=p['boxes']+10000
    b=reference_loss_sums(p,[t])[1]
    assert a==b and a['cls']>0
    t.supervision_grid=torch.zeros(49,40,dtype=torch.bool)
    assert not reference_support(h.references,t).any()


def test_invalid_future_and_empty_targets_finite():
    _,t=example();t.future_valid_mask[:]=False;t.future_xy_in_ego_t0[:]=float('nan')
    h=ReferenceAgentHeads(16,slots=2,classes=2);p=h(torch.randn(1,2,16))
    s,c,_=reference_loss_sums(p,[t]);assert c['motion']==0 and all(torch.isfinite(x) for x in s.values())
    t.current_supervision_mask[:]=False
    s,c,_=reference_loss_sums(p,[t]);sum(s.values()).backward();assert all(torch.isfinite(x) for x in s.values())


def test_accumulation_matches_joint_sum_count_gradient():
    import copy
    torch.manual_seed(11);a=ReferenceAgentHeads(16,slots=2,classes=2);b=copy.deepcopy(a)
    _,t1=example();_,t2=example();t2.current_supervision_mask[:]=False
    xs=[torch.randn(1,2,16),torch.randn(1,2,16)];ts=[t1,t2]
    sums=[];counts=[]
    for x,t in zip(xs,ts):
        s,c,_=reference_loss_sums(a(x),[t]);sums.append(s);counts.append(c)
    total={k:sum(c[k] for c in counts) for k in counts[0]}
    sum(sum(s[k] for s in sums)/max(total[k],1.) for k in total).backward()
    for x,t in zip(xs,ts):
        s,c,_=reference_loss_sums(b(x),[t]);sum(s[k]/max(total[k],1.) for k in total).backward()
    for p,q in zip(a.parameters(),b.parameters()):
        if p.grad is not None:torch.testing.assert_close(p.grad,q.grad,atol=2e-6,rtol=2e-5)

import math
import pytest
import torch
from starVLA.model.modules.qwen_world_postcot.flow import clean_x_l1
from starVLA.rl.qwen_world_postcot.coordinate_objective import CoordinateObjective,tolerance_weights


def test_control_matches_value_and_gradient():
    x=torch.randn(3,8,3,requires_grad=True);y=torch.randn_like(x);mask=torch.rand(3,8)>.2
    old=clean_x_l1(x,y,mask);new,stats=CoordinateObjective([1.,1.,1.])(x,y,mask)
    assert torch.equal(old,new)
    assert torch.equal(torch.autograd.grad(old,x)[0],torch.autograd.grad(new,x)[0])
    assert stats['coordinate_diagnostics']['normalized_sum'].shape==(8,3)


def test_physical_tolerance_and_affine_slopes():
    tol=(.5,.25,math.radians(2));w=tolerance_weights(tol)
    assert sum(w)==pytest.approx(3)
    x=torch.zeros(1,8,3,requires_grad=True);y=torch.tensor([tol[0]/33.37,tol[1]/21,tol[2]/1.765])[None,None].expand_as(x)
    loss,_=CoordinateObjective(w)(x,y)
    g=torch.autograd.grad(loss,x)[0]
    assert (y.abs()*torch.tensor(w)).std()<1e-7
    assert (g<0).all()


def test_weighted_mean_global_mask_and_microbatch_equivalence():
    x=torch.randn(5,8,3);y=torch.randn_like(x);mask=torch.rand(5,8)>.15;objective=CoordinateObjective([.8,1.5,.7],.93)
    full,terms=objective(x,y,mask)
    pieces=[objective(x[:2],y[:2],mask[:2])[1],objective(x[2:],y[2:],mask[2:])[1]]
    assert torch.allclose(full,sum(p['numerator'] for p in pieces)/terms['denominator'])
    assert torch.equal(terms['coordinate_diagnostics']['valid_count'].sum(),terms['denominator'])


def test_invalid_and_zero_valid_do_not_invent_errors():
    x=torch.randn(2,8,3,requires_grad=True);y=torch.randn_like(x)
    loss,terms=CoordinateObjective([1,2,3])(x,y,torch.zeros(2,8,dtype=torch.bool))
    assert loss==0 and terms['denominator']==0
    assert torch.equal(torch.autograd.grad(loss,x)[0],torch.zeros_like(x))


@pytest.mark.parametrize('w,s',[([0,1,2],1),([1,float('nan'),1],1),([1,1,1],-1)])
def test_reject_invalid_constants(w,s):
    with pytest.raises(ValueError):CoordinateObjective(w,s)


def test_unbounded_temporary_grant_still_honors_new_withdrawal(tmp_path,monkeypatch):
    import json
    from starVLA.rl.qwen_world_postcot import contracts as c
    record=tmp_path/'grant.json';policy=tmp_path/'policy.json'
    grant=dict(permanent_hosts=['fixed'],temporary_hosts=[],unbounded_temporary_hosts=['temporary'],starts_unix=1000,expires_unix=1000,checkpoint_guard_seconds=20,record=str(record))
    record.write_text(json.dumps(grant));value=dict(task_authorizations={'study':grant},global_dispatch_prohibitions={})
    policy.write_text(json.dumps(value));monkeypatch.setattr(c,'POLICY',policy);monkeypatch.setattr(c,'TASK','study')
    assert c.allowed_hosts(at=2000,admit_long_job=True)==['fixed','temporary']
    assert not c.boundary_stop_due('temporary',now=2000)
    value['global_dispatch_prohibitions']['temporary']=dict(new_task_dispatch_allowed=False,utc='1970-01-01T00:25:00+00:00')
    policy.write_text(json.dumps(value));assert c.allowed_hosts(at=2000)==['fixed']
    assert c.boundary_stop_due('temporary',now=2000)


def test_no_unbounded_field_retains_expired_lease_semantics(tmp_path,monkeypatch):
    import json
    from starVLA.rl.qwen_world_postcot import contracts as c
    record=tmp_path/'grant.json';policy=tmp_path/'policy.json'
    grant=dict(permanent_hosts=['fixed'],temporary_hosts=['temporary'],starts_unix=1000,expires_unix=1100,checkpoint_guard_seconds=20,record=str(record))
    record.write_text(json.dumps(grant));policy.write_text(json.dumps(dict(task_authorizations={'study':grant})))
    monkeypatch.setattr(c,'POLICY',policy);monkeypatch.setattr(c,'TASK','study')
    assert c.allowed_hosts(at=2000)==['fixed']

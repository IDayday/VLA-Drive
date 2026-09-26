import numpy as np
import torch
from starVLA.model.modules.structured_world.targets import make_targets
from starVLA.model.modules.structured_world.losses import world_losses
from starVLA.model.modules.structured_world.agent_heads import AgentHeads
from starVLA.model.modules.structured_world.scene_agent_reader import SceneAgentReader
from starVLA.model.modules.structured_world.action_adapter import WorldToActionAdapter


def frame(boxes, tracks):
    return {'timestamp':0,'ego2global':np.eye(4),'anns':{'gt_boxes':np.asarray(boxes).reshape(-1,7),'gt_names':['vehicle']*len(tracks),'track_tokens':tracks}}


def test_empty_and_missing_future_are_finite_and_differ_from_missing_annotation():
    targets = [make_targets(frame([],[]),[]),make_targets(frame([[5,0,0,4,2,1,0]],['v']),[])]
    head = AgentHeads(16)
    p = head(torch.randn(2,32,16))
    losses, assignments = world_losses(p,targets)
    assert all(torch.isfinite(x) for x in losses.values())
    assert losses['motion'] == 0 and len(assignments[1][0]) == 1
    sum(losses.values()).backward()
    assert head.box.weight.grad is not None
    missing = make_targets({'anns':None,'timestamp':0,'ego2global':np.eye(4)},[])
    losses,_ = world_losses({k:v[:1] for k,v in p.items()},[missing])
    assert sum(losses.values()) == 0


def test_reader_and_zero_gate_gradient():
    reader = SceneAgentReader(16,16,dim=32,scene_tokens=4,agent_tokens=2,layers=1)
    memory = reader(torch.randn(2,5,16),support=torch.zeros(2,5,dtype=torch.bool))
    assert torch.isfinite(memory.agent_memory).all()
    adapter = WorldToActionAdapter(16,heads=4)
    a = torch.randn(2,3,16)
    w = torch.cat([memory.scene_memory,memory.agent_memory],1)
    out = adapter(a,w)
    torch.testing.assert_close(out,a,rtol=0,atol=0)
    out.square().sum().backward()
    assert adapter.gate.grad.abs() > 0
    with torch.no_grad():adapter.gate.fill_(.1)
    reader.zero_grad();adapter.zero_grad()
    memory = reader(torch.randn(2,5,16))
    adapter(a,memory.agent_memory).square().sum().backward()
    assert reader.queries.grad.abs().sum() > 0


def test_unequal_microbatch_accumulation_matches_full_batch():
    from starVLA.model.modules.structured_world.losses import normalize_accumulated_world_losses
    torch.manual_seed(4)
    head=AgentHeads(16)
    x=torch.randn(2,32,16)
    targets=[make_targets(frame([],[]),[]),make_targets(frame([[5,0,0,4,2,1,0],[8,2,0,4,2,1,0]],['v','w']),[])]
    losses,_=world_losses(head(x),targets)
    sum(losses.values()).backward()
    expected=[p.grad.clone() for p in head.parameters()]
    head.zero_grad()
    accum=[]
    for i in range(2):
        sums,_=world_losses(head(x[i:i+1]),targets[i:i+1],return_sums=True)
        accum.append(sums)
    sum(normalize_accumulated_world_losses(accum).values()).backward()
    for param,reference in zip(head.parameters(),expected):
        torch.testing.assert_close(param.grad,reference,atol=1e-7,rtol=1e-5)

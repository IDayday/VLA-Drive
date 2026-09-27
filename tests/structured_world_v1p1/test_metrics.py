import itertools
import numpy as np
import torch
from starVLA.model.modules.structured_world.metrics import match_geometry, diagnostics
from starVLA.model.modules.structured_world.matching import match_current
from starVLA.model.modules.structured_world.contracts import WorldTargets


def example():
    boxes = torch.tensor([[10.,0,0,4,2,1,0,1],[15.,0,0,4,2,1,0,1]])
    target = WorldTargets(boxes, torch.tensor([0,1]), ('a','b'), boxes[:,:2,None].transpose(-1,-2).repeat(1,8,1), torch.ones(2,8,dtype=torch.bool), torch.ones(2,dtype=torch.bool), torch.tensor(True), torch.ones(2,8,dtype=torch.bool), torch.tensor([0.,-20,50,20]))
    pred = {'boxes': boxes.clone(), 'logits': torch.tensor([[-10.,10,-10],[10.,-10,-10]]), 'future_xy': target.future_xy_in_ego_t0.clone()}
    return pred,target


def test_class_swap_reproduces_legacy_failure():
    pred,target = example()
    r,c = match_current(pred,target)
    assert ((pred['boxes'][r,:2]-target.current_boxes[c,:2]).norm(dim=-1)<2).sum()==0
    d = diagnostics(pred,target)
    assert d['legacy_matched_detection_targets']==0
    assert d['filtered_tp']==2 and d['class_filtered_tp']==0


def test_size_yaw_not_in_geometric_matching():
    p,t = example();p['boxes'][:,3:]=1000.
    d=diagnostics(p,t)
    assert d['filtered_tp']==2


def test_empty_duplicates_and_boundary():
    e=torch.empty(0,2);g=torch.tensor([[0.,0.]])
    assert len(match_geometry(e,g)[0])==len(match_geometry(g,e)[0])==0
    assert len(match_geometry(g.repeat(2,1),g)[0])==1
    assert len(match_geometry(torch.tensor([[2.,0.]]),g)[0])==0
    assert len(match_geometry(torch.tensor([[1.99999,0.]]),g)[0])==1


def test_maximum_cardinality_before_distance():
    # Shortest single pair would consume the only match of the second prediction.
    p=torch.tensor([[0.,0.],[1.8,0.]])
    g=torch.tensor([[0.,0.],[-1.8,0.]])
    r,c=match_geometry(p,g)
    assert len(r)==2 and c.tolist()==[1,0]


def test_lexicographic_objective_bruteforce():
    rng=np.random.default_rng(42)
    for _ in range(25):
        p=rng.normal(size=(4,2));g=rng.normal(size=(3,2))
        dist=np.linalg.norm(p[:,None]-g[None],axis=-1)
        choices=[]
        for cols in itertools.product(range(-1,3),repeat=4):
            used=[v for v in cols if v>=0]
            if len(used)!=len(set(used)) or any(dist[i,j]>=2 for i,j in enumerate(cols) if j>=0):continue
            choices.append((-len(used),sum(dist[i,j] for i,j in enumerate(cols) if j>=0)))
        optimal=min(choices)
        r,c=match_geometry(torch.tensor(p),torch.tensor(g))
        assert -len(r)==optimal[0] and abs(dist[r,c].sum()-optimal[1])<1e-10


def test_masks_and_no_motion_zero_fill():
    p,t=example();t.future_valid_mask[:]=False;t.future_xy_in_ego_t0[:]=float('nan')
    d=diagnostics(p,t)
    assert d['filtered_all_motion_points']==0 and d['filtered_all_ade_sum']==0
    # Objectness collapse and spatial exclusion remain separately visible.
    p['logits'][:]=torch.tensor([-10.,-10.,10.])
    d=diagnostics(p,t)
    assert d['raw_k_tp']==2 and d['filtered_tp']==0 and d['support_predictions']==2
    p['boxes'][:,:2]=100
    d=diagnostics(p,t)
    assert d['outside_roi']==2 and d['support_predictions']==0

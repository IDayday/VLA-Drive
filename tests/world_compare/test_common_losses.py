"""Common objective checks independent of expensive pretrained Qwen loading."""
from types import MethodType

import pytest
import torch
from torch import nn

from starVLA.model.modules.qwen_world_compare.heads import (
    object_track_loss, evaluation_loss, FeatureReadout, ObjectTrackReadout,
    EvaluationReadout)
from starVLA.model.modules.qwen_world_compare.model import FourFrameworkVLA
from starVLA.model.modules.qwen_world_compare.independent_actor import ArchitectureA
from starVLA.model.modules.qwen_world_compare.contracts import CurrentContext, LOSS_WEIGHTS
from starVLA.rl.qwen_world_compare.batching import (
    post_positions, scene_generator, sample_valid_future_queries, update_denominators)


@pytest.fixture(autouse=True)
def small_cpu_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(42)
    yield
    torch.set_num_threads(old)


def evaluation_tensors(batch=2, candidates=3):
    pred = dict(factor_logits=torch.randn(batch*candidates, 6, requires_grad=True),
        score_logits=torch.randn(batch*candidates, requires_grad=True),
        event_logits=torch.randn(batch*candidates, 3, 9, requires_grad=True))
    labels = torch.rand(batch,candidates,6)
    valid = torch.ones_like(labels, dtype=torch.bool)
    score = torch.rand(batch,candidates)
    score_valid = torch.ones_like(score,dtype=torch.bool)
    events = torch.full((batch,candidates,3),8,dtype=torch.long)
    event_valid = torch.ones_like(events,dtype=torch.bool)
    return pred,labels,valid,score,score_valid,events,event_valid


def test_all_invalid_evaluation_nan_markers_produce_finite_zero_gradients():
    pred, labels, valid, score, score_valid, events, event_valid = evaluation_tensors()
    labels.fill_(float('nan'));valid.zero_();score.fill_(float('nan'));score_valid.zero_()
    events.fill_(-1);event_valid.zero_()
    result = evaluation_loss(pred,labels,valid,score,score_valid,events,event_valid)
    assert result['loss'].item() == 0
    result['loss'].backward()
    for value in pred.values():
        assert value.grad is not None and torch.isfinite(value.grad).all()
        assert value.grad.abs().sum() == 0


def test_nc_half_soft_label_and_invalid_other_labels_preserved():
    pred,labels,valid,score,score_valid,events,event_valid = evaluation_tensors(1,1)
    pred = {name: torch.zeros_like(value,requires_grad=True) for name,value in pred.items()}
    labels.fill_(float('nan'));labels[0,0,0]=.5;valid.zero_();valid[0,0,0]=True
    score.fill_(float('nan'));score_valid.zero_();events.fill_(-1);event_valid.zero_()
    result = evaluation_loss(pred,labels,valid,score,score_valid,events,event_valid)
    torch.testing.assert_close(result['factor'],torch.log(torch.tensor(2.)))
    result['loss'].backward()
    assert torch.isfinite(pred['factor_logits'].grad).all()
    assert pred['factor_logits'].grad.abs().sum()==0


def test_pair_ranking_threshold_and_global_score_offset():
    pred,labels,valid,score,score_valid,events,event_valid=evaluation_tensors(1,3)
    labels.zero_();valid.zero_();events.fill_(8);event_valid.zero_()
    score[:]=torch.tensor([[.20,.219,.50]])
    with torch.no_grad():pred['score_logits'][:]=torch.tensor([.1,.3,.7])
    actual=evaluation_loss(pred,labels,valid,score,score_valid,events,event_valid)['rank']
    expected=(torch.nn.functional.softplus(torch.tensor(-.6))+
              torch.nn.functional.softplus(torch.tensor(-.4)))/2
    torch.testing.assert_close(actual,expected)
    other=dict(pred);other['score_logits']=pred['score_logits']+13
    shifted=evaluation_loss(other,labels,valid,score,score_valid,events,event_valid)['rank']
    torch.testing.assert_close(actual,shifted,rtol=1e-6,atol=1e-6)


def object_tensors():
    target_boxes=torch.tensor([[[10.,0.,0.,4.,2.,1.5,0.,1.],
                                 [20.,0.,0.,4.,2.,1.5,0.,1.]]])
    future=torch.arange(1*2*4*6,dtype=torch.float32).reshape(1,2,4,6)/10
    predicted_boxes=torch.cat((target_boxes.flip(1),torch.zeros(1,1,8)),1).requires_grad_()
    predicted_future=torch.cat((future.flip(1),torch.randn(1,1,4,6)),1).requires_grad_()
    pred=dict(boxes=predicted_boxes,class_logits=torch.zeros(1,3,7,requires_grad=True),
              existence_logits=torch.zeros(1,3,requires_grad=True),future_states=predicted_future)
    target=dict(boxes=target_boxes,classes=torch.zeros(1,2,dtype=torch.long),
        valid=torch.ones(1,2,dtype=torch.bool),coverage_valid=torch.tensor([False]),
        future_states=future,future_valid=torch.ones(1,2,4,dtype=torch.bool))
    return pred,target


def test_current_hungarian_assignment_is_reused_for_all_future_times():
    pred,target=object_tensors();result=object_track_loss(pred,target)
    pi,gi=result['assignments'][0]
    assert pi.tolist()==[0,1] and gi.tolist()==[1,0]
    assert result['future_track'].item()==0
    result['object'].backward(retain_graph=True)
    assert pred['existence_logits'].grad[0,2]==0  # unknown coverage: not negative
    assert pred['class_logits'].grad[0,2].abs().sum()==0
    target['future_states']=target['future_states'].flip(1)
    assert object_track_loss(pred,target)['future_track']>0


def test_all_invalid_future_markers_cannot_poison_object_gradients():
    pred,target=object_tensors();target['future_states'].fill_(float('nan'));target['future_valid'].zero_()
    result=object_track_loss(pred,target);assert result['future_track'].item()==0
    (result['object']+result['future_track']).backward()
    for value in pred.values():
        if value.grad is not None:assert torch.isfinite(value.grad).all()


def test_every_common_loss_is_fp32_under_bf16_forward():
    pred,target=object_tensors()
    with torch.autocast('cpu',dtype=torch.bfloat16):
        obj=object_track_loss(pred,target)
        ep=evaluation_loss(*evaluation_tensors())
    for name,value in obj.items():
        if name!='assignments':assert value.dtype==torch.float32
    assert all(value.dtype==torch.float32 for value in ep.values())


def test_future_track_fp32_targets_are_not_rounded_to_prediction_storage():
    # The prediction may come from BF16 linear layers; the FP32 target must not
    # first be copied into a zeros_like(BF16_prediction) assignment buffer.
    boxes=torch.tensor([[[10.,0.,0.,4.,2.,1.5,0.,1.]]])
    target=dict(boxes=boxes,classes=torch.zeros(1,1,dtype=torch.long),
        valid=torch.ones(1,1,dtype=torch.bool),coverage_valid=torch.tensor([False]),
        future_states=torch.zeros(1,1,4,6),future_valid=torch.zeros(1,1,4,dtype=torch.bool))
    target['future_states'][0,0,0,0]=.123
    target['future_valid'][0,0,0]=True
    future=torch.zeros(1,1,4,6,dtype=torch.bfloat16);future[0,0,0,0]=.125
    pred=dict(boxes=boxes.bfloat16(),class_logits=torch.zeros(1,1,7,dtype=torch.bfloat16),
        existence_logits=torch.zeros(1,1,dtype=torch.bfloat16),future_states=future)
    actual=object_track_loss(pred,target)['future_track']
    expected=.5*((future[0,0,0,0].float()-target['future_states'][0,0,0,0])/50).square()/6
    torch.testing.assert_close(actual,expected,rtol=1e-4,atol=1e-15)


class TinyReadouts(nn.Module):
    def __init__(self):
        super().__init__();self.bridge=nn.Sequential(nn.LayerNorm(12),nn.Linear(12,16))
        self.dino=FeatureReadout(4,(1,1,1),width=16,layers=1,heads=2)
        self.wa=FeatureReadout(4,(1,1,1,2),width=16,layers=1,heads=2)
        self.objects=ObjectTrackReadout(width=16,queries=2)
        self.evaluation=EvaluationReadout(width=16)


def tiny_model():
    model=FourFrameworkVLA.__new__(FourFrameworkVLA);nn.Module.__init__(model)
    model.architecture_id='A';model.train_heads=True;model.policy_version='test';model.interface=False
    model.decision=ArchitectureA(actor_config=dict(hidden_dim=24,num_layers=4,num_heads=4,ffn_dim=72),
        world_config=dict(hidden_dim=12,num_layers=4,num_heads=3,ffn_dim=36,bridge_groups=2))
    model.readouts=TinyReadouts()
    def encode(self,current_inputs,ego,scene_identity=()):
        scene=current_inputs['current_scene']
        return CurrentContext(scene,torch.ones(scene.shape[:2],dtype=torch.bool),ego,None,
            tuple(scene_identity),'test','FP32','C160')
    model.encode_current=MethodType(encode,model)
    return model


def tiny_batch(size=8,post=True):
    current=torch.randn(size,160,24,requires_grad=True)
    selected=torch.arange(size)if post else torch.empty(0,dtype=torch.long)
    target=torch.randn(size,8,3)
    objects=dict(boxes=torch.zeros(size,1,8),classes=torch.zeros(size,1,dtype=torch.long),
        valid=torch.zeros(size,1,dtype=torch.bool),coverage_valid=torch.zeros(size,dtype=torch.bool),
        future_states=torch.full((size,1,4,6),float('nan')),future_valid=torch.zeros(size,1,4,dtype=torch.bool))
    return dict(current_inputs=dict(current_scene=current),ego=torch.randn(size,8),
        target=target,noise=torch.randn_like(target),sigma=torch.rand(size),tokens=tuple(map(str,range(size))),
        action_valid=torch.ones(size,8,dtype=torch.bool),
        dino_target=torch.randn(size,1,1,1,4),dino_valid=torch.ones(size,1,1,1,dtype=torch.bool),
        future_queries=torch.tensor([[[0,0,0,0],[0,0,0,1]]]).expand(size,-1,-1),
        future_target=torch.randn(size,2,4),future_valid=torch.ones(size,2,dtype=torch.bool),objects=objects,
        aux_scene_indices=selected,post_trajectories=torch.randn(len(selected),3,8,3),
        post_labels=torch.rand(len(selected),3,6),post_valid=torch.ones(len(selected),3,6,dtype=torch.bool),
        post_score=torch.rand(len(selected),3),post_score_valid=torch.ones(len(selected),3,dtype=torch.bool),
        post_events=torch.full((len(selected),3,3),8,dtype=torch.long),
        post_event_valid=torch.ones(len(selected),3,3,dtype=torch.bool))


def test_a_common_post_loss_updates_actor_world_scene_and_replays_exactly_three_per_scene():
    model=tiny_model();batch=tiny_batch();seen=[]
    original=model.decision.replay_candidates
    def record(candidate,indices,prefix):
        seen.append((candidate.detach().clone(),indices.detach().clone()))
        return original(candidate,indices,prefix)
    model.decision.replay_candidates=record
    weights={key:0. for key in LOSS_WEIGHTS};weights['evaluation']=.2;weights['wa_post']=.25
    result=model(batch,weights=weights);result['loss'].backward()
    assert len(seen)==1 and seen[0][0].shape==(24,8,3)
    assert torch.equal(seen[0][0],batch['post_trajectories'].reshape(-1,8,3))
    assert torch.equal(seen[0][1],torch.arange(8).repeat_interleave(3))
    assert model.decision.actor.blocks[0].cross_attn.q.weight.grad.abs().sum()>0
    assert model.decision.world_stream.blocks[-1].ffn[0].weight.grad.abs().sum()>0
    assert batch['current_inputs']['current_scene'].grad.abs().sum()>0
    assert model.decision.actor.action_output.weight.grad.abs().sum()==0


def test_empty_post_microchunk_has_finite_zero_post_and_no_fake_sampling():
    model=tiny_model();batch=tiny_batch(size=1,post=False)
    def forbidden(*args):raise AssertionError('Empty post microchunk must not replay candidates')
    model.decision.replay_candidates=forbidden
    result=model(batch)
    assert result['losses']['wa_post'].item()==0 and result['losses']['evaluation'].item()==0
    assert torch.isfinite(result['loss'])
    result['loss'].backward()


@pytest.mark.parametrize('update',[0,1,1000,53007])
def test_post_budget_is_global_eight_and_identical_across_micro_layouts(update):
    selected=post_positions(update)
    assert len(selected)==len(set(selected))==8
    assert all(4*rank<=selected[rank]<4*rank+4 for rank in range(8))
    layouts=[]
    for micro in (1,2,4):
        found=[]
        for rank in range(8):
            for start in range(0,4,micro):
                positions=list(range(rank*4+start,rank*4+start+micro))
                found.extend(p for p in positions if p in post_positions(update))
        layouts.append(found)
    assert layouts[0]==layouts[1]==layouts[2]==list(selected)


def test_valid_wa_queries_are_unique_balanced_and_skip_invalid_buckets():
    valid=torch.ones(2,3,4,8,16,dtype=torch.bool)
    valid[0,1]=False;valid[1,0,0,:2]=False
    q,mask=sample_valid_future_queries(valid,128,
        generator=[scene_generator(100,scene,'teacher_query')for scene in range(2)])
    assert mask.all()
    for scene in range(2):
        c,t,y,x=q[scene].unbind(-1)
        assert valid[scene,c,t,y,x].all()
        assert len(torch.unique(q[scene],dim=0))==128
    assert not (q[0,:,0]==1).any()
    buckets=q[0,:,0]*4+q[0,:,1]
    counts=torch.bincount(buckets,minlength=12)
    assert counts[counts>0].tolist()==[16]*8


def test_insufficient_valid_wa_queries_have_explicit_invalid_padding():
    valid=torch.zeros(2,3,4,8,16,dtype=torch.bool)
    valid[0,0,0,0,:7]=True
    q,mask=sample_valid_future_queries(valid,128,
        generator=[scene_generator(1,scene,'teacher_query')for scene in range(2)])
    assert mask.sum(1).tolist()==[7,0]
    assert len(torch.unique(q[0,mask[0]],dim=0))==7
    assert q[0,7:].eq(0).all()and not mask[0,7:].any()
    # Coordinate0 happens to be valid in source: the query validity still wins.
    gathered=valid[(torch.arange(2)[:,None],)+tuple(q.unbind(-1))]
    assert gathered[0,7:].all()
    assert (gathered&mask).sum(1).tolist()==[7,0]


def test_wa_query_rng_is_exact_across_microbatch_layouts():
    valid=torch.ones(32,3,4,8,16,dtype=torch.bool)
    valid[2]=False;valid[5,0]=False;valid[17,:,1:3]=False
    def collect(micro):
        q=[];v=[]
        for rank in range(8):
            for start in range(rank*4,rank*4+4,micro):
                positions=list(range(start,start+micro))
                a,b=sample_valid_future_queries(valid[positions],128,
                    generator=[scene_generator(517,p,'teacher_query')for p in positions])
                q.append(a);v.append(b)
        return torch.cat(q),torch.cat(v)
    base=collect(4)
    for micro in (1,2):
        observed=collect(micro)
        assert all(torch.equal(x,y)for x,y in zip(base,observed))


def test_full_update_denominators_do_not_recount_post_per_microchunk():
    full=tiny_batch(size=32,post=False)
    positions=torch.tensor(post_positions(1521))
    full['aux_scene_indices']=positions
    full['post_score']=torch.tensor([[0.,.3,.9]]).expand(8,-1).clone()
    full['post_score_valid']=torch.ones(8,3,dtype=torch.bool)
    full['post_valid']=torch.ones(8,3,6,dtype=torch.bool)
    full['post_event_valid']=torch.ones(8,3,3,dtype=torch.bool)
    expected=update_denominators([full],torch.device('cpu'))
    assert expected['action_elements']==768
    assert expected['dino']==expected['wa_pre']==32
    for name in ('wa_post','factor','score','rank','event'):assert expected[name]==8
    post_fields={'post_score','post_score_valid','post_valid','post_event_valid'}
    fields={'action_valid','dino_valid','future_valid'}
    for micro in (1,2,4):
        chunks=[]
        for start in range(0,32,micro):
            selected=((positions>=start)&(positions<start+micro)).nonzero().flatten()
            chunk={name:full[name][start:start+micro]for name in fields}
            chunk['objects']={name:value[start:start+micro]for name,value in full['objects'].items()}
            chunk.update({name:full[name].index_select(0,selected)for name in post_fields})
            chunk['aux_scene_indices']=positions.index_select(0,selected)-start
            chunks.append(chunk)
        actual=update_denominators(chunks,torch.device('cpu'))
        assert set(actual)==set(expected)
        for key in actual:assert torch.equal(actual[key],expected[key]),key


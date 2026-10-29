"""Common readouts see P/W or R only; no bypass through current C memory."""
import torch
from torch import nn
from torch.nn import functional as F
from scipy.optimize import linear_sum_assignment
from ..qwen_world_postcot.readouts import CrossReadoutBlock, _scene_reduce

class FeatureReadout(nn.Module):
    def __init__(self, channels, axes, width=512, layers=2, heads=8):
        super().__init__()
        self.axes=tuple(axes)
        self.positions=nn.ModuleList(nn.Embedding(n,width) for n in axes)
        self.blocks=nn.ModuleList(CrossReadoutBlock(width,heads) for _ in range(layers))
        self.output=nn.Sequential(nn.LayerNorm(width),nn.Linear(width,channels))

    def forward(self, reasoning, indices=None):
        full=indices is None
        if full:
            values=[torch.arange(n,device=reasoning.device) for n in self.axes]
            indices=torch.stack(torch.meshgrid(*values,indexing='ij'),-1).reshape(1,-1,len(values))
            indices=indices.expand(len(reasoning),-1,-1)
        if indices.shape[0]!=len(reasoning) or indices.shape[-1]!=len(self.axes):
            raise ValueError('Readout query axes/batch mismatch')
        query=0
        for i,(table,n) in enumerate(zip(self.positions,self.axes)):
            if (indices[...,i]<0).any() or (indices[...,i]>=n).any():raise ValueError('Query outside teacher grid')
            query=query+table(indices[...,i])
        valid=torch.ones(reasoning.shape[:2],dtype=torch.bool,device=reasoning.device)
        for block in self.blocks:query=block(query,reasoning,valid)
        result=self.output(query)
        return result.reshape(len(reasoning),*self.axes,-1) if full else result

class ObjectTrackReadout(nn.Module):
    def __init__(self,width=512,queries=32):
        super().__init__()
        self.query=nn.Parameter(torch.randn(queries,width)*.02)
        self.blocks=nn.ModuleList(CrossReadoutBlock(width,8,self_attention=True) for _ in range(2))
        self.norm=nn.LayerNorm(width)
        self.existence=nn.Linear(width,1);self.classes=nn.Linear(width,7)
        self.boxes=nn.Linear(width,8);self.future=nn.Linear(width,4*6)

    def forward(self,p):
        q=self.query[None].expand(len(p),-1,-1)
        valid=torch.ones(p.shape[:2],dtype=torch.bool,device=p.device)
        for block in self.blocks:q=block(q,p,valid)
        q=self.norm(q);raw=self.boxes(q)
        boxes=torch.cat((raw[...,:3],F.softplus(raw[...,3:6]),raw[...,6:]),-1)
        return dict(existence_logits=self.existence(q).squeeze(-1),class_logits=self.classes(q),
                    boxes=boxes,future_states=self.future(q).reshape(len(q),len(self.query),4,6))

def object_track_loss(pred,target,denominators=None):
    """One current Hungarian assignment, reused at every future horizon."""
    den=denominators or {};boxes=pred['boxes'];classes=pred['class_logits'];exists=pred['existence_logits']
    scale=torch.tensor([50,50,5,10,5,5,1,1],device=boxes.device,dtype=torch.float32);future_scale=torch.tensor([50,50,10,5,1,1],device=boxes.device,dtype=torch.float32)
    ev=target['coverage_valid'][:,None].expand_as(exists).clone();et=torch.zeros_like(exists,dtype=torch.float32)
    match=torch.zeros_like(exists,dtype=torch.bool);bt=torch.zeros_like(boxes,dtype=torch.float32);ct=torch.zeros_like(exists,dtype=torch.long)
    ft=torch.zeros_like(pred['future_states'],dtype=torch.float32);fv=torch.zeros(ft.shape[:-1],device=boxes.device,dtype=torch.bool)
    assignments=[]
    for b in range(len(boxes)):
        ids=target['valid'][b].nonzero().flatten()
        if not len(ids):assignments.append(([],[]));continue
        labels=target['classes'][b,ids]
        cost=torch.cdist(boxes[b].detach().float()/scale,target['boxes'][b,ids].float()/scale,p=1)-classes[b].detach().float().softmax(-1)[:,labels]
        pi,gi=linear_sum_assignment(cost.cpu().numpy());pi=torch.as_tensor(pi,device=boxes.device);gi=ids[torch.as_tensor(gi,device=boxes.device)]
        ev[b,pi]=True;et[b,pi]=1;match[b,pi]=True
        bt[b,pi]=target['boxes'][b,gi];ct[b,pi]=target['classes'][b,gi]
        ft[b,pi]=target['future_states'][b,gi];fv[b,pi]=target['future_valid'][b,gi]
        assignments.append((pi.detach(),gi.detach()))
    existence=_scene_reduce(F.binary_cross_entropy_with_logits(exists.float(),et,reduction='none'),ev,scene_denominator=den.get('object_existence'))
    classification=_scene_reduce(F.cross_entropy(classes.float().transpose(1,2),ct,reduction='none'),match,scene_denominator=den.get('object_present'))
    regression=_scene_reduce(F.smooth_l1_loss(boxes.float()/scale,bt/scale,reduction='none'),match,scene_denominator=den.get('object_present'))
    # Invalid future annotation markers are masked before subtraction.
    mask=fv[...,None];prediction=torch.where(mask,pred['future_states'].float(),0.)
    truth=torch.where(mask,ft.float(),0.)
    future=_scene_reduce(F.smooth_l1_loss(prediction/future_scale,truth/future_scale,reduction='none'),fv,scene_denominator=den.get('future_track'))
    return dict(object=existence+classification+regression,future_track=future,assignments=assignments)

class EvaluationReadout(nn.Module):
    def __init__(self,width=512):
        super().__init__()
        self.factor=nn.Linear(width,1)
        self.events=nn.Linear(width,3)
        self.event_none=nn.Linear(width,3)
        self.score=nn.Sequential(nn.Linear(width+6+24,width),nn.SiLU(),nn.Linear(width,1))

    def forward(self,r,candidate):
        if r.shape[1]!=16:raise ValueError('R16 means eight temporal, six metric, two summary slots')
        factor=self.factor(r[:,8:14]).squeeze(-1)
        summary=r[:,14:16].mean(1)
        score=self.score(torch.cat((summary,factor.sigmoid(),candidate.detach().float().flatten(1)),-1)).squeeze(-1)
        # Each of the eight temporal R slots produces its own physical-time
        # class logit; the two summary slots supply the explicit none class.
        event_times=self.events(r[:,:8]).transpose(1,2)
        event_logits=torch.cat((event_times,self.event_none(summary)[...,None]),-1)
        return dict(factor_logits=factor,score_logits=score,event_logits=event_logits)

def evaluation_loss(pred,labels,valid,score,score_valid,events,event_valid,denominators=None):
    """Input candidates are grouped [scene,candidate]; metadata never enters heads."""
    den=denominators or {};ns,nc,nf=labels.shape
    factor=pred['factor_logits'].reshape(ns,nc,nf)
    if not torch.isfinite(labels[valid]).all():raise FloatingPointError('Nonfinite valid native factor label')
    clean_labels=torch.where(valid,labels.detach().float(),0.)
    factors=_scene_reduce(F.binary_cross_entropy_with_logits(factor.float(),clean_labels,reduction='none'),valid,scene_denominator=den.get('factor'))
    scores=pred['score_logits'].reshape(ns,nc).float()
    if not torch.isfinite(score[score_valid]).all():raise FloatingPointError('Nonfinite valid canonical score')
    clean_score=torch.where(score_valid,score.detach().float(),0.)
    total=_scene_reduce(F.smooth_l1_loss(scores.float().sigmoid(),clean_score,reduction='none'),score_valid,scene_denominator=den.get('score'))
    pair_valid=score_valid[:,:,None]&score_valid[:,None,:]
    diff=clean_score[:,:,None]-clean_score[:,None,:]
    pair_valid&=diff.abs()>=.02
    pair_valid&=torch.triu(torch.ones(nc,nc,device=score.device,dtype=torch.bool),1)[None]
    logits=scores[:,:,None]-scores[:,None,:]
    rank=_scene_reduce(F.softplus(-diff.sign()*logits.float()),pair_valid,scene_denominator=den.get('rank'))
    eventlogits=pred['event_logits'].reshape(ns,nc,3,9)
    if ((events[event_valid]<0)|(events[event_valid]>8)).any():raise ValueError('Invalid official event class')
    clean_events=torch.where(event_valid,events.detach().long(),8)
    error=F.cross_entropy(eventlogits.float().flatten(0,2),clean_events.flatten(),reduction='none').reshape(ns,nc,3)
    event=_scene_reduce(error,event_valid,scene_denominator=den.get('event'))
    return dict(loss=factors+.5*total+.5*rank+.5*event,factor=factors,score=total,rank=rank,event=event)

class CommonReadouts(nn.Module):
    def __init__(self,reasoning_dim):
        super().__init__()
        # Common heads have identical initialization across A-D; bridge input
        # widths differ, so each component gets its own registered seed.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(342);self.bridge=nn.Sequential(nn.LayerNorm(reasoning_dim),nn.Linear(reasoning_dim,512))
            torch.manual_seed(343);self.dino=FeatureReadout(1024,(3,9,16))
            torch.manual_seed(344);self.wa=FeatureReadout(512,(3,4,8,16),layers=4)
            torch.manual_seed(345);self.objects=ObjectTrackReadout()
            torch.manual_seed(346);self.evaluation=EvaluationReadout()

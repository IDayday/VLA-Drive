"""Variable-K, population-level error-cost weighting and natural-scene regression."""
from copy import deepcopy
import numpy as np
import torch
import torch.nn.functional as F
from iqe.contracts import require
from iqe.losses import LossTerm
from iqe.io import digest


def pair_population(scores,valid,tie=.01):
    s=np.asarray(scores,dtype=np.float64);v=np.asarray(valid,dtype=bool)&np.isfinite(s)
    require(s.ndim==2 and v.shape==s.shape,'score/valid dimensions')
    pairs=[];weights=[]
    for x in range(len(s)):
        indices=np.flatnonzero(v[x]);P=len(indices)*(len(indices)-1)//2
        for p,i in enumerate(indices):
            for j in indices[p+1:]:
                gap=s[x,i]-s[x,j]
                if abs(gap)>tie:
                    pairs.append((x,int(i),int(j)));weights.append(abs(gap)/max(P,1))
    return np.asarray(pairs,dtype=np.int64).reshape(-1,3),np.asarray(weights,dtype=np.float64)


def population_ranking(pred,truth,valid):
    i,j=torch.triu_indices(pred.shape[1],pred.shape[1],1,device=pred.device)
    gap=truth[:,i]-truth[:,j];v=valid[:,i]&valid[:,j]&torch.isfinite(gap)&(gap.abs()>.01)
    count=valid.sum(-1);p=(count*(count-1)/2).clamp_min(1)
    a=torch.where(v,gap.abs()/p[:,None],torch.zeros_like(gap))
    loss=F.softplus(-torch.nan_to_num(gap).sign()*(pred[:,i]-pred[:,j])/.1)
    return LossTerm((a*loss).sum(),a.sum())


def masks(batch):
    valid=batch['valid'].bool()&torch.isfinite(batch['scores'])
    natural=valid&batch['natural'][:,None].bool()
    components=natural[...,None]&batch['component_valid'].bool()&torch.isfinite(batch['components'])
    ranking=(batch['comparison'].bool()&valid[:,:2].all(-1)&((batch['scores'][:,0]-batch['scores'][:,1]).abs()>.01)) if valid.shape[1]>1 else batch['comparison'].bool()&False
    return natural,components,ranking


def denominators(batch):
    v,c,r=masks(batch)
    return {'value':v.any(-1).sum().float(),'components':c.flatten(1).any(-1).sum().float(),'ranking':r.sum().float()}


def terms(pred,batch):
    v,c,r=masks(batch);p=pred['values'].float();logits=pred['component_logits'].float()
    require(torch.isfinite(p[batch['valid']]).all().item() and torch.isfinite(logits[batch['valid']]).all().item(),'nonfinite predictions for legal candidates')
    p=torch.where(batch['valid'].bool(),p,torch.zeros_like(p))
    logits=torch.where(batch['valid'].bool()[...,None],logits,torch.zeros_like(logits))
    truth=torch.nan_to_num(batch['scores'].detach().float());clabel=torch.nan_to_num(batch['components'].detach().float())
    require(((truth[v]>=0)&(truth[v]<=1)).all().item() and ((clabel[c]>=0)&(clabel[c]<=1)).all().item(),'labels outside [0,1]')
    value=F.smooth_l1_loss(p,truth,beta=.1,reduction='none')
    component=F.binary_cross_entropy_with_logits(logits,clabel,reduction='none')
    # First normalize within scene; candidate-rich scenes do not dominate natural scene objective.
    vn=(value*v).sum(-1)/v.sum(-1).clamp_min(1)
    cn=(component*c).flatten(1).sum(-1)/c.flatten(1).sum(-1).clamp_min(1)
    pair=F.softplus(-(truth[:,0]-truth[:,1]).sign()*(p[:,0]-p[:,1])/.1) if p.shape[1]>1 else p[:,0]*0
    return {'value':LossTerm(vn.sum(),v.any(-1).sum().float()),
            'components':LossTerm(cn.sum(),c.flatten(1).any(-1).sum().float()),
            'ranking':LossTerm((pair*r).sum(),r.sum().float())}


class SceneSampler:
    def __init__(self,ids,seed,identity):
        require(len(ids)>0,'empty expert data')
        self.ids=tuple(ids);self.rng=np.random.default_rng(seed);self.identity=digest([self.ids,seed,identity]);self.consumed=0;self.pending=None
    def peek(self,n):
        if self.pending is None:
            r=deepcopy(self.rng); entries=[{'scene_id':self.ids[i]} for i in r.integers(len(self.ids),size=n)]
            self.pending=(entries,r.bit_generator.state)
        return deepcopy(self.pending[0])
    def consume(self,n):
        require(self.pending is not None and n==len(self.pending[0]),'sampler consume mismatch')
        self.rng.bit_generator.state=self.pending[1];self.pending=None;self.consumed+=n
    def state_dict(self):return {'identity':self.identity,'rng':self.rng.bit_generator.state,'consumed':self.consumed}
    def load_state_dict(self,s):
        require(s['identity']==self.identity,'changed expert manifest/sampler')
        self.rng.bit_generator.state=s['rng'];self.consumed=s['consumed'];self.pending=None


class StudySampler:
    def __init__(self,ids,scores,valid,*,seed,bank_hash,phase_steps=3000):
        self.ids=tuple(ids);self.pairs,self.weights=pair_population(scores,valid)
        self.q=self.weights/self.weights.sum() if self.weights.sum() else None
        self.natural=np.random.default_rng(seed);self.comparison=np.random.default_rng(seed+1)
        self.identity=digest([self.ids,self.pairs.tolist(),self.weights.tolist(),bank_hash,phase_steps])
        self.phase_steps=phase_steps;self.step=0;self.natural_consumed=0;self.comparison_consumed=0;self.pending=None
    def peek(self,n):
        require(n==128,'study global scene/pair batch128')
        if self.pending is None:
            nr,cr=deepcopy(self.natural),deepcopy(self.comparison)
            count=128 if self.step<self.phase_steps or self.q is None else 64
            entries=[{'scene_id':self.ids[i],'natural':True,'comparison':False,'phase':1 if self.step<self.phase_steps else 2,
                      'pair_i':0,'pair_j':1} for i in nr.integers(len(self.ids),size=count)]
            if count<128:
                for index in cr.choice(len(self.pairs),size=128-count,p=self.q):
                    x,i,j=self.pairs[index]; entries.append({'scene_id':self.ids[x],'natural':False,'comparison':True,'phase':2,
                                                           'pair_i':int(i),'pair_j':int(j)})
            self.pending=(entries,nr.bit_generator.state,cr.bit_generator.state,count)
        return deepcopy(self.pending[0])
    def consume(self,n):
        require(self.pending is not None and n==len(self.pending[0]),'two-flow consumption mismatch')
        _,ns,cs,count=self.pending;self.natural.bit_generator.state=ns;self.comparison.bit_generator.state=cs
        self.natural_consumed+=count;self.comparison_consumed+=n-count;self.step+=1;self.pending=None
    def state_dict(self):return {'identity':self.identity,'step':self.step,'natural_rng':self.natural.bit_generator.state,
            'comparison_rng':self.comparison.bit_generator.state,'natural_consumed':self.natural_consumed,'comparison_consumed':self.comparison_consumed}
    def load_state_dict(self,s):
        require(s['identity']==self.identity,'bank/q/stage identity changed')
        self.step=s['step'];self.natural.bit_generator.state=s['natural_rng'];self.comparison.bit_generator.state=s['comparison_rng']
        self.natural_consumed=s['natural_consumed'];self.comparison_consumed=s['comparison_consumed'];self.pending=None

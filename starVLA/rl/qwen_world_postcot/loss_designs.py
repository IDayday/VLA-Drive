"""Independent, parameter-free IL objectives, never a sampler/codec change.

Horizon statistics are TRAIN residual proxies, not conditional uncertainty.
DELTA matches GT increments; it never encourages zero speed or averaged targets.
"""
import math
import torch
from .contracts import file_sha,fingerprint
from .coordinate_objective import SLOPES,review_loading_adapter

ARMS=('CONTROL','TIME_RESIDUAL','DELTA','HUBER')


class LossDesign:
    def __init__(self,kind,scale=1.,horizon_weights=None,delta_coefficient=0.,beta=None):
        if kind not in ARMS or not math.isfinite(scale) or scale<=0:raise ValueError('Invalid loss design/scale')
        self.kind,self.scale=kind,float(scale)
        self.horizon_weights=tuple(horizon_weights or [1.]*8)
        self.delta_coefficient=float(delta_coefficient)
        self.beta=tuple(beta or [.5/SLOPES[0],.25/SLOPES[1],math.radians(2)/SLOPES[2]])
        if (len(self.horizon_weights)!=8 or any(not math.isfinite(w) or w<=0 for w in self.horizon_weights)
                or len(self.beta)!=3 or any(not math.isfinite(b) or b<=0 for b in self.beta)
                or not math.isfinite(self.delta_coefficient) or self.delta_coefficient<0):
            raise ValueError('Invalid nonnegative supervision constants')

    def __call__(self,prediction,target,valid=None):
        from starVLA.model.modules.qwen_world_postcot.flow import clean_x_l1
        original,stats=clean_x_l1(prediction,target,valid,return_statistics=True)
        mask=torch.ones_like(target,dtype=torch.bool) if valid is None else valid.detach().bool()
        if mask.shape==target.shape[:2]:mask=mask[:,:,None].expand_as(target)
        error=torch.where(mask,prediction.float()-target.float(),torch.zeros_like(prediction).float())
        if self.kind=='CONTROL':numerator=stats['numerator']
        elif self.kind=='TIME_RESIDUAL':numerator=(error.abs()*error.new_tensor(self.horizon_weights)[None,:,None]).sum()
        elif self.kind=='HUBER':
            absolute=error.abs();beta=error.new_tensor(self.beta)
            numerator=torch.where(absolute<beta,.5*absolute.square()/beta,absolute-.5*beta).sum()
        else:
            xy=error[:,:,:2]
            # Error at t=0 is exactly zero for both GT and prediction in the
            # current fixed ego frame. Affine normalization offsets cancel.
            delta=torch.cat((xy[:,:1],xy[:,1:]-xy[:,:-1]),dim=1)/.5
            pairs=mask[:,:,:2].clone();pairs[:,1:] &= mask[:,:-1,:2]
            # Reference denominator stays the existing valid pose-element
            # count;3/2 compensates the two XY channels, with no hidden local
            # per-microbatch normalization or bridges over invalid points.
            term=torch.where(pairs,delta.abs(),torch.zeros_like(delta)).sum()*1.5
            numerator=stats['numerator']+self.delta_coefficient*term
        stats['numerator']=numerator*self.scale
        stats['coordinate_diagnostics']=dict(normalized_sum=error.detach().abs().sum(0),valid_count=mask.sum(0).detach().float())
        return stats['numerator']/stats['denominator'].clamp_min(1),stats


def read_loss_registration(row,arm):
    if (row['schema']!='world_IL_loss_designs_v1' or arm not in ARMS
            or row['parent_updates']!=63000 or row['added_updates']!=2000
            or row['seed']!=42 or row['evaluation_seed']!=2101 or not row['freeze_parent_candidate_bank']):
        raise ValueError('Registered loss-family contract changed')
    if row['arms']['CONTROL']!=dict(kind='CONTROL',scale=1.):raise ValueError('Shared control changed')
    for name,config in row['arms'].items():
        if name not in ARMS or name!=config['kind']:raise ValueError('Unknown/mismatched arm')
        LossDesign(**config)
    review=row['loading_adapter_review']
    from pathlib import Path
    if review_loading_adapter(review['reference'],Path(__file__).with_name('loading.py'))!=review:
        raise ValueError('Loading adapter review changed')
    return row

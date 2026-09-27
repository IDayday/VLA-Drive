"""V1.1 fixed-reference decoding and supervision; GT is used only in loss."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from scipy.optimize import linear_sum_assignment
from .agent_heads import AgentHeads


def reference_grid(count):
    nx=math.ceil(math.sqrt(count));ny=math.ceil(count/nx)
    x=1+(torch.arange(nx)+.5)*49/nx
    y=-20+(torch.arange(ny)+.5)*40/ny
    return torch.stack(torch.meshgrid(x,y,indexing='ij'),-1).reshape(-1,2)[:count]


class ReferenceAgentHeads(AgentHeads):
    def __init__(self,qwen_dim,slots=64,classes=7,steps=8,dim=256):
        super().__init__(qwen_dim,classes,steps,dim)
        self.register_buffer('references',reference_grid(slots))
        self.register_buffer('centre_scale',torch.tensor([5.,5.,2.]))
        self.register_buffer('size_scale',torch.tensor([4.,2.,1.5]))

    def forward(self,hidden):
        x=self.shared(hidden);raw=self.box(x).float()
        ref=torch.cat([self.references,torch.zeros_like(self.references[:,:1])],-1)
        centre=ref+raw[...,:3]*self.centre_scale
        size=F.softplus(raw[...,3:6])*self.size_scale+.01
        yaw=F.normalize(raw[...,6:]+raw.new_tensor([0.,1.]),dim=-1,eps=1e-6)
        boxes=torch.cat([centre,size,yaw],-1)
        offset=self.motion(x).float().reshape(*x.shape[:-1],self.steps,2)*5.
        return {'logits':self.classifier(x).float(),'boxes':boxes,'future_displacement':offset,
                'future_xy':centre[...,:2].detach().unsqueeze(-2)+offset,
                'references':self.references[None].expand(x.shape[0],-1,-1)}


def reference_support(ref,target):
    lo,hi=target.supervision_bounds[:2],target.supervision_bounds[2:]
    valid=((ref>=lo)&(ref<=hi)).all(-1)
    if target.supervision_grid is not None:
        grid=target.supervision_grid;cell=((ref-lo)/target.supervision_resolution).floor().long()
        valid&=(cell>=0).all(-1)&(cell[:,0]<grid.shape[0])&(cell[:,1]<grid.shape[1])
        valid&=grid[cell[:,0].clamp(0,grid.shape[0]-1),cell[:,1].clamp(0,grid.shape[1]-1)]
    return valid


@torch.no_grad()
def match_reference(pred,target):
    cols=torch.where(target.current_supervision_mask.bool())[0]
    empty=torch.empty(0,device=pred['boxes'].device,dtype=torch.long)
    if not bool(target.annotation_valid_mask) or not len(cols):return empty,empty
    mask=target.box_valid_mask[cols].bool();gt=target.current_boxes[cols].masked_fill(~mask,0.)
    scale=gt.new_tensor([5.,5.,2.,4.,2.,2.,1.,1.])
    cost=(((pred['boxes'][:,None]-gt[None])/scale).abs()*mask[None]).sum(-1)
    cost-=.25*pred['logits'].softmax(-1)[:,target.current_classes[cols]]
    if not torch.isfinite(cost).all():raise ValueError('Nonfinite assignment')
    r,c=linear_sum_assignment(cost.cpu().numpy())
    return torch.as_tensor(r,device=cols.device),cols[torch.as_tensor(c,device=cols.device)]


def reference_loss_sums(prediction,targets):
    # Do not attach decoded future-centre addition to motion regression.
    zero=sum(prediction[k].sum()*0 for k in ['logits','boxes','future_displacement'])
    sums={k:zero for k in ['cls','box','motion']};counts={k:0. for k in sums};matches=[]
    for b,t in enumerate(targets):
        p={k:v[b] for k,v in prediction.items()};r,c=match_reference(p,t);matches.append((r,c))
        if not bool(t.annotation_valid_mask):continue
        eligible=reference_support(p['references'],t)
        if t.overflow or int(t.current_supervision_mask.sum())>len(p['boxes']):eligible[:]=False
        eligible[r]=True
        labels=torch.full((len(p['boxes']),),p['logits'].shape[-1]-1,device=p['boxes'].device,dtype=torch.long);labels[r]=t.current_classes[c]
        weight=p['logits'].new_full((len(labels),),.1);weight[r]=1.
        sums['cls']=sums['cls']+(F.cross_entropy(p['logits'],labels,reduction='none')*weight)[eligible].sum()
        counts['cls']+=float(weight[eligible].sum())
        if len(r):
            mask=t.box_valid_mask[c].bool();gt=t.current_boxes[c].masked_fill(~mask,0.)
            scale=gt.new_tensor([5.,5.,2.,4.,2.,2.,1.,1.])
            error=F.smooth_l1_loss(p['boxes'][r]/scale,gt/scale,reduction='none')
            sums['box']=sums['box']+error[mask].sum();counts['box']+=int(mask.sum())
            valid=t.future_valid_mask[c].bool()
            displacement=(t.future_xy_in_ego_t0[c]-t.current_boxes[c,:2,None].transpose(-1,-2)).masked_fill(~valid[...,None],0.)
            error=F.smooth_l1_loss(p['future_displacement'][r]/5.,displacement/5.,reduction='none').sum(-1)
            sums['motion']=sums['motion']+error[valid].sum();counts['motion']+=int(valid.sum())
    return sums,counts,matches

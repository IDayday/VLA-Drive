"""From-scratch trajectory-local visual readout, with the original S11 weights' shapes.

This is a new scorer architecture, not an output-equivalent execution change.
No trained S11 parameters are imported. Base/planner/VLM functions are unchanged.
"""
import torch
from iqe.contracts import require
from iqe.internal_baseline.scorer import InternalScorer
from iqe.specialist_study.models import SpatialReader
from .geometry import project_points, candidate_anchors, neighborhood_indices, gather_neighborhood


class GeometricSpatialReader(SpatialReader):
    def forward(self, visual, valid, xy, tokens, physical, calibration):
        b, v, n, _ = visual.shape
        require(v == self.views == 3 and n == 576 and valid.shape == (b,v,n)
                and valid.all().item() and xy.shape == (b,v,n,2), "audited18x32 three-view V contract")
        require(tokens.shape[2] == 8 and self.queries.shape[1] == 16,
                "two height hypotheses for each of eight physical waypoints")
        points = candidate_anchors(physical.detach())
        uv, visible = project_points(points,calibration['camera_intrinsics'],calibration['camera_to_ego'],
                                     calibration['camera_image_sizes'])
        grid = torch.tensor([32,18],device=visual.device).expand(b,v,2)
        indices, local_valid = neighborhood_indices(uv,visible,calibration['camera_image_sizes'],grid)
        p = xy.detach().float()*torch.pi
        # Expensive feature projection runs once per scene, not once per candidate.
        memory = self.projection(visual.detach().float()) + self.position(torch.cat((p.sin(),p.cos()),-1)) + self.camera[None,:,None]
        local = gather_neighborhood(memory,indices)
        k,q = tokens.shape[1],self.queries.shape[1]
        query = self.queries[None,None] + tokens.repeat_interleave(2,dim=2)[:,:,None]
        z = query.reshape(b*k*v,q,self.hidden)
        layer = self.layer
        norm = layer.norm1(z)
        z = z + layer.dropout1(layer.self_attn(norm,norm,norm,need_weights=False)[0])
        # Each time/height query attends only its own projected3x3 neighborhood.
        # No-view points carry an explicitly masked zero, not a border feature.
        local = local.reshape(b*k*v*q,9,self.hidden)
        mask = local_valid.reshape(b*k*v*q,9)
        observed = mask.any(-1)
        safe_mask = mask.clone();safe_mask[~observed,0] = True
        local = torch.where(mask[...,None],local,torch.zeros_like(local))
        query = layer.norm2(z).reshape(b*k*v*q,1,self.hidden)
        read = layer.multihead_attn(query,local,local,key_padding_mask=~safe_mask,need_weights=False)[0]
        read = read * observed[:,None,None]
        z = z + layer.dropout2(read.reshape(b*k*v,q,self.hidden))
        hidden = layer.norm3(z)
        z = z + layer.dropout3(layer.linear2(layer.dropout(layer.activation(layer.linear1(hidden)))))
        z = self.norm(z).reshape(b,k,v*q,self.hidden)
        observed = observed.reshape(b,k,v*q)
        return z*observed[...,None],observed


class GeometricReadoutScorer(InternalScorer):
    def __init__(self,world_dim,action_dim,trajectory_contract,*,visual_dim,views=3):
        super().__init__(world_dim,action_dim,trajectory_contract,dropout=0.)
        self.spatial=True
        self.spatial_reader=GeometricSpatialReader(visual_dim,views)

    def forward(self,context,trajectories):
        standard=('world','action','current_ego','world_valid','action_valid')
        cameras=('camera_intrinsics','camera_to_ego','camera_image_sizes')
        require(set(context)==set(standard)|set(cameras)|{'visual','visual_valid','visual_xy'},
                "geometric scorer current-input whitelist")
        memory,mask=self.project_context({k:context[k] for k in standard})
        t=trajectories.detach().float()
        require(t.ndim==4 and t.shape[-2:]==(len(self.times),3),"physical candidate contract")
        valid=torch.isfinite(t).all((-1,-2))
        safe=torch.where(valid[...,None,None],t,torch.zeros_like(t))
        b,k,steps,_=t.shape
        xy=(safe[...,:2]-self.xy_mean)/self.xy_scale
        token=self.trajectory_encoder(torch.cat((xy,safe[...,2:3].sin(),safe[...,2:3].cos(),
                                                self.times.expand(b,k,steps)[...,None]),-1))
        visual,visual_mask=self.spatial_reader(context['visual'],context['visual_valid'],context['visual_xy'],
                                               token,safe,{name:context[name].detach().float() for name in cameras})
        combined=torch.cat((memory[:,None].expand(-1,k,-1,-1),visual),2).flatten(0,1)
        masks=torch.cat((mask[:,None].expand(-1,k,-1),visual_mask),2).flatten(0,1)
        z=self.decoder(token.reshape(b*k,steps,-1),combined,memory_key_padding_mask=~masks).mean(1).reshape(b,k,-1)
        return dict(values=self.value_head(z).squeeze(-1).sigmoid(),
                    component_logits=self.component_head(z),valid=valid)

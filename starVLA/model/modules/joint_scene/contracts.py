"""Current-state graph and label-side annotated scene have separate contracts."""
from dataclasses import dataclass, fields
import torch


@dataclass
class LocalSceneGraph:
    boxes: torch.Tensor                 # B,A,8 xyz/lwh/sincos; ego slot0
    classes: torch.Tensor               # B,A; 7 denotes ego
    active_actor_mask: torch.Tensor     # B,A
    source_indices: torch.Tensor        # correspondence only, never embedded
    geometric_support: torch.Tensor    # B,A; calibrated FOV, NOT occlusion visibility
    ego_state: torch.Tensor             # B,7: current speed + history velocity xy + nav onehot4
    edge_mask: torch.Tensor             # B,A,A current-only candidate links
    edge_features: torch.Tensor         # B,A,A,8 relative geometry
    context_boxes: torch.Tensor         # B,K,8 current risk/scene hypotheses
    context_classes: torch.Tensor       # B,K
    context_mask: torch.Tensor          # B,K
    origin: str                         # annotated_supervision or predicted_inference

    def validate(self):
        b,a,_=self.boxes.shape
        if self.boxes.shape!=(b,a,8) or not self.active_actor_mask[:,0].all():
            raise ValueError('A valid ego must occupy slot0')
        if self.classes.shape!=(b,a) or self.geometric_support.shape!=(b,a):
            raise ValueError('Invalid current actor features')
        if self.edge_mask.shape!=(b,a,a) or self.edge_features.shape!=(b,a,a,8):
            raise ValueError('Invalid current relation dimensions')
        if self.origin not in ('annotated_supervision','predicted_inference'):
            raise ValueError('Explicit teacher/inference provenance required')
        if (self.edge_mask & ~(self.active_actor_mask[:,:,None]&self.active_actor_mask[:,None,:])).any():
            raise ValueError('Padding participates in relation graph')
        if not torch.isfinite(self.boxes[self.active_actor_mask]).all():
            raise ValueError('Nonfinite active current state')
        return self

    def to(self,device):
        return LocalSceneGraph(**{f.name:(getattr(self,f.name).to(device) if torch.is_tensor(getattr(self,f.name)) else getattr(self,f.name)) for f in fields(self)})


@dataclass
class AnnotatedLocalScene:
    token: str
    log: str
    graph: LocalSceneGraph
    future: torch.Tensor               # 1,A,T,4: xy metres, yaw sin/cos
    feature_valid: torch.Tensor        # 1,A,T,4; never model input on all-hidden path
    track_ids: tuple                   # audit/association only; never model input
    metadata: dict

    def validate(self):
        self.graph.validate()
        if self.graph.origin!='annotated_supervision':raise ValueError('Annotated scene must declare its privileged current graph')
        if self.future.shape!=self.feature_valid.shape or self.future.shape[:2]!=self.graph.boxes.shape[:2] or self.future.shape[-1]!=4:
            raise ValueError('Label dimensions differ from fixed graph')
        if (self.feature_valid & ~self.graph.active_actor_mask[:,:,None,None]).any():raise ValueError('Padding has labels')
        if not torch.isfinite(self.future[self.feature_valid]).all():raise ValueError('Nonfinite valid labels')
        return self


def stack_graphs(graphs,device='cpu'):
    if len({g.origin for g in graphs})!=1:raise ValueError('Batch teacher/predicted origins separately')
    return LocalSceneGraph(**{f.name:(torch.cat([getattr(g,f.name) for g in graphs]).to(device) if torch.is_tensor(getattr(graphs[0],f.name)) else getattr(graphs[0],f.name)) for f in fields(graphs[0])}).validate()

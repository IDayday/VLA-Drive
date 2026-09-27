"""Schema4 separates generated channels, available labels, known values and actors."""
from dataclasses import dataclass, fields
import torch

SCHEMA_VERSION = 4
STATE_POLICY = 'ego_xy_yaw_neighbors_xy_v1'


def modeled_channels(active):
    if active.dtype!=torch.bool or active.ndim!=2:raise ValueError('active_actor_mask must be bool B,A')
    mask=active[...,None].expand(*active.shape,4).clone()
    mask[:,1:,2:]=False
    return mask


def finite_selected(value,mask,name):
    if not torch.isfinite(value[mask]).all():raise ValueError(f'Nonfinite valid {name}')


@dataclass
class LocalSceneGraph:
    boxes: torch.Tensor
    classes: torch.Tensor
    active_actor_mask: torch.Tensor
    modeled_state_mask: torch.Tensor       # B,A,4 fixed task policy; never future-valid
    source_indices: torch.Tensor
    geometric_support: torch.Tensor
    ego_state: torch.Tensor                # speed, history vx/vy, navigation onehot4
    edge_mask: torch.Tensor
    edge_features: torch.Tensor
    context_boxes: torch.Tensor
    context_classes: torch.Tensor
    context_mask: torch.Tensor
    context_geometric_support: torch.Tensor
    context_source_indices: torch.Tensor
    origin: str
    selection_metadata: tuple
    schema_version: int = SCHEMA_VERSION
    state_policy: str = STATE_POLICY

    def validate(self):
        if self.schema_version!=SCHEMA_VERSION or self.state_policy!=STATE_POLICY:raise ValueError('Graph schema/state policy mismatch')
        if self.boxes.ndim!=3 or self.boxes.shape[-1]!=8:raise ValueError('Boxes must be B,A,8')
        b,a,_=self.boxes.shape
        if b<1 or a<1:raise ValueError('Empty graph batch/ego')
        if self.context_boxes.ndim!=3 or self.context_boxes.shape[0]!=b or self.context_boxes.shape[-1]!=8:raise ValueError('Context boxes must be B,K,8')
        k=self.context_boxes.shape[1]
        for name,shape in [('active_actor_mask',(b,a)),('modeled_state_mask',(b,a,4)),('edge_mask',(b,a,a)),('context_mask',(b,k))]:
            value=getattr(self,name)
            if value.shape!=shape or value.dtype!=torch.bool:raise ValueError(f'{name} must be bool {shape}')
        for name,shape in [('classes',(b,a)),('source_indices',(b,a)),('context_classes',(b,k)),('context_source_indices',(b,k))]:
            value=getattr(self,name)
            if value.shape!=shape or value.dtype!=torch.long:raise ValueError(f'{name} must be int64 {shape}')
        for name,shape in [('geometric_support',(b,a)),('ego_state',(b,7)),('edge_features',(b,a,a,8)),('context_geometric_support',(b,k))]:
            value=getattr(self,name)
            if value.shape!=shape or not value.is_floating_point():raise ValueError(f'{name} shape/dtype mismatch')
        if not self.boxes.is_floating_point() or not self.context_boxes.is_floating_point():raise ValueError('Boxes must be floating point')
        if not self.active_actor_mask[:,0].all() or not (self.classes[:,0]==7).all():raise ValueError('Valid ego class7 required in slot0')
        if not torch.equal(self.modeled_state_mask,modeled_channels(self.active_actor_mask)):raise ValueError('Modeled channels must follow current role, not sample label validity')
        if self.origin not in ('annotated_supervision','predicted_inference'):raise ValueError('Explicit graph provenance required')
        if len(self.selection_metadata)!=b:raise ValueError('Selection provenance batch mismatch')
        for values,mask,classes,support,name in [(self.boxes,self.active_actor_mask,self.classes,self.geometric_support,'actor'),(self.context_boxes,self.context_mask,self.context_classes,self.context_geometric_support,'context')]:
            finite_selected(values,mask,name+' boxes');finite_selected(support,mask,name+' support')
            if (values[...,3:6][mask]<=0).any():raise ValueError(f'Invalid valid {name} dimensions')
            if (values[...,6:8][mask].norm(dim=-1)<1e-6).any():raise ValueError(f'Invalid valid {name} current yaw')
            selected_classes=classes[mask]
            if ((selected_classes<0)|(selected_classes>(7 if name=='actor' else 6))).any():raise ValueError(f'Invalid valid {name} class')
            if ((support[mask]<0)|(support[mask]>1)).any():raise ValueError(f'Invalid valid {name} support')
        neighbor=self.active_actor_mask[:,1:]
        if (self.classes[:,1:][neighbor]>2).any():raise ValueError('Only vehicle/pedestrian/bicycle are trajectory actors')
        if (self.geometric_support[:,1:][neighbor]<=0).any():raise ValueError('Zero support actor belongs in context')
        if not torch.isfinite(self.ego_state).all() or (self.ego_state[:,0]<0).any():raise ValueError('Invalid ego_state')
        nav=self.ego_state[:,3:]
        if not (((nav==0)|(nav==1)).all() and (nav.sum(-1)==1).all()):raise ValueError('Navigation must be a valid onehot')
        permitted=self.active_actor_mask[:,:,None]&self.active_actor_mask[:,None,:]
        if (self.edge_mask & ~permitted).any():raise ValueError('Padding participates in graph')
        if not (self.edge_mask.diagonal(dim1=1,dim2=2)[self.active_actor_mask]).all():raise ValueError('Each active actor needs a self attention edge')
        finite_selected(self.edge_features,self.edge_mask,'edge features')
        if (self.source_indices[:,1:][neighbor]<0).any() or (self.context_source_indices[self.context_mask]<0).any():raise ValueError('Missing current source identity')
        return self

    def to(self,device):
        return LocalSceneGraph(**{f.name:(getattr(self,f.name).to(device) if torch.is_tensor(getattr(self,f.name)) else getattr(self,f.name)) for f in fields(self)})


@dataclass
class AnnotatedLocalScene:
    token: str
    log: str
    graph: LocalSceneGraph
    future: torch.Tensor
    feature_valid: torch.Tensor            # labels only; independent of modeled policy
    track_ids: tuple                       # audit only, never embedded
    metadata: dict

    def validate(self):
        self.graph.validate()
        if self.graph.origin!='annotated_supervision':raise ValueError('Annotated scene requires supervision provenance')
        if self.future.ndim!=4 or self.future.shape!=self.feature_valid.shape or self.future.shape[:2]!=self.graph.boxes.shape[:2] or self.future.shape[-1]!=4 or self.future.shape[2]<1:raise ValueError('Label shape mismatch')
        if self.feature_valid.dtype!=torch.bool:raise ValueError('feature_valid must be bool')
        if (self.feature_valid & ~self.graph.modeled_state_mask[:,:,None]).any():raise ValueError('Labels assigned to inactive/unmodeled state')
        finite_selected(self.future,self.feature_valid,'future labels')
        if len(self.track_ids)!=self.graph.boxes.shape[1]:raise ValueError('Track correspondence shape mismatch')
        return self


def stack_graphs(graphs,device='cpu'):
    if not graphs:raise ValueError('Empty graph batch')
    for g in graphs:g.validate()
    if len({(g.origin,g.state_policy,g.schema_version) for g in graphs})!=1:raise ValueError('Graph origins/schemas differ')
    result={}
    for f in fields(graphs[0]):
        values=[getattr(g,f.name) for g in graphs]
        result[f.name]=torch.cat(values).to(device) if torch.is_tensor(values[0]) else (sum((tuple(v) for v in values),()) if f.name=='selection_metadata' else values[0])
    return LocalSceneGraph(**result).validate()

"""Current-only local candidates: geometry support, participant type, navigation."""
from dataclasses import dataclass,asdict
import math
import torch
from starVLA.model.modules.structured_world.targets import CLASSES
from .contracts import LocalSceneGraph,modeled_channels

# Authoritative label mapping is shared with the verified raw-log target writer.
TRAJECTORY_CLASSES=tuple(CLASSES.index(name) for name in ('vehicle','pedestrian','bicycle'))


@dataclass(frozen=True)
class GraphConfig:
    variant: str = 'decision_local'
    max_neighbors: int = 16
    primary_neighbors: int = 8
    radius_m: float = 50.
    secondary_radius_m: float = 25.
    max_context: int = 64
    existence_floor: float = .20
    corridor_half_width_m: float = 2.
    unknown_motion_margin_m: float = 2.
    turn_radius_m: float = 12.
    duplicate_distance_m: float = .6

    def validate(self):
        if self.variant not in ('nearest','decision_local'):raise ValueError('Unknown current graph variant')
        if not all(isinstance(v,int) and not isinstance(v,bool) for v in (self.max_neighbors,self.primary_neighbors,self.max_context)):raise ValueError('Capacities must be integers')
        if not 1<=self.primary_neighbors<=self.max_neighbors or self.max_context<0:raise ValueError('Invalid candidate/context capacity')
        for name in ('radius_m','secondary_radius_m','turn_radius_m','duplicate_distance_m','corridor_half_width_m','unknown_motion_margin_m'):
            if not math.isfinite(getattr(self,name)) or getattr(self,name)<=0:raise ValueError('Invalid geometry configuration: '+name)
        if not 0<=self.existence_floor<=1:raise ValueError('Invalid existence floor')
        return self


def corridor_points(ego_state,config):
    """Conservative navigated current corridor; no logged neighbor velocity or future."""
    length=min(45.,max(12.,float(ego_state[0])*4.+8.))
    s=torch.linspace(0,length,49,device=ego_state.device,dtype=ego_state.dtype)
    straight=torch.stack([s,torch.zeros_like(s)],-1)
    angle=(s/config.turn_radius_m).clamp(max=math.pi/2)
    extra=(s-config.turn_radius_m*math.pi/2).clamp_min(0)
    left=torch.stack([config.turn_radius_m*torch.sin(angle),config.turn_radius_m*(1-torch.cos(angle))+extra],-1)
    right=left*left.new_tensor([1.,-1.]);nav=int(ego_state[3:].argmax())
    return [left,straight,right,torch.cat([left,straight,right])][nav]


def _build(boxes,classes,current_valid,support,ego_state,config,origin):
    config.validate()
    if boxes.ndim!=2 or boxes.shape[-1]!=8 or not boxes.is_floating_point():raise ValueError('Expected current boxes N,8')
    n=len(boxes)
    if classes.shape!=(n,) or classes.dtype!=torch.long or current_valid.shape!=(n,) or current_valid.dtype!=torch.bool or support.shape!=(n,) or ego_state.shape!=(7,):raise ValueError('Current graph metadata mismatch')
    if not torch.isfinite(ego_state).all() or ego_state[0]<0 or not (((ego_state[3:]==0)|(ego_state[3:]==1)).all() and ego_state[3:].sum()==1):raise ValueError('Invalid ego state/navigation')
    if not torch.isfinite(boxes[current_valid]).all() or (boxes[current_valid,3:6]<=0).any() or (boxes[current_valid,6:8].norm(dim=-1)<1e-6).any():raise ValueError('Invalid valid current box')
    if ((classes[current_valid]<0)|(classes[current_valid]>=len(CLASSES))).any():raise ValueError('Invalid valid current class')
    if not torch.isfinite(support[current_valid]).all() or ((support[current_valid]<0)|(support[current_valid]>1)).any():raise ValueError('Invalid valid geometric support')
    clean=torch.where(current_valid[:,None],boxes,0.);centre=clean[:,:2];distance=centre.norm(dim=-1)
    path=corridor_points(ego_state,config);corridor_distance=torch.cdist(centre,path).amin(-1)
    radius=clean[:,3:5].norm(dim=-1)*.5
    clearance=corridor_distance-radius-config.corridor_half_width_m-config.unknown_motion_margin_m
    direct=current_valid&(distance<=config.radius_m)&(clearance<=0)
    traffic=current_valid & ((classes==TRAJECTORY_CLASSES[0])|(classes==TRAJECTORY_CLASSES[1])|(classes==TRAJECTORY_CLASSES[2]))
    eligible=traffic&(support>0)&(distance<=config.radius_m)
    # Conservative same-class near-identical-box dedup, based only on CURRENT geometry.
    duplicate_of={};unique=[]
    order=torch.where(current_valid)[0].tolist()
    order.sort(key=lambda i:(-float(support[i]),float(distance[i]),i))
    for i in order:
        duplicate=None
        for j in unique:
            if int(classes[i])!=int(classes[j]):continue
            if float((centre[i]-centre[j]).norm())>config.duplicate_distance_m:continue
            ratio=clean[i,3:6]/clean[j,3:6]
            yaw_agreement=float((clean[i,6:8]*clean[j,6:8]).sum()/(clean[i,6:8].norm()*clean[j,6:8].norm()))
            if bool(((ratio>=.75)&(ratio<=1.3334)).all()) and yaw_agreement>.95:duplicate=j;break
        if duplicate is None:unique.append(i)
        else:duplicate_of[i]=duplicate;eligible[i]=False
    ids=torch.where(eligible)[0].tolist()
    primary_pool=ids if config.variant=='nearest' else [i for i in ids if bool(direct[i])]
    def rank(i):return (float(distance[i]),i) if config.variant=='nearest' else (float(clearance[i])+.05*float(distance[i]),float(distance[i]),i)
    primary_pool.sort(key=rank);main=primary_pool[:config.primary_neighbors]
    secondary=[]
    for i in ids:
        if i in main or not main:continue
        proximity=float((centre[main]-centre[i]).norm(dim=-1).min())
        if proximity<=config.secondary_radius_m:secondary.append((i,proximity))
    secondary.sort(key=lambda item:(0 if bool(direct[item[0]]) else 1,item[1],float(distance[item[0]]),item[0]))
    if config.variant=='nearest':secondary.sort(key=lambda item:(float(distance[item[0]]),item[0]))
    selected=main+[i for i,_ in secondary[:config.max_neighbors-len(main)]]
    selected_set=set(selected);candidate_set=set(primary_pool)|{i for i,_ in secondary}
    # Risk context includes static, unsupported and capacity-excluded hypotheses.
    context_pool=[i for i in unique if float(distance[i])<=config.radius_m]
    context_pool.sort(key=lambda i:(0 if bool(direct[i]) and i not in selected_set else 1,float(clearance[i])+.05*float(distance[i]),i))
    context_ids=context_pool[:config.max_context];context_set=set(context_ids)
    records=[]
    for i in range(n):
        reason='selected_primary' if i in main else ('selected_secondary' if i in selected_set else
            ('invalid_current_annotation' if not bool(current_valid[i]) else 'duplicate' if i in duplicate_of else
             'static_context_only' if not bool(traffic[i]) else 'no_geometric_support' if float(support[i])<=0 else
             'outside_radius' if float(distance[i])>config.radius_m else 'capacity' if i in candidate_set else 'outside_decision_local'))
        records.append({'source_index':i,'class_id':int(classes[i]) if bool(current_valid[i]) else None,
            'current_valid':bool(current_valid[i]),'distance_m':float(distance[i]) if bool(current_valid[i]) else None,
            'geometric_support':float(support[i]) if bool(current_valid[i]) else None,
            'corridor_clearance_m':float(clearance[i]) if bool(current_valid[i]) else None,
            'direct_candidate':bool(direct[i]),'trajectory_candidate':i in candidate_set,'selected':i in selected_set,
            'context_retained':i in context_set,'context_overflow':i in context_pool and i not in context_set,
            'duplicate_of':duplicate_of.get(i),'reason':reason,'neighbor_velocity_available':False})
    a=config.max_neighbors+1;k=config.max_context;device=boxes.device
    bx=boxes.new_zeros(1,a,8);bx[0,0]=boxes.new_tensor([0,0,0,4.9,2.,1.6,0,1])
    cl=torch.zeros(1,a,device=device,dtype=torch.long);cl[0,0]=7
    active=torch.zeros(1,a,device=device,dtype=torch.bool);active[0,:len(selected)+1]=True
    source=torch.full((1,a),-1,device=device,dtype=torch.long);source[0,0]=-2
    sup=boxes.new_zeros(1,a);sup[0,0]=1
    if selected:
        bx[0,1:len(selected)+1]=boxes[selected];cl[0,1:len(selected)+1]=classes[selected]
        source[0,1:len(selected)+1]=torch.tensor(selected,device=device);sup[0,1:len(selected)+1]=support[selected]
    allowed=active[:,:,None]&active[:,None,:]
    delta=bx[:,:,None,:2]-bx[:,None,:,:2];sizes=bx[:,:,None,3:5]-bx[:,None,:,3:5];yaw=bx[:,:,None,6:8]-bx[:,None,:,6:8]
    edge=torch.cat([delta/20.,delta.norm(dim=-1,keepdim=True)/20.,sizes/5.,yaw,(cl[:,:,None]==cl[:,None,:])[...,None].to(boxes.dtype)],-1)
    edge=torch.where(allowed[...,None],edge,0.)
    ctx=boxes.new_zeros(1,k,8);cc=torch.zeros(1,k,dtype=torch.long,device=device);cm=torch.zeros(1,k,dtype=torch.bool,device=device)
    cs=boxes.new_zeros(1,k);ci=torch.full((1,k),-1,dtype=torch.long,device=device)
    if context_ids:
        ctx[0,:len(context_ids)]=boxes[context_ids];cc[0,:len(context_ids)]=classes[context_ids];cm[0,:len(context_ids)]=True
        cs[0,:len(context_ids)]=support[context_ids];ci[0,:len(context_ids)]=torch.tensor(context_ids,device=device)
    metadata={'graph_config':asdict(config),'source_current_objects':n,'trajectory_candidates':len(candidate_set),
        'selected_neighbors':len(selected),'context_objects':len(context_ids),'context_overflow':len(context_pool)-len(context_ids),
        'ego_only':not selected,'geometric_support_not_occlusion':True,'class_mapping':list(CLASSES),'objects':records}
    return LocalSceneGraph(bx,cl,active,modeled_channels(active),source,sup,ego_state[None],allowed,edge,ctx,cc,cm,cs,ci,origin,(metadata,)).validate()


def build_supervision_graph(current_boxes,current_classes,current_annotation_valid,geometric_support,ego_state,config=GraphConfig()):
    return _build(current_boxes,current_classes,current_annotation_valid,geometric_support,ego_state,config,'annotated_supervision')


def build_inference_graph(prediction,geometric_support,ego_state,config=GraphConfig()):
    boxes,logits=prediction['boxes'],prediction['logits']
    if boxes.ndim!=2 or logits.shape!=(len(boxes),8) or not torch.isfinite(logits).all():raise ValueError('Invalid current predicted logits/boxes')
    probability=logits.softmax(-1);classes=probability[:,:7].argmax(-1)
    valid=((1-probability[:,-1])>=config.existence_floor)&(probability.argmax(-1)!=7)
    return _build(boxes,classes,valid,geometric_support,ego_state,config,'predicted_inference')

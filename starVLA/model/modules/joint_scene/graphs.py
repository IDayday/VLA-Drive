"""Simple current-only candidates. Targets/future validity cannot enter selection."""
from dataclasses import dataclass
import torch
from .contracts import LocalSceneGraph


@dataclass(frozen=True)
class GraphConfig:
    max_neighbors: int = 16
    primary_neighbors: int = 8
    radius_m: float = 50.
    secondary_radius_m: float = 25.
    max_context: int = 64
    existence_floor: float = .20


def _build(boxes,classes,current_valid,support,ego_state,config,origin):
    if boxes.ndim!=2 or boxes.shape[-1]!=8 or classes.shape!=boxes.shape[:1]:raise ValueError('Expected singleton current boxes/types')
    if current_valid.shape!=classes.shape or support.shape!=classes.shape or ego_state.shape!=(7,):raise ValueError('Current graph metadata mismatch')
    if config.max_neighbors<1 or not 1<=config.primary_neighbors<=config.max_neighbors:raise ValueError('Invalid candidate capacity')
    legal=current_valid & torch.isfinite(boxes).all(-1) & (boxes[:,3:6]>0).all(-1)
    centre=torch.where(legal[:,None],boxes[:,:2],0.)
    distance=centre.norm(dim=-1)
    candidates=torch.where(legal & (distance<=config.radius_m))[0]
    # Stable geometric ordering; class correctness and future labels never gate nodes.
    candidates=candidates[torch.argsort(distance[candidates],stable=True)]
    main=candidates[:config.primary_neighbors]
    other=candidates[config.primary_neighbors:]
    if len(main) and len(other):
        proximity=torch.cdist(centre[other],centre[main]).amin(-1)
        other=other[proximity<=config.secondary_radius_m]
    selected=torch.cat([main,other[:config.max_neighbors-len(main)]])
    a=config.max_neighbors+1;device=boxes.device
    bx=boxes.new_zeros(1,a,8);bx[0,0]=boxes.new_tensor([0,0,0,4.9,2.,1.6,0,1])
    cl=torch.zeros(1,a,device=device,dtype=torch.long);cl[0,0]=7
    active=torch.zeros(1,a,device=device,dtype=torch.bool);active[0,:len(selected)+1]=True
    source=torch.full((1,a),-1,device=device,dtype=torch.long);source[0,0]=-2
    sup=boxes.new_zeros(1,a);sup[0,0]=1
    if len(selected):
        bx[0,1:len(selected)+1]=boxes[selected];cl[0,1:len(selected)+1]=classes[selected]
        source[0,1:len(selected)+1]=selected;sup[0,1:len(selected)+1]=support[selected]
    # Dense within the small current candidate set. Learned attention determines strength.
    allowed=active[:,:,None]&active[:,None,:]
    delta=bx[:,:,None,:2]-bx[:,None,:,:2]
    sizes=bx[:,:,None,3:5]-bx[:,None,:,3:5]
    yaw=bx[:,:,None,6:8]-bx[:,None,:,6:8]
    dist=delta.norm(dim=-1,keepdim=True)
    edge=torch.cat([delta/20.,dist/20.,sizes/5.,yaw,(cl[:,:,None]==cl[:,None,:])[...,None].to(boxes.dtype)],-1)
    edge=torch.where(allowed[...,None],edge,0.)
    # All legal current candidates, including unsupported/capacity-excluded, retain context.
    context_ids=torch.where(legal)[0];context_ids=context_ids[torch.argsort(distance[context_ids],stable=True)][:config.max_context]
    ctx=boxes.new_zeros(1,config.max_context,8);cc=torch.zeros(1,config.max_context,dtype=torch.long,device=device);cm=torch.zeros(1,config.max_context,dtype=torch.bool,device=device)
    ctx[0,:len(context_ids)]=boxes[context_ids];cc[0,:len(context_ids)]=classes[context_ids];cm[0,:len(context_ids)]=True
    result=LocalSceneGraph(bx,cl,active,source,sup,ego_state[None],allowed,edge,ctx,cc,cm,origin)
    return result.validate()


def build_supervision_graph(current_boxes,current_classes,current_annotation_valid,geometric_support,ego_state,config=GraphConfig()):
    """Privileged CURRENT labels or assigned queries; no future arguments accepted."""
    return _build(current_boxes,current_classes,current_annotation_valid,geometric_support,ego_state,config,'annotated_supervision')


def build_inference_graph(prediction,geometric_support,ego_state,config=GraphConfig()):
    """Only current predictions/calibration/ego. No labels or future-valid argument."""
    boxes,logits=prediction['boxes'],prediction['logits']
    if boxes.ndim!=2 or logits.shape!=(len(boxes),8):raise ValueError('Expected current singleton prediction')
    probability=logits.softmax(-1);classes=probability[:,:7].argmax(-1)
    valid=torch.isfinite(logits).all(-1)&((1-probability[:,-1])>=config.existence_floor)
    valid &= probability.argmax(-1)!=7
    return _build(boxes,classes,valid,geometric_support,ego_state,config,'predicted_inference')

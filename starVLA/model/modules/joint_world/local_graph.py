"""Current-only rooted relationship graph. True edge[i,j] means i reads j."""
from dataclasses import dataclass, fields
import hashlib
import json
import numpy as np
import torch
from .observability import measure_observability

CONTRACT_VERSION = 2
EDGE_FEATURE_DIM = 8


@dataclass
class LocalInteractionGraph:
    source_slot_ids: torch.Tensor   # B,A: -1 ego, >=0 original prediction slot, -2 padding
    active_actor_mask: torch.Tensor
    predictable_actor_mask: torch.Tensor
    trajectory_condition_mask: torch.Tensor
    edge_mask: torch.Tensor         # B,A,A, query i reads key j; includes active self
    edge_features: torch.Tensor     # dx,dy,distance,heading cosine,influence i<-j,i->j,direct,relation code
    visual_reliability: torch.Tensor
    relevance_scores: torch.Tensor
    context_only_source_ids: torch.Tensor
    context_only_mask: torch.Tensor
    context_only_entities: list
    selection_reasons: list
    exclusion_reasons: list
    node_records: list
    graph_provenance: list

    def validate(self):
        b, a = self.source_slot_ids.shape
        if self.edge_mask.shape != (b,a,a) or self.edge_features.shape != (b,a,a,EDGE_FEATURE_DIM):
            raise ValueError('Invalid local edge shape')
        for v in (self.active_actor_mask,self.predictable_actor_mask,self.trajectory_condition_mask):
            if v.shape != (b,a) or v.dtype != torch.bool: raise ValueError('Invalid local actor mask')
        if self.edge_mask.dtype != torch.bool: raise ValueError('Edges must be boolean')
        if not (self.source_slot_ids[:,0] == -1).all() or not self.active_actor_mask[:,0].all():
            raise ValueError('Ego must be active slot0')
        active = self.active_actor_mask
        if (self.predictable_actor_mask & ~active).any() or (self.trajectory_condition_mask & ~self.predictable_actor_mask).any():
            raise ValueError('Invalid trajectory eligibility')
        if (self.edge_mask & ~(active[:,:,None]&active[:,None,:])).any():
            raise ValueError('Inactive node cannot have communication edges')
        if not self.edge_mask.diagonal(dim1=1,dim2=2)[active].all():
            raise ValueError('Active nodes need a self path')
        if (self.source_slot_ids[active] < -1).any() or (self.source_slot_ids[~active] != -2).any():
            raise ValueError('Invalid source identity/padding')
        for i in range(b):
            ids = self.source_slot_ids[i,active[i]]
            if len(ids.unique()) != len(ids):raise ValueError('Duplicate source node')
        if self.context_only_mask.shape != self.context_only_source_ids.shape:
            raise ValueError('Invalid risk context shape')
        if not torch.isfinite(self.edge_features[self.edge_mask]).all():
            raise ValueError('Nonfinite active edge features')
        return self

    def to(self, device):
        return type(self)(**{f.name: getattr(self,f.name).to(device) if torch.is_tensor(getattr(self,f.name))
                            else getattr(self,f.name) for f in fields(self)})

    def tensor_state(self):
        return {f.name:getattr(self,f.name) for f in fields(self)}


def stack_graphs(graphs, device='cpu'):
    if not graphs:raise ValueError('Empty graph batch')
    return LocalInteractionGraph(**{
        f.name:torch.cat([getattr(g,f.name) for g in graphs],0).to(device)
        if torch.is_tensor(getattr(graphs[0],f.name)) else sum([getattr(g,f.name) for g in graphs],[])
        for f in fields(LocalInteractionGraph)}).validate()


def rule_identity(config):
    return hashlib.sha256(json.dumps(config,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def corridor_relations(boxes, classes, observation, config):
    """Heading/speed-range geometry, not observed neighbor velocity or a lane-map claim."""
    x,y=boxes[:,:2].T
    extent=float(np.clip(8+observation.ego_speed_mps*config['ego_horizon_s'],config['min_forward_m'],config['max_forward_m']))
    half=config['corridor_half_width_m']+config['corridor_uncertainty_slope']*np.maximum(x,0)
    in_x=(x>=-config['backward_margin_m'])&(x<=extent)
    straight=in_x&(np.abs(y)<=half+boxes[:,4]/2)
    sign={'left':1,'right':-1,'straight':0,'unknown':0}[observation.navigation]
    curve=sign*np.minimum(config['turn_curvature']*np.maximum(x,0)**2,16)
    turn=in_x&(sign!=0)&(np.abs(y-curve)<=half+boxes[:,4]/2+2)
    sine,cosine=boxes[:,6],boxes[:,7]
    same=np.abs(sine)<.5
    merge=in_x&same&(np.abs(y)<=6+boxes[:,4]/2)
    reach=np.asarray(config['unknown_speed_upper_mps'])[classes]*config['unknown_neighbor_reach_s']
    # Speeds are UNKNOWN. Both heading directions and a lateral envelope are retained.
    lateral=np.abs(sine)*reach+np.minimum(reach,.8*config['unknown_neighbor_reach_s'])
    crossing=in_x&(np.abs(y)<=half+lateral+boxes[:,4]/2)&((np.abs(sine)>.35)|(classes==1)|(classes==2))
    if observation.navigation=='unknown':
        turn=in_x&(np.abs(y)<=half+np.minimum(config['turn_curvature']*np.maximum(x,0)**2,10))
    relevant=straight|turn|merge|crossing
    kinds=np.where(straight,'corridor',np.where(turn,'navigation_turn',np.where(merge,'merge',np.where(crossing,'crossing','weak'))))
    distance=np.linalg.norm(boxes[:,:2],axis=-1)
    # Separate current-decision proxy, never multiplied with existence/visibility.
    scores=np.where(relevant,1/(1+distance/20),0)
    return relevant,kinds,scores,extent


def relation_between(a,b,config):
    delta=b[:2]-a[:2];dist=float(np.linalg.norm(delta))
    ha=a[[7,6]];hb=b[[7,6]];alignment=float(ha@hb)
    longitudinal=float(delta@ha);lateral=abs(float(delta@np.array([-ha[1],ha[0]])))
    if alignment>.7 and lateral<config['following_lateral_m'] and dist<config['following_gap_m']:
        return 'following',float(longitudinal>0),float(longitudinal<0)
    # Potential conflict of bounded current heading envelopes; no claim of observed intent.
    if dist<config['neighbor_conflict_radius_m'] and abs(alignment)<.8:
        matrix=np.stack([ha,-hb],-1)
        if abs(np.linalg.det(matrix))>.15:
            times=np.linalg.solve(matrix,delta)
            if np.abs(times).max()<config['following_gap_m']:
                return 'crossing_chain',1.,1.
    return None


def build_local_graph(boxes, logits, observation, config, evidence_2d=None):
    """Deployment entry: the signature has no labels, Scene, GT count or future mask."""
    if config['schema_version']!=CONTRACT_VERSION or config['graph_variant'] not in ('relations','nearest'):
        raise ValueError('Unrecognized local graph contract')
    cap=config['max_non_ego'];direct_cap=config['max_direct_neighbors']
    if not 0<=direct_cap<=cap<=16:raise ValueError('Unsupported local capacity')
    boxes=np.asarray(boxes,dtype=np.float64);obs=measure_observability(boxes,logits,observation,config,evidence_2d)
    n=len(boxes);safe=np.where(np.isfinite(boxes),boxes,0)
    relevant,kinds,relevance,extent=corridor_relations(safe,obs['entity_class'],observation,config)
    exists=obs['object_wins']&(obs['existence_score']>=config['min_existence'])
    plausible=obs['geometry_plausible'];duplicate=np.full(n,-1,dtype=np.int64)
    order=sorted(range(n),key=lambda i:(-float(obs['existence_score'][i]),-int(obs['adequate_views'][i]),i))
    kept=[]
    for i in order:
        if not plausible[i] or obs['existence_score'][i]<config['risk_min_existence']:continue
        for j in kept:
            if (obs['entity_class'][i]==obs['entity_class'][j] and np.linalg.norm(safe[i,:2]-safe[j,:2])<config['duplicate_radius_m']
                    and abs(safe[i,6:8]@safe[j,6:8])>.8):
                duplicate[i]=j;break
        if duplicate[i]<0:kept.append(i)
    invalid=(~plausible)|(obs['existence_score']<config['risk_min_existence'])|(duplicate>=0)
    dynamic_class=np.isin(obs['entity_class'],config['predictable_classes'])
    eligible=(~invalid)&exists&obs['reliable']&dynamic_class
    distance=np.linalg.norm(safe[:,:2],axis=-1)
    rank=lambda i:(-float(relevance[i]),-int(obs['adequate_views'][i]),-float(obs['existence_score'][i]),float(distance[i]),i)
    selected=[];direct=[];second=[];relation_counts={};edges={}
    if config['graph_variant']=='relations':
        candidates=sorted(np.where(eligible&relevant)[0].tolist(),key=rank)
        # Reserve type diversity before filling unused capacity, without forcing any node.
        for i in candidates:
            kind=str(kinds[i])
            if len(direct)<direct_cap and relation_counts.get(kind,0)<config['max_per_relation_initial']:
                direct.append(i);relation_counts[kind]=relation_counts.get(kind,0)+1
        for i in candidates:
            if i not in direct and len(direct)<direct_cap:direct.append(i)
        selected=direct.copy()
        for i in sorted(np.where(eligible)[0].tolist(),key=rank):
            if i in selected or len(selected)>=cap:continue
            if any(relation_between(safe[j],safe[i],config) for j in direct):
                selected.append(i);second.append(i)
    else:
        candidates=[i for i in np.where(eligible)[0] if distance[i]<=config['max_forward_m']]
        selected=sorted(candidates,key=lambda i:(float(distance[i]),i))[:cap];direct=selected.copy()
    for i in direct:edges[(-1,i)]=(str(kinds[i]) if config['graph_variant']=='relations' else 'nearest',1.,1.)
    for ii,i in enumerate(selected):
        for j in selected[ii+1:]:
            relation=relation_between(safe[i],safe[j],config) if config['graph_variant']=='relations' else None
            if config['graph_variant']=='nearest':
                close=sorted((k for k in selected if k!=i),key=lambda k:(float(np.linalg.norm(safe[i,:2]-safe[k,:2])),k))[:2]
                if j in close:relation=('nearest',1.,1.)
            if relation:edges[(i,j)]=relation
    # Capacity-excluded relevant hypotheses remain uncertain context, not free space.
    risks=[i for i in range(n) if not invalid[i] and relevant[i] and i not in selected]
    source=np.full(cap+1,-2,np.int64);source[0]=-1;source[1:len(selected)+1]=selected
    active=source!=-2;edge=np.zeros((cap+1,cap+1),bool);features=np.zeros((cap+1,cap+1,8),np.float32)
    mapping={int(s):l for l,s in enumerate(source) if s!=-2}
    codes={'corridor':1,'navigation_turn':2,'merge':3,'crossing':4,'following':5,'crossing_chain':6,'nearest':7,'weak':8}
    def pos(i):return np.zeros(2) if i==-1 else safe[i,:2]
    def heading(i):return np.array([1.,0.]) if i==-1 else safe[i,[7,6]]
    for (i,j),(kind,ij,ji) in edges.items():
        for a,b,incoming,outgoing in ((i,j,ij,ji),(j,i,ji,ij)):
            ai,bi=mapping[a],mapping[b];edge[ai,bi]=True;delta=pos(b)-pos(a)
            features[ai,bi]=[*list(delta/50),float(np.linalg.norm(delta))/50,float(heading(a)@heading(b)),incoming,outgoing,float(i==-1 or j==-1),codes[kind]/8]
    for i in np.where(active)[0]:edge[i,i]=True
    risk_ids=np.full(n,-2,np.int64);risk_ids[:len(risks)]=risks
    records=[];selection=[];exclusion=[];risk_records=[]
    for i in range(n):
        if invalid[i]:group='D';reason='duplicate' if duplicate[i]>=0 else 'invalid_geometry_or_existence'
        elif i in selected:group='A';reason='second_hop' if i in second else 'direct_'+str(kinds[i])
        elif i in risks:
            group='B';reason='static_or_unknown_context' if not dynamic_class[i] else ('capacity_risk_context' if eligible[i] else 'insufficient_current_evidence')
        else:group='C';reason='weak_current_decision_relation'
        r={'source_slot_id':i,'local_slot':mapping.get(i),'group':group,'reason':reason,'entity_type':config['class_names'][int(obs['entity_class'][i])],
           'entity_class':int(obs['entity_class'][i]),'existence_score':float(obs['existence_score'][i]),'object_wins':bool(obs['object_wins'][i]),
           'observation_support':obs['observation_support'][i].tolist(),'visual_reliability':float(obs['visual_reliability'][i]),
           'relevance_to_ego':float(relevance[i]),'decision_related':bool(relevant[i]),'relation_kind':str(kinds[i]),
           'geometry_plausible':bool(plausible[i]),'reliable_proxy':bool(obs['reliable'][i]),'support_views':int(obs['support_views'][i]),
           'adequate_views':int(obs['adequate_views'][i]),'pixel_area':obs['pixel_area'][i].tolist(),'pixel_height':obs['pixel_height'][i].tolist(),
           'truncation':obs['truncation'][i].tolist(),'projection_legal':obs['projection_legal'][i].tolist(),
           'projected_boxes':obs['projected_boxes'][i].tolist(),'current_2d_available':obs['current_2d_available'][i].tolist(),
           'current_2d_iou':[float(x) if np.isfinite(x) else None for x in obs['current_2d_iou'][i]],
           'duplicate_of':int(duplicate[i]) if duplicate[i]>=0 else None,'speed_known':False,'speed_mps':None,
           'distance_m':float(distance[i]),'confirmed_free_space':False}
        records.append(r)
        (selection if group=='A' else exclusion).append({'source_slot_id':i,'reason':reason})
        if group=='B':risk_records.append(r)
    rel=np.zeros(cap+1,np.float32);rel[0]=1;visual=rel.copy()
    for i in selected:rel[mapping[i]]=relevance[i];visual[mapping[i]]=obs['visual_reliability'][i]
    provenance={'schema_version':CONTRACT_VERSION,'selector_identity':rule_identity(config),'config':config,
                'observation_identity':observation.fingerprint(),'token':observation.token,'targets_read':False,
                'current_ego_speed_mps':observation.ego_speed_mps,'navigation':observation.navigation,
                'corridor_extent_m':extent,'neighbor_speeds':'UNKNOWN; conservative heading envelopes, not zero speed',
                'direct_count':len(direct),'second_hop_count':len(second),'risk_context_count':len(risks),
                'candidate_overflow':int((eligible&relevant).sum())-len([i for i in selected if relevant[i]]),
                'edge_semantics':'True[i,j]:query i reads key j; directional features are geometric influence hypotheses, not causal directions',
                'visual_reliability_semantics':'uncalibrated geometric evidence proxy; no occlusion model or current2D detector unless explicitly provided'}
    ten=lambda x:torch.as_tensor(x)[None]
    return LocalInteractionGraph(ten(source),ten(active),ten(active.copy()),ten(active.copy()),ten(edge),ten(features),ten(visual),ten(rel),
                                 ten(risk_ids),ten(risk_ids>=0),[risk_records],[selection],[exclusion],[records],[provenance]).validate()


def gather_current(current, graph):
    """Pack selected actors; keep full original visual context and explicit current risk context."""
    graph.validate();source=graph.source_slot_ids
    index=(source+1).clamp_min(0);active=graph.active_actor_mask
    def take(value,idx,mask):
        shape=[*idx.shape]+list(value.shape[2:]);ix=idx.reshape(*idx.shape,*([1]*(value.ndim-2))).expand(shape)
        selected=value.gather(1,ix)
        return torch.where(mask.reshape(*mask.shape,*([1]*(value.ndim-2))),selected,torch.zeros_like(selected))
    out={k:take(current[k],index,active) for k in ('actor_features','current_xy','existence')}
    risks=take(current['actor_features'],(graph.context_only_source_ids+1).clamp_min(0),graph.context_only_mask)
    out['context']=torch.cat([current['context'],risks],1)
    out['context_mask']=torch.cat([torch.ones(current['context'].shape[:2],device=source.device,dtype=torch.bool),graph.context_only_mask],1)
    out['local_graph']=graph
    return out

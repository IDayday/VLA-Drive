"""Private planning reader and spatial scoring readout after a frozen VLA."""
from copy import deepcopy
import torch
from torch import nn
from iqe.contracts import FeatureBundle, require
from iqe.io import read_json, file_hash
from iqe.query_base import QueryActionBridge
from iqe.internal_baseline.scorer import InternalScorer, initialization_rng


class Specialist(nn.Module):
    def __init__(self, bridge, *, private_reader, contract_hash, planner_mode):
        super().__init__()
        self.private_reader=private_reader; self.contract_hash=contract_hash; self.planner_mode=planner_mode
        self.expert=deepcopy(bridge.expert)
        if private_reader:
            self.q_former=deepcopy(bridge.q_former)
            self.scene_queries=nn.Parameter(bridge.scene_queries.detach().clone())
            self.ego_encoder=deepcopy(bridge.ego_encoder)
        self.trajectory_contract=bridge.trajectory_contract

    def forward(self,x):
        if self.private_reader:
            action=x['action'].detach().float()
            if self.planner_mode=='action_plus_W': action=torch.cat((action,x['world'].detach().float()),1)
            elif self.planner_mode!='action_only': raise ValueError('unknown original planner input')
            scene=self.q_former(self.scene_queries,action)
            ego=self.ego_encoder(x['current_ego'].detach().float())[:,None]
            f=FeatureBundle(scene,ego,torch.ones(scene.shape[:2],device=scene.device,dtype=torch.bool),self.contract_hash,tuple(map(str,range(len(scene)))))
        else:
            f=FeatureBundle(x['planner_scene'].detach(),x['planner_ego'].detach(),x['planner_mask'],self.contract_hash,tuple(map(str,range(len(x['planner_scene'])))))
        return self.expert(f)


def load_bridge(config):
    from omegaconf import OmegaConf
    c=read_json(config.root/'OFFLINE_BASE_CONTRACT.json')
    with initialization_rng(config.seed):
        bridge=QueryActionBridge(OmegaConf.create(c['source_config']),c['query_architecture'])
    # Reuse a small, source-bound copy of the complete planner branch; no VLM GPU allocation.
    path=config.root/'BASE_PLANNER.pt'; receipt=path.with_suffix('.json')
    if path.exists():
        recorded=read_json(receipt)
        require(file_hash(path)==recorded['checksum'],'Base planner copy corrupted')
        require(recorded['parent_checkpoint_hash']==read_json(config.root/'BASE_BINDING.json')['checkpoint_hash'],
                'Base planner copy belongs to another checkpoint')
        state=torch.load(path,weights_only=True,map_location='cpu')
    else:
        from iqe.io import atomic_torch, atomic_json
        b=read_json(config.root/'BASE_BINDING.json')
        full=torch.load(b['checkpoint'],weights_only=False,map_location='cpu',mmap=True)
        state={k.removeprefix('action_model.'):v.clone() for k,v in full['model'].items() if k.startswith('action_model.')}
        atomic_torch(path,state,immutable=True)
        atomic_json(receipt,{'checksum':file_hash(path),'parent_checkpoint_hash':b['checkpoint_hash']},immutable=True)
    bridge.load_state_dict(state,strict=True)
    return bridge


def make_specialist(config, private_reader):
    c=read_json(config.root/'OFFLINE_BASE_CONTRACT.json')
    return Specialist(load_bridge(config),private_reader=private_reader,
                      contract_hash=read_json(config.root/'BASE_BINDING.json')['contract_hash'],
                      planner_mode=c['source_config']['foresight']['planner_condition_mode'])


class EMAExpert(nn.Module):
    """FP32 EMA belongs to the same specialist checkpoint, not a second candidate."""
    def __init__(self, raw, decay=.999):
        super().__init__(); self.raw=raw; self.ema=deepcopy(raw).requires_grad_(False).eval(); self.decay=decay
        self.register_buffer('ema_updates',torch.zeros((),dtype=torch.long))
    def train(self,mode=True):
        super().train(mode);self.ema.eval();return self
    def forward(self,x):return self.raw(x)
    @torch.no_grad()
    def update(self):
        for p,q in zip(self.ema.parameters(),self.raw.parameters()):p.lerp_(q.detach().float(),1-self.decay)
        for a,b in zip(self.ema.buffers(),self.raw.buffers()):a.copy_(b)
        self.ema_updates.add_(1)


class SpatialReader(nn.Module):
    def __init__(self,visual_dim,views=3,hidden=256,queries=16):
        super().__init__();self.views=views;self.hidden=hidden
        self.projection=nn.Sequential(nn.LayerNorm(visual_dim),nn.Linear(visual_dim,hidden))
        self.queries=nn.Parameter(torch.randn(views,queries,hidden)*.02)
        self.camera=nn.Parameter(torch.randn(views,hidden)*.02)
        self.position=nn.Linear(4,hidden,bias=False)
        self.layer=nn.TransformerDecoderLayer(hidden,4,hidden*4,dropout=0.,activation='gelu',batch_first=True,norm_first=True)
        self.norm=nn.LayerNorm(hidden)
    def forward(self,visual,valid,xy):
        require(visual.ndim==4 and visual.shape[1]==self.views and valid.shape==visual.shape[:3], 'spatial view/mask contract')
        require(xy.shape==(*visual.shape[:3],2) and valid.any(-1).all().item(),'spatial 2D layout/empty view')
        b,v,n,_=visual.shape
        p=xy.detach().float()*torch.pi
        pos=torch.cat((p.sin(),p.cos()),-1)
        memory=self.projection(visual.detach().float())+self.position(pos)+self.camera[None,:,None]
        q=self.queries[None].expand(b,-1,-1,-1).reshape(b*v,-1,self.hidden)
        z=self.layer(q,memory.reshape(b*v,n,-1),memory_key_padding_mask=(~valid).reshape(b*v,n))
        return self.norm(z).reshape(b,v*z.shape[1],self.hidden)


class StudyScorer(InternalScorer):
    def __init__(self,world_dim,action_dim,trajectory_contract,*,spatial=False,visual_dim=None,views=3):
        # Instantiate common modules first: same seed gives identical common initialization.
        super().__init__(world_dim,action_dim,trajectory_contract,dropout=0.)
        self.spatial=spatial
        if spatial:
            require(visual_dim is not None,'actual visual dimension required')
            self.spatial_reader=SpatialReader(visual_dim,views)
    def project_context(self,context):
        standard={k:context[k] for k in ('world','action','current_ego','world_valid','action_valid')}
        allowed=set(standard)|({'visual','visual_valid','visual_xy'} if self.spatial else set())
        require(set(context)==allowed,'study Scorer deployment whitelist')
        memory,mask=super().project_context(standard)
        if self.spatial:
            extra=self.spatial_reader(context['visual'],context['visual_valid'],context['visual_xy'])
            memory=torch.cat((memory,extra),1);mask=torch.cat((mask,torch.ones(extra.shape[:2],device=mask.device,dtype=torch.bool)),1)
        return memory,mask

"""Single scene Flow-DiT. The sampled ego slot is the only executed ego."""
import torch
from torch import nn
from starVLA.model.modules.action_model.flow_matching_head.action_encoder import ActionEncoder
from starVLA.model.modules.joint_world.flow import GraphBlock


class JointSceneFlow(nn.Module):
    def __init__(self,dim=384,heads=6,layers=6,steps=8,condition_dim=2048,xy_scale=20.):
        super().__init__()
        if dim%heads or dim%2 or steps<1 or xy_scale<=0:raise ValueError('Invalid JointSceneFlow dimensions')
        self.config=dict(dim=dim,heads=heads,layers=layers,steps=steps,condition_dim=condition_dim,xy_scale=xy_scale)
        self.steps,self.xy_scale=steps,xy_scale
        # Reuse the original FM action/time encoder, extended over the actor axis.
        self.state=ActionEncoder(4,dim)
        self.current=nn.Sequential(nn.Linear(9,dim),nn.SiLU(),nn.Linear(dim,dim))
        self.kind=nn.Embedding(8,dim);self.role=nn.Embedding(2,dim)
        self.ego=nn.Sequential(nn.Linear(7,dim),nn.SiLU(),nn.Linear(dim,dim))
        self.known=nn.Linear(8,dim) # feature-wise clean known values and mask
        self.horizon=nn.Embedding(steps,dim)
        self.condition=nn.Sequential(nn.LayerNorm(condition_dim),nn.Linear(condition_dim,dim))
        self.blocks=nn.ModuleList(GraphBlock(dim,heads,8) for _ in range(layers))
        self.time_modulation=nn.ModuleList(nn.Sequential(nn.Linear(3,dim),nn.SiLU(),nn.Linear(dim,dim*2)) for _ in range(layers))
        self.velocity=nn.Sequential(nn.LayerNorm(dim),nn.Linear(dim,4))
        self.register_buffer('box_scale',torch.tensor([50.,50.,3.,5.,3.,2.,1.,1.]))

    def encode(self,trajectory,graph):
        # Current-state anchoring; no future endpoint/statistic and no detach.
        xy=(trajectory[...,:2]-graph.boxes[:,:,:2,None].transpose(-1,-2))/self.xy_scale
        return torch.cat([xy,trajectory[...,2:]],-1)

    def decode(self,state,graph):
        xy=state[...,:2]*self.xy_scale+graph.boxes[:,:,:2,None].transpose(-1,-2)
        return torch.cat([xy,state[...,2:]],-1)

    def forward(self,noisy,time,graph,known=None,known_mask=None,actor_features=None,scene_features=None,scene_mask=None):
        graph.validate();b,a,t,d=noisy.shape
        if noisy.shape!=(b,a,self.steps,4) or graph.boxes.shape[:2]!=(b,a) or time.shape!=(b,):raise ValueError('Joint state dimensions mismatch')
        active=graph.active_actor_mask
        if (known is None)!=(known_mask is None):raise ValueError('Known values and feature masks must be paired')
        if known is None:known=torch.zeros_like(noisy);known_mask=torch.zeros_like(noisy,dtype=torch.bool)
        if known.shape!=noisy.shape or known_mask.shape!=noisy.shape:raise ValueError('Known feature mask mismatch')
        if (known_mask & ~active[:,:,None,None]).any():raise ValueError('Padding cannot condition trajectories')
        # Hidden values cleared BEFORE subtraction, projection or attention.
        visible=torch.where(known_mask,known,0.)
        encoded=torch.where(known_mask,self.encode(visible,graph),0.)
        noisy=torch.where(active[:,:,None,None],noisy,0.)
        boxes=torch.where(active[...,None],graph.boxes,0.)
        current=torch.cat([boxes/self.box_scale,graph.geometric_support[...,None]],-1)
        roles=torch.ones(a,dtype=torch.long,device=noisy.device);roles[0]=0
        x=self.state(noisy.reshape(b*a,t,4),time[:,None].expand(b,a).reshape(-1)).reshape(b,a,t,-1)
        actor=self.current(current)+self.kind(graph.classes)+self.role(roles)[None]
        if actor_features is not None:
            if actor_features.shape[:2]!=(b,a):raise ValueError('Actor feature order differs from joint slots')
            actor=actor+self.condition(torch.where(active[...,None],actor_features,0.))
        x=x+actor[:,:,None]+self.known(torch.cat([encoded,known_mask.to(noisy.dtype)],-1))+self.horizon.weight[None,None]
        ego=self.ego(graph.ego_state)
        risk=self.current(torch.cat([graph.context_boxes/self.box_scale,graph.context_mask[...,None].to(noisy.dtype)],-1))+self.kind(graph.context_classes)
        memory=torch.cat([ego[:,None],risk],1)
        mask=torch.cat([torch.ones(b,1,device=noisy.device,dtype=torch.bool),graph.context_mask],1)
        if scene_features is not None:
            if scene_mask is None:scene_mask=torch.ones(scene_features.shape[:2],device=noisy.device,dtype=torch.bool)
            memory=torch.cat([memory,self.condition(torch.where(scene_mask[...,None],scene_features,0.))],1)
            mask=torch.cat([mask,scene_mask],1)
        te=torch.stack([time,torch.sin(torch.pi*time),torch.cos(torch.pi*time)],-1)
        for block,mod in zip(self.blocks,self.time_modulation):
            shift,scale=mod(te).chunk(2,-1)
            x=x*(1+.1*scale[:,None,None])+.1*shift[:,None,None]
            x=block(x,boxes[...,:2]/self.xy_scale,memory,graph,mask)
        return torch.where(active[:,:,None,None],self.velocity(x),0.)

    def sample(self,noise,graph,sampling_steps=20,**current_features):
        # Deployment entry exposes no future or label-valid parameters.
        if set(current_features)-{'actor_features','scene_features','scene_mask'}:raise ValueError('Deployment only accepts current visual features')
        return self.sample_conditional(noise,graph,sampling_steps=sampling_steps,**current_features)

    def sample_conditional(self,noise,graph,known=None,known_mask=None,sampling_steps=20,return_path=False,**current_features):
        if sampling_steps<1:raise ValueError('Positive integration steps required')
        if (known is None)!=(known_mask is None):raise ValueError('Known condition pair missing')
        if known is None:known=torch.zeros_like(noise);known_mask=torch.zeros_like(noise,dtype=torch.bool)
        visible=torch.where(known_mask,known,0.)
        clean=torch.where(known_mask,self.encode(visible,graph),0.)
        active=graph.active_actor_mask[:,:,None,None]
        x=torch.where(active,torch.where(known_mask,clean,noise),0.)
        def result(v):return torch.where(active,torch.where(known_mask,visible,self.decode(v,graph)),0.)
        path=[result(x)] if return_path else None
        for i in range(sampling_steps):
            time=x.new_full((len(x),),i/sampling_steps)
            velocity=self(x,time,graph,visible,known_mask,**current_features)
            x=torch.where(active,torch.where(known_mask,clean,x+velocity/sampling_steps),0.)
            if return_path:path.append(result(x))
        joint=result(x)
        return (joint,path) if return_path else joint

    def predict_action(self,noise,graph,sampling_steps=20,**current_features):
        if graph.origin!='predicted_inference':raise ValueError('Camera deployment cannot use the supervision graph')
        joint=self.sample(noise,graph,sampling_steps,**current_features)
        return execution_from_joint(joint)


def execution_from_joint(joint):
    """One explicit output conversion: ego(t0) xy + atan2(sin yaw,cos yaw)."""
    ego=joint[:,0]
    executed=torch.cat([ego[...,:2],torch.atan2(ego[...,2],ego[...,3])[...,None]],-1)
    if not torch.equal(executed[...,:2],joint[:,0,:,:2]):raise RuntimeError('Executed ego was rewritten')
    return {'joint_trajectories':joint,'ego_xy_sincos':ego,'executed_ego_xyyaw':executed}


def flow_loss_sums(model,target,valid,hidden,noise,time,graph,**features):
    if target.shape!=valid.shape or target.shape!=noise.shape or hidden.shape!=target.shape[:2]:raise ValueError('Joint loss contract mismatch')
    valid=valid&graph.active_actor_mask[:,:,None,None]
    safe=torch.where(valid,target,0.)
    known_mask=valid&~hidden[:,:,None,None]
    clean=torch.where(valid,model.encode(safe,graph),noise)
    noisy=noise+(clean-noise)*time[:,None,None,None]
    noisy=torch.where(known_mask,clean,noisy)
    velocity=model(noisy,time,graph,safe,known_mask,**features)
    expected=clean-noise
    error=(velocity-expected).square()
    selected=valid&hidden[:,:,None,None]
    sums={};counts={}
    for key,sl,channels in [('ego_xy',slice(0,1),slice(0,2)),('ego_yaw',slice(0,1),slice(2,4)),('neighbor_xy',slice(1,None),slice(0,2)),('neighbor_yaw',slice(1,None),slice(2,4))]:
        take=selected[:,sl,:,channels];values=error[:,sl,:,channels]
        sums[key]=values[take].sum();counts[key]=int(take.sum())
    return sums,counts


def normalized_loss(sums,counts):
    # Equal ego/neighbor XY means; yaw auxiliary uses the same feature scale across arms.
    return sum(sums[k]/max(counts[k],1)*(.25 if k.endswith('yaw') else 1.) for k in sums)

"""Cached-current planning uses the original DiT and the online policy's exact bridge."""
import hashlib
from pathlib import Path
import pickle

import numpy as np
from omegaconf import OmegaConf
import torch
from torch import nn

from starVLA.model.modules.action_model.GR00T_ActionHeader import FlowmatchingActionHead
from starVLA.model.modules.joint_world.flow import JointTrajectoryFlow
from starVLA.model.modules.joint_world.policy import JointTrajectoryPolicy
from starVLA.model.modules.structured_world.action_adapter import WorldToActionAdapter


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def load_original_head(checkpoint, expected_sha):
    """No Qwen construction or downloads; every action-head tensor must match exactly."""
    root = Path(checkpoint); path = root / 'pytorch_model.pt'
    if file_sha256(path) != expected_sha: raise ValueError('Original checkpoint hash mismatch')
    config = OmegaConf.load(root / 'config.yaml')
    if config.ver_1225 != 1 or config.framework.action_model.mlp_head != 0:
        raise ValueError('Only the validated absolute-XY/sincos FM baseline is supported')
    c = config.framework.action_model
    # Identical derived configuration to QwenOFT.__init__, not a replacement architecture.
    c.DiTConfig = {'num_layers': c.diffusion_model_cfg.num_layers,
                  'input_embedding_dim': c.hidden_size,
                  'attention_head_dim': 64, 'num_attention_heads': c.hidden_size // 64}
    head = FlowmatchingActionHead(config)
    state = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    selected = {k[len('action_model.'):]: v for k, v in state.items() if k.startswith('action_model.')}
    head.load_state_dict(selected, strict=True)
    if set(selected) != set(head.state_dict()): raise ValueError('Incomplete original action head')
    return head.cuda().eval().requires_grad_(False), config


class CachedCurrentPlanner(nn.Module):
    """Same module names/rollout as JointTrajectoryPolicy; only fixed upstream is cached."""
    def __init__(self, condition_dim, graph_config, bev_enabled=False):
        super().__init__(); self.graph_config = dict(graph_config)
        self.graph = JointTrajectoryFlow(condition_dim, **{k:v for k,v in graph_config.items()
            if k in ('dim','heads','layers','scale_m','trajectory_mode','agent_scale_m')})
        self.graph_to_world = nn.Linear(graph_config['dim'], condition_dim)
        self.adapter = WorldToActionAdapter(condition_dim)
        self.bev_enabled=bev_enabled
        if bev_enabled:
            from starVLA.model.modules.joint_world.bev_tasks import TaskBEVEncoder,BEVGraphFusion,InteractionGeometryHead
            self.bev_encoder=TaskBEVEncoder()
            self.bev_fusion=BEVGraphFusion(condition_dim)
            self.interaction_head=InteractionGeometryHead()

    # Share the deployment implementation, including precision and all-hidden sampling.
    rollout_condition = JointTrajectoryPolicy.rollout_condition
    fuse_bev = JointTrajectoryPolicy.fuse_bev

    def condition_from_frozen_graph(self, native_actions, current, noise):
        if any(p.requires_grad for p in self.graph.parameters()):
            raise ValueError('Frozen-representation probe must not detach a trainable graph')
        with torch.no_grad():
            trajectories, features = self.graph.sample(noise.float(), **current,
                sampling_steps=self.graph_config.get('sampling_steps',10))
        condition = self.adapter(native_actions.float(), self.graph_to_world(features.float()))
        return condition.float(), trajectories


def graph_noise(batch, actors, steps, device, seed=2037):
    # Repeat the online policy's per-scene noise, independently of batch shape/order.
    generator = torch.Generator(device=device).manual_seed(seed)
    one = torch.randn(1, actors, steps, 2, device=device, generator=generator)
    return one.expand(batch,-1,-1,-1).contiguous()


def ego_action_target(token, data_root, act_norm):
    """Exact released NAVSIM ver1225 action contract, loaded only on the label side."""
    from starVLA.dataloader.navsim_dataset import (StateSE2, absolute_to_relative_poses,
        wrap_to_pi, x_mean, x_std, y_mean, y_std)
    with open(Path(data_root)/'meta/train'/(token+'.pkl'),'rb') as f: meta=pickle.load(f)
    poses=meta['glo_status']['global_poses']
    if len(poses)<12: raise ValueError('Missing ego future action labels')
    states=[StateSE2(float(x),float(y),float(yaw)) for x,y,yaw in poses[:12]]
    relative=np.array([[s.x,s.y,s.heading] for s in absolute_to_relative_poses(states,3)])
    if not act_norm:relative[:,0]/=4.5912
    dxy=(relative[4:,:2]-relative[3:4,:2]).astype(np.float32)
    if act_norm:
        dxy[:,0]=(dxy[:,0]-x_mean)/x_std;dxy[:,1]=(dxy[:,1]-y_mean)/y_std
    yaw=wrap_to_pi(relative[4:,2]-relative[3:4,2])
    return torch.from_numpy(np.concatenate([dxy,np.stack([np.sin(yaw),np.cos(yaw)],-1).astype(np.float32)],-1))


@torch.no_grad()
def predict(head, planner, sample, seed=20260926, disable_graph=False):
    from tools.joint_world.train_graph import current_batch
    cache=sample['cache']; native=cache['native_actions'].cuda(); current=current_batch([sample])
    if planner.bev_enabled:
        from tools.joint_world.bev_cache import batch_bev
        bev=batch_bev([sample]);current,_=planner.fuse_bev(current,bev['features'],bev['coordinates'],bev['observation_support'])
    noise=graph_noise(1,current['actor_features'].shape[1],planner.graph.steps,native.device,
                      planner.graph_config.get('sampling_seed',2037))
    condition, joint, _=planner.rollout_condition(native,current,noise)
    if disable_graph:condition=native.float()
    # Match original per-scene BF16 ego noise; graph generation uses its own RNG.
    value=int.from_bytes(hashlib.sha256(f'{seed}:{cache["token"]}'.encode()).digest()[:4],'little')
    gen=torch.Generator(device=native.device).manual_seed(value)
    ego_noise=torch.randn(1,head.config.action_horizon,head.config.action_dim,
                          device=native.device,dtype=native.dtype,generator=gen)
    with torch.autocast('cuda',dtype=torch.float32):action=head.predict_action(condition,initial_noise=ego_noise)
    return action.cpu().numpy(),joint.cpu().numpy()

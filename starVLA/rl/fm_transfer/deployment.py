"""Source ActionDiT under the original Qwen/VLAAgent observation entry.

Only a velocity head is deployed: no training cache, scorer, teacher or LoRA
wrapper. The action encoding/decoding remains the original DDP eight-by-four.
"""
import torch
from torch import nn
from .model import TransferPolicy, SOURCE_ACTION_CONFIG
from .vendor.action_dit import ActionDiT


class SourceActionHead(nn.Module):
    def __init__(self, full_config):
        super().__init__()
        cfg=full_config.framework.action_model
        if cfg.action_dim!=4 or cfg.action_horizon!=8 or cfg.num_inference_timesteps!=10:
            raise ValueError('Source-head transfer keeps original 8x4 / ten-step deployment')
        if full_config.framework.qwenvl.vl_hidden_dim!=2048:
            raise ValueError('Qwen condition width differs from the locked base')
        self.policy=TransferPolicy(ActionDiT(**SOURCE_ACTION_CONFIG),'source_action_dit')
        self.action_dim=4;self.action_horizon=8;self.num_inference_timesteps=10
        self.config=cfg

    def _project_condition(self, raw, global_scene_tokens=None, video_token=None):
        if global_scene_tokens is not None or video_token is not None:
            raise ValueError('New privileged/scene/video conditions are not registered')
        return self.policy.project(raw.float())

    def forward(self, vl_embs, actions, video_token=None, state=None,
                global_scene_tokens=None, explicit_randomness=None):
        raw=self._project_condition(vl_embs,global_scene_tokens,video_token)
        r=explicit_randomness or {}
        sigma=r.get('sigma')
        if 't' in r:
            if sigma is not None:raise ValueError('Ambiguous time convention')
            sigma=1-r['t'].reshape(-1)
        return self.policy(raw,actions,epsilon=r.get('noise'),sigma=sigma,repeat=1)[0]

    @torch.no_grad()
    def _euler_sample(self, actions, projected_condition):
        return self.policy.sample_ode(projected_condition,actions,grid='native')

    @torch.no_grad()
    def sample_shift5(self, actions, projected_condition):
        return self.policy.sample_ode(projected_condition,actions,grid='shift5')

    @torch.no_grad()
    def predict_action(self, vl_embs, state=None, initial_noise=None, **kwargs):
        if kwargs:raise ValueError('Unregistered deployment arguments: '+repr(sorted(kwargs)))
        raw=self._project_condition(vl_embs)
        noise=initial_noise if initial_noise is not None else torch.randn((raw.shape[0],8,4),device=raw.device,dtype=torch.float32)
        return self._euler_sample(noise,raw)

import torch
from torch import nn
from torch.nn import functional as F


class AgentHeads(nn.Module):
    """Input must be post-Qwen agent states; independent of ego action encoding."""
    def __init__(self, qwen_dim, classes=7, steps=8, dim=256):
        super().__init__()
        self.steps = steps
        self.shared = nn.Sequential(nn.LayerNorm(qwen_dim), nn.Linear(qwen_dim,dim), nn.GELU())
        self.classifier = nn.Linear(dim,classes+1)  # final class=no object
        self.box = nn.Linear(dim,8)
        self.motion = nn.Linear(dim,steps*2)

    def forward(self, post_qwen_agents):
        x = self.shared(post_qwen_agents)
        raw = self.box(x)
        boxes = torch.cat([raw[...,:3], F.softplus(raw[...,3:6])+.01, raw[...,6:]],dim=-1)
        # Predict displacement relative to the same slot's current centre.
        future = self.motion(x).reshape(*x.shape[:-1],self.steps,2) + boxes[...,:2].unsqueeze(-2)
        return {'logits':self.classifier(x), 'boxes':boxes, 'future_xy':future}

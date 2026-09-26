import torch
from torch import nn
from .contracts import WorldMemory


class SceneAgentReader(nn.Module):
    def __init__(self, input_dim, qwen_dim, dim=256, scene_tokens=64, agent_tokens=32, layers=2):
        super().__init__()
        self.scene_tokens = scene_tokens
        self.agent_tokens = agent_tokens
        self.project = nn.Linear(input_dim, dim)
        self.position = nn.Sequential(nn.Linear(3, dim), nn.GELU(), nn.Linear(dim, dim))
        self.queries = nn.Parameter(torch.randn(scene_tokens + agent_tokens, dim) * .02)
        self.types = nn.Parameter(torch.randn(2, dim) * .02)
        self.reader = nn.ModuleList([nn.TransformerDecoderLayer(dim, 8, dim*4, dropout=0., batch_first=True, norm_first=True) for _ in range(layers)])
        self.output = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, qwen_dim))
        # A null observation keeps empty FOV samples finite without fabricated evidence.
        self.null = nn.Parameter(torch.zeros(1,1,dim))

    def forward(self, features, coordinates=None, support=None, metadata=None):
        b, n, _ = features.shape
        memory = self.project(features)
        if coordinates is None:
            coordinates = features.new_zeros(b,n,3)
        else:
            memory = memory + self.position(coordinates.to(memory.dtype) / 50.)
        if support is None:
            support = torch.ones(b,n,device=features.device,dtype=torch.bool)
        memory = torch.cat([memory,self.null.expand(b,-1,-1)],dim=1)
        invalid = torch.cat([~support.bool(),torch.zeros(b,1,device=features.device,dtype=torch.bool)],dim=1)
        types = torch.cat([self.types[0:1].expand(self.scene_tokens,-1),self.types[1:2].expand(self.agent_tokens,-1)])
        queries = (self.queries + types).unsqueeze(0).expand(b,-1,-1)
        for layer in self.reader:
            queries = layer(queries, memory, memory_key_padding_mask=invalid)
        output = self.output(queries)
        return WorldMemory(output[:,:self.scene_tokens],output[:,self.scene_tokens:],coordinates,support,metadata or {'provider':'qwen_current_visual'})

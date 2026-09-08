import math
import torch
from torch import nn


def pad_tile_registers(registers, counts, metadata):
    if sum(counts) != len(registers) or metadata.shape != (sum(counts), 5):
        raise ValueError("Per-tile registers/counts/geometry disagree")
    chunks = registers.split(counts)
    geometry = metadata.split(counts)
    content = nn.utils.rnn.pad_sequence([r.flatten(0,1) for r in chunks], batch_first=True)
    position = nn.utils.rnn.pad_sequence([g[:,None,:].expand(-1,16,-1).flatten(0,1) for g in geometry], batch_first=True)
    valid = torch.arange(content.shape[1], device=content.device)[None] < torch.tensor(counts,device=content.device)[:,None]*16
    return content, position, valid


class RichSceneMemory(nn.Module):
    def __init__(self, mode='per_tile_register_memory'):
        super().__init__()
        if mode not in ('per_tile_register_memory', 'compact_memory'):
            raise ValueError(mode)
        self.mode = mode
        self.geometry = nn.Sequential(nn.Linear(5,256), nn.GELU(), nn.Linear(256,256))
        self.planning_norm = nn.LayerNorm(256)
        self.semantic_norm = nn.LayerNorm(256)
        self.semantic_attention = nn.MultiheadAttention(256,8,dropout=0,batch_first=True)
        self.semantic_gate = nn.Parameter(torch.tensor(math.log(.2/.8)))
        self.output_norm = nn.LayerNorm(256)

    def forward(self, content, geometry, valid, semantic):
        if not valid.any(-1).all():
            raise ValueError("Every scene needs at least one visual tile")
        planning = self.planning_norm(content + self.geometry(geometry))
        semantic = self.semantic_norm(semantic)
        context, _ = self.semantic_attention(planning, semantic, semantic, need_weights=False)
        memory = self.output_norm(planning + self.semantic_gate.sigmoid()*context)
        if self.mode == 'compact_memory':
            b,n,d = memory.shape
            slots = memory.reshape(b,n//16,16,d)
            mask = valid.reshape(b,n//16,16)
            memory = (slots*mask[...,None]).sum(1)/mask.sum(1).clamp_min(1)[...,None]
            valid = mask.any(1)
        return memory, valid

"""Shared action-conditioned block-causal TF and differentiable rollout.

Principle transfer: facebookresearch/vjepa2@204698b45b3712590f06245fbfba32d3be539812.
V2 engineering: register slots are NOT a patch grid; time, geometry and slot encodings are separate.
"""
import math
import torch
import torch.nn.functional as F
from torch import nn
from .motion import IntervalMotionEncoder, HORIZONS


def state_norm(x):
    return F.layer_norm(x.float(),(x.shape[-1],),weight=None,bias=None)


class CausalBlock(nn.Module):
    def __init__(self,dim,heads,ffn,layer_index):
        super().__init__()
        self.norm1,self.norm2 = nn.LayerNorm(dim),nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(dim,heads,dropout=0,batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(dim,ffn),nn.GELU(),nn.Linear(ffn,dim))
        for module in self.modules():
            if isinstance(module,nn.Linear):
                nn.init.trunc_normal_(module.weight,std=.02)
                if module.bias is not None: nn.init.zeros_(module.bias)
        # MHA packed Q/K/V is not an nn.Linear child.
        nn.init.trunc_normal_(self.attention.in_proj_weight,std=.02)
        nn.init.zeros_(self.attention.in_proj_bias)
        with torch.no_grad():
            self.attention.out_proj.weight.div_(math.sqrt(2*layer_index))
            self.ffn[-1].weight.div_(math.sqrt(2*layer_index))

    def forward(self,x,mask,padding):
        z = self.norm1(x)
        x = x+self.attention(z,z,z,attn_mask=mask,key_padding_mask=padding,need_weights=False)[0]
        return x+self.ffn(self.norm2(x))


class ActionCausalPredictor(nn.Module):
    def __init__(self,dim=256,layers=2,heads=8,ffn=1024):
        super().__init__()
        self.motion_encoder = IntervalMotionEncoder(dim)
        self.geometry = nn.Sequential(nn.Linear(5,dim),nn.GELU(),nn.Linear(dim,dim))
        self.time = nn.Sequential(nn.Linear(3,dim),nn.GELU(),nn.Linear(dim,dim))
        self.slot_queries = nn.Parameter(torch.empty(16,dim))
        nn.init.trunc_normal_(self.slot_queries,std=.02)
        self.blocks = nn.ModuleList([CausalBlock(dim,heads,ffn,i+1) for i in range(layers)])
        self.output = nn.Linear(dim,dim)
        # Independently initialize every new non-block projection; never touch pretrained/LoRA modules.
        for root in (self.motion_encoder,self.geometry,self.time,self.output):
            for module in root.modules():
                if isinstance(module,nn.Linear):
                    nn.init.trunc_normal_(module.weight,std=.02)
                    nn.init.zeros_(module.bias)

    def forward(self,states,actions,geometry,valid,semantic,horizons=HORIZONS):
        """states [B,S,N,D], actions [B,S,D]; block i cannot read any later block."""
        b,s,n,d = states.shape
        if n % 16 or actions.shape != (b,s,d):
            raise ValueError('Register slots must be real per-tile groups of 16')
        lengths = len(semantic[0])
        slot = self.slot_queries.repeat(n//16,1)
        left = 0.
        sequences = [semantic]
        for i in range(s):
            right = float(horizons[i])
            time = states.new_tensor([left,right,right-left])
            content = state_norm(states[:,i])+self.geometry(geometry)+slot+self.time(time)
            sequences.append(torch.cat((actions[:,i,None],content),1))
            left = right
        x = torch.cat(sequences,1)
        block_ids = torch.cat((torch.full((lengths,),-1,device=x.device),
                              torch.arange(s,device=x.device).repeat_interleave(n+1)))
        causal_mask = block_ids[None,:] > block_ids[:,None]
        padding = torch.cat((torch.zeros(b,lengths,device=x.device,dtype=torch.bool),
                             torch.cat((torch.zeros(b,1,device=x.device,dtype=torch.bool),~valid),1).repeat(1,s)),1)
        for block in self.blocks:
            x = block(x,causal_mask,padding)
        outputs = x[:,lengths:].reshape(b,s,n+1,d)[:,:,1:]
        return state_norm(self.output(outputs))

    def branches(self,z0,targets,actions,geometry,valid,semantic):
        """No teacher feature ever enters rollout. Intermediate predictions retain their graph."""
        teacher_inputs = torch.cat((state_norm(z0)[:,None],state_norm(targets[:,:2]).detach()),1)
        tf = self(teacher_inputs,actions,geometry,valid,semantic)
        states = [state_norm(z0)]
        rollout = []
        for i in range(3):
            prediction = self(torch.stack(states,1),actions[:,:i+1],geometry,valid,semantic)[:,-1]
            rollout.append(prediction)
            states.append(prediction)
        return tf,torch.stack(rollout,1),rollout

    def rollout_candidates(self,z0,codec_output,geometry,valid,semantic,chunk_size=8):
        """Executable K-dimensional kinematics interface, NOT a labeled consequence model."""
        motion,times,mask = [codec_output[n] for n in ('motion_sequence','timestamps','valid_mask')]
        b,k,t,_ = motion.shape
        outputs,coverages = [],[]
        for start in range(0,k,chunk_size):
            count = min(chunk_size,k-start)
            actions,coverage = self.motion_encoder(motion[:,start:start+count],times[:,start:start+count],mask[:,start:start+count],HORIZONS)
            def expand(x):
                return x[:,None].expand(b,count,*x.shape[1:]).reshape(b*count,*x.shape[1:])
            current = expand(z0)
            states = [state_norm(current)]
            predictions = []
            for i in range(3):
                out = self(torch.stack(states,1),actions.reshape(b*count,3,-1)[:,:i+1],
                           expand(geometry),expand(valid),expand(semantic))[:,-1]
                states.append(out); predictions.append(out)
            outputs.append(torch.stack(predictions,1).reshape(b,count,3,*z0.shape[1:]))
            coverages.append(coverage.long().cumprod(-1).bool())
        return torch.cat(outputs,1),torch.cat(coverages,1)

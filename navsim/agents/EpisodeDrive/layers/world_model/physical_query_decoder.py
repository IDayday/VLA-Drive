"""One shared <1M-parameter task decoder. Not part of the deployed policy."""
import torch
from torch import nn


def trajectory_features(trajectories, current_speed, dt=.5):
    """V1.1 fixed scales and measured v0; actions have no gradient boundary."""
    trajectories = trajectories.detach().float()
    xy, theta = trajectories[..., :2], trajectories[..., 2]
    prev_xy = torch.cat((torch.zeros_like(xy[..., :1, :]), xy[..., :-1, :]), -2)
    speed = torch.linalg.vector_norm(xy-prev_xy, dim=-1)/dt
    v0 = current_speed.detach().float()[:, None, None].expand(*speed.shape[:2], 1)
    acc = (speed-torch.cat((v0, speed[..., :-1]), -1))/dt
    return torch.stack((xy[...,0]/30, xy[...,1]/10, theta.sin(), theta.cos(), speed/15, acc/8), -1)


class PhysicalDecoderBlock(nn.Module):
    def __init__(self, dim, heads, ffn):
        super().__init__()
        self.self_attention = nn.MultiheadAttention(dim, heads, dropout=0, batch_first=True)
        self.cross_attention = nn.MultiheadAttention(dim, heads, dropout=0, batch_first=True)
        self.norm1, self.norm2, self.norm3 = (nn.LayerNorm(dim) for _ in range(3))
        self.ffn = nn.Sequential(nn.Linear(dim, ffn), nn.GELU(), nn.Linear(ffn, dim))

    def forward(self, queries, keys, values):
        q = self.norm1(queries)
        queries = queries + self.self_attention(q,q,q,need_weights=False)[0]
        queries = queries + self.cross_attention(self.norm2(queries),keys,values,need_weights=False)[0]
        return queries + self.ffn(self.norm3(queries))


class PhysicalQueryDecoder(nn.Module):
    """Candidate independent, complete-plan-conditioned physics answers.

    `forward` has no future argument. Training hindsight is a separate explicit
    call to the SAME weights. Position/time affects K only, never visual V.
    A known complete action plan is not an observed future ego execution.
    """
    def __init__(self, dim=128, heads=4, layers=2, ffn=512, chunk_size=0):
        super().__init__()
        if (dim,heads,layers,ffn) != (128,4,2,512):
            raise ValueError('Lite fixed small topology is d128/heads4/layers2/ffn512')
        self.dim, self.chunk_size = dim, chunk_size
        self.memory_projection = nn.Linear(256, dim)
        self.trajectory_mlp = nn.Sequential(nn.Linear(6,dim), nn.GELU(), nn.Linear(dim,dim))
        # Status contains command[4], velocity[2], acceleration[2]. Never future pose.
        self.status_mlp = nn.Sequential(nn.Linear(8,dim), nn.GELU(), nn.Linear(dim,dim))
        self.time_embeddings = nn.Embedding(8,dim)
        self.frame_pose_key = nn.Sequential(nn.Linear(5,dim),nn.GELU(),nn.Linear(dim,dim))
        self.blocks = nn.ModuleList([PhysicalDecoderBlock(dim, heads, ffn) for _ in range(layers)])
        self.output_norm = nn.LayerNorm(dim)
        self.gap_head = nn.Linear(dim,5)
        self.road_head = nn.Linear(dim,1)
        self.progress_head = nn.Linear(dim,1)
        if sum(p.numel() for p in self.parameters()) >= 1_000_000:
            raise RuntimeError('Physical decoder exceeded declared <1M parameter budget')

    def _keys_values(self, registers, pose_time):
        values = self.memory_projection(registers.float())
        x,y,heading,time = pose_time.detach().float().unbind(-1)
        # Zero current frame gives zero key offset, despite MLP biases/cos(0).
        identity = torch.stack((x/30,y/10,heading.sin(),heading.cos()-1,time/4), -1)
        offset = self.frame_pose_key(identity)-self.frame_pose_key(torch.zeros_like(identity))
        return values + offset.unsqueeze(-2), values

    def _decode(self, keys, values, trajectories, status_feature, chunk_size):
        if trajectories.ndim != 4 or trajectories.shape[-2:] != (8,3):
            raise ValueError('Candidates must be [B,K,8,3]')
        b,k = trajectories.shape[:2]
        if k < 1 or keys.shape[0] != b or status_feature.shape != (b,8):
            raise ValueError('Nonempty candidates and current status [B,8] required')
        status = status_feature.detach().float()
        points = trajectory_features(trajectories, torch.linalg.vector_norm(status[:,4:6],dim=-1))
        chunk = chunk_size or self.chunk_size or k
        if chunk < 1:
            raise ValueError('chunk_size must be positive')
        result = []
        for start in range(0,k,chunk):
            n = min(chunk,k-start)
            q = (self.trajectory_mlp(points[:,start:start+n]) + self.time_embeddings.weight[None,None]
                 + self.status_mlp(status)[:,None,None]).reshape(b*n,8,self.dim)
            kk = keys[:,None].expand(-1,n,-1,-1).reshape(b*n,-1,self.dim)
            vv = values[:,None].expand(-1,n,-1,-1).reshape(b*n,-1,self.dim)
            for block in self.blocks:
                q=block(q,kk,vv)
            result.append(self.output_norm(q).reshape(b,n,8,self.dim))
        hidden=torch.cat(result,1)
        return dict(gap_logits=self.gap_head(hidden), road_margin=self.road_head(hidden).squeeze(-1),
                    route_progress=self.progress_head(hidden).squeeze(-1))

    def forward(self, current_registers, trajectories, status_feature, *, chunk_size=0):
        if current_registers.shape[1:] != (16,256):
            raise ValueError('Current planning readout must be [B,16,256]')
        zero=current_registers.new_zeros(current_registers.shape[0],4)
        keys,values=self._keys_values(current_registers,zero)
        return self._decode(keys,values,trajectories,status_feature,chunk_size)

    def forward_hindsight(self, ema_current_registers, ema_future_registers, trajectories,
                          status_feature, logged_future_pose, horizon_sec, *, chunk_size=0):
        if ema_current_registers.shape[1:] != (16,256) or ema_future_registers.shape != ema_current_registers.shape:
            raise ValueError('Hindsight consumes current + ONE future [B,16,256] each')
        b=ema_current_registers.shape[0]
        if logged_future_pose.shape != (b,3) or float(horizon_sec) not in (.5,1.5,3.):
            raise ValueError('Logged rear-axle SE(2) pose [B,3] and .5/1.5/3s required')
        zero=ema_current_registers.new_zeros(b,4)
        pose=torch.cat((logged_future_pose.detach(),zero[:,:1]+horizon_sec),-1)
        k0,v0=self._keys_values(ema_current_registers.detach(),zero)
        k1,v1=self._keys_values(ema_future_registers.detach(),pose)
        return self._decode(torch.cat((k0,k1),1),torch.cat((v0,v1),1),trajectories,status_feature,chunk_size)

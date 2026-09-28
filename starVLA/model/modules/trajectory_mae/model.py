"""GT MAE: encode only current states and visible peer futures; decode target Z."""
import torch
from torch import nn
from .tokenizer import TeacherInputs
from starVLA.model.modules.foresight.future_latent_head import CrossReadout


class TrajectoryMAE(nn.Module):
    def __init__(self, dim=512, layers=6, decoder_layers=2, heads=8, steps=8, xy_scale=20.):
        super().__init__()
        if xy_scale <= 0 or steps != 8: raise ValueError('Invalid teacher coordinate/time definition')
        self.steps, self.xy_scale = steps, float(xy_scale)
        self.current = nn.Sequential(nn.Linear(8,dim),nn.GELU(),nn.Linear(dim,dim))
        self.future = nn.Linear(2,dim)
        self.role = nn.Embedding(2,dim)
        self.time = nn.Embedding(steps,dim)
        self.kind = nn.Embedding(2,dim)
        self.navigation = nn.Embedding(4,dim)
        self.ego_state = nn.Linear(4,dim)
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(dim,heads,4*dim,
            dropout=0.,batch_first=True,norm_first=True),layers,norm=nn.LayerNorm(dim),enable_nested_tensor=False)
        self.target_type = nn.Parameter(torch.randn(dim)*.02)
        self.decoder = nn.ModuleList(CrossReadout(dim,heads) for _ in range(decoder_layers))
        self.reconstruct = nn.Sequential(nn.LayerNorm(dim),nn.Linear(dim,dim),nn.GELU(),nn.Linear(dim,2))

    def forward(self, inputs: TeacherInputs):
        current, future = inputs.sanitized()
        b,n,_ = current.shape
        if future.shape[2] != self.steps: raise ValueError('Teacher horizon mismatch')
        scales = current.new_tensor([self.xy_scale,self.xy_scale,5.,10.,10.,5.,1.,1.])
        role = torch.ones(n,dtype=torch.long,device=current.device);role[0]=0
        base = self.current(current/scales) + self.role(role)[None]
        context = self.navigation(inputs.navigation)+self.ego_state(inputs.ego_state)
        base = base + context[:,None]
        now = base+self.kind.weight[0]
        then = (base[:,:,None]+self.future(future/self.xy_scale)+self.time.weight[None,None]
                +self.kind.weight[1])
        memory = torch.cat((now,then.flatten(1,2)),1)
        valid = torch.cat((inputs.active,inputs.visible.flatten(1)),1)
        memory = torch.where(valid[...,None],memory,0.)
        memory = self.encoder(memory,src_key_padding_mask=~valid)
        memory = torch.where(valid[...,None],memory,0.)
        rows = torch.arange(b,device=current.device)
        z = base[rows,inputs.target,None]+self.time.weight[None]+self.target_type
        for block in self.decoder: z = block(z,memory,valid)
        # No bypass from visible trajectories: head sees ONLY Z and target current anchor.
        xy = self.reconstruct(z)*self.xy_scale+current[rows,inputs.target,None,:2]
        if not torch.isfinite(z).all() or not torch.isfinite(xy).all(): raise FloatingPointError('Invalid teacher output')
        return {'latent':z,'xy':xy}

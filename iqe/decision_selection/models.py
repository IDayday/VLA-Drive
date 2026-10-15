"""Scratch decision heads. Scene routing cannot accept a trajectory argument."""
import math
import torch
from torch import nn
from iqe.contracts import require

CONTEXT_KEYS = {'world', 'action', 'current_ego', 'world_valid', 'action_valid'}


def positions(length, width, device):
    p = torch.arange(length, device=device, dtype=torch.float32)[:, None]
    f = torch.exp(torch.arange(0, width, 2, device=device).float()*(-math.log(10000)/width))
    out = torch.zeros(length, width, device=device)
    out[:, 0::2], out[:, 1::2] = (p*f).sin(), (p*f).cos()
    return out


class ContextProjection(nn.Module):
    def __init__(self, world_dim, action_dim, hidden=256):
        super().__init__(); self.hidden = hidden
        self.world = nn.Sequential(nn.LayerNorm(world_dim), nn.Linear(world_dim, hidden))
        self.action = nn.Sequential(nn.LayerNorm(action_dim), nn.Linear(action_dim, hidden))
        self.ego = nn.Sequential(nn.Linear(4, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.types = nn.Parameter(torch.randn(3, hidden)*.02)

    def forward(self, context):
        require(set(context) == CONTEXT_KEYS, 'decision context whitelist: no targets/IDs/trajectories')
        w, a, e = [context[k].detach().float() for k in ('world', 'action', 'current_ego')]
        wm, am = [context[k].detach().bool() for k in ('world_valid', 'action_valid')]
        require(w.ndim == a.ndim == 3 and e.shape == (len(a), 4), 'W/A/ego contract')
        require(wm.shape == w.shape[:2] and am.shape == a.shape[:2], 'context masks')
        require(wm.any(-1).all().item() and am.any(-1).all().item(), 'actual W/A required')
        require(all(torch.isfinite(x).all().item() for x in (w, a, e)), 'nonfinite context')
        memory = [self.world(w)+self.types[0]+positions(w.shape[1], self.hidden, w.device),
                  self.action(a)+self.types[1]+positions(a.shape[1], self.hidden, a.device),
                  self.ego(e)[:, None]+self.types[2]]
        return torch.cat(memory, 1), torch.cat((wm, am, torch.ones(len(e), 1, dtype=torch.bool, device=e.device)), 1)


def decoder(hidden):
    block = nn.TransformerDecoderLayer(hidden, 4, hidden*4, dropout=0., activation='gelu',
                                       batch_first=True, norm_first=True)
    return nn.TransformerDecoder(block, 2, norm=nn.LayerNorm(hidden))


def decision_output(logits, valid):
    require(valid.shape == logits.shape and valid.any(-1).all().item(), 'no available deployment candidate')
    require(torch.isfinite(logits[valid]).all().item(), 'nonfinite legal decision output')
    probabilities = logits.float().masked_fill(~valid, -torch.inf).softmax(-1)
    return {'decision_logits': logits, 'decision_probabilities': probabilities, 'candidate_available': valid}


class SceneRouter(nn.Module):
    def __init__(self, world_dim, action_dim, expert_ids, hidden=256):
        super().__init__(); self.expert_ids = tuple(expert_ids)
        require(len(set(expert_ids)) == len(expert_ids) and 'expert_0' in expert_ids, 'fixed expert registry')
        self.context = ContextProjection(world_dim, action_dim, hidden)
        self.query = nn.Parameter(torch.randn(1, 1, hidden)*.02)
        self.decoder = decoder(hidden)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, len(expert_ids)))

    def forward(self, context):
        memory, valid = self.context(context)
        summary = self.decoder(self.query.expand(len(memory), -1, -1), memory,
                               memory_key_padding_mask=~valid)[:, 0]
        logits = self.head(summary)
        # No bank-validity mask: a pre-router cannot see future Decoder outputs.
        return decision_output(logits, torch.ones_like(logits, dtype=torch.bool))


class SetSelector(nn.Module):
    def __init__(self, world_dim, action_dim, trajectory_contract, hidden=256):
        super().__init__()
        self.context = ContextProjection(world_dim, action_dim, hidden)
        self.trajectory_encoder = nn.Sequential(nn.Linear(5, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.decoder = decoder(hidden)
        self.base_role = nn.Embedding(2, hidden)
        layer = nn.TransformerEncoderLayer(hidden, 4, hidden*4, dropout=0., activation='gelu',
                                           batch_first=True, norm_first=True)
        self.set_encoder = nn.TransformerEncoder(layer, 1, norm=nn.LayerNorm(hidden), enable_nested_tensor=False)
        self.head = nn.Linear(hidden, 1)
        self.register_buffer('xy_mean', torch.tensor(trajectory_contract.mean[:2]))
        self.register_buffer('xy_scale', torch.tensor(trajectory_contract.scale[:2]))
        self.register_buffer('times', torch.arange(1, trajectory_contract.horizon+1).float()*trajectory_contract.dt)

    def forward(self, context, trajectories, base_role, available=None):
        t = trajectories.detach().float(); role = base_role.detach().bool()
        require(t.ndim == 4 and t.shape[-2:] == (len(self.times), 3) and role.shape == t.shape[:2], 'physical candidates/base role')
        require((role.sum(-1) == 1).all().item(), 'exactly one Base role')
        valid = torch.isfinite(t).all((-1, -2))
        if available is not None:
            require(available.dtype == torch.bool and available.shape == valid.shape, 'deployment availability mask')
            valid = valid & available.detach()
        require(valid.any(-1).all().item(), 'no finite available candidate')
        safe = torch.where(valid[..., None, None], t, torch.zeros_like(t))
        b, k, steps, _ = safe.shape
        tokens = torch.cat(((safe[..., :2]-self.xy_mean)/self.xy_scale,
                            safe[..., 2:3].sin(), safe[..., 2:3].cos(),
                            self.times.expand(b, k, steps)[..., None]), -1)
        memory, memory_valid = self.context(context)
        z = self.decoder(self.trajectory_encoder(tokens).reshape(b*k, steps, -1),
                         memory[:, None].expand(-1, k, -1, -1).reshape(b*k, memory.shape[1], -1),
                         memory_key_padding_mask=(~memory_valid)[:, None].expand(-1, k, -1).reshape(b*k, -1))
        z = z.mean(1).reshape(b, k, -1)+self.base_role(role.long())
        # No expert identity, no candidate order encoding or causal candidate mask.
        z = self.set_encoder(z, src_key_padding_mask=~valid)
        logits = self.head(z).squeeze(-1)
        return decision_output(logits, valid)


def choose(logits, expert_ids, available=None, gamma=None):
    require(len(expert_ids) == logits.shape[1] and len(set(expert_ids)) == len(expert_ids), 'stable IDs')
    base = list(expert_ids).index('expert_0')
    valid = torch.ones_like(logits, dtype=torch.bool) if available is None else available.bool()
    require(valid.any(-1).all().item() and torch.isfinite(logits[valid]).all().item(), 'legal logits/candidate availability')
    order = sorted(range(len(expert_ids)), key=lambda i: (i != base, expert_ids[i]))
    perm = torch.tensor(order, device=logits.device)
    adjusted = logits.masked_fill(~valid, -torch.inf)
    winner = perm[adjusted[:, perm].argmax(-1)]
    if gamma is not None:
        require(gamma >= 0 and valid[:, base].all().item(), 'logit margin requires available Base')
        gain = adjusted.gather(1, winner[:, None]).squeeze(1)-adjusted[:, base]
        winner = torch.where(gain > gamma, winner, torch.full_like(winner, base))
    return winner

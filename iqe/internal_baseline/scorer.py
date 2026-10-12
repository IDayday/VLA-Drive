"""Small internal scorer: original frozen W/A, current ego, actual trajectory.

No identity embedding, no extra vision/language network, no external weights.
The context projection is evaluated once before expanding the candidate axis.
"""
from contextlib import contextmanager
import random
import numpy as np
import torch
from torch import nn
from iqe.contracts import require
from iqe.evaluation.retention import module_hash
from iqe.io import atomic_json, read_json, file_hash
from iqe.scoring.reference import COMPONENTS


@contextmanager
def initialization_rng(seed):
    python, numpy = random.getstate(), np.random.get_state()
    with torch.random.fork_rng(devices=[]):
        random.seed(seed); np.random.seed(seed); torch.random.default_generator.manual_seed(seed)
        try:
            yield
        finally:
            random.setstate(python); np.random.set_state(numpy)


class InternalScorer(nn.Module):
    def __init__(self, world_dim, action_dim, trajectory_contract, *, hidden=256, layers=2,
                 heads=4, dropout=.1, context="world_action_ego", components=COMPONENTS):
        super().__init__()
        self.context_kind = context
        self.components = tuple(components)
        self.world_projection = nn.Sequential(nn.LayerNorm(world_dim), nn.Linear(world_dim, hidden))
        self.action_projection = nn.Sequential(nn.LayerNorm(action_dim), nn.Linear(action_dim, hidden))
        self.ego_projection = nn.Sequential(nn.Linear(4, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.type_embeddings = nn.Parameter(torch.empty(3, hidden))
        nn.init.normal_(self.type_embeddings, std=.02)
        self.trajectory_encoder = nn.Sequential(nn.Linear(5, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        layer = nn.TransformerDecoderLayer(hidden, heads, hidden*4, dropout=dropout, batch_first=True,
                                            activation="gelu", norm_first=True)
        self.decoder = nn.TransformerDecoder(layer, layers, norm=nn.LayerNorm(hidden))
        # TransformerDecoder clones its layer; initialize each clone independently.
        for index, block in enumerate(self.decoder.layers):
            if index:
                for m in block.modules():
                    if isinstance(m, nn.Linear):
                        nn.init.xavier_uniform_(m.weight)
                        if m.bias is not None: nn.init.zeros_(m.bias)
        self.value_head = nn.Linear(hidden, 1)
        self.component_head = nn.Linear(hidden, len(self.components))
        self.register_buffer("xy_mean", torch.tensor(trajectory_contract.mean[:2]))
        self.register_buffer("xy_scale", torch.tensor(trajectory_contract.scale[:2]))
        self.register_buffer("times", torch.arange(1, trajectory_contract.horizon+1, dtype=torch.float32)*trajectory_contract.dt)

    @staticmethod
    def positions(length, width, device):
        p = torch.arange(length, dtype=torch.float32, device=device)[:, None]
        f = torch.exp(torch.arange(0, width, 2, dtype=torch.float32, device=device)*(-np.log(10000)/width))
        out = torch.zeros(length, width, device=device)
        out[:, 0::2], out[:, 1::2] = torch.sin(p*f), torch.cos(p*f)
        return out

    def project_context(self, context):
        require(set(context) == {"world", "action", "current_ego", "world_valid", "action_valid"}, "score context whitelist")
        w, a, e = (context[k].detach().float() for k in ("world", "action", "current_ego"))
        require(a.ndim == w.ndim == 3 and e.shape == (len(a), 4), "scorer context shape")
        require(context["action_valid"].any(-1).all().item(), "empty action context")
        ap = self.action_projection(a) + self.type_embeddings[1] + self.positions(a.shape[1], self.type_embeddings.shape[1], a.device)
        ep = self.ego_projection(e)[:, None] + self.type_embeddings[2]
        memory, masks = [ap, ep], [context["action_valid"].bool(), torch.ones(len(e), 1, device=e.device, dtype=torch.bool)]
        if self.context_kind == "world_action_ego":
            require(w.shape[1] > 0 and context["world_valid"].any(-1).all().item(), "W unavailable; use an explicit A-only ablation")
            wp = self.world_projection(w) + self.type_embeddings[0] + self.positions(w.shape[1], self.type_embeddings.shape[1], w.device)
            memory.insert(0, wp); masks.insert(0, context["world_valid"].bool())
        else:
            # Keep optional projection parameters graph-connected for uniform DDP.
            memory[0] = memory[0] + sum(p.sum()*0 for p in self.world_projection.parameters())
        return torch.cat(memory, dim=1), torch.cat(masks, dim=1)

    def score_projected(self, memory, memory_valid, trajectories):
        t = trajectories.detach().float()
        require(t.ndim == 4 and t.shape[-2:] == (len(self.times), 3), "physical candidate trajectory contract")
        valid = torch.isfinite(t).all((-1, -2))
        safe = torch.where(valid[..., None, None], t, torch.zeros_like(t))
        b, k, steps, _ = t.shape
        xy = (safe[..., :2]-self.xy_mean)/self.xy_scale
        times = self.times.expand(b, k, steps)[..., None]
        tokens = self.trajectory_encoder(torch.cat((xy, safe[..., 2:3].sin(), safe[..., 2:3].cos(), times), -1))
        # Each candidate is an independent batch item. No candidate IDs/positions.
        z = self.decoder(tokens.reshape(b*k, steps, -1),
                         memory[:, None].expand(-1, k, -1, -1).reshape(b*k, memory.shape[1], -1),
                         memory_key_padding_mask=(~memory_valid[:, None].expand(-1, k, -1)).reshape(b*k, -1))
        z = z.mean(1).reshape(b, k, -1)
        return {"values": self.value_head(z).squeeze(-1).sigmoid(),
                "component_logits": self.component_head(z), "valid": valid}

    def forward(self, context, trajectories):
        return self.score_projected(*self.project_context(context), trajectories)


def construct(config, trajectory_contract, world_dim, action_dim, *, initialization_receipt=True):
    with initialization_rng(config.module_init_seed):
        m = InternalScorer(world_dim, action_dim, trajectory_contract, dropout=config.scorer_dropout,
                           context=config.score_context)
    identity = {"schema": "random_internal_scorer_v1", "experiment_id": config.experiment_id,
                "init_seed": config.module_init_seed, "initial_state_hash": module_hash(m),
                "external_scorer_weights_loaded": False, "loaded_scorer_weights": [],
                "allowed_resume_lineage": config.experiment_id,
                "parameters": {n: {"shape": list(p.shape), "count": p.numel()} for n, p in m.named_parameters()},
                "parameter_count": sum(p.numel() for p in m.parameters()),
                "base_binding_hash": file_hash(config.root/"BASE_BINDING.json"),
                "loaded_candidate_generator_sources": {"base": read_json(config.root/"BASE_BINDING.json")["checkpoint"],
                   "base_hash": read_json(config.root/"BASE_BINDING.json")["checkpoint_hash"],
                   "expert_experiment_id": config.experiment_id, "expert_initial_head_hash": read_json(config.root/"BASE_BINDING.json")["head_hash"]},
                "context": config.score_context}
    if initialization_receipt:
        atomic_json(config.root / "SCORER_INITIALIZATION.json", identity, immutable=True)
    m._iqe_initialization_identity = identity
    return m

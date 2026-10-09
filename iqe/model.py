"""Stage isolation survives outer train() calls; shared encoding happens once."""
from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import nn
from .contracts import require
from .expert import assert_no_alias


@dataclass
class CandidateBatch:
    trajectories: torch.Tensor
    raw: torch.Tensor
    expert_ids: tuple[str, ...]
    valid: torch.Tensor
    features: object


class IQEModel(nn.Module):
    def __init__(self, adapter, scorer=None, router=None):
        super().__init__()
        self.adapter = adapter
        self.experts = nn.ModuleDict({"expert_0": adapter.clone_s0_expert("expert_0")})
        self.scorer = scorer
        self.router = router
        self.stage, self.active_expert_id = "inference", None
        self.expert_variant = "independent"
        self.set_trainable_stage("inference")

    def append_expert(self, expert_id, variant="independent", bottleneck=32):
        require(self.stage in {"inference", "audit"}, "append only at job boundaries before optimizer/DDP")
        require(expert_id not in self.experts, "expert identity is append-only")
        require(expert_id.startswith("expert_") and expert_id[7:].isdigit() and int(expert_id[7:]) > max(int(e[7:]) for e in self.experts), "stable append-only expert registration sequence")
        expert = self.adapter.clone_s0_expert(expert_id)
        if variant == "adapter":
            from .expert import AdapterExpert
            expert = AdapterExpert(expert, bottleneck)
        elif variant == "residual":
            from .expert import ResidualExpert
            expert = ResidualExpert(expert)
        require(variant in {"independent", "query_only", "adapter", "residual", "single_finetune"}, "expert ablation")
        self.experts[expert_id] = expert
        self.expert_variant = variant
        assert_no_alias(dict(self.experts))
        self.set_trainable_stage(self.stage)
        return expert

    def set_trainable_stage(self, stage, active_expert_id=None):
        require(stage in {"expert_train", "scorer_train", "router_train", "inference", "audit"}, "unknown training stage")
        if stage == "expert_train":
            require(active_expert_id in self.experts and active_expert_id != "expert_0", "train only a new expert")
        if stage in {"scorer_train", "router_train"}:
            require(getattr(self, stage.split("_")[0]) is not None, "missing trainable selector")
        self.stage, self.active_expert_id = stage, active_expert_id
        self.requires_grad_(False)
        if stage == "expert_train":
            expert = self.experts[active_expert_id]
            expert.requires_grad_(True)
            if self.expert_variant == "query_only":
                expert.requires_grad_(False)
                expert.query.requires_grad_(True)
            elif self.expert_variant == "adapter":
                expert.requires_grad_(False)
                expert.query.requires_grad_(True)
                expert.adapter.requires_grad_(True)
            elif self.expert_variant == "residual":
                expert.initial.requires_grad_(False)
        elif stage == "scorer_train":
            self.scorer.requires_grad_(True)
        elif stage == "router_train":
            self.router.requires_grad_(True)
        self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        self.adapter.eval()
        self.experts.eval()
        if self.scorer is not None:
            self.scorer.eval()
        if self.router is not None:
            self.router.eval()
        if mode and self.stage == "expert_train":
            active = self.experts[self.active_expert_id]
            active.train()
            if self.expert_variant == "query_only":
                active.decoder.eval()
                active.heads.eval()
            elif self.expert_variant == "adapter":
                active.decoder.eval()
                active.heads.eval()
        elif mode and self.stage == "scorer_train":
            self.scorer.train()
        elif mode and self.stage == "router_train":
            self.router.train()
        return self

    def encode_scene(self, batch):
        with torch.no_grad():
            return self.adapter.encode_scene(batch).detached()

    def forward_candidates(self, batch, active_expert_ids=None, *, features=None):
        ids = tuple(active_expert_ids or self.experts.keys())
        require(ids and ids[0] == "expert_0" and len(set(ids)) == len(ids), "candidate IDs require Base first and no duplicates")
        require(all(i in self.experts for i in ids), "unknown expert ID")
        f = features.detached() if features is not None else self.encode_scene(batch)
        out = [self.adapter.expert_forward(self.experts[i], f) for i in ids]
        physical, raw = torch.stack([o.physical for o in out], 1), torch.stack([o.raw for o in out], 1)
        return CandidateBatch(physical, raw, ids, torch.isfinite(physical).all((-1, -2)), f)

    def forward_scores(self, features, candidates):
        require(self.scorer is not None, "Scorer weights required for K>1")
        return self.scorer(features, candidates.trajectories, candidates.valid)

    def predict(self, batch, selector, *, features=None):
        candidates = self.forward_candidates(batch, features=features)
        if len(candidates.expert_ids) == 1:
            require(bool(candidates.valid.all()), "Base invalid: use original S0 error policy")
            return candidates.trajectories[:, 0], ["expert_0"] * len(candidates.trajectories), ["base_only"] * len(candidates.trajectories)
        return selector(candidates, self.forward_scores(candidates.features, candidates))

    def forward_prerouted(self, batch, router_rule, *, features=None):
        require(self.router is not None and self.router.expert_ids == tuple(self.experts), "Router/registry mismatch")
        f = features.detached() if features is not None else self.encode_scene(batch)
        selected = router_rule(self.router(f), self.router.expert_ids)
        outputs = []
        order = []
        for idx, expert_id in enumerate(self.router.expert_ids):
            rows = (selected == idx).nonzero().flatten()
            if len(rows):
                outputs.append(self.adapter.expert_forward(self.experts[expert_id], f.subset(rows)).physical)
                order.append(rows)
        concatenated = torch.cat(outputs)
        inverse = torch.argsort(torch.cat(order))
        trajectories = concatenated[inverse]
        require(bool(torch.isfinite(trajectories).all()), "pre-routed expert invalid")
        return trajectories, [self.router.expert_ids[i] for i in selected.cpu().tolist()]

    def optimizer_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def audit_optimizer(self, optimizer):
        actual = [p for group in optimizer.param_groups for p in group["params"]]
        expected = self.optimizer_parameters()
        require(len(actual) == len({id(p) for p in actual}) and {id(p) for p in actual} == {id(p) for p in expected},
                "optimizer parameter set must exactly match active trainable set")

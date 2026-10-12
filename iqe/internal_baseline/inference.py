"""Deployable one-VLM-forward algorithm. No environment/target/label dependencies."""
import torch
from torch import nn
from iqe.contracts import require
from .features import encode_joint


class InternalAlgorithm(nn.Module):
    def __init__(self, adapter, expert, scorer):
        super().__init__()
        self.adapter, self.expert, self.scorer = adapter, expert, scorer
        self.requires_grad_(False).eval()

    def train(self, mode=True):
        super().train(False)
        return self

    @torch.no_grad()
    def forward(self, observations, *, margin=0., return_debug=False):
        require(margin >= 0, "nonnegative selection margin")
        planner, context = encode_joint(self.adapter, observations)
        base = self.adapter.predict_s0(planner)
        candidate = self.expert(planner)
        candidates = torch.stack((base.physical, candidate.physical), dim=1)
        inputs = context.tensors()
        if self.scorer.context_kind == "planner_ego":
            inputs = inputs | {"world": context.world[:, :0], "world_valid": context.world_valid[:, :0],
                               "action": planner.scene, "action_valid": planner.valid_tokens}
        prediction = self.scorer(inputs, candidates)
        require(prediction["valid"][:, 0].all().item(), "Base trajectory invalid; do not invent a safe fallback")
        select = ((prediction["values"][:, 1]-prediction["values"][:, 0] > margin)
                  & prediction["valid"][:, 1]).long()
        final = candidates[torch.arange(len(select), device=select.device), select]
        if return_debug:
            return {"trajectory": final, "selected_expert": select, "predicted_values": prediction["values"],
                    "component_predictions": prediction["component_logits"].sigmoid(), "candidates": candidates}
        return final


def load_bundle(path, device="cuda:0"):
    from pathlib import Path
    from iqe.io import read_json, file_hash
    from iqe.s0_adapter import S0Adapter
    from iqe.query_base import source_trajectory_contract
    from .scorer import InternalScorer, initialization_rng
    path = Path(path)
    manifest = read_json(path/"BUNDLE.json")
    require(manifest["schema"] == "own_internal_scorer_bundle_v1" and manifest["candidate_ids"] == ["expert_0", "expert_1"], "bundle registry/schema")
    require(file_hash(path/"algorithm.pt") == manifest["weights_hash"], "bundle weights corrupted")
    require(manifest["scorer_initialization"]["external_scorer_weights_loaded"] is False, "external scorer lineage prohibited")
    adapter = S0Adapter.load_and_validate_s0(manifest["base_contract"], device)
    expert = adapter.clone_s0_expert("expert_1")
    arch = manifest["scorer_architecture"]
    tc = manifest["base_contract"]["trajectory"]
    with initialization_rng(manifest["scorer_initialization"]["init_seed"]):
        scorer = InternalScorer(arch["world_dim"], arch["action_dim"], source_trajectory_contract(tc["horizon"], tc["dt"]),
                                dropout=arch["dropout"], context=arch["context"])
    weights = torch.load(path/"algorithm.pt", map_location="cpu", weights_only=True)
    require(set(weights) == {"expert_1", "internal_scorer"}, "bundle has unknown/external modules")
    expert.load_state_dict(weights["expert_1"], strict=True)
    scorer.load_state_dict(weights["internal_scorer"], strict=True)
    return InternalAlgorithm(adapter, expert.to(device), scorer.to(device))

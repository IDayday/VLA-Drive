"""Concrete adapter for the original-trajectory Query-migrated S0 framework."""
from __future__ import annotations
from copy import deepcopy
import torch
from torch import nn
from .contracts import require
from .io import read_json, file_hash, digest, BlockedError
from .losses import expert_il_terms
from .query_base import build_query_framework


class S0Adapter(nn.Module):
    def __init__(self, framework, contract):
        super().__init__()
        self.framework = framework
        self.contract = contract
        self.contract_hash = digest(contract)
        self.requires_grad_(False).eval()

    @classmethod
    def load_and_validate_s0(cls, contract, device="cpu"):
        require(contract["binding_status"] == "LOCKED_QUERY_S0", "Query base must be trained and locked before incrementing")
        path = contract["query_checkpoint"]
        if not path:
            raise BlockedError("BLOCKED_QUERY_BASE_CHECKPOINT: framework exists; learned driving weights explicitly not reused")
        require(file_hash(path) == contract["query_checkpoint_hash"], "S0 checkpoint hash mismatch")
        from pathlib import Path
        repo = Path(__file__).resolve().parents[1]
        for name, expected in (contract.get("query_model_code_hashes", {}) | contract.get("reused_M0_module_hashes", {})).items():
            require(file_hash(repo/name) == expected, "locked Query S0 implementation changed: " + name)
        # Read only model tensors; optimizer storages stay file-backed during
        # inference construction. Values and strict state loading are unchanged.
        state = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        require(state.get("kind") == "iqe_query_base" and state.get("framework_contract_hash") == contract["framework_contract_hash"],
                "wrong S0 architecture or framework contract")
        from omegaconf import OmegaConf
        model = build_query_framework(OmegaConf.create(contract["source_config"]), contract["query_architecture"],
                                      contract["source_root"], contract["source_commit"]).float()
        model.load_state_dict(state["model"], strict=True)
        model.strip_auxiliary_heads()
        model.to(device).eval()
        return cls(model, contract)

    def train(self, mode=True):
        super().train(False)
        return self

    def encode_scene(self, batch):
        with torch.no_grad():
            encoded = self.framework.encode_current(batch)
            f = self.framework.action_model.encode_features(self.framework.build_planner_condition(encoded), self.framework.action_model.ego_state,
                                                            [x["token"] for x in batch], self.contract_hash)
        return f.detached()

    def predict_s0(self, features):
        return self.expert_forward(self.framework.action_model.expert, features)

    def clone_s0_expert(self, expert_id):
        require(expert_id.startswith("expert_"), "invalid expert identity")
        return deepcopy(self.framework.action_model.expert)

    def expert_forward(self, expert, features):
        require(features.contract_hash == self.contract_hash, "FeatureBundle/S0 contract mismatch")
        return expert(features)

    def compute_expert_il_loss(self, prediction, target, masks=None):
        return expert_il_terms(prediction, target, prev_weight=self.framework.action_model.prev_weight, masks=masks)

    def to_official_trajectory(self, prediction, scene_meta=None):
        from navsim.common.dataclasses import Trajectory
        from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
        poses = prediction.physical.detach().cpu().numpy()
        require(len(poses) == 1, "official NAVSIM Agent outputs one scene trajectory")
        return Trajectory(poses[0], TrajectorySampling(num_poses=poses.shape[1], interval_length=prediction.dt))

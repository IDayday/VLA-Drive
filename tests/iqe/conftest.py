import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import pytest
import torch
from torch import nn
from copy import deepcopy
from iqe.contracts import FeatureBundle, TrajectoryContract, SceneRecord
from iqe.expert import IndependentExpert
from iqe.losses import expert_il_terms

torch.set_num_threads(2)


class Decoder(nn.Module):
    def __init__(self, width=8):
        super().__init__()
        self.cross = nn.MultiheadAttention(width, 2, dropout=.2, batch_first=True)
        self.norm = nn.LayerNorm(width)
    def forward(self, q, f):
        return [self.norm(q + self.cross(q, f, f)[0])]


class TensorAdapter(nn.Module):
    """Unit-only synthetic adapter, never labelled as real S0 integration."""
    def __init__(self):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(8, 8), nn.BatchNorm1d(8), nn.Dropout(.5))
        self.contract = TrajectoryContract(4, .5, "normalized_xy_sincos", "ego_relative", "rear_axle", "radian", (1.,2.,0.,0.), (2.,3.,1.,1.), "unit")
        self.action = IndependentExpert(nn.Embedding(1,8), Decoder(), nn.ModuleList([nn.Linear(8,16), nn.Linear(8,16)]), self.contract)
        self.encodes = 0
    def encode_scene(self, x):
        self.encodes += 1
        return FeatureBundle(self.shared(x.mean(1))[:,None].expand(-1,3,-1), x[:,0], torch.ones(x.shape[:2], dtype=torch.bool), "unit", tuple(map(str,range(len(x)))))
    def clone_s0_expert(self, expert_id):
        return deepcopy(self.action)
    def expert_forward(self, expert, features):
        return expert(features)
    def compute_expert_il_loss(self, out, target):
        return expert_il_terms(out, target)


@pytest.fixture
def features():
    torch.manual_seed(3)
    return FeatureBundle(torch.randn(3,3,8), torch.randn(3,1,8), torch.ones(3,3,dtype=torch.bool), "unit", ("a","b","c"))


def scene(i=0, role="incremental_fit", group=None, **kw):
    values = dict(schema_version=1, scene_id=f"s{i}", source_log_id=f"log{i}" if group is None else group,
        source_group_id=f"g{i}" if group is None else group, observation_hash=f"ob{i}", split_role=role, source_kind="original",
        source_scene_id=f"s{i}", observation_ref="obs", target_id=f"target{i}", target_ref="target", target_provenance="gt",
        input_consistency_status="verified", metric_context_ref="ctx", metric_context_hash=f"ctx{i}", quality_audit_ref="audit")
    values.update(kw)
    return SceneRecord(**values)

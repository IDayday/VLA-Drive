"""Two-rank executable smoke for the curriculum's unused DrivoR parameters."""

import os
from pathlib import Path
import sys

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from starVLA.model.modules.scene_encoder import GlobalSceneQFormer
from starVLA.model.modules.trajectory_scorer import (
    DriveSuprimCoarseScorer,
    DriveSuprimFineRefiner,
    DrivoRDynamicScorer,
    HierarchicalDrivoRSuprimScorer,
)
from starVLA.model.modules.trajectory_scorer.losses import SUPRIM_METRICS


class StaticCurriculumModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.scene = GlobalSceneQFormer(
            input_dim=12,
            hidden_dim=32,
            output_dim=32,
            num_queries=4,
            num_layers=1,
            num_heads=4,
            ffn_dim=64,
        )
        dynamic = DrivoRDynamicScorer(
            scene_dim=32,
            ego_state_dim=4,
            model_dim=32,
            ffn_dim=64,
            num_layers=1,
            num_heads=4,
        )
        coarse = DriveSuprimCoarseScorer(
            static_vocab=torch.randn(16, 40, 3),
            vocab_size=16,
            scene_dim=32,
            ego_state_dim=4,
            model_dim=32,
            ffn_dim=64,
            num_heads=4,
            num_layers=1,
            coarse_topk=4,
        )
        fine = DriveSuprimFineRefiner(
            scene_dim=32,
            model_dim=32,
            ffn_dim=64,
            num_heads=4,
            num_layers=1,
        )
        self.hierarchy = HierarchicalDrivoRSuprimScorer(dynamic, coarse, fine)

    def forward(self):
        context = self.scene(
            torch.randn(1, 6, 12), torch.ones(1, 6, dtype=torch.bool)
        )
        targets = {name: torch.rand(1, 16) for name in SUPRIM_METRICS}
        output = self.hierarchy.forward_static_only(
            global_scene_tokens=context.global_tokens,
            dense_scene_memory=context.dense_memory,
            memory_key_padding_mask=context.memory_key_padding_mask,
            ego_state=torch.randn(1, 1, 4),
            gt_trajectory_8=torch.randn(1, 8, 3),
            static_targets=targets,
        )
        return output["losses"]["suprim_coarse"] + output["losses"]["suprim_fine"]


def main():
    dist.init_process_group("gloo")
    torch.manual_seed(100 + int(os.environ["RANK"]))
    model = DistributedDataParallel(
        StaticCurriculumModel(), find_unused_parameters=True
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-4)
    optimizer.zero_grad()
    loss = model()
    loss.backward()
    optimizer.step()
    assert torch.isfinite(loss)
    assert all(
        parameter.grad is None
        for parameter in model.module.hierarchy.dynamic_prescorer.parameters()
    )
    received = torch.tensor(
        int(
            model.module.hierarchy.joint_coarse_scorer.candidate_embedding[
                0
            ].weight.grad
            is not None
        )
    )
    dist.all_reduce(received, op=dist.ReduceOp.SUM)
    assert received.item() == dist.get_world_size()
    if dist.get_rank() == 0:
        print("2-rank static-only DDP smoke passed")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

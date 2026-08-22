import torch

from starVLA.model.modules.planning.trajectory_codec import TrajectoryCodec
from starVLA.model.modules.trajectory_scorer import (
    DriveSuprimCoarseScorer,
    DriveSuprimFineRefiner,
)


def _coarse():
    return DriveSuprimCoarseScorer(
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


def test_unified_pool_and_metadata_alignment():
    coarse = _coarse()
    dynamic_8 = torch.randn(2, 4, 8, 3)
    dynamic_40 = TrajectoryCodec().upsample_8_to_40(dynamic_8)
    ids = torch.tensor([[7, 2, 6, 1], [0, 3, 5, 4]])
    candidates, metadata = coarse.build_joint_pool(
        2, torch.randn(2, 4, 32), dynamic_40, ids
    )
    assert candidates.shape == (2, 20, 40, 3)
    assert torch.equal(metadata.source[:, :16], torch.zeros(2, 16, dtype=torch.long))
    assert torch.equal(metadata.source[:, 16:], torch.ones(2, 4, dtype=torch.long))
    assert torch.equal(metadata.source_index[:, 16:], ids)
    assert torch.equal(
        metadata.absolute_index[:, 16:], torch.arange(16, 20)[None].expand(2, -1)
    )


def test_global_topk_and_three_fine_layers_keep_original_candidates():
    coarse = _coarse().eval()
    dynamic_8 = torch.randn(2, 4, 8, 3)
    dynamic_40 = TrajectoryCodec().upsample_8_to_40(dynamic_8)
    output = coarse(
        torch.randn(2, 4, 32),
        torch.randn(2, 1, 4),
        dynamic_trajectories_40=dynamic_40,
        dynamic_candidate_ids=torch.arange(4)[None].expand(2, -1),
    )
    assert output.aggregate_score.shape == (2, 20)
    assert output.topk_indices.shape == (2, 4)
    fine = DriveSuprimFineRefiner(
        scene_dim=32,
        model_dim=32,
        ffn_dim=64,
        num_heads=4,
        num_layers=3,
    ).eval()
    fine_output = fine(
        output,
        torch.randn(2, 7, 32),
        torch.tensor(
            [[False] * 6 + [True], [False] * 5 + [True, True]]
        ),
    )
    assert len(fine_output.layer_metric_logits) == 3
    assert all(next(iter(value.values())).shape == (2, 4) for value in fine_output.layer_metric_logits)
    rows = torch.arange(2)
    torch.testing.assert_close(
        fine_output.selected_trajectory_40,
        output.topk_trajectories_40[rows, fine_output.selected_topk_index],
    )


def test_static_vocabulary_is_not_persistent_in_checkpoint():
    coarse = _coarse()
    assert "static_vocab" not in coarse.state_dict()

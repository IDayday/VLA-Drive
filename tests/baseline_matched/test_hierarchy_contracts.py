import torch
from torch import nn

from starVLA.model.modules.planning.trajectory_codec import TrajectoryCodec
from starVLA.model.modules.scene_encoder import GlobalSceneQFormer
from starVLA.model.modules.trajectory_scorer import (
    DriveSuprimCoarseScorer,
    DriveSuprimFineRefiner,
    DrivoRDynamicScorer,
)
from starVLA.model.modules.trajectory_scorer.losses import (
    SUPRIM_METRICS,
    DriveSuprimMetricLoss,
)


def test_scene_query_shape_is_1_16_256():
    model = GlobalSceneQFormer(
        input_dim=2048,
        hidden_dim=256,
        output_dim=256,
        num_queries=16,
        num_layers=4,
        num_heads=8,
        ffn_dim=1024,
    )
    assert model.scene_queries.shape == (1, 16, 256)


def test_qformer_output_is_B_16_256():
    model = GlobalSceneQFormer(
        input_dim=64,
        hidden_dim=256,
        output_dim=256,
        num_queries=16,
        num_layers=1,
        num_heads=8,
        ffn_dim=1024,
    ).eval()
    output = model(torch.randn(2, 7, 64), torch.ones(2, 7, dtype=torch.long))
    assert output.global_tokens.shape == (2, 16, 256)


def test_dense_memory_is_B_L_256():
    model = GlobalSceneQFormer(
        input_dim=64,
        hidden_dim=256,
        output_dim=256,
        num_queries=16,
        num_layers=1,
        num_heads=8,
        ffn_dim=1024,
    ).eval()
    output = model(torch.randn(2, 7, 64), torch.ones(2, 7, dtype=torch.long))
    assert output.dense_memory.shape == (2, 7, 256)


def test_qformer_parameter_count_below_10m():
    model = GlobalSceneQFormer(
        input_dim=2048,
        hidden_dim=256,
        output_dim=256,
        num_queries=16,
        num_layers=4,
        num_heads=8,
        ffn_dim=1024,
    )
    assert model.parameter_count < 10_000_000


def test_qformer_padding_mask_semantics():
    model = GlobalSceneQFormer(
        input_dim=32,
        hidden_dim=32,
        output_dim=32,
        num_queries=4,
        num_layers=1,
        num_heads=4,
        ffn_dim=64,
    ).eval()
    attention_mask = torch.tensor([[1, 1, 0], [1, 0, 0]])
    output = model(torch.randn(2, 3, 32), attention_mask)
    assert torch.equal(output.memory_key_padding_mask, ~attention_mask.bool())


def _drivor():
    return DrivoRDynamicScorer(
        scene_dim=256,
        ego_state_dim=4,
        model_dim=256,
        ffn_dim=128,
        num_layers=4,
        num_heads=8,
    )


def test_drivor_receives_256_dim_scene_tokens():
    output = _drivor()(
        torch.randn(2, 64, 8, 3),
        torch.randn(2, 16, 256),
        torch.randn(2, 1, 4),
        topm=32,
    )
    assert output.aggregate_score.shape == (2, 64)
    assert output.topm_trajectories_8.shape == (2, 32, 8, 3)


def test_drivor_proposal_geometry_detached():
    proposals = torch.randn(2, 64, 8, 3, requires_grad=True)
    output = _drivor()(
        proposals,
        torch.randn(2, 16, 256),
        torch.randn(2, 1, 4),
        topm=32,
    )
    sum(value.mean() for value in output.metric_logits.values()).backward()
    assert proposals.grad is None


def test_drivor_top32_indices_align_with_trajectories():
    proposals = torch.randn(2, 64, 8, 3)
    output = _drivor()(
        proposals,
        torch.randn(2, 16, 256),
        torch.randn(2, 1, 4),
        topm=32,
    )
    expected = torch.gather(
        proposals,
        1,
        output.topm_indices[..., None, None].expand(-1, -1, 8, 3),
    )
    torch.testing.assert_close(output.topm_trajectories_8, expected)


class _IdentityDecoder(nn.Module):
    def forward(self, query, memory, *, memory_key_padding_mask=None):
        return query


def _production_pool_scorer():
    scorer = DriveSuprimCoarseScorer(
        static_vocab=torch.randn(8192, 40, 3),
        vocab_size=8192,
        num_poses=40,
        scene_dim=256,
        ego_state_dim=4,
        model_dim=256,
        ffn_dim=32,
        num_heads=8,
        num_layers=1,
        coarse_topk=256,
    )
    scorer.coarse_decoder = _IdentityDecoder()
    return scorer


def test_joint_candidate_count_is_8224():
    scorer = _production_pool_scorer()
    dynamic_40 = torch.randn(1, 32, 40, 3)
    candidates, metadata = scorer.build_joint_pool(
        1,
        torch.randn(1, 16, 256),
        dynamic_40,
        torch.arange(32)[None],
    )
    assert candidates.shape == (1, 8224, 40, 3)
    assert metadata.source.shape == (1, 8224)


def test_static_dynamic_share_same_embedding():
    scorer = _production_pool_scorer()
    assert hasattr(scorer, "candidate_embedding")
    assert not hasattr(scorer, "static_candidate_embedding")
    assert not hasattr(scorer, "dynamic_candidate_embedding")


def test_static_dynamic_share_same_heads():
    scorer = _production_pool_scorer()
    assert hasattr(scorer, "metric_heads")
    assert not hasattr(scorer, "static_metric_heads")
    assert not hasattr(scorer, "dynamic_metric_heads")


def test_no_source_embedding():
    scorer = _production_pool_scorer()
    assert all("source" not in name for name, _ in scorer.named_parameters())


def test_global_top256():
    scorer = _production_pool_scorer().eval()
    output = scorer(
        torch.randn(1, 16, 256),
        torch.randn(1, 1, 4),
        dynamic_trajectories_40=torch.randn(1, 32, 40, 3),
        dynamic_candidate_ids=torch.arange(32)[None],
    )
    assert output.aggregate_score.shape == (1, 8224)
    assert output.topk_indices.shape == (1, 256)


def test_fine_refinement_has_three_layers():
    fine = DriveSuprimFineRefiner(
        scene_dim=256,
        model_dim=256,
        ffn_dim=1024,
        num_heads=8,
        num_layers=3,
    )
    assert len(fine.fine_decoder.layers) == 3


def test_each_refinement_layer_has_loss():
    batch, candidates = 1, 4
    logits = [
        {
            name: torch.randn(batch, candidates, requires_grad=True)
            for name in (*SUPRIM_METRICS, "imi")
        }
        for _ in range(3)
    ]
    targets = {
        name: torch.rand(batch, candidates) for name in SUPRIM_METRICS
    }
    loss, details = DriveSuprimMetricLoss().refinement(
        logits,
        targets,
        torch.randn(batch, candidates, 40, 3),
        torch.randn(batch, 8, 3),
    )
    loss.backward()
    assert set(details) == {"layer_0", "layer_1", "layer_2"}
    assert all(layer["imi"].grad is not None for layer in logits)


def test_fine_memory_dim_is_256():
    fine = DriveSuprimFineRefiner(
        scene_dim=256,
        model_dim=256,
        ffn_dim=1024,
        num_heads=8,
        num_layers=3,
    )
    assert fine.scene_dim == 256
    assert all(layer.memory_dim == 256 for layer in fine.fine_decoder.layers)


def test_dynamic_upsampling_preserves_topm_candidate_ids():
    codec = TrajectoryCodec()
    dynamic = torch.randn(1, 32, 8, 3)
    assert codec.upsample_8_to_40(dynamic).shape == (1, 32, 40, 3)

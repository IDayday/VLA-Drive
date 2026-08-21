import torch

from starVLA.model.modules.action_model.multi_trajectory.donor_checkpoints import (
    convert_drivor_donor_state,
    convert_suprim_donor_state,
)


def test_drivor_conversion_transfers_semantic_heads_not_asymmetric_attention():
    target = {
        "trajectory_pos_embed.0.weight": torch.randn(4, 24),
        "metric_heads.pred_score.comfort.0.weight": torch.randn(4, 4),
        "ego_encoder.weight": torch.randn(4, 3),
        "scorer_decoder.layers.0.cross_attn.k_proj_weight": torch.randn(4, 32),
    }
    prefix = "agent._drivor_model."
    donor = {
        prefix + "pos_embed.0.weight": torch.randn(4, 24),
        prefix + "scorer.pred_score.comfort.0.weight": torch.randn(4, 4),
        prefix + "hist_encoding.weight": torch.randn(4, 11),
        prefix + "scorer_attention.layers.0.cross_attn.weight": torch.randn(4, 4),
    }
    converted, report = convert_drivor_donor_state(donor, target)
    torch.testing.assert_close(
        converted["trajectory_pos_embed.0.weight"], donor[prefix + "pos_embed.0.weight"]
    )
    torch.testing.assert_close(
        converted["metric_heads.pred_score.comfort.0.weight"],
        donor[prefix + "scorer.pred_score.comfort.0.weight"],
    )
    torch.testing.assert_close(
        converted["scorer_decoder.layers.0.cross_attn.k_proj_weight"],
        target["scorer_decoder.layers.0.cross_attn.k_proj_weight"],
    )
    assert "scorer_decoder.layers.0.cross_attn.k_proj_weight" in report.requires_training
    assert "ego_encoder.weight" in report.requires_training
    assert report.scene_dim == 2048
    assert report.planning_dim == 256
    assert not report.inference_ready


def test_suprim_conversion_keeps_new_asymmetric_decoders_initialized():
    vocab = torch.randn(2, 4, 3)
    target = {
        "static_vocab": vocab.clone(),
        "status_encoding.weight": torch.randn(4, 3),
        "_trajectory_head.pos_embed.0.weight": torch.randn(8, 120),
        "_trajectory_head.heads.imi.0.weight": torch.randn(8, 4),
        "_trajectory_head.transformer.layers.0.cross_attn.k_proj_weight": torch.randn(4, 32),
    }
    teacher_prefix = "agent.model.teacher.model."
    donor = {
        "agent.model.student.model._trajectory_head.vocab": vocab.clone(),
        teacher_prefix + "_trajectory_head.vocab": vocab.clone(),
        teacher_prefix + "_trajectory_head.pos_embed.0.weight": torch.randn(8, 120),
        teacher_prefix + "_trajectory_head.heads.imi.0.weight": torch.randn(8, 4),
        teacher_prefix + "_trajectory_head.transformer.layers.0.multihead_attn.in_proj_weight": torch.randn(12, 4),
    }
    converted, report = convert_suprim_donor_state(donor, target)
    torch.testing.assert_close(converted["static_vocab"], vocab)
    torch.testing.assert_close(
        converted["_trajectory_head.pos_embed.0.weight"],
        donor[teacher_prefix + "_trajectory_head.pos_embed.0.weight"],
    )
    key = "_trajectory_head.transformer.layers.0.cross_attn.k_proj_weight"
    torch.testing.assert_close(converted[key], target[key])
    assert key in report.requires_training
    assert "status_encoding.weight" in report.requires_training
    assert report.scene_dim == 2048
    assert report.planning_dim == 256
    assert not report.inference_ready

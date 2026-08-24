import inspect
import json
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf
from torch import nn

from starVLA.model.framework.QwenOFT import Qwenvl_OFT
from starVLA.model.framework.QwenOFT_GPSQ3DMix import QwenOFT_GPSQ3DMix
from starVLA.model.framework.QwenOFT_SQ3DMix import (
    QwenOFT_SQ3DMix,
    Qwenvl_OFT_SQ3DMix,
)
from starVLA.model.modules.vggt_query.gp_slot_stats import load_gp_slot_stats, sha256_file
from starVLA.model.modules.vggt_query.scene_query_compressor import SceneQueryCompressor
from starVLA.model.modules.vggt_query.sq_3d_mix import SceneConditionedGatedFusion


def _stats(tmp_path):
    stats = tmp_path / "gp_sq3dmix_pooled_stats.pt"
    torch.save({"pooled_feature_slot_mean": torch.zeros(180, 2048)}, stats)
    manifest = {
        "complete": True,
        "source_cache_manifest_sha256": "cache",
        "datalist_sha256": "data",
        "sample_count": 103288,
        "view_order": ["cam_f0", "cam_l0", "cam_r0"],
        "pooling_layout": [3, 6, 10],
        "feature_dimension": 2048,
        "code_commit": "abc",
        "stats_file_sha256": sha256_file(stats),
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    return stats


def test_slot_stats_manifest_and_hash_validation(tmp_path):
    stats = _stats(tmp_path)
    mean, manifest = load_gp_slot_stats(
        tmp_path,
        expected_source_cache_manifest_sha256="cache",
        expected_datalist_sha256="data",
    )
    assert mean.shape == (180, 2048)
    assert manifest["sample_count"] == 103288
    stats.write_bytes(stats.read_bytes() + b"corrupt")
    with pytest.raises(RuntimeError, match="SHA256"):
        load_gp_slot_stats(tmp_path)


def test_new_framework_directly_inherits_baseline_and_old_classes_remain():
    assert QwenOFT_GPSQ3DMix.__bases__ == (Qwenvl_OFT,)
    assert QwenOFT_SQ3DMix is Qwenvl_OFT_SQ3DMix
    assert Qwenvl_OFT_SQ3DMix is not None
    assert SceneQueryCompressor is not None
    assert SceneConditionedGatedFusion is not None


def test_disabled_mode_is_exact_action_only_identity():
    model = QwenOFT_GPSQ3DMix.__new__(QwenOFT_GPSQ3DMix)
    nn.Module.__init__(model)
    model.gp_mode = "disabled"
    queries = torch.randn(2, 8, 2048)
    conditioned, context, metrics = model._condition_action_queries(queries, {})
    assert conditioned is queries
    assert context is None
    assert metrics == {}


class _FakeActionModel:
    def __init__(self):
        self.states = []
        self.random_draws = []

    def sample_flow_state(self, actions):
        return object()

    def loss_from_flow_state(self, queries, actions, state, **kwargs):
        self.states.append((id(state), kwargs.get("extra_context")))
        draw = torch.rand(())
        self.random_draws.append(draw.detach().clone())
        return (
            queries.float().square().mean(dim=(1, 2))
            + actions.square().mean(dim=(1, 2))
            + 0.0 * draw
        )


def test_stage_a_v2_losses_finite_and_interventions_reuse_flow_noise():
    model = QwenOFT_GPSQ3DMix.__new__(QwenOFT_GPSQ3DMix)
    nn.Module.__init__(model)
    model.gp_mode = "gated_residual"
    model.training_stage = "stage_a"
    model.rank_margin_ratio = 0.05
    model.spatial_margin_ratio = 0.02
    model.fidelity_tolerance = 0.02
    model.config = OmegaConf.create({"framework": {"action_model": {"repeated_diffusion_steps": 1}}})
    model.action_model = _FakeActionModel()
    actions = torch.randn(2, 8, 4)
    extension = {
        "baseline_action_queries": torch.randn(2, 8, 16),
        "real_action_queries": torch.randn(2, 8, 16, requires_grad=True),
        "hard_shuffled_action_queries": torch.randn(2, 8, 16, requires_grad=True),
        "spatial_shuffled_action_queries": torch.randn(2, 8, 16, requires_grad=True),
        "losses": {},
        "metrics": {},
    }
    torch.manual_seed(123)
    _ = torch.rand(())
    expected_next_draw = torch.rand(())
    torch.manual_seed(123)
    action = model._compute_action_loss(extension["real_action_queries"], actions, None, None, extension)
    losses = {"action": action, **extension["losses"]}
    assert set(losses) == {
        "action",
        "geometry_rank_hard",
        "geometry_rank_spatial",
        "baseline_fidelity",
    }
    assert all(value.ndim == 0 and torch.isfinite(value) for value in losses.values())
    assert len({state for state, _ in model.action_model.states}) == 1
    assert all(context is None for _, context in model.action_model.states)
    assert all(
        torch.equal(model.action_model.random_draws[0], draw)
        for draw in model.action_model.random_draws[1:]
    )
    assert torch.equal(torch.rand(()), expected_next_draw)


def test_framework_never_passes_extra_context_to_dit():
    source = inspect.getsource(QwenOFT_GPSQ3DMix._compute_action_loss)
    assert "extra_context=None" in source

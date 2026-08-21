from dataclasses import replace
import inspect

import torch
from torch import nn

from starVLA.model.modules.action_model.multi_trajectory.config import (
    PlanningConfig,
    SuprimConfig,
)
from starVLA.model.modules.action_model.multi_trajectory.ddp_multi_sampler import (
    DDPMultiSampler,
)
from starVLA.model.modules.action_model.multi_trajectory.planner import (
    DDPDrivoRSuprimPlanner,
)
from starVLA.model.modules.action_model.multi_trajectory.scene_context import (
    SceneContext,
)
from starVLA.model.modules.action_model.multi_trajectory.suprim_joint_selector import (
    DriveSuprimJointSelector,
    RefineTrajHead,
    drivesuprim_coarse_aggregate,
    drivesuprim_fine_aggregate,
)
from starVLA.model.modules.action_model.multi_trajectory.trajectory_resampler import (
    STATIC_SAMPLE_INDICES,
    trajectory_8_to_40,
)


SMALL_PLANNING = PlanningConfig(
    planning_dim=12, num_heads=3, ffn_dim=24, dropout=0.0
)


def _small_suprim_config(vocab_size=8, coarse_topk=4, refinement_layers=3):
    return SuprimConfig(
        vocab_size=vocab_size,
        num_trajectory_points=40,
        coarse_topk=coarse_topk,
        coarse_layers=1,
        num_refinement_stages=1,
        refinement_layers=refinement_layers,
        use_mid_output=True,
        use_separate_heads=True,
        use_imitation_head=True,
    )


def _small_selector(vocab_size=8, coarse_topk=4, fine_source="dense_qwen_memory"):
    config = replace(
        _small_suprim_config(vocab_size, coarse_topk),
        fine_memory_source=fine_source,
    )
    return DriveSuprimJointSelector(
        config,
        planning_config=SMALL_PLANNING,
        scene_dim=24,
        ego_status_dim=3,
        static_vocab=torch.randn(vocab_size, 40, 3),
        enforce_fidelity=False,
    )


def _context(batch=2, global_length=5, dense_length=7):
    mask = torch.zeros(batch, dense_length, dtype=torch.bool)
    mask[:, -1] = True
    return SceneContext(
        global_scene_tokens=torch.randn(batch, global_length, 24),
        dense_scene_memory=torch.randn(batch, dense_length, 24),
        memory_key_padding_mask=mask,
    )


def _dynamic_trajectories(batch=2, candidates=2):
    trajectory = torch.randn(batch, candidates, 8, 3)
    trajectory[..., 2] = torch.remainder(
        trajectory[..., 2] + torch.pi, 2 * torch.pi
    ) - torch.pi
    return trajectory


def test_suprim_static_only_candidate_count():
    selector = DriveSuprimJointSelector(
        _small_suprim_config(vocab_size=8192, coarse_topk=256),
        planning_config=SMALL_PLANNING,
        scene_dim=24,
        static_vocab=torch.zeros(8192, 40, 3),
        enforce_fidelity=False,
    )
    candidates, metadata = selector.build_joint_candidates(batch_size=2)
    assert candidates.shape == (2, 8192, 40, 3)
    assert selector.config.coarse_topk == 256
    assert torch.count_nonzero(metadata.source) == 0
    assert not selector.static_vocab.requires_grad


def test_suprim_static_only_output_contract():
    selector = _small_selector().eval()
    output = selector(_context(), ego_status=torch.randn(2, 1, 3))
    assert output.selected_trajectory_40.shape == (2, 40, 3)
    assert output.selected_trajectory_8.shape == (2, 8, 3)
    assert output.top256_indices.shape == (2, 4)
    assert torch.equal(output.selected_source, torch.zeros(2, dtype=torch.long))
    torch.testing.assert_close(
        output.selected_trajectory_8,
        output.selected_trajectory_40[:, list(STATIC_SAMPLE_INDICES)],
    )


def test_suprim_three_refinement_layers_emit_intermediate_outputs():
    selector = _small_selector().eval()
    output = selector(_context(batch=1), ego_status=torch.randn(1, 1, 3))
    assert len(output.fine_scores["layer_results"]) == 3


def test_suprim_official_aggregate_formulas():
    scores = {
        "no_at_fault_collisions": torch.randn(2, 7),
        "drivable_area_compliance": torch.randn(2, 7),
        "time_to_collision_within_bound": torch.randn(2, 7),
        "ego_progress": torch.randn(2, 7),
        "driving_direction_compliance": torch.randn(2, 7),
        "lane_keeping": torch.randn(2, 7),
        "traffic_light_compliance": torch.randn(2, 7),
        "history_comfort": torch.randn(2, 7),
        "imi": torch.randn(2, 7),
    }
    additive = (
        5.0 * scores["time_to_collision_within_bound"].sigmoid()
        + 5.0 * scores["ego_progress"].sigmoid()
        + 2.0 * scores["lane_keeping"].sigmoid()
        + scores["history_comfort"].sigmoid()
    )
    donor = (
        0.1 * scores["traffic_light_compliance"].sigmoid().log()
        + 0.5 * scores["no_at_fault_collisions"].sigmoid().log()
        + 0.5 * scores["drivable_area_compliance"].sigmoid().log()
        + 0.3 * scores["driving_direction_compliance"].sigmoid().log()
        + 6.0 * additive.log()
        + 0.02 * scores["imi"].softmax(-1).log()
    )
    torch.testing.assert_close(drivesuprim_fine_aggregate(scores), donor)
    torch.testing.assert_close(drivesuprim_coarse_aggregate(scores), donor)


def test_joint_candidate_count():
    selector = _small_selector()
    dynamic_8 = _dynamic_trajectories(candidates=3)
    candidates, _ = selector.build_joint_candidates(
        2, trajectory_8_to_40(dynamic_8)
    )
    assert candidates.shape == (2, 11, 40, 3)


def test_suprim_global_top256():
    selector = DriveSuprimJointSelector(
        _small_suprim_config(vocab_size=8192, coarse_topk=256),
        planning_config=SMALL_PLANNING,
        scene_dim=24,
        ego_status_dim=3,
        static_vocab=torch.zeros(8192, 40, 3),
        enforce_fidelity=False,
    )
    dynamic_8 = _dynamic_trajectories(batch=1, candidates=16)
    candidates, metadata = selector.build_joint_candidates(
        1, trajectory_8_to_40(dynamic_8)
    )
    assert candidates.shape == (1, 8208, 40, 3)
    assert metadata.source.shape == (1, 8208)
    scores = torch.arange(8208, dtype=torch.float32)[None]
    top = scores.topk(256, dim=1).indices
    gathered = metadata.gather(top)
    assert top.shape == (1, 256)
    assert torch.equal(gathered.source[:, :16], torch.ones(1, 16, dtype=torch.long))


def test_joint_candidate_source_mapping():
    selector = _small_selector()
    dynamic_8 = _dynamic_trajectories(candidates=3)
    candidate_ids = torch.tensor([[9, 4, 7], [8, 6, 3]])
    _, metadata = selector.build_joint_candidates(
        2, trajectory_8_to_40(dynamic_8), candidate_ids
    )
    assert torch.equal(metadata.source[:, :8], torch.zeros(2, 8, dtype=torch.long))
    assert torch.equal(metadata.source[:, 8:], torch.ones(2, 3, dtype=torch.long))
    assert torch.equal(metadata.source_index[:, 8:], torch.tensor([[0, 1, 2]]).expand(2, -1))
    assert torch.equal(metadata.dynamic_candidate_id[:, 8:], candidate_ids)


def test_joint_topk_metadata_propagation():
    selector = _small_selector().eval()
    dynamic_8 = _dynamic_trajectories(candidates=2)
    dynamic_40 = trajectory_8_to_40(dynamic_8)
    candidate_ids = torch.tensor([[31, 12], [7, 22]])
    output = selector(
        _context(), dynamic_8, dynamic_40, candidate_ids, torch.randn(2, 1, 3)
    )
    _, full_metadata = selector.build_joint_candidates(2, dynamic_40, candidate_ids)
    expected = full_metadata.gather(output.top256_indices.cpu())
    assert torch.equal(output.top256_metadata.source.cpu(), expected.source)
    assert torch.equal(output.top256_metadata.source_index.cpu(), expected.source_index)
    assert torch.equal(
        output.top256_metadata.dynamic_candidate_id.cpu(),
        expected.dynamic_candidate_id,
    )


def test_dynamic_output_uses_original_traj8():
    dynamic_8 = _dynamic_trajectories(batch=2, candidates=3)
    selected = DriveSuprimJointSelector.resolve_selected_trajectory_8(
        torch.zeros(2, 40, 3),
        selected_source=torch.ones(2, dtype=torch.long),
        selected_source_index=torch.tensor([2, 1]),
        dynamic_traj8=dynamic_8,
    )
    torch.testing.assert_close(selected[0], dynamic_8[0, 2])
    torch.testing.assert_close(selected[1], dynamic_8[1, 1])


def test_static_output_uses_expected_sample_indices():
    selected_40 = torch.arange(2 * 40 * 3, dtype=torch.float32).reshape(2, 40, 3)
    selected = DriveSuprimJointSelector.resolve_selected_trajectory_8(
        selected_40,
        selected_source=torch.zeros(2, dtype=torch.long),
        selected_source_index=torch.tensor([3, 5]),
        dynamic_traj8=None,
    )
    torch.testing.assert_close(selected, selected_40[:, list(STATIC_SAMPLE_INDICES)])


def test_suprim_fine_dense_memory_shape():
    head = RefineTrajHead(
        planning_dim=256,
        memory_dim=2048,
        num_heads=8,
        ffn_dim=1024,
        num_layers=3,
        dropout=0.0,
        use_mid_output=True,
        use_imitation=True,
    ).eval()
    mask = torch.tensor([[False, False, False, True]])
    with torch.no_grad():
        layers, aggregate, selected = head(
            torch.randn(1, 4, 2048), mask, torch.randn(1, 256, 256)
        )
    assert len(layers) == 3
    assert all(next(iter(values.values())).shape == (1, 256) for values in layers)
    assert aggregate.shape == (1, 256)
    assert selected.shape == (1,)


def test_suprim_fine_padding_mask_propagation():
    selector = _small_selector().eval()
    context = _context(batch=1)
    seen_masks = []
    for layer in selector._trajectory_offset_head.transformer.layers:
        original = layer.cross_attn.forward

        def wrapped(*args, _original=original, **kwargs):
            seen_masks.append(kwargs.get("key_padding_mask"))
            return _original(*args, **kwargs)

        layer.cross_attn.forward = wrapped
    with torch.no_grad():
        selector(context, ego_status=torch.randn(1, 1, 3))
    assert len(seen_masks) == 3
    assert all(mask is context.memory_key_padding_mask for mask in seen_masks)


def test_suprim_global_memory_ablation():
    selector = _small_selector(fine_source="global_scene_tokens").eval()
    output = selector(_context(batch=1), ego_status=torch.randn(1, 1, 3))
    assert output.selected_trajectory_8.shape == (1, 8, 3)
    assert len(output.fine_scores["layer_results"]) == 3


class _FrozenDDP(nn.Module):
    action_horizon = 8
    action_dim = 3

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))

    def predict_action(self, vl_embs_list, state=None):
        return torch.randn(vl_embs_list[0].shape[0], 8, 3) * self.weight


def test_train_suprim_has_no_ddp_or_drivor_grad():
    ddp = _FrozenDDP()
    dynamic_8 = DDPMultiSampler(ddp, 2).sample(
        [torch.randn(2, 3, 5)], torch.randn(2, 1, 3), seed=4
    )
    scene_projection = nn.Linear(3, 24)
    for parameter in scene_projection.parameters():
        parameter.requires_grad = False
    context = SceneContext(
        scene_projection(torch.randn(2, 5, 3)),
        scene_projection(torch.randn(2, 7, 3)),
        torch.zeros(2, 7, dtype=torch.bool),
    )
    selector = _small_selector().train()
    output = selector(
        context,
        dynamic_8,
        trajectory_8_to_40(dynamic_8),
        ego_status=torch.randn(2, 1, 3),
    )
    output.coarse_scores["aggregate_score"].mean().backward()
    assert ddp.weight.grad is None
    assert all(parameter.grad is None for parameter in scene_projection.parameters())
    assert any(parameter.grad is not None for parameter in selector.parameters())


def test_no_shared_scene_projection_to_256():
    selector = DriveSuprimJointSelector(
        _small_suprim_config(),
        planning_config=PlanningConfig(),
        scene_dim=2048,
        static_vocab=torch.zeros(8, 40, 3),
        enforce_fidelity=False,
    )
    assert not any(
        isinstance(module, nn.Linear)
        and module.in_features == 2048
        and module.out_features == 256
        for name, module in selector.named_modules()
        if "cross_attn" not in name
    )
    assert "scene_memory_adapter" not in dict(selector.named_modules())


def test_no_pseudo_spatial_reshape():
    source = inspect.getsource(DriveSuprimJointSelector)
    assert "patch_feature_map" not in source
    assert "Conv2d" not in source
    assert ".flatten(2)" not in source


def test_inference_does_not_read_metric_cache():
    source = inspect.getsource(DDPDrivoRSuprimPlanner.forward_with_outputs)
    assert "metric_cache" not in source
    assert "cache_schema" not in source
    assert "np.load" not in source


def test_inference_does_not_require_targets():
    parameters = inspect.signature(DDPDrivoRSuprimPlanner.forward).parameters
    assert "targets" not in parameters
    assert "ground_truth" not in parameters


def test_no_dino_dependency():
    planner_source = inspect.getsource(DDPDrivoRSuprimPlanner)
    selector_source = inspect.getsource(DriveSuprimJointSelector)
    forbidden = ("ImgEncoder", "LoRA_ViT_timm", "dinov2", "scene_embeds")
    assert all(name not in planner_source + selector_source for name in forbidden)

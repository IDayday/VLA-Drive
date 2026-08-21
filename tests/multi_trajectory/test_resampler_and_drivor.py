from dataclasses import replace

import torch

from starVLA.model.modules.action_model.multi_trajectory.config import (
    DrivoRConfig,
    PlanningConfig,
)
from starVLA.model.modules.action_model.multi_trajectory.drivor_dynamic_scorer import (
    DrivoRDynamicScorer,
    Scorer,
)
from starVLA.model.modules.action_model.multi_trajectory.trajectory_resampler import (
    STATIC_SAMPLE_INDICES,
    trajectory_8_to_40,
)


def _wrapped_random_trajectories(*leading):
    trajectory = torch.randn(*leading, 8, 3)
    trajectory[..., 2] = torch.remainder(
        trajectory[..., 2] + torch.pi, 2 * torch.pi
    ) - torch.pi
    return trajectory


def _small_scorer():
    return DrivoRDynamicScorer(
        replace(DrivoRConfig(), scorer_layers=1, dynamic_topk=2),
        ego_status_dim=3,
        planning_config=PlanningConfig(
            planning_dim=16, num_heads=4, ffn_dim=32, dropout=0.0
        ),
        scene_dim=32,
    )


def test_trajectory_8_to_40_anchor_invariance():
    trajectory_8 = _wrapped_random_trajectories(2, 5)
    trajectory_40 = trajectory_8_to_40(trajectory_8)
    assert trajectory_40.shape == (2, 5, 40, 3)
    torch.testing.assert_close(
        trajectory_40[..., list(STATIC_SAMPLE_INDICES), :],
        trajectory_8,
        rtol=1e-5,
        atol=1e-6,
    )


def test_heading_wrap_interpolation():
    trajectory_8 = torch.zeros(1, 8, 3)
    trajectory_8[0, 0, 2] = torch.deg2rad(torch.tensor(179.0))
    trajectory_8[0, 1:, 2] = torch.deg2rad(torch.tensor(-179.0))
    trajectory_40 = trajectory_8_to_40(trajectory_8)
    assert trajectory_40[0, 6, 2].abs() > 3.0
    torch.testing.assert_close(
        trajectory_40[0, list(STATIC_SAMPLE_INDICES), 2],
        trajectory_8[0, :, 2],
        rtol=1e-5,
        atol=1e-6,
    )


def test_drivor_score_formula_equivalence():
    config = DrivoRConfig()
    sub_scores = {name: torch.randn(3, 11) for name in Scorer.SCORE_NAMES}
    stable = DrivoRDynamicScorer.aggregate_sub_scores(sub_scores, config)
    donor = (
        config.noc_weight * sub_scores["no_at_fault_collisions"].sigmoid().log()
        + config.dac_weight
        * sub_scores["drivable_area_compliance"].sigmoid().log()
        + config.ddc_weight
        * sub_scores["driving_direction_compliance"].sigmoid().log()
        + (
            config.ttc_weight
            * sub_scores["time_to_collision_within_bound"].sigmoid()
            + config.ep_weight * sub_scores["ego_progress"].sigmoid()
            + config.comfort_weight * sub_scores["comfort"].sigmoid()
        ).log()
    )
    torch.testing.assert_close(stable, donor, rtol=1e-6, atol=1e-6)


def test_drivor_scene_and_planning_dims():
    scorer = _small_scorer().eval()
    output = scorer(
        proposals=_wrapped_random_trajectories(2, 64),
        global_scene_tokens=torch.randn(2, 16, 32),
        ego_status=torch.randn(2, 1, 3),
    )
    assert output.score_states.shape == (2, 64, 16)
    assert output.topk_trajectories.shape == (2, 2, 8, 3)


def test_drivor_proposal_detach():
    scorer = _small_scorer().eval()
    proposals = _wrapped_random_trajectories(2, 4).requires_grad_()
    scene = torch.randn(2, 7, 32, requires_grad=True)
    output = scorer(proposals, scene, torch.randn(2, 1, 3))
    output.aggregate_score.sum().backward()
    assert proposals.grad is None
    assert scene.grad is not None
    assert not output.topk_trajectories.requires_grad
    assert any(parameter.grad is not None for parameter in scorer.parameters())


def test_drivor_topk_metadata():
    scorer = _small_scorer().eval()
    proposals = _wrapped_random_trajectories(2, 5)
    output = scorer(
        proposals,
        global_scene_tokens=torch.randn(2, 6, 32),
        ego_status=torch.randn(2, 3),
        topk=2,
    )
    expected_indices = output.aggregate_score.topk(2, dim=1).indices
    expected = torch.gather(
        proposals,
        1,
        expected_indices[..., None, None].expand(-1, -1, 8, 3),
    )
    assert torch.equal(output.topk_indices, expected_indices)
    torch.testing.assert_close(output.topk_trajectories, expected)

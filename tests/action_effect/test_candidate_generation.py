from __future__ import annotations

import numpy as np
import pytest

from research.action_effect.candidate_generator import (
    CandidateGeneratorConfig,
    KinematicLimits,
    PolicyLocalCandidateGenerator,
)


def straight_anchor() -> np.ndarray:
    x = np.arange(1, 9, dtype=np.float64) * 2.0
    return np.stack([x, np.zeros_like(x), np.zeros_like(x)], axis=1)


def test_generator_is_deterministic_and_keeps_anchor_exact() -> None:
    generator = PolicyLocalCandidateGenerator(CandidateGeneratorConfig())
    first = generator.generate(straight_anchor(), scene_id="scene", anchor_type="expert", seed=7)
    second = generator.generate(straight_anchor(), scene_id="scene", anchor_type="expert", seed=7)

    assert len(first) == 16
    assert [value.candidate_id for value in first] == [value.candidate_id for value in second]
    np.testing.assert_array_equal(first[0].trajectory, straight_anchor().astype(np.float32))
    assert first[0].perturbation_type == "anchor"


def test_candidate_families_cover_requested_local_actions() -> None:
    generator = PolicyLocalCandidateGenerator(CandidateGeneratorConfig())
    candidates = generator.generate(straight_anchor(), scene_id="scene", anchor_type="expert", seed=11)
    types = {value.perturbation_type for value in candidates}
    assert types == {
        "anchor",
        "lateral_terminal_offset",
        "speed_scale",
        "brake_onset_shift",
        "terminal_progress_shift",
        "curvature_scale",
        "turn_inner_outer_offset",
    }
    lateral = [value for value in candidates if value.perturbation_type == "lateral_terminal_offset"]
    assert min(value.trajectory[-1, 1] for value in lateral) < 0
    assert max(value.trajectory[-1, 1] for value in lateral) > 0
    speeds = {value.perturbation_parameters["scale"]: value for value in candidates if value.perturbation_type == "speed_scale"}
    assert np.linalg.norm(speeds[0.8].trajectory[-1, :2]) < np.linalg.norm(straight_anchor()[-1, :2])
    assert np.linalg.norm(speeds[1.1].trajectory[-1, :2]) > np.linalg.norm(straight_anchor()[-1, :2])


def test_validation_rejects_jump_and_yaw_discontinuity() -> None:
    generator = PolicyLocalCandidateGenerator(CandidateGeneratorConfig())
    bad = straight_anchor()
    bad[4, :2] = [100.0, 100.0]
    bad[4, 2] = np.pi
    validation = generator.validate(bad, straight_anchor())
    assert not validation.kinematic_valid
    assert {"max_speed", "position_jump", "yaw_discontinuity"} & set(validation.reasons)


def test_candidate_count_outside_local_budget_is_rejected() -> None:
    config = CandidateGeneratorConfig(
        lateral_offsets_m=(),
        speed_scales=(),
        brake_onset_shifts_s=(),
        terminal_progress_shifts_m=(),
        curvature_scales=(),
        turn_offsets_m=(),
    )
    with pytest.raises(ValueError, match="candidate count"):
        PolicyLocalCandidateGenerator(config)


def test_route_validity_is_separate_from_kinematic_validity() -> None:
    config = CandidateGeneratorConfig(limits=KinematicLimits(route_corridor_m=0.1))
    generator = PolicyLocalCandidateGenerator(config)
    shifted = straight_anchor()
    shifted[:, 1] = np.linspace(0.0, 0.5, 8)
    validation = generator.validate(shifted, straight_anchor())
    assert not validation.route_valid
    assert "anchor_route_corridor" in validation.reasons

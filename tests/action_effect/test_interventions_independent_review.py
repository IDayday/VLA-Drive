"""Independent CPU adversarial checks, not real controller/safety evidence."""
import subprocess
import sys
import types

import numpy as np
import pytest

from starVLA.model.modules.action_effect.intervention_basis import (
    BasisSpec, basis_matrix, intervention_grid, reconstruct, wrap_angle,
)
from starVLA.model.modules.action_effect.trajectory_constraints import (
    ConstraintSpec, geometric_screen, tracking_screen,
)

OLD_OWNER_COMMIT = "00a5b754a6d772d5184697effedbe62be1762788"


def straight(speed=5.):
    return np.column_stack((np.arange(1, 9) * .5 * speed, np.zeros(8), np.zeros(8)))


def historical_module(name, relative_path):
    """Read an exact historical primitive, without editing its old evidence."""
    source = subprocess.check_output(["git", "show", f"{OLD_OWNER_COMMIT}:{relative_path}"], text=True)
    module = types.ModuleType(name)
    module.__package__ = "starVLA.model.modules.action_effect"
    sys.modules[name] = module
    exec(compile(source, f"{OLD_OWNER_COMMIT}:{relative_path}", "exec"), module.__dict__)
    return module


def test_original_early_window_underreports_hold_tail_current_records_both():
    old = historical_module("_review_old_basis", "starVLA/model/modules/action_effect/intervention_basis.py")
    before = old.intervention_grid(old.BasisSpec())[1]
    current = intervention_grid(BasisSpec())[1]
    a = straight()
    modified = reconstruct(a, current["coefficients"], BasisSpec())
    assert before["active_intervals_seconds"] == [[0., 2.]]
    assert current["ramp_intervals_seconds"] == [[0., 2.]]
    assert current["active_intervals_seconds"] == [[0., 4.]]
    assert np.all(np.linalg.norm(modified[4:, :2] - a[4:, :2], axis=-1) > 0.)


def test_original_low_speed_curvature_is_hidden_current_rejects():
    old = historical_module("_review_old_constraints", "starVLA/model/modules/action_effect/trajectory_constraints.py")
    times = np.arange(1, 9) * .5
    radius, yaw_rate = .1, .5
    angle = yaw_rate * times
    poses = np.column_stack((radius * np.sin(angle), radius * (1 - np.cos(angle)), angle))
    prior = old.geometric_screen(poses, BasisSpec(), ConstraintSpec(), initial_velocity_xy=[radius*yaw_rate, 0.])
    current = geometric_screen(poses, BasisSpec(), ConstraintSpec(), initial_velocity_xy=[radius*yaw_rate, 0.])
    assert prior["geometrically_admissible"]
    assert prior["maxima"]["curvature_inv_m"] == 0.
    assert not current["geometrically_admissible"]
    assert "curvature_inv_m" in current["failures"]
    assert current["maxima"]["curvature_inv_m"] > 9.
    assert current["curvature_defined_steps"] == 8


def test_original_turn_in_place_passed_current_reports_undefined_translation():
    old = historical_module("_review_old_stationary", "starVLA/model/modules/action_effect/trajectory_constraints.py")
    poses = np.zeros((8, 3), dtype=np.float64)
    poses[:, 2] = np.arange(1, 9) * .05
    assert old.geometric_screen(poses, BasisSpec(), ConstraintSpec())["geometrically_admissible"]
    current = geometric_screen(poses, BasisSpec(), ConstraintSpec(), initial_velocity_xy=[0., 0.])
    assert not current["geometrically_admissible"]
    assert current["curvature_defined_steps"] == 0
    assert current["undefined_stationary_turn_steps"] == 8
    assert "yaw_change_without_translation" in current["failures"]
    assert "score" not in current and current["is_safety_label"] is False


@pytest.mark.parametrize("heading", [0., 1.2, 3.05, -3.05])
def test_global_rear_axle_curved_execution_rotation_and_wrapped_yaw(heading):
    native_times = np.arange(41) * .1
    radius, omega = 8., .5
    local = np.column_stack((radius * np.sin(omega*native_times), radius*(1-np.cos(omega*native_times))))
    c, s = np.cos(heading), np.sin(heading)
    states = np.zeros((41, 11), dtype=np.float64)
    states[:, :2] = local @ np.array([[c, s], [-s, c]]) + np.array([1234.5, -987.25])
    states[:, 2] = wrap_angle(heading + omega*native_times)
    poses = np.column_stack((local[5::5], omega*native_times[5::5]))
    result = tracking_screen(poses, states, BasisSpec(), ConstraintSpec())
    assert result["controller_tracking_pass"]
    assert result["max_xy_error_m"] < 5e-13
    assert result["max_yaw_error_rad"] < 1e-14
    assert result["sampled_state_indices"] == [5, 10, 15, 20, 25, 30, 35, 40]
    assert result["is_official_metric"] is False


def test_initial_local_velocity_changes_first_acceleration_screen():
    a = straight()
    good = geometric_screen(a, BasisSpec(), ConstraintSpec(), initial_velocity_xy=[5., 0.])
    wrong = geometric_screen(a, BasisSpec(), ConstraintSpec(), initial_velocity_xy=[20., 0.])
    assert good["geometrically_admissible"]
    assert good["maxima"]["acceleration_mps2"] == 0.
    assert wrong["maxima"]["acceleration_mps2"] == 30.
    assert not wrong["geometrically_admissible"]
    assert geometric_screen(a, BasisSpec(), ConstraintSpec())["initial_velocity_observed"] is False


def test_zero_intervention_preserves_float64_bits_and_principal_yaw():
    a = straight()
    a[:, 1] = -0.
    a[:, 2] = [np.pi, -np.pi, 3.12, -3.12, .4, -.4, 0., -0.]
    zero = np.zeros(8, dtype=np.float64)
    zero[::2] = -0.
    b = reconstruct(a, zero, BasisSpec())
    assert a.dtype == b.dtype
    assert a.tobytes() == b.tobytes()
    assert not np.shares_memory(a, b)


def test_path_heading_axes_and_active_tail_change_on_a_curve():
    a = straight()
    a[:, 2] = np.linspace(.1, .8, 8)
    matrix = basis_matrix(a, BasisSpec())
    along_tail, lateral_tail = matrix[-1, :, 0], matrix[-1, :, 4]
    np.testing.assert_allclose(along_tail, [np.cos(.8), np.sin(.8)], rtol=0, atol=1e-15)
    np.testing.assert_allclose(lateral_tail, [-np.sin(.8), np.cos(.8)], rtol=0, atol=1e-15)
    assert abs(float(along_tail @ lateral_tail)) < 1e-15
    assert np.linalg.norm(matrix[-1, :, 0] - matrix[4, :, 0]) > .1


@pytest.mark.parametrize("xy_error,expected", [(1.999999, True), (2.000001, False)])
def test_tracking_threshold_uses_metres_not_normalized_action_units(xy_error, expected):
    a = straight()
    states = np.zeros((41, 11), dtype=np.float64)
    states[:, 0] = np.arange(41) * .1 * 5.
    states[1:, 1] = xy_error
    result = tracking_screen(a, states, BasisSpec(), ConstraintSpec())
    assert result["controller_tracking_pass"] is expected
    assert result["max_xy_error_m"] == xy_error


def test_failed_time_grid_and_nonfinite_execution_do_not_return_a_zero_score():
    states = np.zeros((41, 11), dtype=np.float64)
    with pytest.raises(ValueError, match="time grids"):
        tracking_screen(straight(), states, BasisSpec(), ConstraintSpec(), simulation_dt=.12)
    states[3, 0] = np.nan
    with pytest.raises(ValueError, match="native execution"):
        tracking_screen(straight(), states, BasisSpec(), ConstraintSpec())


def test_undefined_base_tangent_retains_heading_requires_real_tracking_not_safety():
    # Disclosure boundary: reconstruct is array-only and does not emit an
    # undefined-tangent mask. A geometric result is never the executable label.
    a = np.zeros((8, 3), dtype=np.float64)
    coefficients = np.zeros(8)
    coefficients[4] = .3
    b = reconstruct(a, coefficients, BasisSpec())
    assert b[-1, 1] == .3
    np.testing.assert_array_equal(b[:, 2], a[:, 2])
    result = geometric_screen(b, BasisSpec(), ConstraintSpec(), initial_velocity_xy=[0., 0.])
    assert result["executable"] == "REQUIRES_REAL_CONTROLLER_REPLAY"
    assert result["is_safety_label"] is False

import pytest

from scripts.select_formal_training_layout import (
    LAYOUT_SPECS,
    build_layout_lock,
    select_layout,
    validate_layout_metrics,
)


def _metrics(layout, throughput, *, peak=60.0):
    spec = LAYOUT_SPECS[layout]
    return {
        "layout": layout,
        "status": "success",
        "timed_optimizer_steps": 300,
        **spec,
        "num_workers_per_rank": 6,
        "samples_per_second": throughput,
        "peak_allocated_gib": peak,
        "oom": False,
        "deadlock": False,
        "nonfinite_count": 0,
    }


def _all_metrics(global64_throughput=130.0):
    return {
        "8x2": _metrics("8x2", 60.0),
        "8x4": _metrics("8x4", 100.0),
        "16x2": _metrics("16x2", 110.0),
        "16x4": _metrics("16x4", global64_throughput),
    }


def test_prefers_faster_global_batch_32_without_global64_stability():
    selected, decisions = select_layout(_all_metrics())
    assert selected == "16x2"
    assert "missing 1000-step" in decisions["16x4"]


def test_global64_requires_throughput_and_1000_step_stability():
    stability = {
        "optimizer_steps": 1000,
        "trajectory_loss_no_clear_degradation": True,
        "scorer_loss_no_clear_degradation": True,
        "gradient_norm_stable": True,
        "nonfinite_count": 0,
    }
    selected, _ = select_layout(
        _all_metrics(global64_throughput=138.0),
        global64_stability=stability,
    )
    assert selected == "16x4"

    selected, _ = select_layout(
        _all_metrics(global64_throughput=136.0),
        global64_stability=stability,
    )
    assert selected == "16x2"


def test_memory_gate_is_strictly_below_72_gib():
    valid, reason = validate_layout_metrics(
        "8x4", _metrics("8x4", 100.0, peak=72.0)
    )
    assert not valid
    assert "not < 72.0" in reason


def test_layout_lock_contains_exact_budget_lr_and_ema_scaling():
    metrics = _all_metrics()
    selected, decisions = select_layout(metrics)
    lock = build_layout_lock(
        selected,
        metrics,
        decisions,
        source_commit="a" * 40,
        metrics_sha256={name: name for name in metrics},
    )
    assert lock["selected_layout"] == "16x2"
    assert lock["global_batch_size"] == 32
    assert lock["steps_per_epoch"] == 3228
    assert lock["total_steps"] == 87156
    assert lock["logical_peak_learning_rates"]["vision_qv_lora"] == pytest.approx(4e-5)
    assert lock["ema_actual_start_momentum"] == pytest.approx(0.996**2)
    assert lock["ema_actual_end_momentum"] == pytest.approx(0.9999**2)

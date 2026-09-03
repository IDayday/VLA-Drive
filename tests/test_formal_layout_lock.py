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
        "median_step_time": spec["global_batch_size"] / throughput,
        "p90_step_time": spec["global_batch_size"] / throughput * 1.05,
        "peak_allocated_gib": peak,
        "peak_reserved_gib": peak + 2.0,
        "oom": False,
        "deadlock": False,
        "nonfinite_count": 0,
    }


def _all_metrics(batch96_throughput=198.0, batch128_throughput=200.0):
    return {
        "8x2": _metrics("8x2", 60.0),
        "8x4": _metrics("8x4", 100.0),
        "16x2": _metrics("16x2", 110.0),
        "16x4": _metrics("16x4", 130.0),
        "16x6": _metrics("16x6", batch96_throughput, peak=54.0),
        "16x8": _metrics("16x8", batch128_throughput, peak=19.0),
    }


def test_prefers_smaller_batch_within_five_percent_of_peak_throughput():
    selected, decisions = select_layout(_all_metrics())
    assert selected == "16x6"
    assert "smallest global batch within near-peak" in decisions["16x6"]
    assert "fewer optimizer updates" in decisions["16x8"]


def test_selects_batch128_only_for_material_wall_time_gain():
    selected, _ = select_layout(
        _all_metrics(batch96_throughput=180.0, batch128_throughput=200.0),
    )
    assert selected == "16x8"
    assert LAYOUT_SPECS[selected]["gradient_checkpointing"] is False
    assert LAYOUT_SPECS[selected]["scorer_partitions_per_scene"] == 2


def test_memory_gate_is_strictly_below_72_gib():
    valid, reason = validate_layout_metrics(
        "8x4", _metrics("8x4", 100.0, peak=72.0)
    )
    assert not valid
    assert "not < 72.0" in reason


def test_historical_eight_partition_metrics_remain_auditable():
    metrics = _metrics("16x6", 200.0, peak=54.0)
    metrics.pop("scorer_partitions_per_scene")
    valid, reason = validate_layout_metrics("16x6", metrics)
    assert valid
    assert reason == "eligible"

    optimized = _metrics("16x8", 210.0, peak=69.0)
    optimized.pop("scorer_partitions_per_scene")
    valid, reason = validate_layout_metrics("16x8", optimized)
    assert not valid
    assert "partition count mismatch" in reason


def test_step_time_tail_gate_rejects_unstable_layout():
    metrics = _metrics("16x6", 200.0, peak=54.0)
    metrics["p90_step_time"] = metrics["median_step_time"] * 1.36
    valid, reason = validate_layout_metrics("16x6", metrics)
    assert not valid
    assert "tail ratio" in reason


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
    assert lock["selected_layout"] == "16x6"
    assert lock["global_batch_size"] == 96
    assert lock["gradient_checkpointing"] is False
    assert lock["scorer_partitions_per_scene"] == 8
    assert lock["read_only_attention_backend"] == "split_sdpa"
    assert lock["steps_per_epoch"] == 1076
    assert lock["total_steps"] == 29052
    assert lock["logical_peak_learning_rates"]["planning_adapter"] == pytest.approx(3e-4)
    assert lock["logical_peak_learning_rates"]["vision_qv_lora"] == pytest.approx(5e-5)
    assert lock["ema_actual_start_momentum"] == pytest.approx(0.996**6)
    assert lock["ema_actual_end_momentum"] == pytest.approx(0.9999**6)

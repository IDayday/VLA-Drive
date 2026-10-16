import pytest

from tools.st_world_policy.profile_trace import interval_measure, interval_overlap, summarize_trace


def test_stream_overlap_is_not_summed():
    # Two compute kernels overlap; NCCL extends beyond both. The inclusive work
    # is 30 us, but union is 20 us and the independent NCCL portion is only 5 us.
    events = [
        {"ph": "X", "cat": "kernel", "name": "compute_a", "ts": 0, "dur": 15, "args": {"device": 0}},
        {"ph": "X", "cat": "kernel", "name": "compute_b", "ts": 5, "dur": 15, "args": {"device": 0}},
        {"ph": "X", "cat": "kernel", "name": "nccl_AllReduce", "ts": 10, "dur": 15, "args": {"device": 0}},
    ]
    result = summarize_trace({"traceEvents": events})
    device = result["devices"]["0"]
    assert device["compute_union_ms"] == .020
    assert device["nccl_union_ms"] == .015
    assert device["compute_nccl_overlap_ms"] == .010
    assert device["nccl_without_compute_ms"] == .005
    assert device["nccl_after_last_compute_ms"] == .005
    assert device["all_activity_union_ms"] == .025
    assert not result["eligible_for_throughput"]


def test_devices_and_transfers_remain_separate():
    events = [{"ph": "X", "cat": "kernel", "name": "flash_fwd", "ts": 0, "dur": 10,
               "args": {"device": i}} for i in (0, 1)]
    events.append({"ph": "X", "cat": "gpu_memcpy", "name": "HtoD", "ts": 5, "dur": 10,
                   "args": {"device": 0}})
    result = summarize_trace({"traceEvents": events})
    assert result["devices"]["0"]["all_activity_union_ms"] == .015
    assert result["devices"]["1"]["all_activity_union_ms"] == .010
    assert result["observed_attention_kernel_names"] == ["flash_fwd"]


def test_interval_empty_invalid_and_touching():
    assert interval_measure([]) == 0
    assert interval_measure([(0, 10), (10, 20)]) == 20
    assert interval_overlap([(0, 10), (20, 30)], [(5, 25)]) == 10
    with pytest.raises(ValueError):
        interval_measure([(0, float("nan"))])
    with pytest.raises(ValueError):
        summarize_trace({"traceEvents": [{"ph": "X", "ts": 1, "dur": -1}]})

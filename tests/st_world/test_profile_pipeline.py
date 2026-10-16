import copy
import json

import pytest
import torch
from types import SimpleNamespace

from tools.st_world_policy.profile_pipeline import (
    PipelineProfiler, TimingPlan, aggregate_concurrent, dtype_inventory, summarize_ranks,
)


def test_zero_partition_dtype_and_tied_parameter_inventory():
    model = torch.nn.ModuleDict({"input": torch.nn.Linear(2, 2, bias=False),
                                "output": torch.nn.Linear(2, 2, bias=False)})
    model["output"].weight = model["input"].weight
    gradient = torch.ones(4, dtype=torch.float32)
    optimizer = SimpleNamespace(single_partition_of_fp32_groups=[torch.zeros(4)],
        optimizer=SimpleNamespace(state={}), averaged_gradients={0: [gradient, gradient]})
    before = dtype_inventory(model, optimizer)
    assert before["unique_model_tensors"]["parameter:torch.float32:trainable=True"] == 4
    assert before["partition_gradient_tensors"] == {"averaged_gradients:torch.float32": 4}
    assert before["optimizer"]["zero_master:torch.float32"] == 4
    assert not before["optimizer_states_created"]
    optimizer.optimizer.state[0] = {"exp_avg": torch.zeros(4)}
    assert dtype_inventory(model, optimizer)["optimizer_states_created"]
from tools.st_world_policy.profile_resources import cgroup_paths, parse_mounts


def fake_ranks(world=4, start=0, gpus=None):
    identity = {"source_sha": "a"*40, "scientific_config_sha256": "b"*64,
                "data_manifest_sha256": "c"*64, "teacher_identity": "d"*64,
                "precision_policy": "same", "global_batch": 32, "sample_order_sha256": "e"*64}
    result = []
    for rank in range(world):
        rows = []
        for step in range(1, 241):
            begin = start + step * 3_000_000_000
            # Include one second outside measured update kernel time. Window
            # makespan must include it, unlike an average kernel-time shortcut.
            rows.append({"step": step, "start_ns": begin, "end_ns": begin+2_000_000_000,
                "seconds": 2., "local_scenes": 32//world, "checkpoint_seconds": 0.,
                "cpu_start": {"self_seconds": step, "waited_children_seconds": 0},
                "cpu_end": {"self_seconds": step+1, "waited_children_seconds": 0},
                "peak_allocated_bytes": 1024+rank, "peak_reserved_bytes": 2048+rank})
        result.append({"identity": identity, "hostname": "node", "world_size": world,
            "gpu_uuids": gpus or [str(i) for i in range(world)], "mode": "timing",
            "eligible_for_throughput": True, "rank": rank, "updates": rows})
    return result


def test_plan_requires_two_real_windows():
    assert TimingPlan().steady_steps(0) == list(range(21,121))
    assert TimingPlan().steady_steps(1) == list(range(141,241))
    for kwargs in ({"steady":3}, {"warmup":0}, {"windows":1}, {"steady":120}):
        with pytest.raises(ValueError):
            TimingPlan(**kwargs)


def test_summary_includes_between_update_gaps_and_slowest_rank():
    rows = fake_ranks()
    rows[-1]["updates"][20]["seconds"] = 2.5
    summary = summarize_ranks(rows)
    a = summary["windows"][0]
    assert a["complete_wall_seconds"] == 299
    assert a["scenes"] == 3200
    assert a["scenes_per_second"] == 3200/299
    assert a["slowest_rank_update_mean"] == pytest.approx(2.005)
    assert a["peak_allocated_bytes"] == 1027


@pytest.mark.parametrize("mutation", ["trace", "missing_rank", "missing_step", "wrong_identity", "wrong_host", "wrong_batch"])
def test_summary_rejects_incomplete_or_mixed_evidence(mutation):
    rows = fake_ranks()
    if mutation == "trace": rows[0]["mode"] = "trace"
    if mutation == "missing_rank": rows.pop()
    if mutation == "missing_step": rows[0]["updates"].pop(40)
    if mutation == "wrong_identity": rows[0] = copy.deepcopy(rows[0]); rows[0]["identity"]["teacher_identity"] = "f"*64
    if mutation == "wrong_host": rows[0]["hostname"] = "another-node"
    if mutation == "wrong_batch": rows[0]["updates"][20]["local_scenes"] = 1
    with pytest.raises(ValueError): summarize_ranks(rows)


def test_two_by_four_uses_observed_concurrent_makespan():
    first = summarize_ranks(fake_ranks())
    second = summarize_ranks(fake_ranks(start=10_000_000_000,gpus=["4","5","6","7"]))
    aggregate = aggregate_concurrent(first,second)
    row = aggregate["windows"][0]
    assert row["scenes"] == 6400
    assert row["joint_wall_seconds"] == 309
    assert row["aggregate_scenes_per_second"] == 6400/309
    assert row["aggregate_scenes_per_second"] != first["windows"][0]["scenes_per_second"]*2


@pytest.mark.parametrize("mutation", ["serial", "overlap_gpu", "different_data", "single_window"])
def test_two_by_four_rejects_fake_concurrency(mutation):
    first = summarize_ranks(fake_ranks())
    offset = 900_000_000_000 if mutation == "serial" else 0
    second = summarize_ranks(fake_ranks(start=offset,gpus=["4","5","6","7"]))
    if mutation == "overlap_gpu": second["gpu_uuids"][0] = "0"
    if mutation == "different_data": second["identity"]["data_manifest_sha256"] = "x"
    if mutation == "single_window": second["windows"].pop()
    with pytest.raises(ValueError): aggregate_concurrent(first,second)


def test_cpu_trace_hooks_preserve_gradients_rng_and_nested_call_shapes(tmp_path):
    model = torch.nn.Sequential(torch.nn.Linear(3,4),torch.nn.GELU(),torch.nn.Linear(4,2))
    value = torch.ones(2,3)
    state = torch.get_rng_state().clone()
    expected = model(value)
    expected.sum().backward()
    gradients = [p.grad.clone() for p in model.parameters()]
    model.zero_grad()
    profiler = PipelineProfiler(tmp_path,identity={"test":True},rank=0,world_size=1,
                                gpu_uuids=["cpu-test"],mode="trace",trace_steps=(1,),cuda=False)
    profiler.attach_modules({"whole":model,"linear":model[0]})
    profiler.begin_update(1)
    actual=model(value)
    actual.sum().backward()
    profiler.end_update(local_scenes=2)
    result=profiler.finish()
    assert torch.equal(actual,expected)
    assert torch.equal(state,torch.get_rng_state())
    assert all(torch.equal(g,p.grad) for g,p in zip(gradients,model.parameters()))
    assert result["eligible_for_throughput"] is False
    calls={row["module"]:row for row in result["module_calls"]}
    assert calls["whole"]["outputs"][0]["shape"] == [2,2]
    assert calls["linear"]["outputs"][0]["shape"] == [2,4]
    assert (tmp_path/"trace_rank0_step0001.json").exists()


def test_inventory_reports_real_optimizer_state_creation():
    model=torch.nn.Linear(2,1)
    optimizer=torch.optim.AdamW(model.parameters())
    before=dtype_inventory(model,optimizer)
    assert not before["optimizer_states_created"]
    model(torch.ones(1,2)).sum().backward();optimizer.step()
    after=dtype_inventory(model,optimizer)
    assert after["optimizer"]["exp_avg:torch.float32"]==3
    assert after["modules"][""]["gradient:torch.float32"]==3


def test_v1_namespaced_root_resolves_actual_container_mount(tmp_path):
    mount=tmp_path/"cpu,cpuacct";mount.mkdir()
    text=f"1 2 0:1 /kubepods/pod/container {mount} rw - cgroup cgroup rw,cpu,cpuacct\n"
    parsed=parse_mounts(text)
    paths=cgroup_paths("9:cpu,cpuacct:/kubepods/pod/container\n",parsed)
    assert paths[0]["leaf"]==str(mount)
    assert paths[0]["type"]=="cgroup"


def test_cgroup2_namespaced_root_and_escaped_mount(tmp_path):
    mount=tmp_path/"with space";mount.mkdir()
    escaped=str(mount).replace(' ',r'\040')
    text=f"1 2 0:1 / {escaped} rw - cgroup2 cgroup rw\n"
    paths=cgroup_paths("0::/\n",parse_mounts(text))
    assert paths[0]["leaf"]==str(mount)

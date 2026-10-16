"""CPU native-layout/installed-linkage tests; these are not GPU engine evidence."""
from copy import deepcopy
from datetime import timedelta
import json
import os
from pathlib import Path
import random
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from tools.ddpolicy_vehicle.optimizer_safety import bounded_parameter_groups
from tools.st_world_policy.fp32_master_init import (
    capture_fp32_master_initialization, restore_fp32_master_initialization)


class Source(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.parent = torch.nn.Parameter(torch.linspace(-.712345, .934567, 9).reshape(3, 3))
        self.parent_bias = torch.nn.Parameter(torch.tensor([-.0, .124537, -.175467, .372165, .542375]))
        self.new_world = torch.nn.Parameter(torch.linspace(.017123, .823467, 6).reshape(2, 3))
        self.new_history = torch.nn.Parameter(torch.tensor([.314159]))
        self.frozen_visual = torch.nn.Parameter(torch.tensor([.123457, .876543]), requires_grad=False)
        self.register_buffer("frozen_buffer", torch.tensor([.271828]))


def native_layout_engine(model, groups):
    """Reproduce installed stage_1_and_2.py's layout and use its real linker.

    No DeepSpeed engine, backward, CUDA or fake optimizer step is claimed. The
    class and fragment mapping are the installed official implementation; CPU
    AdamW exercises the unchanged local master slots.
    """
    from deepspeed.runtime.zero.stage_1_and_2 import DeepSpeedZeroOptimizer
    from deepspeed.utils.mixed_precision_linkage import link_hp_params

    rank, world = (dist.get_rank(), dist.get_world_size()) if dist.is_initialized() else (0, 1)
    group_handle = dist.group.WORLD if dist.is_initialized() else None
    optimizer = object.__new__(DeepSpeedZeroOptimizer)
    optimizer.dtype = torch.bfloat16
    optimizer.gradient_accumulation_dtype = torch.bfloat16
    optimizer.partition_gradients = True
    optimizer.cpu_offload = optimizer.round_robin_gradients = optimizer.has_moe_layers = False
    optimizer.fp16_master_weights_and_gradients = False
    optimizer.mpu = None
    optimizer.micro_step_id = -1
    optimizer.nccl_start_alignment_factor = 2
    optimizer.averaged_gradients = {}
    optimizer.param_names = {p: name for name, p in model.named_parameters()}
    optimizer.bit16_groups, optimizer.bit16_groups_flat = [], []
    optimizer.round_robin_bit16_groups, optimizer.round_robin_bit16_indices = [], []
    optimizer.single_partition_of_fp32_groups, optimizer.parallel_partitioned_bit16_groups = [], []
    optimizer.partition_size, optimizer.groups_padding = [], []
    optimizer.real_dp_process_group, optimizer.partition_count = [], []
    model.bfloat16()
    optimizer_groups = []
    for index, group in enumerate(groups):
        params = group["params"]
        count = sum(p.numel() for p in params)
        padded = ((count + 2 * world - 1) // (2 * world)) * (2 * world)
        flat = torch.cat([p.detach().flatten() for p in params] + [torch.zeros(padded - count, dtype=torch.bfloat16)])
        offset = 0
        for p in params:
            p.data = flat[offset:offset + p.numel()].reshape(p.shape)
            p.grad_accum = None
            offset += p.numel()
        size = padded // world
        partitions = list(flat.split(size))
        master = partitions[rank].clone().float().detach().requires_grad_(True)
        optimizer.bit16_groups.append(params)
        optimizer.bit16_groups_flat.append(flat)
        optimizer.round_robin_bit16_groups.append(params)
        optimizer.round_robin_bit16_indices.append(list(range(len(params))))
        optimizer.single_partition_of_fp32_groups.append(master)
        optimizer.parallel_partitioned_bit16_groups.append(partitions)
        optimizer.partition_size.append(float(size))
        optimizer.groups_padding.append(min(size, max(0, (rank + 1) * size - count)))
        optimizer.real_dp_process_group.append(group_handle)
        optimizer.partition_count.append(world)
        optimizer_groups.append({**{k: v for k, v in group.items() if k != "params"}, "params": [master]})
        link_hp_params(params, master, {}, {}, False, index, rank * size, size, group_handle)
    optimizer.optimizer = torch.optim.AdamW(optimizer_groups, lr=1e-5, betas=(.9, .95), eps=1e-8, weight_decay=.01)
    return SimpleNamespace(module=model, optimizer=optimizer, global_steps=0, global_samples=0,
        micro_steps=0, skipped_steps=0, zero_optimization_stage=lambda: 2, bfloat16_enabled=lambda: True)


def setup(chunk=3):
    model = Source()
    groups, _ = bounded_parameter_groups(model.named_parameters(), maximum=15)
    groups[0]["lr"] = 2e-5
    snapshot = capture_fp32_master_initialization(model, groups, is_fresh_run=True, chunk_elements=chunk)
    expected = [g.values.clone() for g in snapshot.groups]
    engine = native_layout_engine(model, groups)
    return model, groups, snapshot, engine, expected


def test_true_parent_and_new_fp32_values_preserved_BF16_execution_and_rng_unchanged():
    # Import installed linkage before measuring RNG; production imports DS first.
    from deepspeed.utils.mixed_precision_linkage import link_hp_params  # noqa: F401
    before_rng = (random.getstate(), np.random.get_state(), torch.get_rng_state().clone())
    model, _, snapshot, engine, expected = setup()
    before_model = {name: value.clone() for name, value in model.state_dict().items()}
    before_master_ids = [id(p) for p in engine.optimizer.single_partition_of_fp32_groups]
    before_groups = [{k: deepcopy(v) for k, v in g.items() if k != "params"} for g in engine.optimizer.optimizer.param_groups]
    assert any(not torch.equal(p, q) for p, q in zip(engine.optimizer.single_partition_of_fp32_groups, expected))
    audit = restore_fp32_master_initialization(engine, snapshot, is_fresh_run=True)
    assert audit["status"] == "RESTORED_EXACT_FP32_SOURCE"
    assert audit["ranks"][0]["changed_from_rounded_elements"] > 0
    assert audit["ranks"][0]["gradient_accumulation_dtype"] == "torch.bfloat16"
    assert audit["execution_unchanged"] and not audit["ranks"][0]["gradient_precision_changed"]
    assert before_master_ids == [id(p) for p in engine.optimizer.single_partition_of_fp32_groups]
    for p, q in zip(engine.optimizer.single_partition_of_fp32_groups, expected):
        assert torch.equal(p.detach().view(torch.int32), q.view(torch.int32))
    for name, value in model.state_dict().items():
        assert torch.equal(value.view(torch.uint8), before_model[name].view(torch.uint8))
    assert before_groups == [{k: v for k, v in g.items() if k != "params"} for g in engine.optimizer.optimizer.param_groups]
    assert not engine.optimizer.optimizer.state
    assert snapshot.consumed and all(g.values is None for g in snapshot.groups)
    assert random.getstate() == before_rng[0]
    assert np.array_equal(np.random.get_state()[1], before_rng[1][1])
    assert torch.equal(torch.get_rng_state(), before_rng[2])
    json.dumps(audit)
    with pytest.raises(ValueError, match="consumed"):
        restore_fp32_master_initialization(engine, snapshot, is_fresh_run=True)


def test_source_hash_and_values_independent_of_copy_chunk_and_frozen_weights():
    first = Source()
    second = deepcopy(first)
    with torch.no_grad():
        second.frozen_visual.add_(.1)
        second.frozen_buffer.add_(.2)
    groups1, _ = bounded_parameter_groups(first.named_parameters(), maximum=15)
    groups2, _ = bounded_parameter_groups(second.named_parameters(), maximum=15)
    one = capture_fp32_master_initialization(first, groups1, is_fresh_run=True, chunk_elements=1)
    two = capture_fp32_master_initialization(second, groups2, is_fresh_run=True, chunk_elements=7)
    assert one.identity == two.identity
    assert one.retained_cpu_bytes == sum(g.partition_size * 4 for g in one.groups)
    for left, right in zip(one.groups, two.groups):
        assert left.sha256 == right.sha256


@pytest.mark.parametrize("case", ["missing", "extra_frozen", "duplicate", "bf16", "nonfinite", "overflow", "gradient", "noncontiguous"])
def test_capture_rejects_invalid_or_incomplete_source(case):
    model = Source()
    groups, _ = bounded_parameter_groups(model.named_parameters(), maximum=15)
    if case == "missing": groups[-1]["params"].pop()
    if case == "extra_frozen": groups[0]["params"].append(model.frozen_visual)
    if case == "duplicate": groups[0]["params"].append(model.parent)
    if case == "bf16": model.bfloat16()
    if case == "nonfinite": model.parent.data[0, 0] = float("nan")
    if case == "overflow": model.parent.data[0, 0] = torch.finfo(torch.float32).max
    if case == "gradient": model.parent.grad = torch.zeros_like(model.parent)
    if case == "noncontiguous": model.parent.data = model.parent.data.T
    with pytest.raises(ValueError):
        capture_fp32_master_initialization(model, groups, is_fresh_run=True)


@pytest.mark.parametrize("case", ["version", "progress", "microstep", "moments", "master_gradient", "LP_gradient",
    "round_robin", "offload", "moe", "mpu", "stage", "precision", "name", "order", "padding", "mapping",
    "snapshot", "master_values", "execution_values", "uncovered", "already_applied"])
def test_restore_rejects_unsupported_or_changed_state_before_any_write(case, monkeypatch):
    model, _, snapshot, engine, _ = setup()
    opt = engine.optimizer
    if case == "version": monkeypatch.setattr("tools.st_world_policy.fp32_master_init.version", lambda _: "0.17.0")
    if case == "progress": engine.global_steps = 1
    if case == "microstep": opt.micro_step_id = 0
    if case == "moments": opt.optimizer.state[opt.single_partition_of_fp32_groups[0]]["step"] = 1
    if case == "master_gradient": opt.single_partition_of_fp32_groups[0].grad = torch.zeros_like(opt.single_partition_of_fp32_groups[0])
    if case == "LP_gradient": model.parent.grad = torch.zeros_like(model.parent)
    if case == "round_robin": opt.round_robin_gradients = True
    if case == "offload": opt.cpu_offload = True
    if case == "moe": opt.has_moe_layers = True
    if case == "mpu": opt.mpu = object()
    if case == "stage": engine.zero_optimization_stage = lambda: 1
    if case == "precision": engine.bfloat16_enabled = lambda: False
    if case == "name": opt.param_names[model.parent] = "wrong"
    if case == "order": opt.bit16_groups[0] = list(reversed(opt.bit16_groups[0]))
    if case == "padding": opt.groups_padding[-1] += 1
    if case == "mapping": model.parent._hp_mapping.hp_fragment_address.start += 1
    if case == "snapshot": snapshot.groups[0].values[0] += .000001
    if case == "master_values": opt.single_partition_of_fp32_groups[0].data[0] += .000001
    if case == "execution_values": model.parent.data[0, 0] += .125
    if case == "uncovered": model.new_history.requires_grad_(False)
    if case == "already_applied": engine._st_fp32_master_initialization = {"status": "old"}
    before = [p.detach().clone() for p in opt.single_partition_of_fp32_groups]
    with pytest.raises(ValueError):
        restore_fp32_master_initialization(engine, snapshot, is_fresh_run=True)
    for p, q in zip(opt.single_partition_of_fp32_groups, before):
        assert torch.equal(p.view(torch.int32), q.view(torch.int32))
    assert not snapshot.consumed


def test_native_resume_and_budget_continuation_are_unconditional_noops():
    model, groups, snapshot, engine, _ = setup()
    restore_fp32_master_initialization(engine, snapshot, is_fresh_run=True)
    # A real AdamW update materializes moments; resume must leave them intact.
    for p in engine.optimizer.single_partition_of_fp32_groups:
        p.grad = torch.full_like(p, .125)
    engine.optimizer.optimizer.step()
    engine.global_steps = 7
    before = deepcopy(engine.optimizer.optimizer.state_dict())
    values = [p.detach().clone() for p in engine.optimizer.single_partition_of_fp32_groups]
    rng = torch.get_rng_state().clone()
    assert capture_fp32_master_initialization(model, groups, is_fresh_run=False) is None
    audit = restore_fp32_master_initialization(engine, None, is_fresh_run=False)
    assert audit["status"] == "SKIPPED_NATIVE_RESUME" and not audit["mutated"]
    assert torch.equal(torch.get_rng_state(), rng)
    for p, q in zip(engine.optimizer.single_partition_of_fp32_groups, values):
        assert torch.equal(p, q)
    after = engine.optimizer.optimizer.state_dict()
    assert before["param_groups"] == after["param_groups"]
    for key in before["state"]:
        for item, value in before["state"][key].items():
            assert torch.equal(value, after["state"][key][item])
    with pytest.raises(ValueError, match="resume/continuation"):
        restore_fp32_master_initialization(engine, snapshot, is_fresh_run=False)


def test_first_actual_AdamW_update_and_moments_start_from_original_FP32_not_rounded():
    _, _, snapshot, engine, expected = setup()
    reference = [p.clone().requires_grad_() for p in expected]
    ref_groups = [{**{k: v for k, v in g.items() if k != "params"}, "params": [p]}
                  for g, p in zip(engine.optimizer.optimizer.param_groups, reference)]
    reference_optimizer = torch.optim.AdamW(ref_groups)
    restore_fp32_master_initialization(engine, snapshot, is_fresh_run=True)
    for p, q in zip(engine.optimizer.single_partition_of_fp32_groups, reference):
        # Represents the existing backend's BF16 gradient converted to its
        # FP32 optimizer input, without claiming FP32 backward accumulation.
        grad = torch.linspace(.0123, .0971, p.numel()).bfloat16().float()
        p.grad, q.grad = grad.clone(), grad.clone()
    engine.optimizer.optimizer.step()
    reference_optimizer.step()
    for p, q in zip(engine.optimizer.single_partition_of_fp32_groups, reference):
        assert torch.equal(p, q)
        for name, value in engine.optimizer.optimizer.state[p].items():
            assert torch.equal(value, reference_optimizer.state[q][name])


def distributed_worker(rank, rendezvous, output, case):
    dist.init_process_group("gloo", init_method="file://" + rendezvous, rank=rank, world_size=2,
                            timeout=timedelta(seconds=60))
    try:
        model = Source()
        groups, _ = bounded_parameter_groups(model.named_parameters(), maximum=15)
        if case == "different_FP32_source" and rank == 1:
            model.new_world.data[0, 0] += 1e-7  # Same BF16 value, different required master.
        rejected = False
        try:
            snapshot = capture_fp32_master_initialization(model, groups,
                is_fresh_run=(rank == 0) if case == "different_fresh_flags" else True, chunk_elements=3)
        except ValueError:
            rejected = True
        if case in {"different_FP32_source", "different_fresh_flags"}:
            assert rejected
            Path(output, f"{rank}.json").write_text(json.dumps({"all_rank_rejected": True}))
            return
        expected = [g.values.clone() for g in snapshot.groups]
        engine = native_layout_engine(model, groups)
        before = [p.detach().clone() for p in engine.optimizer.single_partition_of_fp32_groups]
        if case == "one_rank_bad_mapping" and rank == 1:
            model.parent._hp_mapping.hp_fragment_address.start += 1
        try:
            audit = restore_fp32_master_initialization(engine, snapshot, is_fresh_run=True)
        except ValueError:
            rejected = True
        if case == "one_rank_bad_mapping":
            assert rejected
            for p, q in zip(engine.optimizer.single_partition_of_fp32_groups, before):
                assert torch.equal(p, q)
            result = {"all_rank_rejected": True, "no_rank_mutated": True}
        else:
            assert not rejected
            for p, q in zip(engine.optimizer.single_partition_of_fp32_groups, expected):
                assert torch.equal(p.view(torch.int32), q.view(torch.int32))
            # Gather the actual repaired flat masters and compare to the full
            # unrounded source, including parameters crossing rank boundaries.
            original = Source()
            original_groups, _ = bounded_parameter_groups(original.named_parameters(), maximum=15)
            for index, (p, g) in enumerate(zip(engine.optimizer.single_partition_of_fp32_groups, original_groups)):
                parts = [torch.empty_like(p) for _ in range(2)]
                dist.all_gather(parts, p.detach())
                actual = torch.cat(parts)
                target = torch.cat([param.detach().flatten() for param in g["params"]])
                assert torch.equal(actual[:target.numel()].view(torch.int32), target.view(torch.int32))
                assert not torch.count_nonzero(actual[target.numel():])
            assert audit["ranks"][1]["local_padding_elements"] == 3
            result = {"exact_reassembled": True, "audit": audit}
        Path(output, f"{rank}.json").write_text(json.dumps(result))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("case", ["exact", "different_FP32_source", "different_fresh_flags", "one_rank_bad_mapping"])
def test_two_rank_gloo_source_partition_padding_and_fail_together(tmp_path, case):
    mp.spawn(distributed_worker, args=(str(tmp_path / "rendezvous"), str(tmp_path), case), nprocs=2, join=True)
    records = [json.loads((tmp_path / f"{rank}.json").read_text()) for rank in range(2)]
    assert all(item.get("exact_reassembled") if case == "exact" else item["all_rank_rejected"] for item in records)


def test_all_padding_partition_is_valid(tmp_path):
    # The capture formula must also support a rank owning only padding. This
    # case is checked without pretending it was a real distributed engine.
    from unittest.mock import patch
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad(): model.weight.fill_(.1234567)
    groups, _ = bounded_parameter_groups(model.named_parameters())
    with patch("tools.st_world_policy.fp32_master_init._topology", return_value=(1, 2)):
        snapshot = capture_fp32_master_initialization(model, groups, is_fresh_run=True)
    group = snapshot.groups[0]
    assert group.partition_size == group.padding == 2
    assert torch.equal(group.values, torch.zeros(2))


@pytest.mark.skipif(os.environ.get("ST_TEST_NATIVE_CPU") != "1",
                    reason="Explicit ST_TEST_NATIVE_CPU=1 permits installed DS CPU communication extension build")
def test_actual_deepspeed_CPU_two_rank_init_step_save_and_native_resume(tmp_path):
    environment = {**os.environ, "DS_ACCELERATOR": "cpu", "CUDA_VISIBLE_DEVICES": "",
                   "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "MAX_JOBS": "1"}
    repo = Path(__file__).resolve().parents[2]
    environment["PYTHONPATH"] = str(repo)
    probe = Path(__file__).with_name("fp32_master_native_cpu_probe.py")
    result = subprocess.run([sys.executable, str(probe), str(tmp_path)], env=environment,
                            cwd=repo, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=180)
    (tmp_path / "NATIVE_CPU_ENGINE.log").write_text(result.stdout)
    assert result.returncode == 0, result.stdout[-12000:]
    for rank in (0, 1):
        record = json.loads((tmp_path / f"native_rank{rank}.json").read_text())
        assert record["status"] == "PASS" and record["native_checkpoint_reload_exact"]
        assert record["real_optimizer_steps"] == 1 and not record["CUDA_tested"]

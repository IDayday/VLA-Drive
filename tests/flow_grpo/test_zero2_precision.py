import torch
import pytest
from starVLA.rl.flow_grpo.zero2_precision import partition_in_requested_dtype


def test_actual_partition_api_honors_dtype_offset_padding_and_accumulation():
    from deepspeed.runtime.zero.stage_1_and_2 import DeepSpeedZeroOptimizer

    optimizer = DeepSpeedZeroOptimizer.__new__(DeepSpeedZeroOptimizer)
    optimizer.use_grad_accum_attribute = False
    optimizer.flatten = lambda values: torch.cat([v.flatten() for v in values])
    params = [torch.nn.Parameter(torch.zeros(3, dtype=torch.bfloat16)) for _ in range(2)]
    params[0].grad = torch.tensor([9, 1, 1], dtype=torch.bfloat16)
    params[1].grad = torch.ones(3, dtype=torch.bfloat16)
    fn = optimizer.get_flat_partition
    original = fn(params, 1, 6, torch.float32, "cpu", return_tensor_list=True)
    assert original[0].dtype == torch.bfloat16  # reproduces installed API defect
    parts = partition_in_requested_dtype(fn, params, 1, 6, torch.float32, "cpu", True)
    assert all(p.dtype == torch.float32 for p in parts)
    assert torch.equal(torch.cat(parts), torch.tensor([1, 1, 1, 1, 1, 0], dtype=torch.float32))
    for p in params:
        p.grad.fill_(1 / 512)
    for _ in range(3):
        new = partition_in_requested_dtype(fn, params, 1, 6, dtype=torch.float32, device="cpu", return_tensor_list=True)
        for accumulated, contribution in zip(parts, new):
            accumulated.add_(contribution)
    assert torch.equal(torch.cat(parts), torch.tensor([1 + 3 / 512] * 5 + [0]))
    assert params[0].grad.dtype == torch.bfloat16  # no .data dtype/autograd mutation
    flattened = partition_in_requested_dtype(fn, params, 1, 6, torch.float32, "cpu")
    assert flattened.dtype == torch.float32 and flattened.numel() == 6


def test_partition_correction_rejects_unrecognized_optimizer():
    from types import SimpleNamespace
    from starVLA.rl.flow_grpo.zero2_precision import install_fp32_partitions

    with pytest.raises(RuntimeError, match="audited"):
        install_fp32_partitions(SimpleNamespace(optimizer=object()))


@pytest.mark.parametrize("field,value", [("cpu_offload", True), ("partition_gradients", False), ("communication_data_type", torch.bfloat16)])
def test_partition_correction_rejects_unvalidated_paths(field, value):
    from types import SimpleNamespace
    from deepspeed.runtime.zero.stage_1_and_2 import DeepSpeedZeroOptimizer
    from starVLA.rl.flow_grpo.zero2_precision import install_fp32_partitions

    optimizer = DeepSpeedZeroOptimizer.__new__(DeepSpeedZeroOptimizer)
    optimizer.partition_gradients = True
    optimizer.cpu_offload = False
    optimizer.dtype = torch.bfloat16
    optimizer.gradient_accumulation_dtype = torch.float32
    optimizer.communication_data_type = torch.float32
    optimizer.overlap_comm = False
    setattr(optimizer, field, value)
    with pytest.raises(ValueError, match="requires"):
        install_fp32_partitions(SimpleNamespace(optimizer=optimizer))


def test_only_observation_and_save_frequency_excluded_from_config_identity():
    import copy
    from starVLA.rl.flow_grpo.config import config_hash

    cfg = {"runtime": {"save_every": 1, "log_every": 1, "diagnostic_optimizer_gradients": True, "accumulation_steps": 4, "numerical_profile": "v2"}}
    other = copy.deepcopy(cfg)
    other["runtime"].update(save_every=100, log_every=10, diagnostic_optimizer_gradients=False)
    assert config_hash(cfg) == config_hash(other)
    for key, value in (("accumulation_steps", 8), ("numerical_profile", "v1")):
        changed = copy.deepcopy(other)
        changed["runtime"][key] = value
        assert config_hash(cfg) != config_hash(changed)

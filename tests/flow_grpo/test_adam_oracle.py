import torch
from scripts.flow_grpo.check_adam_update import comparison


def test_optimizer_oracle_detects_scaling_and_nonfinite_errors():
    expected = torch.tensor([1.0, -0.01, 1e-9, 0.0])
    assert comparison(expected, expected.clone(), master=False)["pass"]
    assert not comparison(expected, expected * 2, master=False)["pass"]
    changed = expected.clone()
    changed[-1] = 1e-12
    assert not comparison(expected, changed, master=False)["pass"]
    changed[0] = float("nan")
    assert not comparison(expected, changed, master=True)["pass"]


def test_master_bound_is_fp32_roundoff_not_bf16_layout_tolerance():
    expected = torch.tensor([1.0, 0.5])
    one_ulp = torch.nextafter(expected, torch.full_like(expected, float("inf")))
    assert comparison(expected, one_ulp, master=True)["pass"]
    assert not comparison(expected, expected + 1e-5, master=True)["pass"]

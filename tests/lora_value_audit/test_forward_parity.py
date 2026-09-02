import torch

from tools.lora_value_audit.drivor_adapter import aggregate_score


class _Config:
    noc = 1
    dac = 1
    ddc = 0
    ttc = 5
    ep = 5
    comfort = 2


def test_aggregate_score_is_logspace_drivor_formula() -> None:
    logits = torch.zeros(3, 64, 6)
    result = aggregate_score(logits, _Config())
    expected = 2 * torch.log(torch.tensor(0.5)) + torch.log(torch.tensor(6.0))
    torch.testing.assert_close(result, expected.expand_as(result))

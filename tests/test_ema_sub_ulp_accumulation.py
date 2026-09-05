import torch
from test_ema_register_target import _StudentBackbone
from navsim.agents.EpisodeDrive.layers.world_model import EMARegisterTarget


def test_1000_sub_bf16_ulp_updates_match_float64_reference():
    student = _StudentBackbone()
    with torch.no_grad():
        for p in student.parameters():
            p.fill_(1.0)
    teacher = EMARegisterTarget(student).bfloat16()
    name = teacher.master.names[0]
    source = teacher.student_parameters(student)[name]
    with torch.no_grad():
        source.fill_(1.125)
    reference = torch.ones_like(source, dtype=torch.float64)
    before_copy = dict(teacher.named_parameters())[name].clone()
    for step in range(1000):
        reference += 0.0001 * (source.double() - reference)
        teacher.update(student, 0.9999)
        if step == 0:
            assert torch.equal(before_copy, dict(teacher.named_parameters())[name])
    master = teacher.master.tensors()[name]
    assert master.dtype == torch.float32
    torch.testing.assert_close(master.double(), reference, atol=2e-6, rtol=0)
    assert not torch.equal(before_copy, dict(teacher.named_parameters())[name])
    teacher.to(dtype=torch.float16)
    assert all(t.dtype == torch.float32 for t in teacher.master.tensors().values())

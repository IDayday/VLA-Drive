import copy
import torch
from test_ema_register_target import _StudentBackbone
from navsim.agents.EpisodeDrive.layers.world_model import EMARegisterTarget


def test_resume_is_bitwise_equivalent_and_legacy_is_explicit():
    student = _StudentBackbone()
    teacher = EMARegisterTarget(student)
    for step in range(15):
        with torch.no_grad():
            student.planning_register_adapter.planning_registers.add_(0.003)
        teacher.update(student, 0.999)
    resumed = EMARegisterTarget(student)
    resumed.load_state_dict(copy.deepcopy(teacher.state_dict()), strict=True)
    for step in range(25):
        with torch.no_grad():
            student.planning_register_adapter.planning_registers.add_(0.003)
        teacher.update(student, 0.999)
        resumed.update(student, 0.999)
    for key, value in teacher.state_dict().items():
        assert torch.equal(value, resumed.state_dict()[key])
    legacy = {k: v.bfloat16() if v.is_floating_point() else v
              for k, v in teacher.state_dict().items() if not k.startswith('master.')}
    assert resumed.migrate_legacy_state_dict(legacy)
    resumed.load_state_dict(legacy)
    assert resumed.master.legacy_bf16_ema_history_unrecoverable

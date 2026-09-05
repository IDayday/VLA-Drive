from test_ema_register_target import _StudentBackbone
from navsim.agents.EpisodeDrive.layers.world_model import EMARegisterTarget


def test_student_teacher_and_master_map_have_identical_readout_topology():
    student = _StudentBackbone('global_local_8_8')
    teacher = EMARegisterTarget(student)
    assert tuple(student.planning_register_adapter.state_dict()) == tuple(teacher.planning_register_adapter.state_dict())
    expected = {'planning_register_adapter.' + n for n, p in student.planning_register_adapter.named_parameters() if p.requires_grad}
    assert expected <= set(teacher.master.names)
    assert any('global_local_readout.local_attention' in name for name in teacher.master.names)

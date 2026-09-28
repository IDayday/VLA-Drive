import torch
from starVLA.model.modules.vehicle_joint.graphs import select_vehicles, VehicleGraphConfig
from starVLA.model.modules.vehicle_joint.masks import VehicleRoleScheduler, auxiliary_due
from starVLA.model.modules.vehicle_joint.action_head import modeled_mask


def boxes(xy):
    b = torch.tensor([[x, y, 1., 4., 2., 1.5, 0., 1.] for x,y in xy])
    return b


def test_support_context_and_navigation_without_future_inputs():
    current = boxes([(12., 13.), (28., 0.), (8., 0.)])
    confidence = torch.ones(3)
    support = torch.tensor([True, True, False])
    cfg = VehicleGraphConfig(max_vehicles=1)
    left, context, _ = select_vehicles(current, confidence, support, 8., 0, cfg)
    straight, _, _ = select_vehicles(current, confidence, support, 8., 1, cfg)
    assert left == [0] and straight == [1]
    assert 2 not in left and 2 in context
    # No speed of any neighboring vehicle is supplied, including parked cars.
    selected, _, _ = select_vehicles(boxes([(8., 0.)]), torch.ones(1), torch.ones(1, dtype=torch.bool), 0., 1)
    assert selected == [0]


def test_batch1_odd_tail_and_resume_roles_do_not_consume_global_rng():
    active = torch.tensor([[True, True, True]])
    valid = modeled_mask(active)
    scheduler = VehicleRoleScheduler(71)
    before = torch.random.get_rng_state()
    seen = []
    for _ in range(5):
        known, tasks = scheduler.known_mask(active, valid)
        seen.append(tasks[0]["role"])
        assert not known[0, tasks[0]["target"]].any()
    assert set(seen) == {"ego", "neighbor"}
    restored = VehicleRoleScheduler(999); restored.load_state_dict(scheduler.state_dict())
    expected = scheduler.known_mask(active.expand(3,-1), valid.expand(3,-1,-1,-1))
    actual = restored.known_mask(active.expand(3,-1), valid.expand(3,-1,-1,-1))
    assert torch.equal(expected[0], actual[0]) and expected[1] == actual[1]
    assert torch.equal(before, torch.random.get_rng_state())


def test_half_xy_point_is_not_neighbor_task_and_all_hidden_final_phase():
    active = torch.tensor([[True, True]])
    valid = modeled_mask(active); valid[:, 1, :, 1] = False
    known, tasks = VehicleRoleScheduler(4).known_mask(active, valid)
    assert not known.any() and tasks[0]["role"] == "all_hidden_fallback"
    assert [i+1 for i in range(16) if auxiliary_due(i, 12)] == [4, 8, 12]

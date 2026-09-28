import torch
from tools.ddpolicy_vehicle.training_state import epoch_batches
from starVLA.model.modules.vehicle_joint.action_head import VehicleJointActionHead, modeled_mask
from test_action_head import config


def test_epoch_manifest_exact_coverage_and_deterministic_tail():
    batches = epoch_batches(101592, 32, 42, 0)
    assert len(batches) == 3175 and len(batches[-1]) == 24
    assert sorted(sum(batches, [])) == list(range(101592))
    assert batches == epoch_batches(101592, 32, 42, 0)
    assert batches != epoch_batches(101592, 32, 42, 1)


def test_per_scene_joint_loss_gradients_equal_microbatch_partition():
    torch.manual_seed(33)
    model = VehicleJointActionHead(config()).eval()
    active = torch.tensor([[True, True, False], [True, True, True]])
    valid = modeled_mask(active); valid[1, 1:, 3:, :2] = False
    shape = valid.shape
    y, noise = torch.randn(shape), torch.randn(shape)
    action, query = torch.randn(2, 8, 24), torch.randn(2, 3, 24)
    boxes, times = torch.ones(2, 3, 8), torch.tensor([.3, .7])
    args = [action, query, boxes, active, y, valid, noise, times]
    loss, _ = model.loss(*args); loss.backward()
    expected = {k:p.grad.clone() for k,p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none=True)
    for i in range(2):
        part, _ = model.loss(*[x[i:i+1] for x in args]); (part/2).backward()
    for name, p in model.named_parameters():
        if p.grad is not None: torch.testing.assert_close(p.grad, expected[name], atol=2e-6, rtol=2e-5)

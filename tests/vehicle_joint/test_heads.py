import torch
from starVLA.model.modules.vehicle_joint.heads import VehicleHeads


def test_metric_decoding_keeps_same_track_center_and_checkpoint():
    torch.manual_seed(37)
    legacy = VehicleHeads(24, dim=16, xy_scale=1.)
    scaled = VehicleHeads(24, dim=16, xy_scale=20.)
    scaled.load_state_dict(legacy.state_dict(), strict=True)
    hidden = torch.randn(2, 3, 24, requires_grad=True)
    old, new = legacy(hidden), scaled(hidden)
    torch.testing.assert_close(new['boxes'][...,:2],old['boxes'][...,:2]*20)
    torch.testing.assert_close(new['boxes'][...,2:],old['boxes'][...,2:])
    torch.testing.assert_close(new['future_xy']-new['boxes'][...,:2].unsqueeze(-2),
        (old['future_xy']-old['boxes'][...,:2].unsqueeze(-2))*20,atol=2e-6,rtol=2e-6)
    torch.testing.assert_close(new['logits'],old['logits'])
    new['future_xy'].square().mean().backward()
    assert torch.isfinite(hidden.grad).all() and hidden.grad.norm()>0
    assert scaled.box.weight.grad.norm()>0 and scaled.motion.weight.grad.norm()>0

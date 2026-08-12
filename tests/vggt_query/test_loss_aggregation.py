from types import SimpleNamespace

import torch

from starVLA.training.trainer_utils.trainer_tools import aggregate_output_losses


def test_named_losses_use_explicit_weights():
    output = {
        "losses": {
            "action": torch.tensor(2.0),
            "vggt_alignment": torch.tensor(3.0),
        }
    }
    trainer = SimpleNamespace(loss_weights={"action": 1.0, "vggt_alignment": 0.2})
    cfg = SimpleNamespace(trainer=trainer)
    total, weighted = aggregate_output_losses(output, cfg)
    torch.testing.assert_close(total, torch.tensor(2.6))
    torch.testing.assert_close(weighted["action"], torch.tensor(2.0))
    torch.testing.assert_close(weighted["vggt_alignment"], torch.tensor(0.6))


def test_unknown_named_loss_weight_fails_instead_of_silent_fallback():
    output = {"losses": {"action": torch.tensor(1.0), "new_loss": torch.tensor(1.0)}}
    cfg = SimpleNamespace(trainer=SimpleNamespace(loss_weights={"action": 1.0}))
    try:
        aggregate_output_losses(output, cfg)
    except KeyError as error:
        assert "new_loss" in str(error)
    else:
        raise AssertionError("a missing loss weight must fail")


def test_auxiliary_named_losses_warm_up_without_scaling_action():
    output = {
        "losses": {
            "action": torch.tensor(2.0),
            "vggt_geometry": torch.tensor(4.0),
        }
    }
    trainer = SimpleNamespace(
        loss_weights={"action": 1.0, "vggt_geometry": 0.1},
        named_loss_warmup_steps=100,
    )
    cfg = SimpleNamespace(trainer=trainer)
    total, weighted = aggregate_output_losses(output, cfg, optimizer_step=49)
    torch.testing.assert_close(weighted["action"], torch.tensor(2.0))
    torch.testing.assert_close(weighted["vggt_geometry"], torch.tensor(0.2))
    torch.testing.assert_close(total, torch.tensor(2.2))

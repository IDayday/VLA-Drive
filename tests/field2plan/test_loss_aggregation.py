from types import SimpleNamespace

import torch

from starVLA.training.train_starvla import aggregate_losses, json_scalar_metrics


def _legacy_cfg():
    return SimpleNamespace(
        datasets=SimpleNamespace(
            video_data=SimpleNamespace(load_2d_data=1),
            gs_data=SimpleNamespace(load_3d_data=0),
            reward_data=SimpleNamespace(load_reward_data=0),
            vla_data=SimpleNamespace(load_act_data=1),
        ),
        w_depth=1,
        trainer=SimpleNamespace(),
    )


def test_legacy_loss_aggregation_is_unchanged() -> None:
    output = {
        "action_loss": torch.tensor(1.0),
        "rgb_loss": torch.tensor(2.0),
        "gs_loss": torch.tensor(3.0),
        "reward_loss": torch.tensor(100.0),
    }
    total, named = aggregate_losses(output, _legacy_cfg())
    assert total.item() == 6.0
    assert set(named) == {"action", "rgb", "gs"}


def test_structured_losses_use_named_weights() -> None:
    cfg = _legacy_cfg()
    cfg.trainer.loss_weights = {"plan": 2.0, "delta_reg": 0.25}
    output = {
        "losses": {
            "plan": torch.tensor(3.0),
            "delta_reg": torch.tensor(4.0),
        }
    }
    total, named = aggregate_losses(output, cfg)
    assert total.item() == 7.0
    assert set(named) == {"plan", "delta_reg"}


def test_json_scalar_metrics_keeps_checkpoint_diagnostics_machine_readable() -> None:
    record = json_scalar_metrics(
        {
            "future_temporal_margin": torch.tensor(0.25),
            "world_delta_norm": 0.1,
            "generated_videos": [object()],
        },
        step=30000,
    )
    assert record == {
        "step": 30000,
        "future_temporal_margin": 0.25,
        "world_delta_norm": 0.1,
    }

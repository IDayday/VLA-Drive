import inspect

import torch

from starVLA.model.modules.trajectory_scorer.drivor_dynamic_scorer import (
    DrivoRDynamicScorer,
)
from starVLA.training import train_register_drivor
from starVLA.training.register_stage_utils import selector_statistics
from starVLA.training.train_register_drivor import (
    build_drivor_scorer,
    drivor_training_step,
)


def _scorer():
    return DrivoRDynamicScorer(
        scene_dim=32,
        model_dim=32,
        ffn_dim=64,
        num_layers=2,
        num_heads=1,
        decoder_style="donor_register",
        proj_drop=0.0,
        drop_path=0.0,
    )


def _batch(requires_grad=False):
    proposals = torch.randn(2, 4, 8, 3, requires_grad=requires_grad)
    metrics = {
        name: torch.rand(2, 4)
        for name in (
            "no_at_fault_collisions",
            "drivable_area_compliance",
            "time_to_collision_within_bound",
            "ego_progress",
            "driving_direction_compliance",
            "comfort",
            "aggregate_score",
        )
    }
    return {
        "proposals": proposals,
        "scene_global_tokens": torch.randn(2, 4, 32),
        "ego_state": torch.randn(2, 4),
        "metrics": metrics,
    }


def test_scorer_training_does_not_instantiate_qwen():
    source = inspect.getsource(train_register_drivor)
    assert "QwenRegister" not in source
    assert "build_framework" not in source


def test_scorer_training_does_not_instantiate_generator():
    assert "RegisterTrajectoryGenerator(" not in inspect.getsource(train_register_drivor)


def test_scorer_training_does_not_call_metric_supervisor():
    assert "DynamicMetricSupervisor" not in inspect.getsource(train_register_drivor)


def test_scorer_proposal_detach():
    batch = _batch(requires_grad=True)
    output = _scorer()(
        batch["proposals"],
        batch["scene_global_tokens"],
        batch["ego_state"],
        topm=2,
    )
    output.aggregate_score.sum().backward()
    assert batch["proposals"].grad is None


def test_scorer_top1_and_regret():
    predicted = torch.tensor([[0.1, 0.9, 0.2]])
    true = torch.tensor([[0.8, 0.3, 1.0]])
    metrics = selector_statistics(predicted, true)
    assert metrics["selected_true_score"].item() == torch.tensor(0.3).item()
    assert metrics["regret"].item() == torch.tensor(0.7).item()


def test_drivor_bank_forward_backward():
    scorer, batch = _scorer(), _batch()
    loss, statistics = drivor_training_step(scorer, batch, topm=4)
    loss.backward()
    assert torch.isfinite(loss)
    assert "regret" in statistics


def test_navsim_v2_aggregate_weights_reach_scorer():
    weights = {
        "noc": 10.0,
        "dac": 13.0,
        "ddc": 6.0,
        "ttc": 14.0,
        "ep": 15.0,
        "comfort": 2.0,
    }
    scorer = build_drivor_scorer(
        {
            "model": {
                "scene_dim": 32,
                "model_dim": 32,
                "ffn_dim": 64,
                "num_layers": 2,
                "num_heads": 1,
                "decoder_style": "donor_register",
                "proj_drop": 0.0,
                "drop_path": 0.0,
                **weights,
            }
        }
    )
    assert scorer.aggregate_weights == weights

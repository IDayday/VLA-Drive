import math

import pytest

from test_planreg_optimizer_groups import _agent


def _formal_optimizer_agent(global_batch: int):
    agent = _agent()
    agent._formal_initialization = True
    agent.batch_size = global_batch // 8
    agent.num_gpus = 8
    agent._lr_args = {
        "name": "AdamW",
        "reference_global_batch": 32,
        "scale_with_batch_size": "sqrt",
        "planning_adapter_lr": 2.0e-4,
        "semantic_fusion_lr": 2.0e-4,
        "action_generator_lr": 2.0e-4,
        "scorer_lr": 2.0e-4,
        "future_predictor_lr": 2.0e-4,
        "semantic_qformer_lr": 1.0e-4,
        "vision_qv_lora_lr": 4.0e-5,
        "language_model_lr": 0.0,
        "new_module_lr_cap": 3.0e-4,
        "action_scorer_lr_cap": 3.0e-4,
        "qformer_lr_cap": 1.5e-4,
        "vision_lora_lr_cap": 5.0e-5,
        "decay_weight_decay": 0.01,
        "no_decay_weight_decay": 0.0,
        "betas": (0.9, 0.999),
        "eps": 1e-8,
    }
    return agent


@pytest.mark.parametrize("global_batch", [16, 32, 64])
def test_formal_lr_uses_sqrt_global_batch_scaling_and_caps(global_batch):
    agent = _formal_optimizer_agent(global_batch)
    optimizer = agent.get_optimizers()[0]
    logical_lrs = {
        group["logical_name"]: group["lr"] for group in optimizer.param_groups
    }
    scale = math.sqrt(global_batch / 32)
    expected_new = min(2.0e-4 * scale, 3.0e-4)
    assert logical_lrs["planning_adapter"] == pytest.approx(expected_new)
    assert logical_lrs["semantic_fusion"] == pytest.approx(expected_new)
    assert logical_lrs["future_predictor"] == pytest.approx(expected_new)
    assert logical_lrs["action_generator"] == pytest.approx(expected_new)
    assert logical_lrs["scorer"] == pytest.approx(expected_new)
    assert logical_lrs["semantic_qformer"] == pytest.approx(
        min(1.0e-4 * scale, 1.5e-4)
    )
    assert logical_lrs["vision_qv_lora"] == pytest.approx(
        min(4.0e-5 * scale, 5.0e-5)
    )
    assert agent._planreg_optimizer_runtime["actual_global_batch"] == global_batch
    assert agent._planreg_optimizer_runtime["lr_scale"] == pytest.approx(scale)


def test_formal_optimizer_has_seven_declared_logical_groups():
    optimizer = _formal_optimizer_agent(32).get_optimizers()[0]
    assert {group["logical_name"] for group in optimizer.param_groups} == {
        "planning_adapter",
        "semantic_fusion",
        "action_generator",
        "scorer",
        "future_predictor",
        "semantic_qformer",
        "vision_qv_lora",
    }
    for group in optimizer.param_groups:
        if group["name"].endswith("no_decay"):
            assert group["weight_decay"] == 0.0
        else:
            assert group["weight_decay"] == pytest.approx(0.01)


def test_formal_optimizer_rejects_non_sqrt_scaling():
    agent = _formal_optimizer_agent(32)
    agent._lr_args["scale_with_batch_size"] = False
    with pytest.raises(ValueError, match="scale_with_batch_size=sqrt"):
        agent.get_optimizers()

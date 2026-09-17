import pytest
from starVLA.rl.flow_grpo.diagnostic_loss import selected_loss, validate_scope
from starVLA.rl.flow_grpo.config import config_hash


def test_default_joint_loss_is_unchanged_and_scope_is_explicit():
    result = {"loss": object(), "grpo": object(), "sft": object(), "reference": object()}
    runtime = {"run_mode": "diagnostic", "max_updates": 2}
    assert selected_loss(result, runtime, 0) is result["loss"]
    for scope, field in (("rl_only", "grpo"), ("sft_only", "sft")):
        cfg = {**runtime, "diagnostic_loss_scope": scope}
        assert selected_loss(result, cfg, 0) is result[field]
    cfg["diagnostic_loss_scope"] = "reference_after_joint"
    assert selected_loss(result, cfg, 0) is result["loss"]
    assert selected_loss(result, cfg, 1) is result["reference"]
    assert config_hash({"runtime": cfg}) != config_hash({"runtime": runtime})


@pytest.mark.parametrize("mode,budget,scope", [("formal", 2, "rl_only"), ("paired_short", 2, "sft_only"), ("diagnostic", 9, "rl_only"), ("diagnostic", 2, "invented")])
def test_isolation_cannot_enter_formal_training_or_exact_resume(mode, budget, scope):
    with pytest.raises(ValueError, match="bounded diagnostics"):
        validate_scope({"run_mode": mode, "max_updates": budget, "diagnostic_loss_scope": scope})

import numpy as np
import pytest
from torch import nn
from starVLA.rl.flow_grpo.contracts import (
    check_optimizer,
    parameter_manifest,
    check_manifest,
)
from starVLA.rl.flow_grpo.observation import prepare_policy_observation
from starVLA.rl.flow_grpo.data import keys_for_positions, SceneStream


def test_manifest_tied_parameters_optimizer_and_negative_cases():
    model = nn.Module()
    model.a = nn.Linear(3, 3)
    model.b = model.a
    model.frozen = nn.Linear(3, 3).requires_grad_(False)
    groups = [dict(params=list(model.a.parameters()), lr=0.1, name="a")]
    check_optimizer(model, groups)
    manifest = parameter_manifest(model, groups)
    assert manifest["parameters"][0]["aliases"] == ["a.weight", "b.weight"]
    check_manifest(manifest, parameter_manifest(model, groups))
    for bad in (
        [dict(params=[model.a.weight], lr=0.1)],
        groups + groups,
        [dict(params=list(model.a.parameters()), lr=0)],
    ):
        with pytest.raises(ValueError):
            check_optimizer(model, bad)
    model.frozen.requires_grad_(True)
    with pytest.raises(ValueError):
        check_manifest(manifest, parameter_manifest(model, groups))


def test_future_labels_cannot_enter_observation():
    sample = dict(
        image=["current_l", "current_f", "current_r"],
        lang="command",
        state=np.zeros((1, 4)),
        token="scene",
        action="future",
        depth_data="future",
        future_images="future",
    )
    obs = prepare_policy_observation([sample])
    assert set(obs.examples()[0]) == {"image", "lang", "state", "token"}
    sample["action"] = "different"
    sample["depth_data"] = "different"
    sample["future_images"] = "different"
    assert obs.examples()[0]["image"] == ("current_l", "current_f", "current_r")


def test_sampler_resume_is_consumption_not_prefetch():
    tokens = ["a", "b", "c", "d", "e"]

    class Dataset:
        def __getitem__(self, key):
            return key

    a = SceneStream(Dataset(), tokens, 17, rank=1, world_size=2)
    first = a.take(3)
    state = a.state_dict()
    expected = a.take(7)
    b = SceneStream(Dataset(), tokens, 17, rank=1, world_size=2)
    b.load_state_dict(state)
    assert b.take(7) == expected
    c = SceneStream(Dataset(), tokens, 17, rank=1, world_size=3)
    with pytest.raises(ValueError):
        c.load_state_dict(state)
    assert len(first) == 3


def test_scene_keys_independent_of_chunks():
    tokens = list("abcdefg")
    together = list(keys_for_positions(tokens, 12, range(20)))
    separate = list(keys_for_positions(tokens, 12, range(7))) + list(
        keys_for_positions(tokens, 12, range(7, 20))
    )
    assert together == separate


def test_user_authorized_visual_freeze_resolves_alias_and_nothing_else():
    from starVLA.rl.flow_grpo.contracts import apply_rl_freezes

    class WrappedQwen(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Module()
            self.model.visual = nn.Linear(3, 3)
            self.model.language_model = nn.Linear(3, 3)

        @property
        def visual(self):
            return self.model.visual

    model = nn.Module()
    model.qwen_vl_interface = nn.Module()
    model.qwen_vl_interface.model = WrappedQwen()
    before = parameter_manifest(model, [dict(params=list(model.parameters()), lr=1e-5)])
    allowed = apply_rl_freezes(
        model, {"rl_freeze_modules": ["qwen_vl_interface.model.visual"]}
    )
    groups = [dict(params=[p for p in model.parameters() if p.requires_grad], lr=1e-5)]
    after = parameter_manifest(model, groups)
    check_manifest(before, after, allowed)
    check_optimizer(model, groups)
    assert allowed and all(".model.model.visual." in name for name in allowed)
    with pytest.raises(ValueError):
        check_manifest(before, after)
    with pytest.raises(ValueError):
        apply_rl_freezes(
            model,
            {"rl_freeze_modules": ["qwen_vl_interface.model.model.language_model"]},
        )
    model.qwen_vl_interface.model.model.language_model.requires_grad_(False)
    with pytest.raises(ValueError):
        check_manifest(before, parameter_manifest(model, groups), allowed)

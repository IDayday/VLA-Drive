from types import SimpleNamespace
import random
import numpy as np
import pytest
import torch
from starVLA.rl.flow_grpo.checkpoint import capture_rng, restore_rng
from starVLA.rl.flow_grpo.data import SceneStream, KeyedDataset
from starVLA.rl.flow_grpo.config import resolve_config, split_tokens, config_hash


def test_rng_restores_auxiliary_generators_and_optimizer_scheduler():
    torch.manual_seed(123)
    np.random.seed(123)
    random.seed(123)
    policy = SimpleNamespace(
        rgb_model=SimpleNamespace(
            rng=np.random.default_rng(123), torch_rng=torch.Generator().manual_seed(123)
        )
    )
    p = torch.nn.Parameter(torch.ones(3))
    opt = torch.optim.AdamW([p], lr=0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.9**s)

    def update():
        opt.zero_grad()
        target = (
            torch.randn(3)
            + random.random()
            + np.random.random()
            + policy.rgb_model.rng.random()
            + torch.rand(3, generator=policy.rgb_model.torch_rng)
        )
        loss = (p - target).square().mean()
        loss.backward()
        opt.step()
        scheduler.step()

    update()
    import copy

    rng = capture_rng(policy)
    state = copy.deepcopy(opt.state_dict())
    schedule = copy.deepcopy(scheduler.state_dict())
    weight = p.detach().clone()
    update()
    expected = p.detach().clone()
    expected_opt = copy.deepcopy(opt.state_dict())
    expected_rng = capture_rng(policy)
    p.data.copy_(weight)
    opt.load_state_dict(state)
    scheduler.load_state_dict(schedule)
    restore_rng(rng, policy)
    update()
    torch.testing.assert_close(p, expected, atol=0, rtol=0)
    for key in ["exp_avg", "exp_avg_sq", "step"]:
        torch.testing.assert_close(
            opt.state_dict()["state"][0][key],
            expected_opt["state"][0][key],
            atol=0,
            rtol=0,
        )
    assert torch.equal(capture_rng(policy)["torch"], expected_rng["torch"])


def test_real_data_workers_prefetch_resume_keys():
    cfg, sft = resolve_config("configs/flow_grpo/full_sft.yaml")
    train, _ = split_tokens(cfg)
    dataset = KeyedDataset(sft)
    direct = SceneStream(dataset, train[:8], 99, workers=0)
    workers = SceneStream(dataset, train[:8], 99, workers=2, prefetch=3)
    first = workers.take(1)
    state = workers.state_dict()
    second = workers.take(1)
    resume = SceneStream(dataset, train[:8], 99, workers=2, prefetch=2)
    resume.load_state_dict(state)
    repeated = resume.take(1)
    serial = direct.take(2)
    assert [first[0]["token"], second[0]["token"]] == [s["token"] for s in serial]
    assert repeated[0]["token"] == second[0]["token"]
    for a, b in [
        (first[0], serial[0]),
        (second[0], serial[1]),
        (repeated[0], second[0]),
    ]:
        np.testing.assert_array_equal(a["action"], b["action"])
        np.testing.assert_array_equal(
            np.asarray(a["image"][0]), np.asarray(b["image"][0])
        )
        torch.testing.assert_close(
            a["2d_gen_data"]["pixel_values"],
            b["2d_gen_data"]["pixel_values"],
            atol=0,
            rtol=0,
        )


def test_config_rejects_shortcuts_and_hash_resume_contract():
    cfg, sft = resolve_config("configs/flow_grpo/full_sft.yaml")
    import copy

    second = copy.deepcopy(cfg)
    second["runtime"]["max_updates"] = 50
    second["runtime"]["output_dir"] = "/new/run"
    assert config_hash(cfg) == config_hash(second)
    second["algorithm"]["logprob_reduction"] = "joint_sum"
    assert config_hash(cfg) != config_hash(second)
    for override in [
        "retention.original_sft_enabled=false",
        "trainable_policy=action_head",
        "lora=enabled",
        "sampling.train_step_fraction=0.5",
        "sampling.raw_noise_clipping=true",
    ]:
        with pytest.raises(ValueError):
            resolve_config("configs/flow_grpo/full_sft.yaml", overrides=[override])

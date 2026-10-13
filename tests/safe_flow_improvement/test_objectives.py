import ast
import hashlib
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from starVLA.rl.safe_flow_improvement.contracts import seed_for
from starVLA.rl.safe_flow_improvement.nft_adapter import nft_forward_loss
from starVLA.rl.safe_flow_improvement.weighted_flow_objective import interpolation, positive_flow_loss, sample_targets


def test_scene_weighted_sampling_matches_exact_enumeration():
    actions = torch.tensor([[[[0.0]], [[2.0]], [[4.0]]], [[[1.0]], [[3.0]], [[7.0]]]])
    weights = torch.tensor([[0.25, 0.5, 0.25], [0.8, 0.15, 0.05]])
    generator = torch.Generator().manual_seed(71)
    draws = torch.multinomial(weights, 60000, replacement=True, generator=generator)
    estimates = torch.stack([actions[i, draws[i]].square().mean() for i in range(2)])
    exact = (weights * actions.square().flatten(2).mean(2)).sum(1)
    assert estimates == pytest.approx(exact, abs=0.07)
    selected, _ = sample_targets(actions.requires_grad_(), weights, generator=generator)
    assert not selected.requires_grad
    with pytest.raises(ValueError):
        sample_targets(actions, torch.tensor([[-0.1, 0.6, 0.5], [0.8, 0.15, 0.05]]), generator=generator)


def test_interpolation_direction_and_repeat_scene_reduction():
    y, epsilon = torch.ones(4, 8, 4), torch.zeros(4, 8, 4)
    time = torch.tensor([0.0, 0.25, 0.75, 1.0])
    x, target = interpolation(y, epsilon, time)
    assert x[:, 0, 0] == pytest.approx(time)
    assert torch.equal(target, y)
    prediction = torch.arange(4.0)[:, None, None].expand_as(y)
    loss, per_scene = positive_flow_loss(prediction, target, scenes=2, repeats=2)
    assert per_scene == pytest.approx([1.0, 2.0])
    assert float(loss) == 1.5


def test_rng_namespaces_are_distinct_and_scene_bound():
    namespaces = ("candidate_root", "sde_innovations", "fm_noise", "fm_time", "target_choice", "evaluation")
    keys = {seed_for(name, 42, "aabb", 0) for name in namespaces}
    assert len(keys) == len(namespaces)
    assert seed_for("fm_noise", 42, "aabb", 0) == seed_for("fm_noise", 42, "aabb", 0)


def _official_loss_source(path):
    """Execute the literal upstream AST arithmetic, not a second hand-written oracle."""
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == 'a91116f5a0cd873e4ef63c45eb152de43ed071d0c4c872217d326b9597c5a47a'
    tree = ast.parse(Path(path).read_text())
    nodes = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "positive_prediction" for t in node.targets):
            start = node.lineno
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "policy_loss" for t in node.targets):
            stop = node.end_lineno
    # Top-level statements of the loss block include the two no-grad contexts.
    for node in ast.walk(tree):
        if isinstance(node, ast.For) and any(isinstance(child, ast.Assign) and child.lineno == start for child in node.body):
            nodes = [child for child in node.body if start <= child.lineno <= stop]
            break
    assert nodes and nodes[-1].end_lineno == stop
    return compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), str(path), "exec")


@pytest.mark.parametrize("beta", [0.1, 1.0])
def test_nft_matches_pinned_official_arithmetic_and_gradients(beta):
    source = os.environ.get("OFFICIAL_NFT_SOURCE")
    if not source:
        pytest.skip("set OFFICIAL_NFT_SOURCE to the pinned official train_nft_sd3.py")
    g = torch.Generator().manual_seed(72)
    clean, noisy = [torch.randn(7, 8, 4, generator=g, dtype=torch.float64) for _ in range(2)]
    current = torch.randn(7, 8, 4, generator=g, dtype=torch.float64, requires_grad=True)
    old = torch.randn(7, 8, 4, generator=g, dtype=torch.float64, requires_grad=True)
    time = torch.linspace(0, 0.9, 7, dtype=torch.float64)
    advantage = torch.tensor([-8, -5, -1, 0, 1, 5, 8], dtype=torch.float64)
    loss, parts = nft_forward_loss(current, old, noisy, clean, time, advantage, beta=beta)
    official_current = (-current).detach().requires_grad_()
    scope = {"torch": torch, "config": SimpleNamespace(beta=beta, train=SimpleNamespace(adv_clip_max=5)),
             "forward_prediction": official_current, "old_prediction": -old.detach(),
             "xt": noisy, "x0": clean, "t_expanded": (1-time)[:, None, None],
             "r": (advantage.clamp(-5, 5) / 10 + 0.5).clamp(0, 1)}
    exec(_official_loss_source(source), scope)
    assert float(loss) == pytest.approx(float(scope["policy_loss"]), abs=1e-12)
    ours_grad = torch.autograd.grad(loss, current, retain_graph=True)[0]
    official_grad = torch.autograd.grad(scope["policy_loss"], official_current)[0]
    assert torch.allclose(ours_grad, -official_grad, atol=1e-12, rtol=0)
    assert torch.autograd.grad(loss, old, allow_unused=True)[0] is None
    assert (parts["per_draw"] >= 0).all()


def test_original_time_distribution_negative_tail_is_preserved():
    target = torch.ones(1, 8, 4)
    time = torch.tensor([-0.0009])
    noisy, _ = interpolation(target, torch.zeros_like(target), time)
    assert float(noisy[0, 0, 0]) == pytest.approx(-0.0009)
    with pytest.raises(ValueError):
        interpolation(target, torch.zeros_like(target), torch.tensor([-0.01]))

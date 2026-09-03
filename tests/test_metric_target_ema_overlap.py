from __future__ import annotations

from types import MethodType

import numpy as np
import pytest
import torch
from torch import nn

from navsim.agents.EpisodeDrive.drivevla_base_agent import DriveVLABaseAgent
from navsim.planning.training.formal_timing import PhaseTimer


class _ImmediateFuture:
    def __init__(self, value):
        self._value = value
        self.cancelled = False

    def result(self):
        return self._value

    def cancel(self):
        self.cancelled = True
        return True


class _InlinePool:
    def submit(self, function, *args):
        return _ImmediateFuture(function(*args))


class _FailingPool:
    def __init__(self):
        self.futures = []

    def submit(self, function, *args):
        if self.futures:
            raise RuntimeError("submit failed")
        future = _ImmediateFuture(function(*args))
        self.futures.append(future)
        return future


def _fake_sub_score(token, poses, test):
    del test
    token_offset = 1.0 if token == "cache-a" else 2.0
    base = poses.reshape(len(poses), -1).sum(axis=1) + token_offset
    scores = np.stack([base + index for index in range(7)], axis=-1)
    corners = np.broadcast_to(base[:, None, None], (len(poses), 2, 2)).copy()
    labels = np.zeros((len(poses), 2), dtype=bool)
    ego_areas = np.ones((len(poses), 1), dtype=bool)
    return scores, corners, labels, ego_areas


def _score_agent(*, process_count: int):
    agent = DriveVLABaseAgent.__new__(DriveVLABaseAgent)
    nn.Module.__init__(agent)
    agent.ray = False
    agent.score_process_count = process_count
    agent.score_partition_count = 2
    agent.score_start_method = "spawn"
    agent._score_process_pool = _InlinePool() if process_count else None
    agent.train_metric_cache_paths = {"a": "cache-a", "b": "cache-b"}
    agent.test_metric_cache_paths = dict(agent.train_metric_cache_paths)
    agent.get_sub_score = _fake_sub_score
    agent.get_scores = lambda points: [
        _fake_sub_score(point["token"], point["poses"], point["test"])
        for point in points
    ]
    agent._formal_phase_timer = PhaseTimer(False)
    nn.Module.train(agent, True)
    return agent


def test_partitioned_async_score_is_exactly_equal_to_sync_score():
    torch.manual_seed(7)
    proposals = torch.randn(2, 4, 8, 3, requires_grad=True)
    targets = {
        "token": ["a", "b"],
        "trajectory": torch.randn(2, 8, 3),
    }

    async_agent = _score_agent(process_count=2)
    request = async_agent._submit_score_request(targets, proposals, test=False)
    assert request["proposals"].requires_grad is False
    async_result = async_agent._resolve_score_request(request)

    sync_agent = _score_agent(process_count=0)
    sync_result = sync_agent.compute_score(targets, proposals, test=False)
    assert len(async_result) == len(sync_result) == 6
    for async_tensor, sync_tensor in zip(async_result, sync_result):
        torch.testing.assert_close(async_tensor, sync_tensor, rtol=0, atol=0)
    assert proposals.grad is None


def test_partial_async_submission_is_cancelled_on_failure():
    agent = _score_agent(process_count=2)
    pool = _FailingPool()
    agent._score_process_pool = pool
    targets = {
        "token": ["a", "b"],
        "trajectory": torch.randn(2, 8, 3),
    }
    with pytest.raises(RuntimeError, match="submit failed"):
        agent._submit_score_request(
            targets,
            torch.randn(2, 4, 8, 3),
            test=False,
        )
    assert len(pool.futures) == 1
    assert pool.futures[0].cancelled is True


class _Loss(nn.Module):
    def __init__(self, events):
        super().__init__()
        self.events = events

    def forward(self, targets, pred, config, scoring_function):
        del config
        self.events.append("base_loss")
        final_scores, *_ = scoring_function(
            targets, pred["proposals"], test=False
        )
        return {"loss": pred["planning_registers"].sum() * 0 + final_scores.mean()}


def _overlap_loss_agent(events):
    agent = DriveVLABaseAgent.__new__(DriveVLABaseAgent)
    nn.Module.__init__(agent)
    agent.world_model_enabled = True
    agent.overlap_metric_target_with_ema = True
    agent.ray = False
    agent.score_process_count = 1
    agent.action_head_config = object()
    agent.loss = _Loss(events)
    agent._formal_phase_timer = PhaseTimer(False)
    agent.register_buffer("_ema_current_momentum", torch.tensor(0.9))

    def submit(self, targets, proposals, test=False):
        del targets, proposals, test
        events.append("submit")
        return {"cancelled": False}

    def resolve(self, request):
        del request
        events.append("resolve")
        score = torch.tensor([[3.0]])
        return score, score, score[..., None], torch.zeros(1), torch.zeros(1), torch.zeros(1)

    def encode_ema(self, features, targets, batch_size):
        del features, targets
        events.append("ema")
        return (
            torch.zeros(batch_size, 16, 4),
            torch.zeros(batch_size, 3, 16, 4),
            torch.ones(batch_size, 3, dtype=torch.bool),
        )

    def wm_loss(self, current, trajectory, target_current, target_future, valid, current_speed=None):
        del trajectory, target_current, target_future, valid, current_speed
        events.append("wm")
        value = current.sum() * 0 + 2.0
        return {"wm_loss": value, "predicted_future_registers": current[:, None, None]}

    agent._submit_score_request = MethodType(submit, agent)
    agent._resolve_score_request = MethodType(resolve, agent)
    agent._encode_ema_register_targets = MethodType(encode_ema, agent)
    agent._compute_world_model_loss_from_registers = MethodType(wm_loss, agent)
    agent.current_world_model_weight = MethodType(lambda self: 0.1, agent)
    nn.Module.train(agent, True)
    return agent


def test_compute_loss_submits_score_before_ema_and_resolves_afterward():
    events = []
    agent = _overlap_loss_agent(events)
    current = torch.randn(1, 16, 4, requires_grad=True)
    predictions = {
        "planning_registers": current,
        "proposals": torch.randn(1, 1, 8, 3, requires_grad=True),
    }
    targets = {"trajectory": torch.randn(1, 8, 3)}
    features = {"status_feature": torch.zeros(1, 6)}

    losses = agent.compute_loss(features, targets, predictions)
    assert events == ["submit", "ema", "wm", "base_loss", "resolve"]
    assert losses["loss"].item() == pytest.approx(3.2)
    losses["loss"].backward()
    assert current.grad is not None
    assert predictions["proposals"].grad is None


def test_ema_failure_cancels_outstanding_score_request():
    events = []
    agent = _overlap_loss_agent(events)
    request = {"cancelled": False}

    def submit(self, targets, proposals, test=False):
        del targets, proposals, test
        return request

    def cancel(score_request):
        score_request["cancelled"] = True

    def fail_ema(self, features, targets, batch_size):
        del features, targets, batch_size
        raise RuntimeError("teacher failed")

    agent._submit_score_request = MethodType(submit, agent)
    agent._cancel_score_request = cancel
    agent._encode_ema_register_targets = MethodType(fail_ema, agent)
    predictions = {
        "planning_registers": torch.randn(1, 16, 4),
        "proposals": torch.randn(1, 1, 8, 3),
    }
    with pytest.raises(RuntimeError, match="teacher failed"):
        agent.compute_loss(
            {"status_feature": torch.zeros(1, 6)},
            {"trajectory": torch.randn(1, 8, 3)},
            predictions,
        )
    assert request["cancelled"] is True

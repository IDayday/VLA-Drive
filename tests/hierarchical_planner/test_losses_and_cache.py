import pickle

import numpy as np
import pytest
import torch

from starVLA.model.modules.trajectory_scorer.losses import (
    DRIVOR_METRICS,
    SUPRIM_METRICS,
    DriveSuprimMetricLoss,
    DrivoRMetricLoss,
    aggregate_drivor_score,
    aggregate_drivesuprim_score,
)
from starVLA.model.modules.trajectory_scorer.static_score_store import (
    StaticVocabScoreStore,
)


def test_score_aggregates_are_finite_for_extreme_logits():
    drivor = {name: torch.tensor([[-1.0e4, 1.0e4]]) for name in DRIVOR_METRICS}
    suprim = {name: torch.tensor([[-1.0e4, 1.0e4]]) for name in SUPRIM_METRICS}
    suprim["imi"] = torch.tensor([[-1.0e4, 1.0e4]])
    assert torch.isfinite(aggregate_drivor_score(drivor)).all()
    assert torch.isfinite(aggregate_drivesuprim_score(suprim)).all()


def test_score_aggregates_match_donor_formulas():
    torch.manual_seed(17)
    drivor = {name: torch.randn(2, 7) for name in DRIVOR_METRICS}
    expected_drivor = (
        torch.log(torch.sigmoid(drivor["no_at_fault_collisions"]))
        + torch.log(torch.sigmoid(drivor["drivable_area_compliance"]))
        + 0.0 * torch.log(torch.sigmoid(drivor["driving_direction_compliance"]))
        + torch.log(
            5.0 * torch.sigmoid(drivor["time_to_collision_within_bound"])
            + 5.0 * torch.sigmoid(drivor["ego_progress"])
            + 2.0 * torch.sigmoid(drivor["comfort"])
        )
    )
    torch.testing.assert_close(aggregate_drivor_score(drivor), expected_drivor)

    suprim = {name: torch.randn(2, 11) for name in SUPRIM_METRICS}
    suprim["imi"] = torch.randn(2, 11)
    expected_suprim = (
        0.02 * torch.log_softmax(suprim["imi"], dim=-1)
        + 0.1 * torch.log(torch.sigmoid(suprim["traffic_light_compliance"]))
        + 0.5 * torch.log(torch.sigmoid(suprim["no_at_fault_collisions"]))
        + 0.5 * torch.log(torch.sigmoid(suprim["drivable_area_compliance"]))
        + 0.3 * torch.log(torch.sigmoid(suprim["driving_direction_compliance"]))
        + 6.0
        * torch.log(
            5.0 * torch.sigmoid(suprim["time_to_collision_within_bound"])
            + 5.0 * torch.sigmoid(suprim["ego_progress"])
            + 2.0 * torch.sigmoid(suprim["lane_keeping"])
            + torch.sigmoid(suprim["history_comfort"])
        )
    )
    torch.testing.assert_close(aggregate_drivesuprim_score(suprim), expected_suprim)


def test_losses_do_not_mutate_targets_and_supervise_every_fine_layer():
    torch.manual_seed(3)
    batch, candidates = 2, 5
    drivor_targets = {name: torch.rand(batch, candidates) for name in DRIVOR_METRICS}
    drivor_targets["no_at_fault_collisions"][0, 0] = 0.5
    drivor_targets["time_to_collision_within_bound"][0, 1] = 2.0
    drivor_copy = {name: value.clone() for name, value in drivor_targets.items()}
    drivor_logits = {
        name: torch.randn(batch, candidates, requires_grad=True)
        for name in DRIVOR_METRICS
    }
    drivor_loss, _ = DrivoRMetricLoss()(drivor_logits, drivor_targets)
    assert torch.isfinite(drivor_loss)
    for name in DRIVOR_METRICS:
        torch.testing.assert_close(drivor_targets[name], drivor_copy[name])

    suprim_targets = {name: torch.rand(batch, candidates) for name in SUPRIM_METRICS}
    suprim_targets["driving_direction_compliance"][0, 0] = 0.5
    suprim_copy = {name: value.clone() for name, value in suprim_targets.items()}
    layer_logits = []
    for _ in range(3):
        logits = {
            name: torch.randn(batch, candidates, requires_grad=True)
            for name in (*SUPRIM_METRICS, "imi")
        }
        layer_logits.append(logits)
    candidate_40 = torch.randn(batch, candidates, 40, 3)
    gt_8 = torch.randn(batch, 8, 3)
    total, details = DriveSuprimMetricLoss().refinement(
        layer_logits, suprim_targets, candidate_40, gt_8
    )
    total.backward()
    assert set(details) == {"layer_0", "layer_1", "layer_2"}
    assert all(layer["imi"].grad is not None for layer in layer_logits)
    for name in SUPRIM_METRICS:
        torch.testing.assert_close(suprim_targets[name], suprim_copy[name])


def _cache_row(size):
    return {
        name: np.linspace(0.0, 1.0, size, dtype=np.float32)
        for name in SUPRIM_METRICS
    }


def test_static_score_store_reads_shards_and_fails_on_missing_token(tmp_path):
    split = tmp_path / "train"
    split.mkdir()
    np.savez(split / "known.npz", **_cache_row(16))
    store = StaticVocabScoreStore(str(tmp_path), split="train", vocab_size=16)
    values = store.get(["known"], device=torch.device("cpu"), dtype=torch.float32)
    assert all(value.shape == (1, 16) for value in values.values())
    with pytest.raises(FileNotFoundError, match="missing"):
        store.get(["missing"], device=torch.device("cpu"), dtype=torch.float32)


def test_static_score_store_accepts_official_aggregate_pickle(tmp_path):
    path = tmp_path / "navtrain.pkl"
    with path.open("wb") as stream:
        pickle.dump({"token": _cache_row(8)}, stream)
    store = StaticVocabScoreStore(str(path), vocab_size=8)
    values = store.get(["token"], device=torch.device("cpu"), dtype=torch.float64)
    assert all(value.dtype == torch.float64 for value in values.values())
    assert all(value.shape == (1, 8) for value in values.values())

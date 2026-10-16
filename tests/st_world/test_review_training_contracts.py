"""Independent counterexamples at actual ST trainer boundaries (CPU only)."""
import json

import pytest
import torch

from starVLA.model.framework.st_world_model import SpaceTimeStudent
from starVLA.model.modules.st_world.losses import optimizer_batch_counts
from starVLA.model.modules.st_world.requests import gather_current, gather_future, sample_requests
from tools.st_world_policy.training_state import checkpoint_receipt, verify_checkpoint
from starVLA.model.modules.vehicle_joint.initialization import identity_hash, file_sha256

from test_student_wrapper import Parent, inputs, options


def test_temporal_training_rejects_other_physical_label_times():
    """Four labels alone cannot establish the registered .5/1/2/4 table."""
    cfg = options()
    cfg.update(cell="T2", spatial_parent="S-P2", future=True, future_weight=1.)
    model = SpaceTimeStudent(Parent(), cfg, initialization_seed_value=42)
    requests = sample_requests(2, 3, (18, 24), 96,
        patch_generator=torch.Generator().manual_seed(11), future=True,
        horizon_generator=torch.Generator().manual_seed(12))
    targets = {
        "ego": torch.zeros(2, 8, 4),
        "current_dense_features": torch.zeros(2, 3, 18, 24, 2),
        "current_valid": torch.ones(2, 3, 18, 24, dtype=torch.bool),
        "future_dense_features": torch.zeros(2, 4, 3, 18, 24, 2),
        "future_valid": torch.ones(2, 4, 3, 18, 24, dtype=torch.bool),
        # A real cache can carry another sorted positive four-time table. Its
        # own timestamp/manifest hashes are internally valid, but not this T2.
        "future_times_s": torch.tensor([[.5, 1., 3., 4.], [.5, 1., 3., 4.]]),
    }
    _, current = gather_current(targets["current_dense_features"], targets["current_valid"], requests)
    _, future = gather_future(targets["future_dense_features"], targets["future_valid"], requests)
    counts = optimizer_batch_counts(2, current_valid=current, future_valid=future)
    counts["ego_scenes"] = 2
    with pytest.raises(ValueError, match="[Tt]ime|[Hh]orizon"):
        model(inputs(), targets, requests=requests, global_counts=counts,
            noise_generator=torch.Generator().manual_seed(13),
            time_generator=torch.Generator().manual_seed(14))


def test_native_two_rank_receipt_requires_both_optimizer_shards(tmp_path):
    """Rank RNG files cannot substitute for a missing ZeRO-2 Adam shard."""
    folder = tmp_path / "step_000002"
    folder.mkdir()
    for name in ("mp_rank_00_model_states.pt", "bf16_zero_pp_rank_0_mp_rank_00_optim_states.pt",
                 "rng_rank0.pt", "rng_rank1.pt"):
        (folder / name).write_bytes(name.encode())
    state = {"identity": "review", "completed": 2, "epoch": 0, "offset": 2,
        "tag": folder.name, "termination_reason": "RUNNING_CHECKPOINT"}
    with pytest.raises(ValueError, match="[Oo]ptim|[Ss]hard|[Rr]ank"):
        checkpoint_receipt(folder, state, world_size=2)
    assert not (folder / "COMPLETE.json").exists()


def test_native_reader_refuses_legacy_incomplete_optimizer_inventory(tmp_path):
    """An old incomplete COMPLETE receipt must not enter engine.load_checkpoint."""
    folder = tmp_path / "step_000002"
    folder.mkdir()
    names = ("mp_rank_00_model_states.pt", "bf16_zero_pp_rank_0_mp_rank_00_optim_states.pt",
             "rng_rank0.pt", "rng_rank1.pt")
    for name in names:
        (folder / name).write_bytes(name.encode())
    state = {"identity": "review", "completed": 2, "epoch": 0, "offset": 2,
        "tag": folder.name, "termination_reason": "RUNNING_CHECKPOINT"}
    receipt = {"schema": "st_native_checkpoint_v1", "state": state, "world_size": 2,
        "files": {name: {"sha256": file_sha256(folder / name), "bytes": (folder / name).stat().st_size}
                  for name in names}}
    receipt["receipt_identity"] = identity_hash(receipt)
    (folder / "COMPLETE.json").write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="[Oo]ptim|[Ss]hard|[Rr]ank"):
        verify_checkpoint(folder, identity="review", world_size=2)

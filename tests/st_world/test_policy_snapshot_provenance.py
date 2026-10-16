"""On-policy memory snapshots cannot be misrepresented as saved checkpoints."""
from dataclasses import replace

import numpy as np
import pytest

from starVLA.model.modules.st_world import consequence as c


def request():
    trajectory = np.zeros((8, 3), dtype=np.float64)
    spec = c.ConsequenceSpec("/unused", "/unused", ("/unused",), "campaign", "a" * 64)
    origin = {"kind": "in_memory_parameter_snapshot", "state_sha256": "b" * 64,
        "optimizer_step": 3, "policy_version": "old_3", "parent_checkpoint_sha256": "c" * 64,
        "parent_receipt_sha256": "d" * 64, "numeric_execution_hash": "e" * 64}
    row = c.ConsequenceRequest("campaign", "scene", "log", "candidate", "old_3", None,
        "e" * 64, c.trajectory_sha256(trajectory), "f" * 64, "a" * 64, "/unused", trajectory,
        generator_kind="policy_snapshot", generator_source_hash=c.fingerprint(origin), generator_identity=origin)
    return spec, row


def test_memory_snapshot_has_explicit_provenance_and_no_fictitious_file_hash():
    spec, row = request()
    assert np.array_equal(c._validate_request(row, spec, "old_3"), row.trajectory)
    assert row.identity()["generator_checkpoint_sha256"] is None
    assert row.identity()["generator_identity"]["state_sha256"] == "b" * 64
    with pytest.raises(c.ContractMismatch, match="pretend"):
        c._validate_request(replace(row, generator_checkpoint_sha256="b" * 64), spec, "old_3")


@pytest.mark.parametrize("patch", [
    {"optimizer_step": True}, {"optimizer_step": -1}, {"policy_version": "old_2"},
    {"numeric_execution_hash": "f" * 64}, {"state_sha256": "not-a-sha"},
    {"kind": "external_teacher"}, {"secret_extra": "unregistered"},
])
def test_snapshot_version_numeric_contract_and_schema_cannot_drift(patch):
    spec, row = request()
    origin = {**row.generator_identity, **patch}
    with pytest.raises(c.ContractMismatch):
        c._validate_request(replace(row, generator_identity=origin,
            generator_source_hash=c.fingerprint(origin)), spec, "old_3")


def test_snapshot_content_and_file_backed_legacy_identity_are_bound():
    spec, row = request()
    with pytest.raises(c.ContractMismatch, match="checksum"):
        c._validate_request(replace(row, generator_source_hash="a" * 64), spec, "old_3")
    legacy = replace(row, generator_kind="policy", generator_checkpoint_sha256="b" * 64,
                     generator_source_hash=None, generator_identity=None)
    assert "generator_identity" not in legacy.identity()
    c._validate_request(legacy, spec, "old_3")

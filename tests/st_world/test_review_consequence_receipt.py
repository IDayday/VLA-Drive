"""Independent verifier counterexample; no simulation or GPU work is performed."""
import json
from types import SimpleNamespace

import pytest

from tools.st_world_policy import consequence_smoke


def test_existing_result_does_not_bypass_phase_integrity(tmp_path, monkeypatch):
    # Model the state after a prior result was written and its underlying phase
    # record was lost/corrupted. Source-registration checks remain separately
    # covered by the production reader, so isolate the phase replay boundary.
    (tmp_path / "RESULT.json").write_text(json.dumps({"status": "PASS"}))
    (tmp_path / "phase_0.json.gz").write_bytes(b"truncated former evidence")
    monkeypatch.setattr(consequence_smoke, "read_registration", lambda _root: {})
    with pytest.raises((FileNotFoundError, ValueError, KeyError, consequence_smoke.ContractMismatch)):
        consequence_smoke.verify(SimpleNamespace(output=str(tmp_path)))


@pytest.mark.parametrize("mutated", ["phase_0.json.gz", "phase_1_receipt.json", "RESULT.json"])
def test_complete_evidence_receipt_rechecks_changed_files(tmp_path, monkeypatch, mutated):
    names = {"REGISTRATION.json", "RESULT.json", "PER_SCENE_PARITY.json", "NEGATIVE_REQUESTS.json"}
    names |= {f"phase_{i}{suffix}" for i in range(3) for suffix in (".json.gz", "_receipt.json")}
    for name in names:
        (tmp_path / name).write_text(json.dumps({"status": "PASS", "fixture": name}))
    receipt = {"schema": "st_consequence_evidence_complete_v1",
               "files": {name: consequence_smoke.file_sha256(tmp_path / name) for name in names}}
    (tmp_path / "EVIDENCE_COMPLETE.json").write_text(json.dumps(receipt))
    monkeypatch.setattr(consequence_smoke, "read_registration", lambda _root: {})
    args = SimpleNamespace(output=str(tmp_path))
    assert consequence_smoke.verify(args)["status"] == "PASS"
    (tmp_path / mutated).write_bytes(b"changed after complete receipt")
    with pytest.raises(consequence_smoke.ContractMismatch, match="Changed consequence"):
        consequence_smoke.verify(args)

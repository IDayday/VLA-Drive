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

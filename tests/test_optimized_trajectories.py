import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


path = Path(__file__).resolve().parents[1] / "tools" / "optimized_trajectories.py"
spec = importlib.util.spec_from_file_location("portable_trajectories", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_full_release_and_token_alignment():
    metadata, arrays = module.load_release()
    assert metadata["scene_count"] == 103288
    assert arrays["accepted"].sum() == 92015
    requested = arrays["tokens"][[10, 0, -1]].tolist()
    selected = module.select_tokens(arrays, requested)
    np.testing.assert_array_equal(selected["tokens"], requested)
    np.testing.assert_array_equal(selected["trajectories"], arrays["trajectories"][[10, 0, -1]])
    with pytest.raises(ValueError, match="Missing"):
        module.select_tokens(arrays, ["unknown_token"])
    with pytest.raises(ValueError, match="not unique"):
        module.select_tokens(arrays, [requested[0], requested[0]])


def test_corrupted_file_is_rejected(tmp_path):
    metadata = json.loads((module.DEFAULT_RELEASE / "release.json").read_text())
    (tmp_path / "release.json").write_text(json.dumps(metadata))
    (tmp_path / "trajectories.npz").write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="checksum"):
        module.load_release(tmp_path)


def test_export_has_portable_identity_and_valid_hashes(tmp_path):
    destination = module.export_campaign(module.DEFAULT_RELEASE, tmp_path / "campaign")
    final = destination / "final"
    manifest = json.loads((final / "manifest.json").read_text())
    summary = json.loads((final / "summary.json").read_text())
    hashes = json.loads((final / "artifact_hashes.json").read_text())
    metadata, arrays = module.load_release()
    assert manifest["tokens"] == arrays["tokens"].tolist()
    assert summary["complete"] and summary["accepted_geometry_failures"] == 0
    assert manifest["run_identity"] == summary["run_identity"]
    assert manifest["run_identity"] != metadata["source_run_identity"]
    assert summary["source_run_identity"] == metadata["source_run_identity"]
    for name, expected in hashes.items():
        assert module.digest(final / name) == expected
    with pytest.raises(FileExistsError):
        module.export_campaign(module.DEFAULT_RELEASE, destination)

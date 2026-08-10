import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from starVLA.model.modules.field2plan.trajectory_codec import TrajectoryCodec
from tools.field2plan.materialize_diagnostic_predictions import materialize_predictions


def test_materialize_cli_imports_from_outside_repo(tmp_path: Path):
    script = (
        Path(__file__).resolve().parents[2]
        / "tools/field2plan/materialize_diagnostic_predictions.py"
    )
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_materialize_draft_predictions(tmp_path: Path):
    token = "sample-token"
    datalist = tmp_path / "test_meta.json"
    datalist.write_text(json.dumps([token]), encoding="utf-8")
    diagnostics = tmp_path / "diagnostics"
    diagnostics.mkdir()
    physical = np.zeros((8, 3), dtype=np.float32)
    physical[:, 0] = np.arange(8, dtype=np.float32)
    normalized = TrajectoryCodec().encode_trajectory(physical)
    np.savez(diagnostics / f"{token}.npz", draft_action=normalized)

    output = tmp_path / "draft_predictions"
    summary = materialize_predictions(
        diagnostics_dir=diagnostics,
        output_dir=output,
        datalist_path=datalist,
        key="draft_action",
    )

    np.testing.assert_allclose(
        np.load(output / f"{token}.npy"), physical, rtol=1e-6, atol=1e-6
    )
    assert summary["entry_count"] == 1
    manifest = json.loads((output / "materialization_manifest.json").read_text())
    assert manifest["diagnostic_key"] == "draft_action"


def test_materialize_predictions_fails_on_missing_entry(tmp_path: Path):
    datalist = tmp_path / "test_meta.json"
    datalist.write_text(json.dumps(["missing"]), encoding="utf-8")
    try:
        materialize_predictions(
            diagnostics_dir=tmp_path / "diagnostics",
            output_dir=tmp_path / "output",
            datalist_path=datalist,
            key="draft_action",
        )
    except FileNotFoundError as error:
        assert "missing" in str(error)
    else:
        raise AssertionError("a missing diagnostic must fail")

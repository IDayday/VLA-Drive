import csv
import json
import os
import subprocess
from pathlib import Path

import numpy as np

from starVLA.training.export_register_navtest_predictions import (
    _atomic_candidate,
    _valid_candidate,
)
from tools.score_register64_oracle_pdms import _individual_equivalent_scores


ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = ROOT / "run_register64_oracle64_pdms_dlc.sh"


def test_candidate_archive_roundtrip(tmp_path):
    path = tmp_path / "token.npz"
    proposals = np.arange(4 * 8 * 3, dtype=np.float32).reshape(4, 8, 3)
    _atomic_candidate(path, proposals, 2)
    assert _valid_candidate(path, 4)
    assert not _valid_candidate(path, 64)
    with np.load(path, allow_pickle=False) as payload:
        np.testing.assert_array_equal(payload["proposals"], proposals)
        assert int(payload["selected_index"]) == 2


def test_pool_scores_use_official_pairwise_progress_normalization():
    # Column zero is the PDM reference; the other columns are candidates.
    multi = np.asarray(
        [
            [1.0, 1.0, 1.0, 0.0],
            [1.0, 1.0, 0.5, 1.0],
        ]
    )
    weighted = np.asarray(
        [
            [0.0, 0.0, 0.0, 0.0],  # progress is overwritten below
            [0.8, 0.9, 0.4, 0.7],
            [1.0, 1.0, 0.5, 1.0],
        ]
    )
    raw_progress = np.asarray([8.0, 4.0, 20.0, 30.0])
    weights = np.asarray([5.0, 5.0, 2.0])
    actual = _individual_equivalent_scores(
        multi,
        weighted,
        raw_progress,
        weights,
        progress_index=0,
        progress_distance_threshold=5.0,
    )

    expected = []
    for index in range(1, multi.shape[1]):
        multiplicative = multi[:, [0, index]].prod(axis=0)
        gated = raw_progress[[0, index]] * multiplicative
        if gated.max() > 5.0:
            progress = gated[1] / gated.max()
        else:
            progress = float(multiplicative[1] != 0.0)
        values = weighted[:, index].copy()
        values[0] = progress
        expected.append(multiplicative[1] * (values * weights).sum() / weights.sum())
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1.0e-12)

    # A shared K-way denominator would penalize candidate one by candidate two's
    # larger progress. The exact protocol deliberately does not do that.
    assert actual[0] > 0.7


def test_oracle_launcher_dry_run_is_portable_and_write_free(tmp_path):
    source = tmp_path / "source run"
    output = tmp_path / "must not exist"
    datalist = tmp_path / "test tokens.json"
    environment = os.environ.copy()
    environment.update(
        {
            "VLA_DRIVE_SKIP_ENV_LOCAL": "1",
            "DRIVEDREAMER_ROOT": "/tmp/stale-worktree",
        }
    )
    completed = subprocess.run(
        [
            "bash",
            str(LAUNCHER),
            "--dry-run",
            "--source-run",
            str(source),
            "--output-root",
            str(output),
            "--datalist",
            str(datalist),
            "--workers",
            "17",
        ],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    text = completed.stdout + completed.stderr
    assert f"project_root={ROOT}" in text
    assert f"source_run={source}" in text
    assert "protocol=navsim-v1.1-pdms candidates=64 scenes=12146" in text
    assert "cpu_workers=17" in text
    assert "--export-candidates" in text
    assert "score_register64_oracle_pdms.py" in text
    assert "navsim_v1.1/navsim/navsim/planning/script/run_pdm_score.py" in text
    assert "accelerator/model/evaluation=NOT_RUN" in text
    assert not output.exists()


def test_oracle_result_collector_cross_checks_official_score(tmp_path):
    tokens = ["a", "b"]
    datalist = tmp_path / "tokens.json"
    datalist.write_text(json.dumps(tokens), encoding="utf-8")
    report = tmp_path / "oracle_report.json"
    report.write_text(
        json.dumps(
            {
                "num_scenes": 2,
                "mean_oracle_at_64_pdms": 0.9,
                "mean_drivor_selected_pdms": 0.7,
                "mean_proposal0_pdms": 0.6,
                "mean_deterministic_random_pdms": 0.5,
                "drivor_exact_oracle_rate": 0.125,
            }
        ),
        encoding="utf-8",
    )
    source = tmp_path / "source.csv"
    with source.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("protocol", "score", "score_percent", "num_scenarios", "official_csv"),
        )
        writer.writeheader()
        writer.writerow(
            {
                "protocol": "pdms_v1_1",
                "score": 0.7,
                "score_percent": 70.0,
                "num_scenarios": 2,
                "official_csv": "baseline.csv",
            }
        )
    official = tmp_path / "official"
    official.mkdir()
    with (official / "result.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("token", "valid", "score"))
        writer.writeheader()
        writer.writerow({"token": "a", "valid": True, "score": 0.8})
        writer.writerow({"token": "b", "valid": True, "score": 1.0})
        writer.writerow({"token": "average", "valid": True, "score": 0.9})
    output = tmp_path / "summary"
    command = [
        "python",
        str(ROOT / "tools/collect_register64_oracle_pdms_results.py"),
        "--oracle-report",
        str(report),
        "--official-results-dir",
        str(official),
        "--source-summary",
        str(source),
        "--datalist",
        str(datalist),
        "--output-dir",
        str(output),
        "--expected-scenarios",
        "2",
    ]
    subprocess.run(command, check=True, text=True, capture_output=True)
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert np.isclose(summary["selector_gap_to_oracle"], 0.2)
    assert summary["oracle_at_64_official_pdms"] == 0.9
    assert (output / "summary.md").is_file()

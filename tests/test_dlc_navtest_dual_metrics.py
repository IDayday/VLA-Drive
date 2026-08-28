from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
from pathlib import Path
import pickle
import subprocess

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_helper():
    path = REPO_ROOT / "scripts" / "base_navtest_dual_metrics.py"
    spec = importlib.util.spec_from_file_location("base_navtest_dual_metrics", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


HELPER = _load_helper()


class FakeTrajectory:
    def __init__(self, poses: np.ndarray):
        self.poses = poses


def _write_tokens(path: Path, tokens: list[str]) -> None:
    path.write_text(json.dumps(tokens), encoding="utf-8")


def _write_score(
    path: Path, tokens: list[str], aggregate: str, *, fail: bool = False
) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("token", "valid", "score"))
        writer.writeheader()
        for index, token in enumerate(tokens):
            writer.writerow(
                {
                    "token": token,
                    "valid": "False" if fail and index == 0 else "True",
                    "score": "0.5",
                }
            )
        writer.writerow({"token": aggregate, "valid": str(not fail), "score": "0.5"})


def test_prediction_pickle_conversion_is_exact_and_rank_manifested(
    tmp_path: Path,
) -> None:
    tokens = ["token-a", "token-b"]
    datalist = tmp_path / "tokens.json"
    _write_tokens(datalist, tokens)
    source = tmp_path / "predictions.pkl"
    mapping = {
        token: {"trajectory": FakeTrajectory(np.full((8, 3), index, dtype=np.float32))}
        for index, token in enumerate(tokens)
    }
    with source.open("wb") as stream:
        pickle.dump(mapping, stream)
    destination = tmp_path / "converted"
    arguments = argparse.Namespace(
        pickle=source,
        prediction_root=destination,
        datalist=datalist,
        checkpoint_sha256="a" * 64,
        checkpoint_step="174312",
        expected_count=2,
    )

    HELPER.convert_predictions(arguments)
    manifest = HELPER.validate_prediction_directory(destination, tokens, "a" * 64)

    assert manifest["world_size"] == 1
    assert manifest["rank"] == 0
    assert (destination / "test/inference_manifest.rank0.json").is_file()
    assert (destination / "submission.pkl").is_file()
    assert np.array_equal(
        np.load(destination / "test/token-b.npy", allow_pickle=False),
        mapping["token-b"]["trajectory"].poses,
    )


def test_prediction_conversion_refuses_wrong_shape_and_dtype(tmp_path: Path) -> None:
    datalist = tmp_path / "tokens.json"
    _write_tokens(datalist, ["token"])
    for poses in (
        np.zeros((7, 3), dtype=np.float32),
        np.zeros((8, 3), dtype=np.float64),
    ):
        source = tmp_path / f"{poses.shape[0]}-{poses.dtype}.pkl"
        with source.open("wb") as stream:
            pickle.dump({"token": {"trajectory": FakeTrajectory(poses)}}, stream)
        with pytest.raises(HELPER.ContractError):
            HELPER.load_prediction_pickle(source, ["token"])


def test_metric_cache_view_relocates_metadata_without_touching_source(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source-cache"
    tokens = ["token-a", "token-b"]
    metadata = source / "metadata/cache_metadata_node_0.csv"
    metadata.parent.mkdir(parents=True)
    recorded: list[str] = []
    for token in tokens:
        cache_file = source / "log-name/unknown" / token / "metric_cache.pkl"
        cache_file.parent.mkdir(parents=True)
        cache_file.write_bytes(token.encode("utf-8"))
        recorded.append(
            f"/unmounted/old/cache/log-name/unknown/{token}/metric_cache.pkl"
        )
    with metadata.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("file_name",))
        writer.writeheader()
        writer.writerows({"file_name": value} for value in recorded)
    original = metadata.read_bytes()
    view = tmp_path / "view"

    HELPER.prepare_cache_view(source, view, set(tokens))

    assert metadata.read_bytes() == original
    with next((view / "metadata").glob("*.csv")).open(
        newline="", encoding="utf-8"
    ) as stream:
        rows = list(csv.DictReader(stream))
    assert all(Path(row["file_name"]).is_file() for row in rows)
    assert HELPER.inspect_cache(view, set(tokens), resolve_files=True)["rows"] == 2


def test_pdms_and_epdms_require_exact_tokens_aggregate_and_zero_failures(
    tmp_path: Path,
) -> None:
    tokens = ["token-a", "token-b"]
    pdms = tmp_path / "pdms.csv"
    epdms = tmp_path / "epdms.csv"
    _write_score(pdms, tokens, "average")
    _write_score(epdms, tokens, "average_all_frames")

    assert HELPER.validate_score_csv(pdms, "average", set(tokens))["score"] == 0.5
    assert (
        HELPER.validate_score_csv(epdms, "average_all_frames", set(tokens))["score"]
        == 0.5
    )
    with pytest.raises(HELPER.ContractError):
        HELPER.validate_score_csv(pdms, "average_all_frames", set(tokens))

    failed = tmp_path / "failed.csv"
    _write_score(failed, tokens, "average", fail=True)
    with pytest.raises(HELPER.ContractError, match="failed_scenarios=1"):
        HELPER.validate_score_csv(failed, "average", set(tokens))


def test_summary_is_atomic_and_keeps_protocols_separate(tmp_path: Path) -> None:
    tokens = ["token-a", "token-b"]
    datalist = tmp_path / "tokens.json"
    pdms = tmp_path / "pdms.csv"
    epdms = tmp_path / "epdms.csv"
    output = tmp_path / "result"
    _write_tokens(datalist, tokens)
    _write_score(pdms, tokens, "average")
    _write_score(epdms, tokens, "average_all_frames")

    HELPER.summarize(
        argparse.Namespace(
            run_id="test-run",
            datalist=datalist,
            pdms_csv=pdms,
            epdms_csv=epdms,
            output_root=output,
            expected_count=2,
        )
    )
    payload = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert payload["PDMS"]["aggregate_token"] == "average"
    assert payload["EPDMS"]["aggregate_token"] == "average_all_frames"
    assert not list(output.glob(".summary*.tmp-*"))


def test_runtime_and_launchers_never_mutate_vendor_stack() -> None:
    entry = (REPO_ROOT / "scripts/run_base_navtest_dual_metrics_dlc.sh").read_text()
    submit = (REPO_ROOT / "scripts/submit_base_navtest_dual_metrics_dlc.sh").read_text()
    runtime = (REPO_ROOT / "scripts/check_ppu_runtime.py").read_text()
    combined = entry + submit
    assert "pip install" not in combined
    assert "AUTO_GENERATE_CACHES=0" in entry
    assert (
        "agent.vlm_config.use_flash_attn=false"
        in (REPO_ROOT / "scripts/run_base_pdms.sh").read_text()
    )
    assert HELPER.EXPECTED_FLASH_ATTN_VERSION in runtime
    assert "--worker_gpu 1" in submit
    assert "--workers 1" in submit


def test_entrypoint_dry_run_has_no_write_or_import(tmp_path: Path) -> None:
    output = tmp_path / "must-not-exist"
    command = [
        "bash",
        str(REPO_ROOT / "scripts/run_base_navtest_dual_metrics_dlc.sh"),
        "--run-id",
        "cpu-contract",
        "--checkpoint",
        "/fake/checkpoint",
        "--model-dir",
        "/fake/model",
        "--vlm-config",
        "/fake/vlm",
        "--dino-weights",
        "/fake/dino",
        "--datalist",
        "/fake/test_meta.json",
        "--data-root",
        "/fake/data",
        "--maps-root",
        "/fake/maps",
        "--pdms-cache",
        "/fake/pdms",
        "--epdms-cache",
        "/fake/epdms",
        "--navsim-v2-root",
        "/fake/navsim-v2",
        "--output-root",
        str(output),
        "--dry-run",
    ]
    result = subprocess.run(
        command, cwd="/", check=True, capture_output=True, text=True
    )
    assert "No files were written" in result.stdout
    assert "12,146" in result.stdout
    assert not output.exists()


def test_submit_dry_run_is_one_worker_one_ppu_and_does_not_submit(
    tmp_path: Path,
) -> None:
    output = tmp_path / "result"
    result = subprocess.run(
        [
            "bash",
            str(REPO_ROOT / "scripts/submit_base_navtest_dual_metrics_dlc.sh"),
            "--run-id",
            "cpu-contract",
            "--project-root",
            str(REPO_ROOT),
            "--output-root",
            str(output),
            "--workspace-id",
            "workspace",
            "--worker-image",
            "registry/image:ppu",
            "--dry-run",
        ],
        cwd="/",
        env={**os.environ, "DLC_WORKER_SPEC": ""},
        check=True,
        capture_output=True,
        text=True,
    )
    assert "dlc submit pytorchjob" in result.stdout
    assert "--workers 1" in result.stdout
    assert "--worker_gpu 1" in result.stdout
    assert "no DLC job was submitted" in result.stdout
    assert not output.exists()

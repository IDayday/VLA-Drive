"""CPU-only contract tests for the frozen 193514 trajectory baseline.

These tests deliberately avoid model construction and external checkpoints.  If
the locally audited prediction fixture is present, its hash is checked as an
additional (optional) inference artifact contract.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO_ROOT / "artifacts/field2plan/baseline_manifest.example.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@pytest.fixture(scope="module")
def manifest() -> dict:
    with MANIFEST_PATH.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def test_manifest_locks_the_local_baseline_source(manifest: dict) -> None:
    assert manifest["schema_version"] == 1
    assert manifest["git"]["commit"] == "30505ee3a86326892f8be6c2cc04ca30ab18c93f"
    assert manifest["git"]["branch"] == "main"
    assert manifest["field2plan"]["config_key_present"] is False
    assert manifest["field2plan"]["enabled"] is False

    extension_points = set(manifest.get("phase1_extension_points", []))
    for relative_path, expected_hash in manifest["source_sha256"].items():
        source_path = REPO_ROOT / relative_path
        assert source_path.is_file(), relative_path
        if relative_path not in extension_points:
            assert _sha256(source_path) == expected_hash, relative_path


def test_training_config_action_and_batch_contract(manifest: dict) -> None:
    contract = manifest["baseline"]["config_contract"]
    assert contract["framework_name"] == "QwenOFT"
    assert contract["ver_1225"] == 1
    assert contract["act_norm"] == 1
    assert contract["action_horizon"] == 8
    assert contract["action_dim"] == 4
    assert contract["action_query_tokens"] == 8
    assert contract["qwen_hidden_dim"] == 2048
    assert contract["action_hidden_dim"] == 1536
    assert contract["action_dit_layers"] == 24

    config_path = REPO_ROOT / manifest["baseline"]["config_path"]
    if config_path.is_file():
        config = OmegaConf.load(config_path)
        assert config.framework.name == contract["framework_name"]
        assert config.ver_1225 == contract["ver_1225"]
        assert config.datasets.vla_data.act_norm == contract["act_norm"]
        assert config.framework.action_model.action_horizon == contract["action_horizon"]
        assert config.framework.action_model.action_dim == contract["action_dim"]
        assert config.act_tok == contract["action_query_tokens"]
        assert config.framework.qwenvl.vl_hidden_dim == contract["qwen_hidden_dim"]
        assert config.framework.action_model.hidden_size == contract["action_hidden_dim"]
        assert (
            config.framework.action_model.diffusion_model_cfg.num_layers
            == contract["action_dit_layers"]
        )

    topology = manifest["baseline"]["training_topology"]
    effective_batch = (
        topology["global_processes"]
        * topology["per_device_batch_size"]
        * topology["gradient_accumulation_steps"]
    )
    assert effective_batch == topology["effective_batch_size"] == 32


def test_ver_1225_normalization_and_output_contract(manifest: dict) -> None:
    stats = manifest["trajectory_contract"]["normalization"]
    x_mean, x_std = stats["x"]["mean"], stats["x"]["std"]
    y_mean, y_std = stats["y"]["mean"], stats["y"]["std"]

    physical_xy = np.array(
        [[0.0, 0.0], [10.0, -2.0], [20.0, 4.0]], dtype=np.float64
    )
    heading = np.array([-np.pi + 0.1, 0.0, np.pi - 0.2], dtype=np.float64)
    normalized = np.concatenate(
        [
            ((physical_xy[:, 0] - x_mean) / x_std)[:, None],
            ((physical_xy[:, 1] - y_mean) / y_std)[:, None],
            np.sin(heading)[:, None],
            np.cos(heading)[:, None],
        ],
        axis=-1,
    )

    decoded_xy = np.stack(
        [
            normalized[:, 0] * x_std + x_mean,
            normalized[:, 1] * y_std + y_mean,
        ],
        axis=-1,
    )
    decoded_heading = (np.arctan2(normalized[:, 2], normalized[:, 3]) + np.pi) % (
        2 * np.pi
    ) - np.pi

    np.testing.assert_allclose(decoded_xy, physical_xy, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(decoded_heading, heading, rtol=0.0, atol=1e-12)
    assert manifest["trajectory_contract"]["model_action_shape"] == ["B", 8, 4]
    assert manifest["trajectory_contract"]["navsim_file_shape"] == [8, 3]
    assert manifest["trajectory_contract"]["navsim_columns"] == ["x_m", "y_m", "heading_rad"]


def test_optional_audited_prediction_artifact(manifest: dict) -> None:
    reference = manifest["baseline"]["reference_prediction"]
    assert reference["seed"] == 20260808
    assert reference["qwen_forward_mode"] == "optimized"
    prediction_path = REPO_ROOT / reference["path"]
    if not prediction_path.is_file():
        pytest.skip("Audited prediction artifact is not present in this checkout")

    prediction = np.load(prediction_path, allow_pickle=False)
    assert list(prediction.shape) == reference["shape"] == [8, 3]
    assert str(prediction.dtype) == reference["dtype"] == "float32"
    assert _sha256(prediction_path) == reference["sha256"]

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

from local_stage2.compare_native_proposal_banks import (
    _bank_rows,
    _cross_bank_geometry,
)
from local_stage2.merge_drivor_native_proposals import main as merge_main


def test_native_bank_rows_separate_selection_and_oracle() -> None:
    matrix = {
        "candidate_scores": np.asarray([[0.2, 0.9, 0.5]], dtype=np.float32),
        "predicted_scores": np.asarray([[3.0, 1.0, 2.0]], dtype=np.float32),
        "candidate_factors": np.asarray(
            [[[0.1, 0.2], [0.8, 0.9], [0.4, 0.5]]], dtype=np.float32
        ),
        "candidate_factor_names": np.asarray(["risk", "score"]),
    }
    rows = _bank_rows(matrix, "bank")
    assert rows["bank_selected_index"].tolist() == [0]
    assert rows["bank_oracle_index"].tolist() == [1]
    assert np.allclose(rows["bank_selected_pdms"], [0.2])
    assert np.allclose(rows["bank_oracle_pdms"], [0.9])
    assert np.allclose(rows["bank_regret"], [0.7])
    assert np.allclose(rows["bank_candidate_mean_risk"], [13.0 / 30.0])
    assert np.allclose(rows["bank_oracle_aggregate_score"], [0.9])


def test_cross_bank_geometry_is_zero_for_identical_banks() -> None:
    proposals = np.zeros((64, 8, 3), dtype=np.float32)
    proposals[:, :, 0] = np.arange(64, dtype=np.float32)[:, None]
    metrics = _cross_bank_geometry(
        ["token"],
        {"token": proposals},
        {"token": proposals.copy()},
        np.asarray([7]),
        np.asarray([7]),
    )
    assert all(np.array_equal(value, np.zeros(1)) for value in metrics.values())


def test_merge_native_drivor_shards_preserves_fp32_bank(
    tmp_path: Path, monkeypatch
) -> None:
    root = tmp_path / "native"
    lineage = {
        "native_proposals": True,
        "shard_count": 2,
        "drivor_checkpoint": "/checkpoint.pth",
        "drivor_checkpoint_sha256": "a" * 64,
        "drivor_config": "/config.yaml",
        "drivor_config_sha256": "b" * 64,
    }
    for shard_index in range(2):
        shard = root / f"shard_{shard_index:03d}-of-002"
        shard.mkdir(parents=True)
        shard_lineage = {
            **lineage,
            "created_utc": f"time-{shard_index}",
            "shard_index": shard_index,
        }
        (shard / "manifest.json").write_text(
            json.dumps(
                {
                    "lineage": shard_lineage,
                    "native_proposals": True,
                }
            )
        )
        token = f"token-{shard_index}"
        torch.save(
            {
                "tokens": [token],
                "log_names": [f"log-{shard_index}"],
                "proposals": torch.full(
                    (1, 64, 8, 3), float(shard_index), dtype=torch.float32
                ),
                "scores": torch.zeros((1, 64), dtype=torch.float32),
                "factor_logits": torch.zeros((1, 64, 6), dtype=torch.float16),
            },
            shard / "chunk_000000.pt",
        )

    output = root / "proposal_predictions.pkl"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "merge_drivor_native_proposals.py",
            "--input-root",
            str(root),
            "--output",
            str(output),
            "--expected-scenes",
            "2",
        ],
    )
    merge_main()
    with output.open("rb") as stream:
        merged = pickle.load(stream)
    assert sorted(merged) == ["token-0", "token-1"]
    assert merged["token-0"]["proposals"].dtype == np.float32
    assert merged["token-1"]["proposals"].shape == (64, 8, 3)
    manifest = json.loads((root / "proposal_cache_manifest.json").read_text())
    assert manifest["scene_count"] == 2
    assert manifest["current_observation_only"] is True
    assert manifest["external_proposal_input"] is False

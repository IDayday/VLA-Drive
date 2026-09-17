"""CPU control-flow evidence only: the real publication/controller functions."""

import json
import numpy as np
import pandas as pd
import pytest
from scripts.flow_grpo.paired_experiment import request_evaluation
from starVLA.rl.flow_grpo.evaluation_transaction import (
    evaluation_transaction,
    completed_evaluation,
)
from starVLA.rl.flow_grpo.contracts import digest
from starVLA.rl.flow_grpo.transactions import atomic_json


TOKENS = ["a", "b"]
CACHES = {t: "a" * 64 for t in TOKENS}
IDENTITY = {
    "schema_version": 2,
    "metric_assets_sha256": digest(CACHES),
    "input": "fixture",
}


def result(root, identity=IDENTITY):
    predictions = root / "predictions/rl_dev"
    predictions.mkdir(parents=True)
    physical = np.zeros((2, 8, 3))
    for token, row in zip(TOKENS, physical):
        np.save(predictions / f"{token}.npy", row)
    np.savez(
        root / "trajectories.npz",
        tokens=TOKENS,
        physical=physical,
        normalized=np.zeros((2, 8, 4)),
    )
    pd.DataFrame(
        {
            "token": TOKENS,
            "log_name": ["log", "log"],
            "score": [0.0, 1.0],
            "valid": [True, True],
        }
    ).to_csv(root / "original_protocol_scores.csv", index=False)
    atomic_json(root / "metric_cache_identity.json", CACHES)
    return dict(
        identity=identity,
        split="rl_dev",
        scene_count=2,
        valid=2,
        epdms=0.5,
        complete_split=True,
        metric_cache_identity=digest(CACHES),
    )


def test_first_run_and_five_seed_reuse_are_independent_of_resume(tmp_path):
    executed = []

    def evaluate(seed):
        dest, identity = tmp_path / f"sft_rl_dev_seed{seed}", {**IDENTITY, "seed": seed}

        def executor():
            executed.append(seed)
            return evaluation_transaction(
                dest, identity, TOKENS, lambda root: result(root, identity)
            )

        return request_evaluation(dest, identity, TOKENS, executor)

    evaluate(42)  # actual controller's initial dev request
    for seed in (42, 43, 44, 45, 46):
        evaluate(seed)
    assert executed == [42, 43, 44, 45, 46]
    # A fresh request closure after restart has exactly the same reuse semantics.
    evaluate(42)
    assert len(executed) == 5


def test_identity_conflict_never_overwrites(tmp_path):
    dest = tmp_path / "eval"
    evaluation_transaction(dest, IDENTITY, TOKENS, result)
    with pytest.raises(ValueError, match="identity conflict"):
        request_evaluation(
            dest,
            {**IDENTITY, "input": "different"},
            TOKENS,
            lambda: pytest.fail("executed"),
        )
    assert completed_evaluation(dest, IDENTITY, TOKENS)


def test_interrupted_and_failed_attempts_are_preserved(tmp_path):
    dest = tmp_path / "eval"
    dest.mkdir()
    atomic_json(
        dest / "evaluation_state.json", {"identity": IDENTITY, "status": "RUNNING"}
    )
    (dest / "failure_evidence.txt").write_text("interrupted process")

    def fail(root):
        result(root)
        raise RuntimeError("injected after writing CSV")

    with pytest.raises(RuntimeError, match="injected"):
        evaluation_transaction(dest, IDENTITY, TOKENS, fail)
    assert not (dest / "COMPLETE").exists()
    assert (
        json.loads((dest / "evaluation_state.json").read_text())["status"] == "FAILED"
    )
    evaluation_transaction(dest, IDENTITY, TOKENS, result)
    attempts = list(tmp_path.glob("eval.attempt-*"))
    assert len(attempts) == 2
    assert any((p / "failure_evidence.txt").exists() for p in attempts)
    assert completed_evaluation(dest, IDENTITY, TOKENS)


@pytest.mark.parametrize(
    "damage",
    ["missing_token", "nan", "missing_file", "extra_prediction", "same_size_edit"],
)
def test_complete_but_corrupt_results_are_not_reused(tmp_path, damage):
    dest = tmp_path / "eval"
    evaluation_transaction(dest, IDENTITY, TOKENS, result)
    csv = dest / "original_protocol_scores.csv"
    if damage == "missing_token":
        pd.read_csv(csv).iloc[:1].to_csv(csv, index=False)
    elif damage == "nan":
        frame = pd.read_csv(csv)
        frame.loc[0, "score"] = np.nan
        frame.to_csv(csv, index=False)
    elif damage == "missing_file":
        (dest / "trajectories.npz").unlink()
    elif damage == "extra_prediction":
        np.save(dest / "predictions/rl_dev/extra.npy", np.zeros((8, 3)))
    else:
        csv.write_text(csv.read_text().replace("log", "bad"))
    with pytest.raises((ValueError, FileNotFoundError)):
        request_evaluation(
            dest,
            IDENTITY,
            TOKENS,
            lambda: pytest.fail("must not reuse or overwrite corruption"),
        )


def test_actual_input_identity_binds_data_root_metadata_images_processor(tmp_path):
    import pickle
    from starVLA.rl.flow_grpo.config import resolve_config
    from starVLA.rl.flow_grpo.evaluation import evaluation_identity

    cfg, sft = resolve_config("configs/flow_grpo/paired_frozen_visual.yaml")
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "pytorch_model.pt").write_bytes(b"fixture weight identity only")
    (checkpoint / "config.yaml").write_text("fixture: true")
    processor = tmp_path / "processor"
    processor.mkdir()
    (processor / "config.json").write_text('{"size":1}')
    cfg["paths"].update(base_vlm=str(processor), asset_manifest=None)
    data = tmp_path / "data"
    (data / "meta/train").mkdir(parents=True)
    image = data / "image"
    image.write_bytes(b"a")
    metadata = {
        "glo_images": {
            v: {"image_paths": [str(image)] * 4} for v in ("cam_f0", "cam_l0", "cam_r0")
        }
    }
    meta = data / "meta/train/a.pkl"
    meta.write_bytes(pickle.dumps(metadata))
    tokens = tmp_path / "tokens.json"
    tokens.write_text('["a"]')
    cache = tmp_path / "cache/log/type/a"
    cache.mkdir(parents=True)
    (cache / "metric_cache.pkl").write_bytes(b"fixture metric identity only")

    def identity():
        return evaluation_identity(
            cfg, sft, checkpoint, tokens, 42, "rl_dev", data, tmp_path / "cache"
        )

    old = identity()
    (data / "unrelated").write_text("not an input")
    assert identity() == old
    image.write_bytes(b"b")
    assert identity() != old
    old = identity()
    metadata["history"] = "changed"
    meta.write_bytes(pickle.dumps(metadata))
    assert identity() != old
    old = identity()
    (processor / "config.json").write_text('{"size":2}')
    assert identity() != old

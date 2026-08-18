import argparse
import csv
import importlib.util
import json
import pickle
from pathlib import Path

import numpy as np


MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "tools"
    / "evaluate_action_only_visual_comparison.py"
)
SPEC = importlib.util.spec_from_file_location("visual_comparison_tools", MODULE_PATH)
TOOLS = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(TOOLS)


def test_train_subset_and_physical_metrics(tmp_path):
    data_root = tmp_path / "data"
    metadata_root = data_root / "meta" / "train"
    metadata_root.mkdir(parents=True)
    tokens = [f"token-{index}" for index in range(4)]
    datalist = tmp_path / "train.json"
    datalist.write_text(json.dumps(tokens), encoding="utf-8")

    for index, token in enumerate(tokens):
        poses = np.zeros((14, 3), dtype=np.float64)
        poses[:, 0] = np.arange(14) + index
        with (metadata_root / f"{token}.pkl").open("wb") as stream:
            pickle.dump({"glo_status": {"global_poses": poses}}, stream)

    output_dir = tmp_path / "subset"
    TOOLS.make_train_subset(
        argparse.Namespace(
            datalist=str(datalist),
            data_root=str(data_root),
            output_dir=str(output_dir),
            size=2,
            seed=7,
        )
    )
    with np.load(output_dir / "train_subset_ground_truth.npz") as payload:
        selected = payload["tokens"].astype(str).tolist()
        target = payload["trajectory"]
    assert target.shape == (2, 8, 3)
    np.testing.assert_allclose(
        target[..., 0], np.broadcast_to(np.arange(1, 9), (2, 8))
    )
    np.testing.assert_allclose(target[..., 1:], 0)

    prediction_dir = tmp_path / "predictions"
    prediction_dir.mkdir()
    prediction = target.copy()
    prediction[..., 1] += 1.0
    for token, value in zip(selected, prediction):
        np.save(prediction_dir / f"{token}.npy", value)
    metrics_path = tmp_path / "metrics.json"
    TOOLS.score_train(
        argparse.Namespace(
            ground_truth=str(output_dir / "train_subset_ground_truth.npz"),
            prediction_dir=str(prediction_dir),
            arm="visual",
            step=30000,
            output=str(metrics_path),
        )
    )
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics["sample_count"] == 2
    assert metrics["ade"] == 1.0
    assert metrics["fde"] == 1.0
    assert metrics["heading_mae_rad"] == 0.0


def test_paired_summary_reports_visual_minus_frozen(tmp_path):
    root = tmp_path / "comparison"
    header = list(TOOLS.PDMS_COLUMNS)
    for arm, pdms, ade in (("frozen", 0.80, 2.0), ("visual", 0.83, 1.5)):
        summary = root / "pdms" / arm / "step30000" / "summary.csv"
        summary.parent.mkdir(parents=True)
        with summary.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=header)
            writer.writeheader()
            writer.writerow({column: pdms for column in header})
        train = root / "train_metrics" / arm / "step30000.json"
        train.parent.mkdir(parents=True)
        train.write_text(
            json.dumps({"ade": ade, "fde": ade + 1, "heading_mae_rad": 0.1}),
            encoding="utf-8",
        )

    TOOLS.summarize(argparse.Namespace(root=str(root), steps=[30000]))
    with (root / "paired_summary.csv").open(newline="", encoding="utf-8") as stream:
        row = next(csv.DictReader(stream))
    assert np.isclose(float(row["delta_pdms"]), 0.03)
    assert np.isclose(float(row["delta_train_ade"]), -0.5)
    assert "Best trainable-visual checkpoint" in (root / "REPORT.md").read_text(
        encoding="utf-8"
    )

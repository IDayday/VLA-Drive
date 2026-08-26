import csv
import json
import os
import subprocess
from pathlib import Path

import pytest

from starVLA.training.config_loader import load_training_config
from starVLA.training.navtest_score_io import validate_score_directory


ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = ROOT / "starVLA/config/training"
OFF = ROOT / "run_register64_drivor_off_dlc.sh"
ON = ROOT / "run_register64_drivor_suprim_on_dlc.sh"
COMMON = ROOT / "train_register64_drivor_pipeline_dlc.sh"


def _dry_run(launcher: Path, tmp_path: Path) -> str:
    output_root = tmp_path / "must-not-be-created"
    environment = os.environ.copy()
    environment.update(
        {
            "VLA_DRIVE_SKIP_ENV_LOCAL": "1",
            "REGISTER64_OUTPUT_ROOT": str(output_root),
            "REGISTER64_RUN_ID": "dry-run-contract",
            "DRIVEDREAMER_ROOT": "/tmp/stale-root",
        }
    )
    completed = subprocess.run(
        ["bash", str(launcher), "--dry-run"],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert not output_root.exists()
    return completed.stdout + completed.stderr


def test_complete_wrappers_are_fixed_portable_16_device_arms(tmp_path):
    off = _dry_run(OFF, tmp_path)
    on = _dry_run(ON, tmp_path)
    assert "arm=off drivesuprim=0" in off
    assert "arm=on drivesuprim=1" in on
    for output in (off, on):
        assert f"project_root={ROOT}" in output
        assert "local:16 total:16" in output
        assert "official_eval=v1.1:average v2:average_all_frames" in output
        assert "formal_training=NOT_RUN" in output
    assert "train_register_suprim.py" not in off
    assert "train_register_suprim.py" in on
    for launcher in (OFF, ON):
        source = launcher.read_text(encoding="utf-8")
        assert "${BASH_SOURCE[0]}" in source
        assert "/mnt/" not in source
        assert (
            'exec bash "$project_root/train_register64_drivor_pipeline_dlc.sh" "$@"'
            in source
        )


def test_common_pipeline_contains_training_bank_and_both_official_protocols():
    source = COMMON.read_text(encoding="utf-8")
    for required in (
        "train_register_generator.py",
        "build_register_candidate_bank.py",
        "train_register_drivor.py",
        "train_register_suprim.py",
        "export_register_navtest_predictions.py",
        "navsim_v1.1/navsim/navsim/planning/script/run_pdm_score.py",
        "navsim/navsim/planning/script/run_pdm_score_one_stage.py",
        "average_all_frames",
        "best_minade_generator.pt",
        "best_regret.pt",
        "training_complete.json",
        "check_ppu_runtime.py",
        "PDMS_METRIC_CACHE_PATH",
        "EPDMS_METRIC_CACHE_PATH",
        "QDS_NAVSIM_SENSOR_PATH",
        "summary.csv",
    ):
        assert required in source
    assert "best_oracle_generator.pt" not in source
    assert source.index("register64_phase=stage-g\n") < source.index(
        "register64_phase=cache-navtrain-v2\n"
    )
    assert source.index(
        "full preflight passed; formal_training=NOT_RUN"
    ) < source.index("run_distributed()")


def test_navtest_export_has_strict_resume_and_rank_manifests():
    source = (
        ROOT / "starVLA/training/export_register_navtest_predictions.py"
    ).read_text(encoding="utf-8")
    assert "prediction_identity.json" in source
    assert "inference_manifest.rank" in source
    assert "CHECKPOINT_MIN_AGE_SECONDS" in source
    assert "Refusing to mix prediction artifacts" in source


def test_logical_val_split_reads_physical_navtrain_records():
    generator = load_training_config(CONFIG_ROOT / "qwen_register64_generator.yaml")
    bank = load_training_config(CONFIG_ROOT / "register64_candidate_bank.yaml")
    assert generator.validation.split == "val"
    assert generator.validation.dataset_split == "train"
    assert bank.candidate_bank.splits.val.dataset_split == "train"


def test_deterministic_train_val_split_roundtrip(tmp_path):
    source = tmp_path / "source.json"
    source.write_text(
        json.dumps([f"token-{index}" for index in range(30)]), encoding="utf-8"
    )
    output = tmp_path / "split"
    command = [
        "python",
        str(ROOT / "tools/prepare_register64_train_val_split.py"),
        "--source",
        str(source),
        "--output-dir",
        str(output),
        "--validation-size",
        "7",
        "--seed",
        "42",
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)
    first_train = (output / "train.json").read_bytes()
    first_val = (output / "val.json").read_bytes()
    subprocess.run(
        command + ["--validate-only"], check=True, capture_output=True, text=True
    )
    subprocess.run(command, check=True, capture_output=True, text=True)
    assert (output / "train.json").read_bytes() == first_train
    assert (output / "val.json").read_bytes() == first_val
    train = set(json.loads(first_train))
    val = set(json.loads(first_val))
    assert len(train) == 23 and len(val) == 7 and not train & val


def _write_score_csv(
    root: Path, summary_token: str, scenarios: int, score: float = 0.75
):
    root.mkdir(parents=True, exist_ok=True)
    path = root / "2026.01.01.csv"
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("", "token", "valid", "score"))
        writer.writeheader()
        for index in range(scenarios):
            writer.writerow(
                {"": index, "token": f"t{index}", "valid": True, "score": 0.5}
            )
        writer.writerow(
            {"": scenarios, "token": summary_token, "valid": True, "score": score}
        )
    return path


@pytest.mark.parametrize(
    ("protocol", "summary_token"),
    (("pdms", "average"), ("epdms", "average_all_frames")),
)
def test_official_score_validation(protocol, summary_token, tmp_path):
    _write_score_csv(tmp_path, summary_token, 5)
    result = validate_score_directory(tmp_path, protocol=protocol, expected_scenarios=5)
    assert result["score"] == pytest.approx(0.75)
    assert result["score_percent"] == pytest.approx(75.0)


def test_official_score_validation_rejects_incomplete_navtest(tmp_path):
    _write_score_csv(tmp_path, "average", 4)
    with pytest.raises(RuntimeError, match="scenario count mismatch"):
        validate_score_directory(tmp_path, protocol="pdms", expected_scenarios=5)


def test_official_score_validation_rejects_wrong_token_set(tmp_path):
    _write_score_csv(tmp_path, "average", 5)
    with pytest.raises(RuntimeError, match="token set mismatch"):
        validate_score_directory(
            tmp_path,
            protocol="pdms",
            expected_scenarios=5,
            expected_tokens=["t0", "t1", "t2", "t3", "different"],
        )


def test_metric_cache_validator_checks_expected_coverage(tmp_path):
    cache = tmp_path / "cache"
    metadata = cache / "metadata"
    metadata.mkdir(parents=True)
    tokens = ["a", "b"]
    paths = []
    for token in tokens:
        path = cache / token / "metric_cache.pkl"
        path.parent.mkdir()
        path.write_bytes(b"cache")
        paths.append(path)
    (metadata / "cache_metadata_node_0.csv").write_text(
        "cache_path\n" + "\n".join(str(path) for path in paths) + "\n",
        encoding="utf-8",
    )
    datalist = tmp_path / "tokens.json"
    datalist.write_text(json.dumps(tokens), encoding="utf-8")
    completed = subprocess.run(
        [
            "python",
            str(ROOT / "tools/validate_navsim_metric_cache.py"),
            "--cache-root",
            str(cache),
            "--expected-datalist",
            str(datalist),
            "--check-cache-files",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert json.loads(completed.stdout)["cache_tokens"] == 2

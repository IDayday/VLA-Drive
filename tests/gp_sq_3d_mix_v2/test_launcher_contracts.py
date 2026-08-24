from pathlib import Path
import json
import subprocess


ROOT = Path(__file__).resolve().parents[2]


def test_single_ppu_batch1_accumulation32_is_supported():
    source = (ROOT / "tools/launch_gp_sq3dmix_training.sh").read_text()
    assert "effective_batch=$((topology_product * gradient_accumulation))" in source
    assert '[[ "$effective_batch" == 32 ]]' in source
    smoke = (ROOT / "25-smoke_gp_sq3dmix_stage_a_v2.sh").read_text()
    assert "--devices 1 --per-device-batch 1 --gradient-accumulation 32" in smoke


def test_stage_b_binds_both_required_seeds_and_matched_control():
    source = (ROOT / "28-train_gp_sq3dmix_stage_b_multiseed.sh").read_text()
    assert "20260824 20260825" in source
    assert 'for variant in "$selected_variant" control' in source


def test_formal_launchers_enforce_permission_flags():
    formal = (ROOT / "31-train_gp_sq3dmix_formal_30k.sh").read_text()
    extension = (ROOT / "33-continue_gp_sq3dmix_to_100k.sh").read_text()
    assert "formal_30k_allowed" in formal
    assert "formal_100k_allowed" in extension
    assert "--dry-run" in formal and "--dry-run" in extension


def test_formal_30k_launcher_rejects_false_permission_before_launch(tmp_path):
    permission = tmp_path / "permission.json"
    stage_a = tmp_path / "stage_a.json"
    permission.write_text(json.dumps({"formal_30k_allowed": False}))
    stage_a.write_text(json.dumps({"all_passed": True}))
    result = subprocess.run(
        [
            "bash",
            str(ROOT / "31-train_gp_sq3dmix_formal_30k.sh"),
            "--permission-report",
            str(permission),
            "--stage-a-decision",
            str(stage_a),
            "--dry-run",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert "formal_30k_allowed is false" in result.stderr


def test_formal_100k_permission_has_one_full_navtest_issuer():
    summary = (ROOT / "tools/summarize_gp_sq3dmix_formal_v2.py").read_text()
    evaluator = (ROOT / "32-eval_gp_sq3dmix_formal_30k.sh").read_text()
    assert '"formal_100k_allowed": passed' in summary
    assert "formal_30k_full_navtest" in summary
    assert "--select-final" in evaluator
    assert "--full-navtest" in evaluator
    assert "eval_gp_sq3dmix_formal_full_navtest.sh" in evaluator


def test_all_new_launchers_bind_per_token_noise_and_decisions():
    for name in (
        "29-eval_gp_sq3dmix_stage_b_multiseed.sh",
        "30-eval_gp_sq3dmix_stage_b_full_navtest.sh",
        "tools/eval_gp_sq3dmix_formal_full_navtest.sh",
    ):
        source = (ROOT / name).read_text()
        assert "INFER_NOISE_MODE=per_token" in source
        assert "hard_negative_map.json" in source


def test_plain_python_stage_a_evaluator_initializes_accelerate_logging():
    source = (ROOT / "tools/evaluate_gp_sq3dmix_stage_a.py").read_text()
    assert "from accelerate import PartialState" in source
    assert source.index("PartialState()") < source.index("build_dataloader(")

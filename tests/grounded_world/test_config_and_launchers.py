import os
import subprocess
from pathlib import Path

from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[2]


def _config(name: str):
    return OmegaConf.load(ROOT / "starVLA/config/training" / name)


def _select_arm(
    name: str, stage: str = "stage3", phase: str = "A"
) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        GROUNDEDWORLD_EXPERIMENT=name,
        GROUNDEDWORLD_STAGE=stage,
        GROUNDEDWORLD_RUN_SEED="47",
        GROUNDEDWORLD_MATRIX_PRINT_ONLY="1",
        GROUNDEDWORLD_STAGE3_PHASE=phase,
    )
    result = subprocess.run(
        ["bash", "scripts/grounded_world/04_run_experiment.sh"],
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    return dict(line.split("=", 1) for line in result.stdout.splitlines())


def test_stage_configs_separate_prior_future_and_planning() -> None:
    stage1 = _config("cfg_groundedworld_stage1.yaml")
    stage2 = _config("cfg_groundedworld_stage2.yaml")
    stage3 = _config("cfg_groundedworld_stage3.yaml")

    assert stage1.framework.name == "QwenOFT_GroundedWorld"
    assert stage1.grounded_world.training.stage == "prior"
    assert stage1.grounded_world.prior.enabled is True
    assert stage1.grounded_world.future.enabled is False
    assert stage1.grounded_world.planner.enabled is False

    assert stage2.grounded_world.training.stage == "predictive"
    assert stage2.grounded_world.prior.retention_enabled is True
    assert stage2.grounded_world.future.enabled is True
    assert stage2.grounded_world.future.target.source == "student_ema"
    assert stage2.grounded_world.future.target.shared_across_teacher_controls is True
    assert stage2.grounded_world.planner.enabled is False

    assert stage3.grounded_world.training.stage == "planning"
    assert stage3.grounded_world.planner.enabled is True
    assert stage3.grounded_world.planner.zero_init is True
    assert stage3.grounded_world.consequence.inference_enabled is False
    assert list(stage3.grounded_world.consequence.target_scales) == [
        10.0,
        4.0,
        1.0,
        5.0,
        20.0,
        1.0,
    ]
    assert stage3.trainer.learning_rate["baseline_model.action_model"] == 1.0e-5
    assert stage3.trainer.learning_rate["baseline_model.qwen_vl_interface"] <= 1.0e-5
    assert "action_model" not in stage3.trainer.learning_rate

    debug = _config("cfg_groundedworld_debug.yaml")
    assert debug.is_debug is True
    assert debug.datasets.vla_data.max_samples == 8
    assert debug.grounded_world.memory.field_size == [24, 24]


def test_b0_to_b5_matrix_matches_revised_research_plan() -> None:
    expected = {
        "b0_pure_vlm_dit": ("none", "0", "0", "0"),
        "b1_geometry_aux": ("vggt", "0", "0", "0"),
        "b2_geometry_access": ("vggt", "0", "1", "0"),
        "b3_current_world": ("vggt_driving_jepa", "0", "1", "0"),
        "b4_predictive_world": ("vggt_driving_jepa", "1", "1", "0"),
        "b5_full": ("vggt_driving_jepa", "1", "1", "1"),
    }
    for arm, values in expected.items():
        selected = _select_arm(arm)
        assert (
            selected["external_prior"],
            selected["future_enabled"],
            selected["world_access"],
            selected["consequence_enabled"],
        ) == values
        assert selected["run_seed"] == "47"
    assert _select_arm("b0_pure_vlm_dit")["refiner_enabled"] == "0"
    assert _select_arm("b1_geometry_aux")["refiner_enabled"] == "0"
    assert _select_arm("b2_geometry_access")["refiner_enabled"] == "1"


def test_teacher_and_access_controls_are_explicit_matrix_arms() -> None:
    real = _select_arm("control_real_sup_access")
    no_teacher = _select_arm("control_no_teacher_same_future")
    shuffled = _select_arm("control_scene_shuffled_same_future")
    no_access = _select_arm("control_real_sup_noaccess")
    random_frozen = _select_arm("control_random_frozen_same_future")
    gt_task = _select_arm("control_gt_task_mlp_same_future")

    assert real["teacher_mode"] == "real"
    assert no_teacher["teacher_mode"] == "none"
    assert shuffled["teacher_mode"] == "scene_shuffled"
    assert all(value["future_target"] == "student_ema" for value in (real, no_teacher, shuffled, no_access))
    assert real["world_access"] == "1"
    assert no_access["world_access"] == "0"
    assert random_frozen["external_prior"] == "vggt_random_frozen"
    assert gt_task["external_prior"] == "vggt_gt_task_mlp"


def test_consequence_cache_pipeline_has_concrete_navsim_provider() -> None:
    metric_script = ROOT / "scripts/grounded_world/00_cache_navsim_metrics_train.sh"
    consequence_script = ROOT / "scripts/grounded_world/00_build_consequence_labels.sh"
    assert metric_script.is_file()
    metric_text = metric_script.read_text(encoding="utf-8")
    consequence_text = consequence_script.read_text(encoding="utf-8")
    assert "TRAIN_TEST_SPLIT=navtrain" in metric_text
    assert "run_metric_caching.sh" in metric_text
    assert "navsim_consequence_provider:build_navsim_nonreactive_provider" in consequence_text


def test_external_cache_launchers_share_groundedworld_paths_and_sensor_roots() -> None:
    geometry = (ROOT / "scripts/grounded_world/00_cache_geometry_vggt.sh").read_text(
        encoding="utf-8"
    )
    assert "FIELD2PLAN_VGGT_CACHE" in geometry
    assert "GROUNDEDWORLD_GEOMETRY_CACHE" in geometry
    for name in (
        "00_cache_current_prior.sh",
        "00_cache_generic_vjepa_control.sh",
        "00_cache_future_ema.sh",
    ):
        text = (ROOT / "scripts/grounded_world" / name).read_text(encoding="utf-8")
        assert '--runtime-raw-root "$OPENSCENE_DATA_ROOT"' in text
        assert '--trainval-sensor-root "$NAVSIM_TRAINVAL_SENSOR_ROOT"' in text
    generic_tool = (
        ROOT / "tools/grounded_world/cache_current_prior_vjepa.py"
    ).read_text(encoding="utf-8")
    assert "INPUT_FRAMES = HISTORY" in generic_tool
    assert 'input_frame_indices=np.asarray(INPUT_FRAMES' in generic_tool


def test_stage3_b1_uses_direct_world_plus_baseline_init() -> None:
    b1_phase_b = _select_arm("b1_geometry_aux", phase="B")
    b5_phase_a = _select_arm("b5_full", phase="A")
    assert b1_phase_b["stage3_direct_init"] == "1"
    assert b5_phase_a["stage3_direct_init"] == "0"
    common = (ROOT / "scripts/grounded_world/train_stage_common.sh").read_text(
        encoding="utf-8"
    )
    assert "GROUNDEDWORLD_STAGE3_DIRECT_INIT" in common
    assert "GROUNDEDWORLD_STAGE3A_CHECKPOINT" in common


def test_navsim_v2_eval_launcher_keeps_navtest_and_navhard_protocols_separate() -> None:
    launcher = ROOT / "scripts/grounded_world/05_eval_checkpoint_navsim_v2_16gpu.sh"
    cache_launcher = ROOT / "scripts/grounded_world/00_cache_navsim_metrics_eval.sh"
    assert launcher.is_file()
    assert cache_launcher.is_file()

    text = launcher.read_text(encoding="utf-8")
    assert "run_pdm_score_one_stage.py" in text
    assert "run_pdm_score.py" in text
    assert "navhard_two_stage" in text
    assert "NAVHARD_DATALIST" in text
    assert "validate_navsim_v2_results.py" in text
    assert "WORLD_SIZE=16" in text

    cache_text = cache_launcher.read_text(encoding="utf-8")
    assert "navtest|navhard_two_stage" in cache_text
    assert "run_metric_caching.sh" in cache_text


def test_navhard_metadata_launcher_is_explicit_and_atomic() -> None:
    launcher = ROOT / "scripts/grounded_world/00_prepare_navhard_metadata.sh"
    assert launcher.is_file()
    text = launcher.read_text(encoding="utf-8")
    assert "navsim_data_process/make_data.py" in text
    assert "navhard_two_stage/sensor_blobs" in text
    assert "navhard_two_stage/synthetic_scene_pickles" in text
    assert "build_datalist.py" in text


def test_inference_constructs_groundedworld_and_loads_combined_checkpoint_strictly() -> None:
    text = (ROOT / "infer.py").read_text(encoding="utf-8")
    assert "Qwenvl_OFT_GroundedWorld" in text
    assert 'framework_name == "QwenOFT_GroundedWorld"' in text
    assert "load_checkpoints=False" in text
    assert '"QwenOFT_Field2Plan",' in text
    assert '"QwenOFT_GroundedWorld",' in text
    assert "strict=strict_field2plan" in text
    assert "grounded_world_intervention" in text

    launcher = (ROOT / "4-infer.sh").read_text(encoding="utf-8")
    assert "GROUNDEDWORLD_INFERENCE_INTERVENTION" in launcher
    assert "SAVE_DIAGNOSTICS" in launcher

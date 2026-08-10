"""Static contracts for the non-interactive Field2Plan launchers."""

import os
from pathlib import Path
import subprocess


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _script(name: str) -> str:
    return (PROJECT_ROOT / "scripts" / "field2plan" / name).read_text(
        encoding="utf-8"
    )


def test_debug_geometry_launcher_can_smoke_formal_train_caches() -> None:
    script = _script("04_debug_geometry.sh")
    assert "FIELD2PLAN_DEBUG_SPLIT" in script
    assert '--datasets.vla_data.split "$debug_split"' in script
    assert "--field2plan.proposal.cache_splits \"[$debug_split]\"" in script


def test_formal_geometry_launcher_pins_cache_manifests() -> None:
    script = _script("05_train_geometry.sh")
    assert "FIELD2PLAN_DRAFT_MANIFEST_SHA256" in script
    assert "--field2plan.geometry.manifest_sha256" in script
    assert '--seed "$run_seed"' in script


def _matrix_selection(name: str) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        FIELD2PLAN_EXPERIMENT=name,
        FIELD2PLAN_RUN_SEED="43",
        FIELD2PLAN_MATRIX_PRINT_ONLY="1",
    )
    result = subprocess.run(
        ["bash", "scripts/field2plan/07_run_phase2_experiment.sh"],
        cwd=PROJECT_ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    return dict(line.split("=", 1) for line in result.stdout.splitlines())


def test_phase2_matrix_maps_scientific_controls() -> None:
    no_access = _matrix_selection("p2_10_sup_noaccess_da3")
    assert no_access["supervision"] == "1"
    assert no_access["disable_access"] == "1"
    assert no_access["teacher_mode"] == "real"
    assert no_access["teacher_type"] == "da3"

    full = _matrix_selection("p2_11_sup_access_vggt")
    assert full["supervision"] == "1"
    assert full["disable_access"] == "0"
    assert full["teacher_mode"] == "real"
    assert full["teacher_type"] == "vggt"
    assert full["run_seed"] == "43"


def test_phase2_matrix_keeps_equal_capacity_control() -> None:
    control = _matrix_selection("p2_00_nosup_noaccess")
    assert control["supervision"] == "0"
    assert control["disable_access"] == "1"
    assert control["teacher_mode"] == "equal_capacity"


def test_each_phase2_dlc_job_has_an_explicit_wrapper() -> None:
    wrappers = {
        "train_p2_00_nosup_noaccess.sh": "p2_00_nosup_noaccess",
        "train_p2_10_sup_noaccess_da3.sh": "p2_10_sup_noaccess_da3",
        "train_p2_01_nosup_access.sh": "p2_01_nosup_access",
        "train_p2_11_sup_access_da3.sh": "p2_11_sup_access_da3",
        "train_p2_11_sup_access_vggt.sh": "p2_11_sup_access_vggt",
        "train_p2_random_access_da3.sh": "p2_random_access_da3",
        "train_p2_shuffled_access_da3.sh": "p2_shuffled_access_da3",
        "train_p2_state_mlp_access.sh": "p2_state_mlp_access",
    }
    for script_name, experiment in wrappers.items():
        environment = os.environ.copy()
        environment.update(
            FIELD2PLAN_RUN_SEED="42",
            FIELD2PLAN_MATRIX_PRINT_ONLY="1",
        )
        result = subprocess.run(
            ["bash", f"scripts/field2plan/{script_name}"],
            cwd=PROJECT_ROOT,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
        )
        selected = dict(
            line.split("=", 1) for line in result.stdout.splitlines()
        )
        assert selected["experiment"] == experiment


def test_16gpu_eval_launcher_preserves_two_shard_seed_protocol() -> None:
    script = _script("10_eval_all_ckpts_16gpu.sh")
    assert "infer_world_size=2" in script
    assert "--world_size \"$infer_world_size\"" in script
    assert "--seed \"$infer_seed\"" in script
    assert "p2_shuffled_access_da3" in script
    assert "p2_state_mlp_access" in script
    assert "record_pdms_result.py" in script
    assert "summary.csv" in script
    assert "summary.md" in script
    assert "unset WORLD_SIZE RANK LOCAL_RANK MASTER_ADDR MASTER_PORT" in script
    assert 'orchestration_revision="${EVAL_ORCHESTRATION_REVISION:-distenvfix-v1}"' in script
    assert '${protocol_id}-${orchestration_revision}' in script


def test_phase3_eval_launcher_uses_explicit_suite_and_separate_outputs() -> None:
    generic = _script("10_eval_all_ckpts_16gpu.sh")
    assert "FIELD2PLAN_EVAL_EXPERIMENTS" in generic
    assert "FIELD2PLAN_EXPERIMENT_SEED" in generic

    phase3 = _script("15_eval_phase3_ckpts_16gpu.sh")
    assert "p3_dyn_only_real" in phase3
    assert "p3_geo_dyn_real" in phase3
    assert "p3_geo_dyn_temporal_shuffle" in phase3
    assert "field2plan_phase3_eval_16gpu_live" in phase3
    assert "field2plan_phase3_all_ckpts" in phase3
    assert "10_eval_all_ckpts_16gpu.sh" in phase3


def test_vjepa_cache_launcher_pins_local_teacher_and_uses_16_independent_ranks() -> None:
    script = _script("11_cache_dynamics_vjepa.sh")
    assert "VJEPA_REPO_COMMIT" in script
    assert "VJEPA_CHECKPOINT_SHA256" in script
    assert "7ea9b7cb4a75d10644a8a8d42cff9e177b10dca8f02173f0eaf2b0bed82838c6" in script
    assert "VJEPA_CACHE_PROCESSES" in script
    assert '--nproc-per-node="$VJEPA_CACHE_PROCESSES"' in script
    assert "field2plan_cache/dynamics_vjepa2_1_vitl384_c96_16_v1" in script
    assert "cache_dynamics_vjepa.py" in script


def _phase3_selection(name: str) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        FIELD2PLAN_PHASE3_EXPERIMENT=name,
        FIELD2PLAN_RUN_SEED="47",
        FIELD2PLAN_MATRIX_PRINT_ONLY="1",
    )
    result = subprocess.run(
        ["bash", "scripts/field2plan/14_run_phase3_experiment.sh"],
        cwd=PROJECT_ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    return dict(line.split("=", 1) for line in result.stdout.splitlines())


def test_phase3_matrix_covers_dynamics_access_supervision_and_shuffle() -> None:
    full = _phase3_selection("p3_geo_dyn_real")
    assert full["geometry_supervision"] == "1"
    assert full["geometry_access"] == "1"
    assert full["dynamics_supervision"] == "1"
    assert full["dynamics_access"] == "1"
    assert full["dynamics_teacher_mode"] == "real"

    no_access = _phase3_selection("p3_dyn_sup_noaccess")
    assert no_access["dynamics_supervision"] == "1"
    assert no_access["dynamics_access"] == "0"

    shuffled = _phase3_selection("p3_geo_dyn_temporal_shuffle")
    assert shuffled["dynamics_teacher_mode"] == "temporal_shuffled"


def test_phase3_matrix_has_dynamics_only_and_equal_capacity_controls() -> None:
    no_access = _phase3_selection("p3_dyn_nosup_noaccess")
    assert no_access["dynamics_supervision"] == "0"
    assert no_access["dynamics_access"] == "0"
    assert no_access["dynamics_teacher_mode"] == "equal_capacity"

    dynamics_only = _phase3_selection("p3_dyn_only_real")
    assert dynamics_only["geometry_supervision"] == "0"
    assert dynamics_only["geometry_access"] == "0"
    assert dynamics_only["dynamics_supervision"] == "1"
    assert dynamics_only["dynamics_access"] == "1"

    no_teacher = _phase3_selection("p3_dyn_access_nosup")
    assert no_teacher["dynamics_supervision"] == "0"
    assert no_teacher["dynamics_access"] == "1"
    assert no_teacher["dynamics_teacher_mode"] == "equal_capacity"


def test_phase3_formal_and_debug_launchers_use_cache_only_and_preserve_batch() -> None:
    formal = _script("13_train_dynamics.sh")
    assert "DynamicsCacheReader" in formal
    assert "cfg_field2plan_phase3.yaml" in formal
    assert "target_effective_batch=32" in formal
    assert "teacher_runtime=offline_cache_only" in formal
    assert "FIELD2PLAN_DYNAMICS_MANIFEST_SHA256" in formal

    debug = _script("12_debug_dynamics.sh")
    assert "cfg_field2plan_phase3_debug.yaml" in debug
    assert "DynamicsCacheReader" in debug
    assert "target_effective_batch=32" in debug
    assert "--trainer.max_train_steps 1" in debug

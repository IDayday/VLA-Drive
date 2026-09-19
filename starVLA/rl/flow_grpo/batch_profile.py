"""Independent acceptance of one fixed saved-transition compute layout.

Historical serial/BF16 batch comparisons stay FAIL. This profile instead requires
FP32 objective-gradient equivalence AND bitwise equality to a FP32 leaf oracle
rounded once to the actual BF16 gradient format. No tolerance is enlarged.
Native target-world optimizer/resume/immutability checks remain in full_epoch.
"""
from pathlib import Path
from .loading import file_sha
from .research_budget import artifact

KERNEL_FILES = (
    "starVLA/rl/flow_grpo/rollout.py", "starVLA/rl/flow_grpo/model.py",
    "starVLA/rl/flow_grpo/math.py", "starVLA/rl/flow_grpo/temporal_noise.py",
    "starVLA/model/modules/action_model/GR00T_ActionHeader.py",
    "starVLA/model/modules/action_model/flow_matching_head/cross_attention_dit.py",
)


def kernel_identity():
    return {p: file_sha(p) for p in KERNEL_FILES}


def validate_batch_profile(pointer, cfg, context):
    if not pointer or str(Path(pointer["path"]).resolve()) != str(Path(cfg["runtime"]["batch_profile_evidence"]).resolve()):
        raise ValueError("missing/misbound transition batch evidence")
    proof = artifact(pointer)
    if (proof.get("schema_version") != 1 or proof.get("status") != "QUALIFIED_FIXED_LAYOUT"
            or proof.get("context") != context or proof.get("kernels") != kernel_identity()
            or cfg["runtime"].get("transition_evaluation") != "flat_saved_chain"
            or cfg["runtime"]["activation_checkpointing"] is not False):
        raise ValueError("transition batch code/config/profile identity mismatch")
    names = set(proof.get("trainable_names", []))
    if len(names) != 359 or any(not n.startswith("action_model.") for n in names):
        raise ValueError("transition batch parameter coverage mismatch")
    # Mathematical gradients are tested with original tolerances, independently
    # of the raw near-zero elementwise log-density diagnostic (retained separately).
    math_refs = proof.get("fp32_gradient_comparisons", [])
    if len(math_refs) != 2: raise ValueError("two fixed mathematical scenes required")
    for position, ref in enumerate(math_refs, 1):
        report = artifact(ref)
        rows = report.get("parameters", {})
        if (report.get("status") != "PASS" or set(rows) != names
                or any(r.get("allclose") is not True or r.get("finite") is not True
                       or r.get("atol") != 2e-6 or r.get("rtol") != 2e-3 for r in rows.values())):
            raise ValueError("FP32 objective gradient equivalence failed")
        if report.get("scene_position") != position or report.get("input_runs") != {
            "left": proof["fp32_serial_run"]["sha256"], "right": proof["fp32_batch_run"]["sha256"]}:
            raise ValueError("FP32 gradient comparison is detached from its CUDA runs")
    scene_sets = []
    for key in ("fp32_serial_run", "fp32_batch_run", "fp32_rounding_oracle", "bf16_batch_run"):
        run = artifact(proof[key])
        expected_dtype = "bf16" if key == "bf16_batch_run" else "fp32"
        if (run.get("status") != "MEASURED_NOT_QUALIFIED" or not run.get("device") or run.get("device_type") != "cuda"
                or run.get("checkpoint_sha256") != cfg["checkpoint_contract"]["sha256"]
                or run.get("kernels") != kernel_identity() or run.get("head_storage") != expected_dtype
                or run.get("observed_head_dtypes") != ["torch.bfloat16" if expected_dtype == "bf16" else "torch.float32"]
                or run.get("mode") != ("serial_on" if key == "fp32_serial_run" else "flat160_off")
                or len(run.get("scenes", [])) != 2 or run.get("trainable_tensors") != 359
                or run.get("trainable_numel") != 819503620):
            raise ValueError("missing matching real CUDA mathematical run: " + key)
        for scene in run["scenes"]:
            if (not scene.get("official_nonzero_advantage") or len(scene.get("repeats", [])) != 2
                    or any(r.get("gradient_tensors") != 359 or not r.get("gradient_finite") for r in scene["repeats"])):
                raise ValueError("incomplete fixed-scene real gradients")
        if key == "fp32_rounding_oracle" and run.get("preserve_bf16_time_input") is not True:
            raise ValueError("rounding oracle changed the timestep input")
        scene_sets.append([s.get("tokens") for s in run["scenes"]])
    if any(s != scene_sets[0] for s in scene_sets) or len({tuple(s) for s in scene_sets[0]}) != 2:
        raise ValueError("mathematical runs did not use the same two fixed scenes")
    rounded = artifact(proof["bf16_rounding_comparison"])
    if rounded.get("status") != "PASS" or set(rounded.get("scenes", {})) != {"1", "2"}:
        raise ValueError("BF16 rounding oracle failed")
    if rounded.get("input_runs") != {"oracle":proof["fp32_rounding_oracle"]["sha256"], "actual":proof["bf16_batch_run"]["sha256"]}:
        raise ValueError("rounding comparison is detached from CUDA oracle")
    for scene in rounded["scenes"].values():
        if (scene.get("status") != "PASS" or scene.get("forward_exact") is not True
                or set(scene.get("parameters", {})) != names
                or any(r.get("equal") is not True or r.get("nonidentical") != 0 for r in scene["parameters"].values())):
            raise ValueError("BF16 gradient is not the correctly rounded FP32 leaf result")
    adam = artifact(proof["native_adam"])
    if (adam.get("status") != "PASS" or adam.get("world_size") != context["world_size"] or set(adam.get("parameters", {})) != names
            or adam.get("execution_context") != context or adam.get("config_hash") != context["config_sha256"]):
        raise ValueError("native optimizer coverage/world mismatch")
    for row in adam["parameters"].values():
        if (row.get("forward_equals_actual_master_cast") is not True
                or any(row.get(k, {}).get("pass") is not True for k in ("master", "exp_avg", "exp_avg_sq"))):
            raise ValueError("native optimizer arithmetic failed")
    exported = artifact(proof["export"])
    if exported.get("status") != "TESTED" or exported.get("tensors_identical") != 989 or exported.get("output_max_abs") != 0:
        raise ValueError("original interface export failed")
    if Path(exported.get("checkpoint", "")).parents[1] != Path(adam["run"]):
        raise ValueError("export is detached from the qualified native pilot")
    # Explicitly preserve the failed historical comparison; never rewrite it as
    # an equality claim to obtain release of this independently tested layout.
    if artifact(proof["historical_serial_comparison"]).get("status") != "FAIL":
        raise ValueError("historical serial comparison must remain FAIL")
    return proof

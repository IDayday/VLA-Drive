"""Explicit bounded gradient-isolation probes; never an alternative training recipe."""

SCOPES = {"rl_only", "sft_only", "reference_after_joint"}


def validate_scope(runtime):
    scope = runtime.get("diagnostic_loss_scope")
    if scope is not None and (
        runtime.get("run_mode") != "diagnostic"
        or runtime["max_updates"] > 8
        or scope not in SCOPES
    ):
        raise ValueError("loss isolation is restricted to explicit bounded diagnostics")
    return scope


def selected_loss(result, runtime, completed_updates):
    scope = validate_scope(runtime)
    if scope == "rl_only":
        return result["grpo"]
    if scope == "sft_only":
        return result["sft"]
    if scope == "reference_after_joint" and completed_updates >= 1:
        # The first real joint update supplies a nonzero current/reference KL.
        # No fabricated advantage or parameter perturbation is used.
        return result["reference"]
    return result["loss"]

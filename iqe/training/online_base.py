"""Public online-image execution policy. No persistent visual payload dependency.

The model construction, sampler and optimizer-step loop remain in the
existing Base trainer. This policy only selects the already measured FP32
execution options and validates resumes within this separate IL-only experiment.
"""
from dataclasses import asdict, dataclass
from pathlib import Path
import copy
import torch

from ..contracts import require
from ..io import digest, file_hash
from .runtime_identity import backend_flags


@dataclass(frozen=True)
class OnlineBasePolicy:
    optimizer: str = "fused"
    gradient_as_bucket_view: bool = True
    bucket_cap_mb: int = 50
    checkpoint_policy: str = "none"
    visual_precision: str = "legacy_fp32"
    query_training_semantics: str = "legacy_repeat8"
    execution_weights: str = "fp32"

    def __post_init__(self):
        require(self.optimizer == "fused" and self.gradient_as_bucket_view is True and
                self.bucket_cap_mb == 50, "portable Base uses the measured fused/bucket-view/50MiB runtime")
        require(self.checkpoint_policy in {"none", "all"}, "unsupported language recomputation policy")
        require(self.visual_precision == "legacy_fp32" and self.execution_weights == "fp32" and
                self.query_training_semantics == "legacy_repeat8", "portable Base preserves precision and repeat8")

    def identity(self, contract, world):
        options = asdict(self)
        from .checkpoint import environment
        identity = {"schema": "query_base_online_v1", "options": options,
                    "backend_flags": backend_flags(), "environment": environment(), "world_size": world,
                    "framework_contract_hash": contract["framework_contract_hash"],
                    "implementation": {p.name: file_hash(p) for p in
                        (Path(__file__), Path(__file__).with_name("base_trainer.py"),
                         Path(__file__).with_name("execution.py"))},
                    "input_source": "online_images", "persistent_visual_payloads": False}
        if torch.cuda.is_available():
            identity["device"] = {"name": torch.cuda.get_device_name(),
                                  "capability": list(torch.cuda.get_device_capability())}
        identity["identity_hash"] = digest(identity)
        return identity

    def validate_state(self, state, identity):
        require(type(state.get("learned_driving_weights_loaded")) is bool,
                "checkpoint lacks initialization provenance; supply the verified original checkpoint")
        require(state["kind"] == "iqe_query_base" and state["resume_boundary"] == "optimizer_step",
                "only complete Query Base optimizer-boundary states may resume")
        previous = state.get("online_base_identity")
        if previous == identity:
            return {"kind": "SAME_ONLINE_RUNTIME", "exact_resume": True}
        require(False, "only this IL-only run can resume; start the separate experiment with --from-generic")

    def configure_model(self, model):
        from ..performance.variants import configure_model
        return configure_model(model, self)

    def make_optimizer(self, model, cfg, contract):
        from ..performance.variants import adam
        return adam(model, cfg, contract, self)

    def load_optimizer(self, optimizer, state):
        from ..performance.variants import load_adam
        load_adam(optimizer, state, self)


def checkpoint_origin(state, parent_path, parent_hash):
    """Carry actual initialization provenance rather than inventing a default."""
    if state is None:
        return {"learned_driving_weights_loaded": False, "initialization": "pinned_generic_qwen_random_driving_heads"}
    require(type(state.get("learned_driving_weights_loaded")) is bool, "missing parent initialization provenance")
    fields = {k: copy.deepcopy(state[k]) for k in
              ("learned_driving_weights_loaded", "initialization", "initialization_source", "training_lineage") if k in state}
    fields.update(parent_checkpoint=str(parent_path), parent_checkpoint_hash=parent_hash)
    return fields

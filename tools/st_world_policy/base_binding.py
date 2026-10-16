"""Bind an original, completed S-series DiT checkpoint once for this campaign.

Byte integrity and architecture identity are checked here. A separate real-input
load/output receipt is required for implementation qualification. This operation
never changes or resumes the parent experiment.
"""
from __future__ import annotations

import argparse
import json
import os
import time
import uuid
from pathlib import Path

from starVLA.model.modules.vehicle_joint.initialization import identity_hash, file_sha256
from tools.foresight.checkpoints import checkpoint_identity


def immutable_json(path: Path, value: dict) -> None:
    """Atomically publish without overwriting an earlier experiment identity."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp." + uuid.uuid4().hex)
    try:
        with temporary.open("x") as stream:
            json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if json.loads(path.read_text()) != value:
                raise ValueError(f"Immutable artifact conflict: {path}")
    finally:
        temporary.unlink(missing_ok=True)


def validate_parent(identity: dict, checkpoint: dict) -> None:
    cfg = identity["config"]
    action = cfg["framework"]["action_model"]
    if (cfg["framework"]["name"] != "DDPActionVideoForesight"
            or checkpoint["model_class"] !=
            "starVLA.model.framework.ddp_action_video_foresight.DDPActionVideoForesight"):
        raise ValueError("The common parent must be the actual S-series DiT S0")
    if identity["scope"] != "formal" or checkpoint["completed"] < 1:
        raise ValueError("Diagnostic/random checkpoints cannot initialize this campaign")
    if (action["action_dim"], action["action_horizon"],
            action["num_inference_timesteps"], action["repeated_diffusion_steps"]) != (4, 8, 10, 8):
        raise ValueError("Parent DiT/trajectory/inference definition differs")
    if (cfg["foresight"]["planner_condition_mode"] != "action_only"
            or cfg["foresight"]["num_queries"] != 144
            or cfg.get("enable_image_aug", 0)):
        raise ValueError("Parent input and planning contract differs from audited S0")
    if any(k not in checkpoint["files"] for k in
           ["mp_rank_00_model_states.pt", "rng_rank0.pt"]):
        raise ValueError("Missing original model/RNG checkpoint artifacts")


def bind_base(run: str | Path, tag: str, destination: str | Path,
              *, expected_checkpoint_sha256: str) -> dict:
    run = Path(run).resolve()
    destination = Path(destination).resolve()
    if destination.exists():
        existing = read_binding(destination)
        if (existing["run"] != str(run) or existing["tag"] != tag
                or existing["checkpoint"]["sha256"] != expected_checkpoint_sha256):
            raise ValueError("Changing the parent requires a new campaign namespace")
        return existing
    started = time.perf_counter()
    identity, checkpoint = checkpoint_identity(run, tag)
    validate_parent(identity, checkpoint)
    if checkpoint["sha256"] != expected_checkpoint_sha256:
        raise ValueError("Parent bytes differ from the explicitly selected checkpoint")
    folder = run / "checkpoints" / tag
    complete = json.loads((folder / "COMPLETE.json").read_text())
    source = Path(__file__).resolve().parents[2]
    contracts = ["starVLA/dataloader/foresight_dataset.py",
                 "starVLA/model/framework/DDPForesight.py",
                 "starVLA/model/framework/QwenOFT.py",
                 "starVLA/model/modules/action_model/GR00T_ActionHeader.py"]
    value = dict(schema="st_world_bound_s0_v1", run=str(run), tag=tag,
                 checkpoint=checkpoint, parent_identity=identity,
                 complete=complete, complete_sha256=file_sha256(folder / "COMPLETE.json"),
                 identity_file_sha256=file_sha256(run / "identity.json"),
                 source_contracts={k: file_sha256(source / k) for k in contracts},
                 initialized_from="learned_FP32_master_tensors_strict_load_required",
                 exploration=True, parent_has_seen_training_population=True,
                 inference_contract={"precision": "FP32_master_eval_TF32_off",
                                     "flow_steps": 10, "scene_seed_protocol": "foresight-action-v1",
                                     "raw": "8x4_xy_sin_cos", "physical": "8x3_xy_yaw",
                                     "dt_seconds": 0.5, "frame": "ego_t0_rear_axle"})
    value["binding_sha256"] = identity_hash(value)
    immutable_json(destination, value)
    # A diagnostic receipt is separate from the immutable model identity.
    immutable_json(destination.with_name("BASE_BYTE_AUDIT.json"), {
        "schema": "st_world_base_byte_audit_v1", "binding_sha256": value["binding_sha256"],
        "status": "BYTES_VERIFIED", "strict_model_load": "NOT_RUN",
        "bytes": sum((folder / name).stat().st_size for name in checkpoint["files"]),
        "elapsed_seconds": time.perf_counter() - started,
        "measured_unix": time.time(), "parent_changed": False})
    return value


def read_binding(path: str | Path, *, verify_files: bool = False) -> dict:
    value = json.loads(Path(path).read_text())
    if value.get("schema") != "st_world_bound_s0_v1":
        raise ValueError("Unsupported Base binding")
    expected = identity_hash({k: v for k, v in value.items() if k != "binding_sha256"})
    if expected != value.get("binding_sha256"):
        raise ValueError("Base binding content changed")
    run = Path(value["run"])
    folder = run / "checkpoints" / value["tag"]
    if (file_sha256(run / "identity.json") != value["identity_file_sha256"]
            or file_sha256(folder / "COMPLETE.json") != value["complete_sha256"]):
        raise ValueError("Parent run/complete identity changed")
    validate_parent(value["parent_identity"], value["checkpoint"])
    if verify_files:
        actual = {p.name: file_sha256(p) for p in sorted(folder.glob("*.pt"))}
        if actual != value["checkpoint"]["files"]:
            raise ValueError("Parent weight/optimizer/RNG bytes changed")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-run", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        binding = bind_base(args.base_run, args.tag, args.output,
                            expected_checkpoint_sha256=args.expected_checkpoint_sha256)
    except (OSError, ValueError, KeyError) as error:
        print(json.dumps({"status": "BLOCKED", "reason": str(error)}))
        return 2
    print(json.dumps({"status": "BOUND", "binding_sha256": binding["binding_sha256"],
                      "optimizer_step": binding["checkpoint"]["completed"],
                      "output": str(Path(args.output).resolve()), "strict_load": "SEPARATE_REQUIRED_AUDIT"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

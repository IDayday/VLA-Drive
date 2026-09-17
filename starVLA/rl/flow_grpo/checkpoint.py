"""Atomic boundary checkpoints through Accelerator's distributed state API."""
from pathlib import Path
import json
import os
import random
import numpy as np
import torch
from omegaconf import OmegaConf
from .config import config_hash
from .distributed import rank0_call, synchronized_call
from .reproducibility import assert_resume_identity


def capture_rng(policy=None):
    result = dict(
        python=random.getstate(),
        numpy=np.random.get_state(),
        torch=torch.get_rng_state(),
        cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    )
    if policy is not None and hasattr(policy, "rgb_model"):
        result["wan_numpy"] = policy.rgb_model.rng.bit_generator.state
        result["wan_torch"] = policy.rgb_model.torch_rng.get_state()
    return result


def restore_rng(state, policy=None):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
    if policy is not None and "wan_numpy" in state:
        policy.rgb_model.rng.bit_generator.state = state["wan_numpy"]
        policy.rgb_model.torch_rng.set_state(state["wan_torch"])


def save_boundary(
    accelerator,
    actor,
    cfg,
    sft,
    manifest,
    provenance,
    update,
    version,
    streams,
    pending=None,
):
    root = Path(cfg["runtime"]["output_dir"]) / "checkpoints"
    final = root / f"update_{update:06d}"
    temporary = root / f".update_{update:06d}.incomplete"

    def prepare_directory():
        if final.exists() or temporary.exists():
            raise FileExistsError(
                f"checkpoint must not overwrite {final} / {temporary}"
            )
        temporary.mkdir(parents=True)

    rank0_call(prepare_directory, accelerator.device)
    # save_state/load_state own collectives. Fatal failures escape to torchrun;
    # do not wrap them in a control-plane collective or a finally barrier.
    accelerator.save_state(str(temporary), safe_serialization=False)
    policy = accelerator.unwrap_model(actor).policy
    synchronized_call(
        lambda: torch.save(
            {
                "rng": capture_rng(policy),
                "streams": [s.state_dict() for s in streams],
                "pending": pending,
            },
            temporary / f"rank_{accelerator.process_index}.pt",
        ),
        accelerator.device,
    )

    def write_metadata():
        metadata = dict(
            schema=2,
            boundary="inner_epoch"
            if pending is not None
            else "complete_rollout_update",
            next_inner_epoch=pending["next_inner"] if pending else 0,
            update=update,
            policy_version=version,
            world_size=accelerator.num_processes,
            config_hash=config_hash(cfg),
            provenance=provenance,
            reduction=cfg["algorithm"]["logprob_reduction"],
        )
        (temporary / "trainer_state.json").write_text(json.dumps(metadata, indent=2))
        (temporary / "rl_config.json").write_text(json.dumps(cfg, indent=2))
        (temporary / "sft_parameter_manifest.json").write_text(
            json.dumps(manifest, indent=2)
        )
        OmegaConf.save(sft, temporary / "config.yaml")
        policy.qwen_vl_interface.processor.save_pretrained(temporary / "processor")
        import shutil

        for name in (
            "source_sft_parameter_manifest.json",
            "actor_parameter_manifest.json",
            "parameter_contract_exceptions.json",
        ):
            source = Path(cfg["runtime"]["output_dir"]) / name
            if source.is_file():
                shutil.copy2(source, temporary / name)
        (temporary / "normalization.json").write_text(
            json.dumps(
                {
                    "ver_1225": 1,
                    "act_norm": int(sft.datasets.vla_data.act_norm),
                    "x_mean": 10.172484,
                    "x_std": 8.805105,
                    "y_mean": 0.360762,
                    "y_std": 2.277741,
                    "horizon": 8,
                    "interval": 0.5,
                    "action_semantics": "absolute ego XY + sin/cos relative yaw",
                },
                indent=2,
            )
        )

    rank0_call(write_metadata, accelerator.device)

    def publish():
        (temporary / "COMPLETE").write_text(
            "complete optimizer boundary; pending behavior state included\n"
        )
        os.replace(temporary, final)

    rank0_call(publish, accelerator.device)
    return final


def resume_boundary(path, accelerator, actor, cfg, provenance, streams):
    path = Path(path)

    def validate_local():
        if not (path / "COMPLETE").is_file():
            raise ValueError("incomplete checkpoint")
        metadata = json.loads((path / "trainer_state.json").read_text())
        if metadata["world_size"] != accelerator.num_processes:
            raise ValueError("world-size change: exact resume rejected")
        if metadata["config_hash"] != config_hash(cfg):
            raise ValueError("resume configuration changed")
        assert_resume_identity(metadata["provenance"], provenance)
        state = torch.load(
            path / f"rank_{accelerator.process_index}.pt",
            map_location="cpu",
            weights_only=False,
        )
        if len(streams) != len(state["streams"]):
            raise ValueError("resume stream count changed")
        for stream, saved in zip(streams, state["streams"]):
            stream.load_state_dict(saved)
        if metadata.get("boundary") == "inner_epoch" and not state.get("pending"):
            raise ValueError("missing pending behavior batch")
        if state.get("pending"):
            pending = state["pending"]
            if not 0 < pending["next_inner"] < cfg["algorithm"].get("inner_epochs", 2):
                raise ValueError("invalid pending inner-epoch cursor")
            if pending["next_inner"] != metadata["next_inner_epoch"]:
                raise ValueError("pending inner-epoch metadata mismatch")
            for rollout in pending["buffers"]:
                if hasattr(rollout, "validate"):
                    rollout.validate(metadata["policy_version"])
        return metadata, state

    metadata, state = synchronized_call(validate_local, accelerator.device)
    accelerator.load_state(str(path))
    restore_rng(state["rng"], accelerator.unwrap_model(actor).policy)
    return metadata["update"], metadata["policy_version"], state.get("pending")


def export_checkpoint(checkpoint, output):
    import shutil

    checkpoint = Path(checkpoint)
    output = Path(output)
    if not (checkpoint / "COMPLETE").exists():
        raise ValueError("export requires completed checkpoint")
    if output.exists():
        raise FileExistsError(output)
    temporary = output.with_name(output.name + ".incomplete")
    temporary.mkdir(parents=True)
    if (checkpoint / "pytorch_model.bin").exists():
        state = torch.load(
            checkpoint / "pytorch_model.bin",
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
    else:
        paths = list(checkpoint.glob("*/mp_rank_00_model_states.pt"))
        if len(paths) != 1:
            raise ValueError(
                "expected a complete ZeRO 1/2 replicated module checkpoint"
            )
        with torch.serialization.safe_globals([set]):
            payload = torch.load(
                paths[0], map_location="cpu", weights_only=True, mmap=True
            )
        state = payload["module"]
    # Export actual forward weights, not higher-precision optimizer masters.
    manifest = json.loads((checkpoint / "sft_parameter_manifest.json").read_text())
    for entry in manifest["parameters"]:
        key = "policy." + entry["name"]
        if key not in state or list(state[key].shape) != entry["shape"]:
            raise ValueError(f"incomplete export: {key}")
    if not state or any(not key.startswith("policy.") for key in state):
        raise ValueError("unexpected actor checkpoint namespace")
    state = {key.removeprefix("policy."): value for key, value in state.items()}
    torch.save(state, temporary / "pytorch_model.pt")
    for name in (
        "config.yaml",
        "normalization.json",
        "sft_parameter_manifest.json",
        "trainer_state.json",
        "rl_config.json",
    ):
        shutil.copy2(checkpoint / name, temporary / name)
    shutil.copytree(checkpoint / "processor", temporary / "processor")
    for name in (
        "source_sft_parameter_manifest.json",
        "actor_parameter_manifest.json",
        "parameter_contract_exceptions.json",
    ):
        if (checkpoint / name).is_file():
            shutil.copy2(checkpoint / name, temporary / name)
    (temporary / "COMPLETE").write_text("original QwenOFT inference format\n")
    os.replace(temporary, output)
    return output

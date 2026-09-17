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
from .loading import file_sha
from .transactions import atomic_json, publication_lock, preserve_attempt


def directory_seal(root, excluded=()):
    root = Path(root)
    return {
        str(p.relative_to(root)): file_sha(p)
        for p in sorted(root.rglob("*"))
        if p.is_file() and str(p.relative_to(root)) not in excluded
    }


def validate_checkpoint(path, cfg, provenance, world_size):
    """Read-only controller validation; no model construction or optimizer update."""
    path = Path(path)
    if not (path / "COMPLETE").is_file():
        raise ValueError("incomplete checkpoint")
    state = json.loads((path / "trainer_state.json").read_text())
    if state.get("schema") != 2 or path.name != f"update_{state['update']:06d}":
        raise ValueError("checkpoint name/update/schema mismatch")
    if state["world_size"] != world_size or state["config_hash"] != config_hash(cfg):
        raise ValueError("checkpoint configuration/world size mismatch")
    if config_hash(json.loads((path / "rl_config.json").read_text())) != config_hash(
        cfg
    ):
        raise ValueError("checkpoint saved configuration mismatch")
    assert_resume_identity(state["provenance"], provenance)
    if state["provenance"].get("sft_sha256") != cfg["checkpoint_contract"]["sha256"]:
        raise ValueError("checkpoint initialization mismatch")
    seal_path = path / "checkpoint_files.json"
    if seal_path.is_file():
        seal = json.loads(seal_path.read_text())
        if seal != directory_seal(path, ("COMPLETE", "checkpoint_files.json")):
            raise ValueError("checkpoint file integrity changed")
    for name in ["config.yaml", "normalization.json", "sft_parameter_manifest.json"]:
        if not (path / name).is_file():
            raise ValueError(f"checkpoint missing {name}")
    if not (path / "processor").is_dir():
        raise ValueError("checkpoint processor missing")
    for rank in range(world_size):
        rank_path = path / f"rank_{rank}.pt"
        data = torch.load(rank_path, map_location="cpu", weights_only=False)
        if not {"rng", "streams", "pending"} <= data.keys():
            raise ValueError("checkpoint rank state incomplete")
        if state["boundary"] == "inner_epoch" and (
            not data["pending"]
            or data["pending"]["next_inner"] != state["next_inner_epoch"]
        ):
            raise ValueError("checkpoint pending behavior mismatch")
    if cfg["runtime"]["deepspeed_stage"]:
        models = list(path.glob("*/mp_rank_00_model_states.pt"))
        optimizers = list(path.glob("*/*optim_states.pt"))
        if len(models) != 1 or len(optimizers) != world_size:
            raise ValueError("checkpoint missing DeepSpeed model/optimizer partitions")
        for rank in range(world_size):
            if not any(f"zero_pp_rank_{rank}_" in p.name for p in optimizers):
                raise ValueError("checkpoint optimizer rank partition missing")
    else:
        models, optimizers = [path / "pytorch_model.bin"], [path / "optimizer.bin"]
        if not (path / "scheduler.bin").is_file():
            raise ValueError("checkpoint scheduler missing")
    for file in models + optimizers:
        if not file.is_file() or file.stat().st_size == 0:
            raise ValueError(f"checkpoint model/optimizer file missing: {file}")
        # Trusted own checkpoints only. Read zip/pickle structures without materializing
        # multi-GB tensor data; the content manifest already checks every byte.
        payload = torch.load(file, map_location="cpu", mmap=True, weights_only=False)
        if not isinstance(payload, dict) or not payload:
            raise ValueError("checkpoint model/optimizer state invalid")
    return state


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
        atomic_json(temporary / "checkpoint_files.json", directory_seal(temporary))
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
    output = Path(output)
    with publication_lock(output.with_name(output.name + ".lock")):
        existing = validate_export(checkpoint, output)
        if existing:
            return output
        if output.exists():
            preserve_attempt(output)
        temporary = output.with_name(output.name + ".incomplete")
        if temporary.exists():
            preserve_attempt(temporary)
        return _export_checkpoint(checkpoint, output)


def export_identity(checkpoint):
    checkpoint = Path(checkpoint)
    if not (checkpoint / "COMPLETE").is_file():
        raise ValueError("export requires completed checkpoint")
    model = checkpoint / "pytorch_model.bin"
    if not model.exists():
        paths = list(checkpoint.glob("*/mp_rank_00_model_states.pt"))
        if len(paths) != 1:
            raise ValueError("export source model is incomplete")
        model = paths[0]
    return {
        "source": str(checkpoint.resolve()),
        "trainer_state_sha256": file_sha(checkpoint / "trainer_state.json"),
        "forward_weights_sha256": file_sha(model),
    }


def validate_export(checkpoint, output):
    output = Path(output)
    if not output.exists():
        return False
    receipt = output / "export_state.json"
    if receipt.is_file():
        saved = json.loads(receipt.read_text())
        if saved["identity"] != export_identity(checkpoint):
            raise ValueError("export identity conflict")
    if not (output / "COMPLETE").is_file():
        return False
    if not receipt.is_file():
        raise ValueError("completed export has no verifiable provenance")
    required = {
        "pytorch_model.pt",
        "config.yaml",
        "normalization.json",
        "sft_parameter_manifest.json",
        "trainer_state.json",
        "rl_config.json",
    }
    if not required <= set(saved.get("files", {})) or not any(
        name.startswith("processor/") for name in saved.get("files", {})
    ):
        raise ValueError("completed export required files missing from receipt")
    if saved["files"] != directory_seal(output, ("COMPLETE", "export_state.json")):
        raise ValueError("export file integrity changed")
    return True


def _export_checkpoint(checkpoint, output):
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
    atomic_json(
        temporary / "export_state.json",
        {
            "schema_version": 1,
            "identity": export_identity(checkpoint),
            "files": directory_seal(temporary),
        },
    )
    (temporary / "COMPLETE").write_text("original QwenOFT inference format\n")
    os.replace(temporary, output)
    return output

"""Auditable generic-weight loading and RNG-independent driving initialization."""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import random
import numpy as np
import torch

GENERIC_REPOSITORIES = frozenset({
    "Qwen/Qwen3-VL-2B-Instruct", "alibaba-pai/Wan2.1-Fun-V1.1-1.3B-InP",
    "gangweix/Pixel-Perfect-Depth", "depth-anything/Depth-Anything-V2-Large",
    "depth-anything/DA3METRIC-LARGE",
})


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def verify_generic_source(root, record):
    """Validate pinned public artifact hashes BEFORE passing a path to a loader."""
    if record.get("repository") not in GENERIC_REPOSITORIES:
        raise ValueError("Initialization source is not an allowed generic repository")
    revision = record.get("revision", "")
    if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        raise ValueError("Generic source requires an immutable public revision")
    artifacts = record.get("files", {})
    if not artifacts or not any(p.endswith((".safetensors", ".pth", ".bin")) for p in artifacts):
        raise ValueError("Source manifest must include actual weight files")
    root = Path(root).resolve()
    verified = {}
    for relative, expected in artifacts.items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Artifact escapes generic model root")
        actual = file_sha256(path)
        if actual != expected:
            raise ValueError(f"Generic artifact hash mismatch: {relative}")
        verified[relative] = actual
    for pattern in ("*.safetensors", "pytorch_model*.bin", "*.pth"):
        for path in root.glob(pattern):
            if str(path.relative_to(root)) not in artifacts:
                raise ValueError(f"Undeclared weight artifact: {path.name}")
    return {"repository": record["repository"], "revision": revision,
            "files": verified, "generic_pretraining_data_fully_auditable": False}


def validate_pinned_sources(sources):
    """External asset paths may change; the reviewed public revision/hashes may not."""
    reference = Path(__file__).parents[4] / "reports/ddpolicy_vehicle_from_scratch/GENERIC_SOURCES.json"
    pinned = json.loads(reference.read_text())
    if set(sources) != set(pinned):
        raise ValueError("Generic source module inventory differs from the pinned campaign")
    for name, expected in pinned.items():
        actual = {k:v for k,v in sources[name].items() if k != "root"}
        if actual != expected:
            raise ValueError(f"Generic source differs from public pinned manifest: {name}")


@contextmanager
def initialization_seed(seed):
    """New modules must not consume the RNG stream used for shared modules."""
    python_state, numpy_state = random.getstate(), np.random.get_state()
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_initialized() else []
    with torch.random.fork_rng(devices=devices):
        random.seed(seed)
        np.random.seed(seed % (2**32))
        torch.manual_seed(seed)
        try:
            yield
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)


def tensor_hash(tensor):
    tensor = tensor.detach().cpu().contiguous()
    h = hashlib.sha256(f"{tensor.dtype}:{tuple(tensor.shape)}".encode())
    h.update(tensor.view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def module_manifest(module):
    return {name: {"shape": list(p.shape), "dtype": str(p.dtype),
                   "sha256": tensor_hash(p), "trainable": p.requires_grad}
            for name, p in module.named_parameters()}


def add_random_driving_tokens(model, tokenizer, tokens, seed):
    """Randomize exact token IDs, including rows in Qwen's spare vocabulary.

    Never load WorldAction embeddings. Native image/video tokens remain intact.
    """
    if len(tokens) != len(set(tokens)) or any(t in tokenizer.get_vocab() for t in tokens):
        raise ValueError("Driving tokens must be unique and absent from generic tokenizer")
    tokenizer.add_special_tokens({"additional_special_tokens": list(tokens)},
                                 replace_additional_special_tokens=False)
    ids = tokenizer.convert_tokens_to_ids(list(tokens))
    if len(set(ids)) != len(ids):
        raise ValueError("Non-unique driving token IDs")
    with initialization_seed(seed):
        if max(ids) >= model.get_input_embeddings().weight.shape[0]:
            model.resize_token_embeddings(max(ids) + 1, mean_resizing=False)
        weight = model.get_input_embeddings().weight
        std = getattr(model.config.text_config, "initializer_range", 0.02)
        with torch.no_grad():
            fresh = torch.randn(len(ids), weight.shape[1], device=weight.device,
                                dtype=weight.dtype) * std
            weight[ids] = fresh
            output = model.get_output_embeddings()
            if output is not None and output.weight.data_ptr() != weight.data_ptr():
                output.weight[ids] = fresh
    return {"tokens": dict(zip(tokens, ids)), "initialization_seed": seed,
            "embedding_sha256": tensor_hash(weight[ids])}


def identity_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def driving_initialization_manifest(model):
    """Hash every new driving module before DeepSpeed casting or updates."""
    modules = ["action_model", "action_input_model", "vehicle_reader", "vehicle_heads",
               "traj_emb", "rgb_act_pre", "gs_traj_emb", "gs_act_pre"]
    result = {name:module_manifest(getattr(model, name)) for name in modules if hasattr(model, name)}
    for name in ("rgb_query", "gs_query", "traj_emb_h0", "gs_traj_emb_h0"):
        if hasattr(model, name): result[name] = tensor_hash(getattr(model, name))
    if hasattr(model, "rgb_model"):
        result["rgb_model.qwen_proj_video"] = module_manifest(model.rgb_model.qwen_proj_video)
    if hasattr(model, "gs_model"):
        for name in ("qwen_proj", "qwen_cross_attn"):
            result["gs_model.dit."+name] = module_manifest(getattr(model.gs_model.dit, name))
    result["qwen_driving_token_embeddings"] = model.qwen_vl_interface.driving_token_initialization
    if hasattr(model, "vehicle_token_initialization"):
        result["qwen_vehicle_token_embeddings"] = model.vehicle_token_initialization
    return result

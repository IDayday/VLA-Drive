"""Exact-resume identities for assets and numerical execution, not path labels."""

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import importlib.metadata
import json
import os
import pickle
import torch
from .contracts import digest
from .loading import file_sha

PACKAGES = (
    "torch",
    "transformers",
    "accelerate",
    "deepspeed",
    "diffusers",
    "flash-attn",
    "numpy",
    "scipy",
    "tokenizers",
    "pillow",
    "torchvision",
    "triton",
    "omegaconf",
    "hydra-core",
    "shapely",
)


def dependency_versions():
    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def backend_source_identity():
    files = {}
    for package, names in {
        "torch": ["torch/amp/autocast_mode.py", "torch/nn/utils/clip_grad.py"],
        "deepspeed": [
            "deepspeed/runtime/engine.py",
            "deepspeed/runtime/config.py",
            "deepspeed/runtime/zero/stage_1_and_2.py",
        ],
        "accelerate": ["accelerate/accelerator.py", "accelerate/utils/deepspeed.py"],
    }.items():
        distribution = importlib.metadata.distribution(package)
        for name in names:
            files[name] = file_sha(distribution.locate_file(name))
    return files


def processor_identity(root):
    root = Path(root)
    # Include tokenizer, vocabulary, merges, special tokens, Jinja chat template,
    # image/video processor and all model configs. Never hash base model weights here.
    files = {
        str(p.relative_to(root)): file_sha(p)
        for p in sorted(root.rglob("*"))
        if p.is_file() and p.suffix in {".json", ".jinja", ".txt", ".model", ".yaml"}
    }
    if not files or not any("config" in p for p in files):
        raise ValueError(f"missing processor/model configuration: {root}")
    return {"files": files, "sha256": digest(files)}


def numerical_profile(cfg, sft):
    runtime = cfg["runtime"]
    return {
        "profile": runtime.get("numerical_profile", "bf16_zero2_fp32_accum_v1"),
        "model_dtype": "bfloat16",
        "probability_dtype": "float32",
        "qwen_autocast": "bfloat16",
        "action_history_projector_autocast": "cuda_float32_torch2.5",
        "gradient_accumulation_dtype": "float32",
        "partition_dtype_correction": (
            "deepspeed_0.16.9_partition_list_v1"
            if runtime.get("numerical_profile") == "bf16_zero2_fp32_partition_v2"
            else None
        ),
        "communication_dtype": "float32",
        "attention_backend": sft.framework.qwenvl.attn_implementation,
        "deterministic": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_tf32": torch.backends.cudnn.allow_tf32,
        "bf16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "flash_deterministic": os.getenv("FLASH_ATTENTION_DETERMINISTIC"),
        "cublas_workspace": os.getenv("CUBLAS_WORKSPACE_CONFIG"),
        "activation_checkpointing": runtime["activation_checkpointing"],
        "zero_stage": runtime["deepspeed_stage"],
        "optimizer_offload": runtime["optimizer_offload"],
        "candidate_chunk": cfg["sampling"]["candidate_chunk_size"],
        "transition_chunk": cfg["sampling"]["transition_chunk_size"],
    }


def configure_numerics():
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False


def write_asset_manifest(paths, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)

    def entry(p):
        return {
            "path": str(Path(p).resolve()),
            "size": Path(p).stat().st_size,
            "sha256": file_sha(p),
        }

    entries = list(asset_map(entry, sorted(set(map(str, paths)))))
    payload = {"schema": 1, "files": entries, "identity": digest(entries)}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2))
    return payload


def verify_asset_manifest(path, expected=None):
    payload = json.loads(Path(path).read_text())
    if digest(payload["files"]) != payload["identity"] or (
        expected and payload["identity"] != expected
    ):
        raise ValueError("asset manifest identity changed")

    # One content check at startup/resume, never at each optimizer step. Stat-only
    # fingerprints cannot detect same-path same-size replacement.
    def check_entry(entry):
        p = Path(entry["path"])
        if (
            not p.is_file()
            or p.stat().st_size != entry["size"]
            or file_sha(p) != entry["sha256"]
        ):
            raise ValueError(f"immutable asset changed: {p}")

    for _ in asset_map(check_entry, payload["files"]):
        pass
    return payload["identity"]


def asset_map(function, items):
    """Bounded I/O overlap, deterministic manifest order, no hash shortcuts.

    Eight threads on rank0, batches of256 avoid hundreds of thousands of queued
    futures for full navtrain. Every file is still read and checked on resume.
    """
    with ThreadPoolExecutor(max_workers=8) as pool:
        for start in range(0, len(items), 256):
            yield from pool.map(function, items[start : start + 256])


def selected_asset_identity(paths, manifest=None):
    """Hash only the actual selected inputs; reuse and verify locked entries.

    A path/mtime-only memo is deliberately not an identity. Same-path replacement
    is checked on every request, including requests that reuse completed outputs.
    """
    locked = {}
    if manifest and Path(manifest).is_file():
        payload = json.loads(Path(manifest).read_text())
        if digest(payload["files"]) != payload["identity"]:
            raise ValueError("asset manifest identity changed")
        locked = {entry["path"]: entry for entry in payload["files"]}

    def entry(path):
        item = {
            "path": str(Path(path).resolve()),
            "size": Path(path).stat().st_size,
            "sha256": file_sha(path),
        }
        if item["path"] in locked and item != locked[item["path"]]:
            raise ValueError(f"immutable asset changed: {path}")
        return item

    entries = list(asset_map(entry, sorted({str(Path(p).resolve()) for p in paths})))
    return {"identity": digest(entries), "files": entries}


def observation_asset_identity(data_root, tokens, split, manifest=None):
    metadata = [
        Path(data_root)
        / "meta"
        / ("test" if split == "navtest" else "train")
        / f"{t}.pkl"
        for t in tokens
    ]
    paths = list(metadata)
    # Exactly the current three camera inputs used by NavSimDataset._get_sample.
    # The full metadata also binds history/commands and preprocessing supervision.
    for path in metadata:
        with path.open("rb") as stream:
            item = pickle.load(stream)  # trusted local project metadata only
        paths.extend(
            item["glo_images"][v]["image_paths"][3]
            for v in ("cam_f0", "cam_l0", "cam_r0")
        )
    return selected_asset_identity(paths, manifest)


def resume_assets(cfg, sft):
    paths = cfg["paths"]
    manifest = paths.get("asset_manifest")
    if not manifest:
        raise ValueError(
            "training requires a content-locked data/metric asset_manifest"
        )
    return {
        "processor": processor_identity(paths["base_vlm"]),
        "numerics": numerical_profile(cfg, sft),
        "dependencies": dependency_versions(),
        "backend_source_sha256": backend_source_identity(),
        "data_metric_manifest": verify_asset_manifest(
            manifest, paths.get("asset_manifest_identity")
        ),
        "train_tokens_file": file_sha(paths["train_list"]),
        "test_tokens_file": file_sha(paths["test_list"]),
    }


def assert_resume_identity(saved, current):
    if saved != current:
        differences = [
            key
            for key in saved.keys() | current.keys()
            if saved.get(key) != current.get(key)
        ]
        raise ValueError("exact resume identity changed: " + ", ".join(differences))


def training_provenance(cfg, sft, assets):
    from .audit import source_fingerprints
    from .config import config_hash
    from .reward import VERSION, reward_metadata

    sha = cfg["checkpoint_contract"]["sha256"]
    return dict(
        assets=assets,
        implementation_sha256=digest(source_fingerprints()),
        sft_sha256=sha,
        reference_sha256=sha,
        normalization=digest(
            {
                "ver_1225": int(sft.ver_1225),
                "act_norm": int(sft.datasets.vla_data.act_norm),
            }
        ),
        reward_version=VERSION,
        config_hash=config_hash(cfg),
        reference_lock=digest(json.loads(Path("reference_lock.json").read_text())),
        reward_config_hash=digest(reward_metadata(Path("navsim").resolve())),
    )

"""Capture reproducible source/environment evidence without importing the SFT trainer."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from datetime import datetime, timezone
import torch


def source_fingerprints():
    paths = list(Path("starVLA/rl").rglob("*.py")) + [
        Path(name)
        for name in (
            "starVLA/model/framework/QwenOFT.py",
            "starVLA/model/framework/baseline_qwen.py",
            "starVLA/model/modules/action_model/GR00T_ActionHeader.py",
            "starVLA/model/modules/vlm/qwen3_vl/modeling_qwen3_vl.py",
            "starVLA/model/modules/video_model/videox_fun/models/attention_utils.py",
            "infer.py",
            "reference_lock.json",
        )
    ]
    return {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(paths)
    }


def capture_source_environment(output, cfg):
    versions = {}
    for package in (
        "torch",
        "transformers",
        "accelerate",
        "deepspeed",
        "diffusers",
        "flash-attn",
        "numpy",
    ):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    result = dict(
        utc=datetime.now(timezone.utc).isoformat(),
        git_sha=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        branch=subprocess.check_output(
            ["git", "branch", "--show-current"], text=True
        ).strip(),
        source_sha256=source_fingerprints(),
        argv=sys.argv,
        launch_environment={
            name: os.environ.get(name)
            for name in (
                "CUDA_VISIBLE_DEVICES",
                "RANK",
                "WORLD_SIZE",
                "LOCAL_RANK",
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MASTER_ADDR",
                "MASTER_PORT",
            )
        },
        dirty_diff_sha256=hashlib.sha256(
            subprocess.check_output(["git", "diff", "--binary"])
        ).hexdigest(),
        python=platform.python_version(),
        packages=versions,
        cuda=torch.version.cuda,
        devices=[
            dict(
                name=torch.cuda.get_device_name(i),
                capability=torch.cuda.get_device_capability(i),
            )
            for i in range(torch.cuda.device_count())
        ],
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        flash_attention_deterministic=os.getenv("FLASH_ATTENTION_DETERMINISTIC", "0"),
        cublas_workspace_config=os.getenv("CUBLAS_WORKSPACE_CONFIG"),
        checkpoint_contract=cfg["checkpoint_contract"],
    )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    path = output / (
        "source_environment_"
        + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        + ".json"
    )
    path.write_text(json.dumps(result, indent=2))
    return result

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import subprocess

from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str):
    path = REPO_ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_base_no_memory_topology() -> None:
    config = OmegaConf.load(REPO_ROOT / "configs" / "base_model_navtest.yaml")
    assert config._target_.endswith("EpisodeDriveAgent")
    assert config.vlm_config.vlm_type == "internvl"
    assert config.vlm_config.initialize_from_config is True
    assert config.vlm_config.use_flash_attn is False
    assert config.action_head_config.proposal_num == 64
    assert config.action_head_config.scorer_ref_num == 4
    assert config.action_head_config.num_poses == 8


def test_setup_never_resolves_dependencies() -> None:
    setup = (REPO_ROOT / "scripts" / "setup_local.sh").read_text()
    assert "--system-site-packages" in setup
    assert "--no-deps" in setup
    assert "requirements.txt" not in setup
    assert "requirements-episode-drive.txt" not in setup

    launcher = (REPO_ROOT / "scripts" / "run_base_pdms.sh").read_text()
    assert "agent.action_head_config.proposal_num=64" in launcher
    assert "agent.action_head_config.scorer_ref_num=4" in launcher
    assert "agent.vlm_config.use_flash_attn=false" in launcher


def test_machine_local_files_are_ignored() -> None:
    patterns = (REPO_ROOT / ".gitignore").read_text().splitlines()
    assert "env.local.sh" in patterns
    assert ".venv/" in patterns
    assert ".cache/" in patterns
    assert "outputs/" in patterns


def _source_environment(env_file: Path, inherited: dict[str, str] | None = None) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(inherited or {})
    environment["DRIVEVLA_ENV_FILE"] = str(env_file)
    command = (
        f'source "{REPO_ROOT / "load_env.sh"}"; '
        "printf '%s\\n' \"$DRIVEVLA_REPO_ROOT\" \"$BATCH_SIZE\""
    )
    result = subprocess.run(
        ["bash", "-c", command],
        cwd="/tmp",
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    repo_root, batch_size = result.stdout.splitlines()
    return {"repo_root": repo_root, "batch_size": batch_size}


def test_environment_precedence_and_foreign_working_directory(tmp_path: Path) -> None:
    env_file = tmp_path / "machine.sh"
    env_file.write_text("export BATCH_SIZE=99\n")

    local_value = _source_environment(env_file)
    explicit_value = _source_environment(
        env_file,
        {"BATCH_SIZE": "7", "DRIVEVLA_REPO_ROOT": "/stale/worktree"},
    )

    assert local_value["batch_size"] == "99"
    assert explicit_value["batch_size"] == "7"
    assert explicit_value["repo_root"] == str(REPO_ROOT)


def test_runtime_contract_keeps_vendor_ppu_builds() -> None:
    module = _load_script("verify_runtime_versions.py")
    assert module.EXPECTED_RUNTIME["torch"] == "2.4.0"
    assert module.EXPECTED_RUNTIME["torchvision"] == "0.19.0"
    assert module.EXPECTED_RUNTIME["torchaudio"] == "2.4.0"
    assert "+ppu" in module.EXPECTED_RUNTIME["triton"]
    assert ".ppu" in module.EXPECTED_RUNTIME["flash-attn"]


def test_preflight_pins_released_assets() -> None:
    module = _load_script("preflight_base.py")
    assert module.EXPECTED_CHECKPOINT_BYTES == 4_271_779_662
    assert len(module.EXPECTED_CHECKPOINT_SHA256) == 64
    assert len(module.EXPECTED_DINO_SHA256) == 64
    assert module.EXPECTED_TOKENIZER_SIZE == 151_682
    assert module.EXPECTED_CACHE_ROWS == 12_146

    source = (REPO_ROOT / "scripts" / "preflight_base.py").read_text()
    assert "retrieval_or_memory_key_count" in source
    assert "memory_attention" in source

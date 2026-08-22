"""Small relative-inheritance loader for matched experiment YAML files."""

from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig, OmegaConf


def load_training_config(path: str | Path, _seen=None) -> DictConfig:
    """Load a YAML and recursively merge its optional relative ``base_config``."""

    config_path = Path(path).expanduser().resolve()
    seen = set() if _seen is None else _seen
    if config_path in seen:
        raise ValueError(f"cyclic base_config reference: {config_path}")
    seen.add(config_path)
    config = OmegaConf.load(config_path)
    base = config.get("base_config")
    if base is None:
        seen.remove(config_path)
        return config
    base_path = Path(str(base)).expanduser()
    if not base_path.is_absolute():
        base_path = config_path.parent / base_path
    inherited = load_training_config(base_path, seen)
    overlay = OmegaConf.create(OmegaConf.to_container(config, resolve=False))
    del overlay["base_config"]
    seen.remove(config_path)
    return OmegaConf.merge(inherited, overlay)

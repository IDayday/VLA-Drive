from dataclasses import asdict, dataclass
from pathlib import Path
import yaml

from iqe.contracts import require
from iqe.io import atomic_json, digest, read_json
from iqe.specialist_study.config import StudyConfig


CELLS = {
    "R00": {"objective": "mean_bce", "dropout": 0.0},
    "R01": {"objective": "mean_bce", "dropout": 0.1},
    "R10": {"objective": "zero_hurdle", "dropout": 0.0},
    "R11": {"objective": "zero_hurdle", "dropout": 0.1},
}


@dataclass(frozen=True)
class Config:
    experiment_id: str
    output_root: str
    source_campaign: str
    seed: int = 20261014
    training_seed: int = 20261034
    sampler_seed: int = 20261024
    updates: int = 6000
    phase_switch: int = 3000
    global_batch: int = 128
    micro_batch: int = 16
    lr: float = 1e-4
    weight_decay: float = 0.01
    warmup: int = 100
    checkpoint_interval: int = 1000
    validation_steps: tuple = (1000, 2000, 3000, 4000, 5000, 6000)
    feature_memory_gib: int = 16

    def __post_init__(self):
        require(self.experiment_id and Path(self.output_root).is_absolute(), "explicit experiment namespace")
        require(Path(self.source_campaign).is_absolute(), "explicit immutable source campaign")
        require(Path(self.output_root).resolve() != Path(self.source_campaign).resolve(), "source campaign is read-only")
        require(self.updates == 6000 and self.phase_switch == 3000 and self.global_batch == 128,
                "registered four-cell experiment budget")
        require(self.micro_batch > 0 and self.global_batch % self.micro_batch == 0, "microbatch/global batch")
        require(tuple(self.validation_steps) == (1000, 2000, 3000, 4000, 5000, 6000), "fixed selection checkpoints")
        require(self.lr > 0 and self.weight_decay >= 0 and 0 < self.warmup < self.updates, "optimizer configuration")
        require(self.checkpoint_interval > 0 and 1 <= self.feature_memory_gib <= 64, "bounded checkpoint/cache budget")

    @property
    def root(self):
        return Path(self.output_root)

    @property
    def identity(self):
        return digest(asdict(self))

    def source(self):
        source = StudyConfig(**read_json(Path(self.source_campaign)/"resolved_config.json"))
        require(source.root.resolve() == Path(self.source_campaign).resolve(), "source namespace mismatch")
        return source


def load(path):
    content = yaml.safe_load(Path(path).read_text())
    if "validation_steps" in content:
        content["validation_steps"] = tuple(content["validation_steps"])
    config = Config(**content)
    atomic_json(config.root/"resolved_config.json", asdict(config), immutable=True)
    return config

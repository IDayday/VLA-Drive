from dataclasses import dataclass, asdict
from pathlib import Path
import yaml
from iqe.contracts import require
from iqe.io import atomic_json, digest, read_json
from iqe.specialist_study.config import StudyConfig


@dataclass(frozen=True)
class Config:
    experiment_id: str
    output_root: str
    source_campaign: str
    updates: int = 6000
    diagnostic_updates: int = 300
    diagnostic_scenes: int = 64
    micro_batch: int = 16
    feature_memory_gib: int = 16

    def __post_init__(self):
        require(self.experiment_id and Path(self.output_root).is_absolute()
                and Path(self.source_campaign).is_absolute(), 'explicit experiment and actual roots')
        require(Path(self.output_root).resolve() != Path(self.source_campaign).resolve(), 'read-only control campaign')
        require(self.updates == 6000 and self.diagnostic_updates == 300 and self.diagnostic_scenes == 64,
                'one registered6000-step cell and two300-step diagnostics only')
        require(self.micro_batch == 16 and 1 <= self.feature_memory_gib <= 64, 'matched/bounded execution')

    @property
    def root(self): return Path(self.output_root)

    @property
    def identity(self): return digest(asdict(self))

    def source(self):
        result = StudyConfig(**read_json(Path(self.source_campaign)/'resolved_config.json'))
        require(result.root.resolve() == Path(self.source_campaign).resolve(), 'source experiment mismatch')
        return result


def load(path):
    config = Config(**yaml.safe_load(Path(path).read_text()))
    atomic_json(config.root/'resolved_config.json', asdict(config), immutable=True)
    return config

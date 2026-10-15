from dataclasses import asdict, dataclass
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
    reference_run: str
    geometric_run: str
    expected_base_hash: str
    expected_pool_hash: str
    seed: int = 20261015
    training_seed: int = 20261035
    sampler_seed: int = 20261025
    partition_seed: int = 314159
    bootstrap_seed: int = 11235
    bootstrap_repetitions: int = 2000
    micro_batch: int = 16
    feature_memory_gib: int = 32
    global_batch: int = 128
    diagnostic_updates: int = 3000
    refit_limit: int = 5000
    validation_steps: tuple = (500, 1000, 1500, 2000, 2500, 3000)
    lr: float = 1e-4
    weight_decay: float = .01
    warmup: int = 100
    grad_clip: float = 1.
    checkpoint_interval: int = 500

    def __post_init__(self):
        paths = [self.output_root, self.source_campaign, self.reference_run, self.geometric_run]
        require(self.experiment_id and all(Path(p).is_absolute() for p in paths), 'absolute explicit study paths')
        require(Path(self.output_root).resolve() not in [Path(p).resolve() for p in paths[1:]], 'isolated new study output')
        require(self.global_batch == 128 and self.diagnostic_updates == 3000 and self.refit_limit == 5000,
                'registered two-model bounded budget')
        require(tuple(self.validation_steps) == (500, 1000, 1500, 2000, 2500, 3000), 'fixed internal validation points')
        require(0 < self.micro_batch <= 128 and self.global_batch % self.micro_batch == 0, 'microbatch/global batch')
        require(self.lr == 1e-4 and self.weight_decay == .01 and self.warmup == 100 and self.grad_clip == 1., 'registered optimizer')
        require(1 <= self.feature_memory_gib <= 64 and self.checkpoint_interval > 0, 'bounded execution config')
        require(self.bootstrap_repetitions == 2000, 'registered log bootstrap repetitions')
        require(len(self.expected_base_hash) == len(self.expected_pool_hash) == 64, 'explicit source identities')

    @property
    def root(self): return Path(self.output_root)

    @property
    def identity(self): return digest(asdict(self))

    def source(self):
        s = StudyConfig(**read_json(Path(self.source_campaign)/'resolved_config.json'))
        require(s.root.resolve() == Path(self.source_campaign).resolve(), 'source namespace changed')
        return s


def load(path):
    d = yaml.safe_load(Path(path).read_text())
    if 'validation_steps' in d: d['validation_steps'] = tuple(d['validation_steps'])
    c = Config(**d)
    atomic_json(c.root/'resolved_config.json', asdict(c), immutable=True)
    return c

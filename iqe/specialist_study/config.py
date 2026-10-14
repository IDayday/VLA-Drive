from dataclasses import dataclass, asdict
from pathlib import Path
import yaml
from iqe.contracts import require
from iqe.io import atomic_json, digest


@dataclass(frozen=True)
class StudyConfig:
    campaign_id: str
    output_root: str
    base_checkpoint: str
    base_run: str
    reuse_root: str
    contract: str
    scene_manifest: str
    historical_summary: str
    historical_root: str
    optimized_root: str
    scorer_fit_limit: int = 20000
    seed: int = 20261014
    partition_seed: int = 271828
    selection_seed: int = 1729
    bootstrap_seed: int = 11235
    bootstrap_repetitions: int = 2000
    expert_updates: int = 5000
    scorer_updates: int = 6000
    global_batch: int = 128
    micro_batch: int = 16
    scoring_workers: int = 8
    scoring_attempts: int = 3
    scoring_python: str = '/root/miniconda3/envs/navsim/bin/python'
    navsim_root: str = '/mnt/project/local-interaction-mask-v2-artifacts/20260927/reference_sources/navsim-v1.1'
    historical_commit: str = '5f241ec78f7505af924c8b24bd09ee78ad45a6d8'
    max_expert_runs: int = 6
    max_scorer_runs: int = 4
    max_rich_candidates: int = 12
    cache_batch: int = 4
    feature_memory_gib: int = 16
    allow_navtest: bool = False
    auto_active: bool = False

    def __post_init__(self):
        require(self.campaign_id and Path(self.output_root).is_absolute(), 'explicit campaign identity/root required')
        require(self.expert_updates == 5000 and self.scorer_updates == 6000 and self.global_batch == 128,
                'fixed study update/exposure budgets; bounded diagnostics use stop_after')
        require(self.max_expert_runs == 6 and self.max_scorer_runs == 4 and self.max_rich_candidates == 12,
                'study search limits are fixed')
        require(0 < self.scorer_fit_limit <= 20000 and 1 <= self.scoring_workers <= 16,
                'scene/CPU worker budget')
        require(self.micro_batch > 0 and self.global_batch % self.micro_batch == 0, 'microbatch divides global batch')
        require(not self.allow_navtest and not self.auto_active, 'research campaign cannot dispatch navtest/ACTIVE')

    @property
    def root(self): return Path(self.output_root)
    @property
    def identity(self): return digest(asdict(self))
    @property
    def experiment_id(self): return self.campaign_id
    @property
    def checkpoint(self): return self.base_checkpoint
    @property
    def expert_micro_batch(self): return self.micro_batch
    @property
    def scorer_micro_batch(self): return self.micro_batch


def load_config(path):
    config = StudyConfig(**yaml.safe_load(Path(path).read_text()))
    atomic_json(config.root/'resolved_config.json', asdict(config), immutable=True)
    return config

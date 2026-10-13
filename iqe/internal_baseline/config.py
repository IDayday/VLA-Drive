from dataclasses import dataclass, asdict
from pathlib import Path
import yaml
from iqe.contracts import require
from iqe.io import digest, atomic_json


@dataclass(frozen=True)
class Config:
    experiment_id: str
    output_root: str
    contract: str
    base_run: str
    scene_manifest: str
    checkpoint: str = "latest_complete"
    fit_limit: int | str = 20000
    val_limit: int = 1024
    cal_limit: int = 1024
    selection_seed: int = 1729
    module_init_seed: int = 31415
    training_rng_seed: int = 27182
    natural_seed: int = 16180
    comparison_seed: int = 14142
    bootstrap_seed: int = 11235
    bootstrap_repetitions: int = 2000
    cache_batch: int = 4
    scorer_micro_batch: int = 16
    expert_micro_batch: int = 32
    expert_updates: int = 1000
    scorer_updates: int = 2000
    pointwise_updates: int = 500
    validation_interval: int = 250
    scorer_dropout: float = 0.1
    scoring_python: str = "/root/miniconda3/envs/navsim/bin/python"
    navsim_root: str = "/mnt/project/local-interaction-mask-v2-artifacts/20260927/reference_sources/navsim-v1.1"
    scoring_workers: int = 16
    scoring_attempts: int = 3
    score_context: str = "world_action_ego"
    diagnostic_updates: int = 200
    research_protocol: str = "internal_legacy_v1"
    expert_fit_limit: int = 16000
    scorer_fit_limit: int = 4000
    partition_seed: int = 271828

    def __post_init__(self):
        require(self.experiment_id and Path(self.output_root).is_absolute(), "explicit experiment/output identity required")
        require((self.fit_limit == "all_fit" or (type(self.fit_limit) is int and 0 < self.fit_limit <= 20000))
                and 0 < self.val_limit <= 1024 and 0 < self.cal_limit <= 1024, "bounded scene budget or explicit all_fit")
        if self.research_protocol == "two_candidate_v1":
            require(all(type(value) is int for value in (self.expert_updates, self.scorer_updates, self.pointwise_updates))
                    and 0 < self.expert_updates <= 10000 and 0 < self.scorer_updates <= 20000
                    and self.scorer_updates == 4*self.pointwise_updates,
                    "two-candidate bounded budgets <=10000/20000 with 1:3 scoring phases")
        else:
            require(0 < self.expert_updates <= 3000 and self.scorer_updates == 2000 and self.pointwise_updates == 500,
                    "legacy expert budget <=3000; scorer fixed 500+1500 recipe")
        require(self.validation_interval == 250 and 1 <= self.scoring_workers <= 32 and 1 <= self.scoring_attempts <= 5,
                "validation/scoring budget")
        require(self.score_context in {"world_action_ego", "action_ego", "planner_ego"}, "explicit scorer input ablation required")
        require(self.scorer_micro_batch > 0 and self.expert_micro_batch > 0 and self.cache_batch > 0, "batch budgets")
        require(0 <= self.scorer_dropout < 1 and 0 < self.diagnostic_updates <= 500, "diagnostic/dropout budget")
        require(self.research_protocol in {"internal_legacy_v1", "two_candidate_v1"}, "unknown research protocol")
        require(0 < self.expert_fit_limit <= 16000 and 0 < self.scorer_fit_limit <= 4000, "D_E/D_S scene budgets")
        if self.research_protocol == "two_candidate_v1":
            require(self.scorer_dropout == 0 and self.diagnostic_updates <= 300, "fixed two-candidate scorer recipe")

    @property
    def root(self):
        return Path(self.output_root)

    @property
    def identity(self):
        return digest(asdict(self))


def load_config(path):
    with open(path) as f:
        config = Config(**yaml.safe_load(f))
    atomic_json(config.root / "resolved_config.json", asdict(config), immutable=True)
    return config

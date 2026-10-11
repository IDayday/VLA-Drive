"""Strict, versioned execution options, separate from the 100k optimization schedule."""
from dataclasses import dataclass, asdict
import yaml
from ..contracts import require

@dataclass(frozen=True)
class PerformanceOptions:
    schema_version: int = 1
    optimizer: str = 'default'
    gradient_as_bucket_view: bool = False
    bucket_cap_mb: int = 25
    checkpoint_policy: str = 'all'
    visual_precision: str = 'legacy_fp32'
    query_training_semantics: str = 'legacy_repeat8'
    execution_weights: str = 'fp32'
    deterministic: bool = True
    vision_cache: str | None = None

    def validate(self):
        require(self.schema_version == 1, 'performance schema version')
        require(self.optimizer in {'default', 'fused'}, 'unsupported Adam implementation')
        require(type(self.gradient_as_bucket_view) is bool and type(self.deterministic) is bool, 'boolean runtime flags required')
        require(self.bucket_cap_mb in {25, 50, 100}, 'unsupported bucket size')
        require(self.checkpoint_policy in {'all', 'alternating', 'none'}, 'unsupported checkpoint policy')
        require(self.visual_precision in {'legacy_fp32','bf16_compute','bf16_compute_fp32_mergers'}, 'unsupported frozen visual precision')
        require(self.query_training_semantics in {'legacy_repeat8','native_repeat1'}, 'unsupported Query training semantics')
        require(self.execution_weights in {'fp32','bf16_master'}, 'unsupported execution weight protocol')
        require(self.vision_cache is None or isinstance(self.vision_cache,str), 'cache path must be explicit')
        return self

    @classmethod
    def load(cls,path):
        with open(path) as f: raw=yaml.safe_load(f)
        require(isinstance(raw,dict), 'runtime config must be a mapping')
        return cls(**raw).validate()

    def identity(self): return asdict(self)

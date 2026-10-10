"""Inference-only Navtest records deliberately cannot enter SceneRecord training."""
from dataclasses import dataclass
from ..contracts import require


@dataclass(frozen=True)
class NavtestInput:
    schema_version: int
    scene_id: str
    source_log_id: str
    observation_hash: str
    observation_ref: str
    metric_context_ref: str
    metric_context_hash: str
    split_role: str = 'navtest'

    def __post_init__(self):
        require(self.schema_version == 1 and self.split_role == 'navtest', 'Navtest input schema')
        for key in ('scene_id','source_log_id','observation_hash','observation_ref','metric_context_ref','metric_context_hash'):
            require(isinstance(getattr(self,key),str) and getattr(self,key), 'missing Navtest field '+key)

"""Explicit opt-in runtime composition; each arm retains a sealed full recipe."""
from pathlib import Path
import yaml
from .group_runtime import GroupRuntime
from .quality import QualityDiagnosticsMixin


def runtime_for_config(path):
    cfg=yaml.safe_load(Path(path).read_text());r=cfg['research']
    if r.get('combined_consolidation_retention') is not None:
        from .combined_consolidation_retention_runtime import combined_runtime
        return combined_runtime(cfg)
    if r.get('mixed_exploration') is not None:
        from .mixed_exploration_runtime import mixed_runtime
        return mixed_runtime(cfg)
    if r.get('precision_training') is not None:
        from .precision_runtime import precision_runtime
        return precision_runtime(cfg)
    if r.get('verified_retention') is not None:
        from .retention_runtime import retention_runtime
        return retention_runtime(cfg)
    if r.get('consolidation_training') is not None:
        from .consolidation_runtime import consolidation_runtime
        return consolidation_runtime(cfg)
    if r.get('batch32') is not None:
        from .batch32_runtime import batch32_runtime
        return batch32_runtime(cfg)
    if r.get('revival_training') is not None:
        from .revival_runtime import revival_runtime
        return revival_runtime(cfg)
    if r.get('standardized_training') is not None:
        from .standardized_runtime import standardized_runtime
        return standardized_runtime(cfg)
    if r.get('endpoint_policy') is not None:
        # Endpoint likelihood has its own quality/audit fields; the historical
        # wrapper explicitly assumes source W3 clamped transition probabilities.
        from .endpoint_runtime import EndpointRuntimeMixin,validate_endpoint_recipe
        validate_endpoint_recipe(cfg)
        return type('EndpointExplorationRuntime',(EndpointRuntimeMixin,GroupRuntime),{})()
    mixins=[QualityDiagnosticsMixin]
    if r['data_policy']=='online_priority':
        from .curriculum import CurriculumRuntimeMixin
        mixins.append(CurriculumRuntimeMixin)
    from .credit import STRUCTURED_SCHEMA
    if r['structured_advantage']!=STRUCTURED_SCHEMA:
        from .dense_credit import DenseRuntimeMixin,SCHEMA as DENSE_SCHEMA
        from .saturated_credit import SaturatedRuntimeMixin,SCHEMA as SATURATED_SCHEMA
        from .protected_credit import ProtectedRuntimeMixin,SCHEMA as PROTECTED_SCHEMA
        from .inversion_credit import InversionRuntimeMixin,SCHEMA as INVERSION_SCHEMA
        if r['structured_advantage']==DENSE_SCHEMA:mixins.append(DenseRuntimeMixin)
        elif r['structured_advantage']==SATURATED_SCHEMA:mixins.append(SaturatedRuntimeMixin)
        elif r['structured_advantage']==PROTECTED_SCHEMA:mixins.append(ProtectedRuntimeMixin)
        elif r['structured_advantage']==INVERSION_SCHEMA:mixins.append(InversionRuntimeMixin)
        else:raise ValueError('Unknown separately registered credit')
    return type('SafetyExplorationRuntime',tuple(mixins+[GroupRuntime]),{})()

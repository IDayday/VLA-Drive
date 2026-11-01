"""Explicit opt-in runtime composition; each arm retains a sealed full recipe."""
from pathlib import Path
import yaml
from .group_runtime import GroupRuntime
from .quality import QualityDiagnosticsMixin


def runtime_for_config(path):
    cfg=yaml.safe_load(Path(path).read_text());r=cfg['research']
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

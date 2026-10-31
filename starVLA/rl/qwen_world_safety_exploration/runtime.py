"""Explicit opt-in runtime composition; each arm retains a sealed full recipe."""
from pathlib import Path
import yaml
from .group_runtime import GroupRuntime
from .quality import QualityDiagnosticsMixin


def runtime_for_config(path):
    cfg=yaml.safe_load(Path(path).read_text());r=cfg['research']
    mixins=[QualityDiagnosticsMixin]
    if r['data_policy']=='online_priority':
        from .curriculum import CurriculumRuntimeMixin
        mixins.append(CurriculumRuntimeMixin)
    from .credit import STRUCTURED_SCHEMA
    if r['structured_advantage']!=STRUCTURED_SCHEMA:
        from .dense_credit import DenseRuntimeMixin,SCHEMA
        if r['structured_advantage']!=SCHEMA:raise ValueError('Unknown separately registered credit')
        mixins.append(DenseRuntimeMixin)
    return type('SafetyExplorationRuntime',tuple(mixins+[GroupRuntime]),{})()

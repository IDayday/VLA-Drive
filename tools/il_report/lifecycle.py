"""Explicit per-run user closure, independent of stale progress or job state."""
import json
from pathlib import Path


def user_termination(run):
    run=Path(run)
    for marker in (run/'USER_TERMINATION.json',run.parent.parent/'USER_TERMINATION.json'):
        try:value=json.loads(marker.read_text())
        except FileNotFoundError:continue
        if (value.get('schema')!='experiment_user_termination_v1'or
                value.get('status')!='TERMINATED_BY_USER'or
                Path(value.get('run','')).resolve()!=run.resolve()):
            raise ValueError('Invalid or misbound termination decision: '+str(marker))
        return value
    return None

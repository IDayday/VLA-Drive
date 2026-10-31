"""Milestone-only local refresh requests to the existing sole HTML renderer."""
import json
from pathlib import Path
import time
from starVLA.rl.qwen_world_safety_exploration.contracts import ROOT, authorization
from starVLA.rl.flow_grpo.transactions import atomic_json


def refresh(evidence, reason):
    try:
        grant, _ = authorization()
        report = Path(grant['report'])
        target = report.parent/'reporting/refresh_request.json'
        if target.parent.is_dir():
            atomic_json(target,dict(id='safety-exploration-'+str(time.time_ns()),at=time.time(),
                scientific_result=False,evidence=str(evidence),reason=reason))
    except (OSError,KeyError,ValueError,PermissionError):
        # Presentation cannot interrupt a native recoverable boundary.
        pass

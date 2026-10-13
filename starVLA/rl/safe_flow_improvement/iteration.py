"""Finite experiment boundaries and the preregistered DEV safety pause rule."""
import json
from pathlib import Path
import socket

from .contracts import SAFETY_METRICS


def require_dispatch():
    policy = json.loads(Path('/mnt/project/server_dispatch_policy.json').read_text())
    host = socket.gethostname().removesuffix('-worker-0')
    authorization = policy['task_authorizations']['flow_safe_improvement_20261013']
    if host not in authorization['allowed_hosts']:
        raise RuntimeError('Current host is outside this task authorization')
    return host


def safety_decision(report, previous=None):
    if report['status'] != 'COMPLETE_MATCHED_EVALUATION' or report['scenes'] != 1696 or not report['no_best_of_N']:
        raise ValueError('Safety decisions require the complete matched DEV evaluation')
    previous = previous or {}
    costs, pause = {}, []
    for metric in SAFETY_METRICS:
        change = report['paired_changes'][metric]
        increase, lower = -change['mean'], -change['CI95'][1]
        sustained = increase >= .005 and lower > 0
        count = previous.get('costs', {}).get(metric, {}).get('consecutive_sustained', 0) + 1 if sustained else 0
        costs[metric] = dict(mean_increase=increase, lower_CI95=lower, consecutive_sustained=count)
        if increase >= .02 or count >= 2:
            pause.append(metric)
    return dict(status='PAUSE_VARIANT' if pause else 'CONTINUE_REGISTERED_WINDOW',
                report_sha256=report['sha256'], costs=costs, triggered_metrics=pause,
                thresholds_raw=dict(immediate=.02, sustained=.005, consecutive=2),
                never_relax_permanent_SFT_budgets=True)

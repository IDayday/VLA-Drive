"""Task-scoped temporary capacity; withdrawn hosts stay closed to other queues."""
import json
from pathlib import Path


def authorized_hosts(policy, root):
    auth=policy['task_authorizations']['qwen_sourceflow_native_20261018']
    temporary=auth.get('temporary_host_exceptions',{})
    normal=[h for h in auth['allowed_hosts'] if h not in temporary and h in policy['allowed_hosts'] and
        policy.get('global_dispatch_prohibitions',{}).get(h,{}).get('new_task_dispatch_allowed') is not False]
    # The user's temporary grant ends with BOTH already-registered RL runs.
    # It does not reopen the historical general vla-zt3 queues or any zt2/zt6.
    for host,lease in temporary.items():
        if host!='training-vla-zt3' or lease.get('enabled') is not True:continue
        if host not in policy['allowed_hosts'] or host not in auth['allowed_hosts']:continue
        if policy.get('global_dispatch_prohibitions',{}).get(host,{}).get('new_task_dispatch_allowed') is False:continue
        if lease.get('scope')!='auxiliary_native_RL_completion':continue
        if lease.get('end_condition')!='both_registered_RL_runs_reach_50000':continue
        states=[]
        try:
            for name in ('RL_CANONICAL_seed42','RL_SOURCE_seed42'):
                state=json.loads((Path(root)/'runs'/name/'progress.json').read_text())
                if not isinstance(state['updates'],int) or state['updates']<0:raise ValueError('Invalid progress')
                states.append(state)
        except (OSError,ValueError,KeyError):continue
        if any(s['updates']<50000 for s in states):normal.append(host)
    return list(dict.fromkeys(normal))

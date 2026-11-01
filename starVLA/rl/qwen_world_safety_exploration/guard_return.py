"""Return an already admitted native job at its existing exact boundary.

This grants no new work on a withdrawn host. Startup still uses long-job
admission; after binding, a resource stop may only save and return.
"""


def native_dispatch(runtime):
    from .contracts import boundary_stop_due, require_dispatch
    host = getattr(runtime, '_admitted_native_host', None)
    bound = (host is not None and hasattr(runtime, 'training_state')
             and hasattr(runtime, 'training_identity'))
    if bound and boundary_stop_due(host):
        runtime.request_stop()
        return host
    try:
        host = require_dispatch(host=host, admit_long_job=not bound)
    except PermissionError:
        # A withdrawal can arrive between the read above and admission check.
        # Re-check only the bound job's resource-return condition; malformed
        # grants still fail here and never acquire a new admission.
        if bound and boundary_stop_due(host):
            runtime.request_stop()
            return host
        raise
    runtime._admitted_native_host = host
    return host


def boundary_signature(state, rollout):
    """Only compare boundaries already returned by strict native restore/save."""
    return (state['updates'], state['next_inner'], state['collections'],
            state['scene_cursor'], state['phase'],
            None if rollout is None else rollout['chain_identity'])


def stopped_checkpoint(state, rollout, last_saved, last_signature, save):
    """Do not overwrite an unchanged COMPLETE when stopping before the next step."""
    signature = boundary_signature(state, rollout)
    if last_saved is not None and last_signature == signature:
        return last_saved, signature
    return save(), signature

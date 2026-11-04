"""Read the declared container allowlist without expanding resource authority.

Older receipts store host metadata directly in ``allowed_hosts``. Current
receipts store an explicit list there and metadata in ``hosts``. Neither form
permits discovering additional hosts by iterating the metadata registry.
"""
import time


def authorized_host_rows(policy, *, now=None):
    declared = policy.get('allowed_hosts', {})
    registry = policy.get('hosts', {})
    if not isinstance(registry, dict):
        raise ValueError('Resource policy hosts must be a mapping')
    if isinstance(declared, dict):
        rows = declared.items()
    elif isinstance(declared, list):
        if any(not isinstance(host, str) or not host for host in declared):
            raise ValueError('Resource allowlist entries must be host names')
        rows = ((host, {}) for host in declared)
    else:
        raise ValueError('Resource allowed_hosts must be a mapping or list')
    now = time.time() if now is None else now
    result = {}
    prohibitions = policy.get('global_dispatch_prohibitions', {})
    for host, legacy in rows:
        metadata = registry.get(host, {})
        if (not isinstance(host, str) or not host or
                not isinstance(legacy, dict) or not isinstance(metadata, dict)):
            raise ValueError('Malformed resource host metadata')
        # The current per-host record can explicitly withdraw legacy authority.
        row = {**legacy, **metadata}
        if (row.get('allowed') is False or row.get('new_task_dispatch_allowed') is False or
                prohibitions.get(host, {}).get('new_task_dispatch_allowed') is False):
            continue
        if (row.get('temporary') and not row.get('unbounded') and
                row.get('expires_unix', 0) <= now):
            continue
        result[host] = row
    return result

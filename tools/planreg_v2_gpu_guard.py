"""Read-only GPU admission, restricted to explicitly visible physical devices."""
import os
import subprocess


def visible_gpu_processes(visible=None, query=None):
    query = query or (lambda fields, kind: subprocess.check_output(
        ['nvidia-smi', '--query-'+kind+'='+fields, '--format=csv,noheader,nounits'], text=True))
    inventory = {}
    for row in query('index,uuid', 'gpu').splitlines():
        index, uuid = [s.strip() for s in row.split(',')]
        inventory[index] = uuid
    visible = os.environ.get('CUDA_VISIBLE_DEVICES') if visible is None else visible
    if visible is None:
        selected = set(inventory.values())
    else:
        selected = set()
        for item in visible.split(','):
            item = item.strip()
            if item in inventory:
                selected.add(inventory[item])
            elif item in inventory.values():
                selected.add(item)
            else:
                raise ValueError('Use explicit full GPU indices/UUIDs, not ambiguous visibility: '+item)
    processes = []
    for row in query('pid,gpu_uuid', 'compute-apps').splitlines():
        pid, uuid = [s.strip() for s in row.split(',')]
        if uuid in selected:
            processes.append(int(pid))
    return sorted(set(processes))

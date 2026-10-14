"""Inspect physical use before dispatch; unknown owners are never signalled."""
import subprocess
import time

from iqe.contracts import require


def assess(gpu_rows, process_rows, physical_gpu, minimum_free_mib=8192):
    rows = [row.split(',') for row in gpu_rows.strip().splitlines() if row.strip()]
    matches = [row for row in rows if int(row[0]) == physical_gpu]
    require(len(matches) == 1, 'physical GPU not visible to capacity probe')
    _, uuid, used, total, util = [s.strip() for s in matches[0]]
    used, total, util = float(used), float(total), float(util)
    owners = []
    for line in process_rows.strip().splitlines():
        if not line.strip(): continue
        pid, key, memory = [s.strip() for s in line.split(',')]
        if key == uuid:
            owners.append({'pid': int(pid), 'memory_mib': float(memory)})
    unexplained = max(0., used-sum(p['memory_mib'] for p in owners))
    reasons = []
    if unexplained > 1024: reasons.append('UNIDENTIFIED_PHYSICAL_OCCUPANCY')
    if total-used < minimum_free_mib: reasons.append('INSUFFICIENT_FREE_MEMORY')
    if util > 60: reasons.append('BUSY_PHYSICAL_GPU')
    return {'ready': not reasons, 'reasons': reasons, 'physical_gpu': physical_gpu,
            'used_mib': used, 'free_mib': total-used, 'utilization_percent': util,
            'visible_owners': owners, 'unexplained_mib': unexplained,
            'minimum_free_mib': minimum_free_mib, 'sampled_unix': time.time(), 'process_signals': False}


def probe(physical_gpu):
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu',
                                   '--format=csv,noheader,nounits'], text=True)
    owners = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid,used_memory',
                                      '--format=csv,noheader,nounits'], text=True)
    return assess(gpu, owners, physical_gpu)

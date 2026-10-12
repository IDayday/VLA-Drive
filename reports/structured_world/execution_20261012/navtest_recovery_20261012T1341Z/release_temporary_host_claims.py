"""Release our routing claims; never launch pressure or signal any process."""
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import time

ROOT = Path('/mnt/project/structured-world-fgtr-round1-artifacts/20261008/navtest_recovery_20261012T1341Z')
QUEUE = ROOT.parent / 'navsim_navtest_dispatch_20261011T0225'
AUTH = json.loads((ROOT / 'temporary_hosts_9h/AUTHORIZATION.json').read_text())
POLICY = Path('/mnt/project/server_dispatch_policy.json')
KEY = 'G3_EVENT_LOCAL_0090000'

def atomic(path, value):
    tmp = path.with_name(path.name + '.claim_release.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    os.replace(tmp, path)

released = False
while time.time() < AUTH['expires_unix'] + 15:
    marker = QUEUE / 'completed' / (KEY + '.json')
    proof = QUEUE / 'jobs' / KEY / 'OWNED_WORKERS_EXITED.json'
    if not released and marker.exists() and proof.exists():
        complete, exited = json.loads(marker.read_text()), json.loads(proof.read_text())
        if (complete.get('valid') is True and exited.get('key') == KEY and
            exited.get('all_gpu_launchers_exited') and exited.get('scorer_exited') and
            all(x['returncode'] == 0 for x in exited['launchers'])):
            with POLICY.with_suffix('.lock').open('a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                data = json.loads(POLICY.read_text())
                removed = []
                for host in AUTH['hosts']:
                    prior = data.get('host_reservations', {}).get(host, {})
                    if prior.get('lease_id') == AUTH['lease_id']:
                        del data['host_reservations'][host]
                        removed.append(host)
                atomic(POLICY, data)
            atomic(ROOT / 'temporary_hosts_9h/CLAIMS_RELEASED.json', dict(
                utc=datetime.now(timezone.utc).isoformat(), removed=removed,
                reason='Complete G3 90k score bank and waited GPU/scorer exit proof',
                pressure_restoration='External user monitor; no assistant pressure action'))
            released = True
    time.sleep(15)

cost = QUEUE / 'COHOST_COST_POLICY.json'
before = json.loads((ROOT / 'BEFORE_TEMPORARY_WINDOW_HANDOFF_COHOST.json').read_text())
with (ROOT / 'pacing_policy.lock').open('a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    data = json.loads(cost.read_text())
    if data.get('temporary_idle_window', {}).get('authorization') == str(ROOT / 'temporary_hosts_9h/AUTHORIZATION.json'):
        quarantine = data['quarantined_hosts']
        for host in ['training-vla-zt', 'training-vlawm-zt2', 'training-rl-zt2']:
            if quarantine.get(host, '').startswith('Prefer the newly authorized, externally cleared idle GPUs'):
                if host in before['quarantined_hosts']:
                    quarantine[host] = before['quarantined_hosts'][host]
                else:
                    quarantine.pop(host, None)
        for host in AUTH['hosts']:
            quarantine.setdefault(host, 'Nine-hour user authorization expired; no new Navtest workers.')
        data.pop('temporary_idle_window', None)
        atomic(cost, data)
atomic(ROOT / 'temporary_hosts_9h/WINDOW_EXPIRED.json', dict(
    utc=datetime.now(timezone.utc).isoformat(), claims_released_after_exit_proof=released,
    expired_hosts=AUTH['hosts'], all_GPU_commands_had_absolute_deadline=True,
    pressure_restoration='External user monitor; no assistant pressure action',
    requires_review=not released))

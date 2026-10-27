"""Receipt-bound co-location with existing jobs; never signal an occupant."""
import json
import time
from pathlib import Path
from .contracts import authorization, file_sha, fingerprint, TASK


def validate_registered_occupant(row, live, receipt, *, host, authorization_sha256, now):
    body = dict(receipt)
    if body.pop('sha256', None) != fingerprint(body):
        raise ValueError('Changed co-location receipt')
    if (receipt.get('schema') != 'world_registered_GPU_occupants_v1'
            or receipt.get('task') != TASK or receipt.get('host') != host
            or receipt.get('authorization_sha256') != authorization_sha256
            or not receipt['valid_from'] <= now <= receipt['valid_until']):
        raise ValueError('Co-location receipt scope/authorization/window changed')
    matches = [v for v in receipt['occupants'] if v['pid'] == row['pid'] and v['uuid'] == row['uuid']]
    if len(matches) != 1:
        raise ValueError('Unregistered GPU occupant')
    selected = matches[0]
    if any(live.get(k) != selected.get(k) for k in ('pid', 'start_ticks', 'argv_sha256', 'cwd')):
        raise ValueError('GPU occupant was replaced or changed its command')
    if (type(row['used_MiB']) is not int or not 0 <= row['used_MiB'] <= selected['maximum_used_MiB'] <= 8192):
        raise ValueError('Registered small occupant exceeded its admitted memory envelope')
    return True


def check_registered_occupant(row, argv, path, host):
    grant, _ = authorization()
    receipt = json.loads(Path(path).read_text())
    process = Path('/proc') / str(row['pid'])
    state = (process / 'stat').read_text().rsplit(')', 1)[1].split()
    if state[0] in ('Z', 'X'):
        raise ValueError('GPU owner exited during admission')
    live = dict(pid=row['pid'], start_ticks=int(state[19]),
                argv_sha256=fingerprint([x for x in argv if x]), cwd=str((process / 'cwd').resolve()))
    return validate_registered_occupant(row, live, receipt, host=host,
            authorization_sha256=file_sha(grant['record']), now=time.time())

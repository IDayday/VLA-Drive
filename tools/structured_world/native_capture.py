"""Pilot-only checkpoint observations and bounded storage admission; CPU only.

No model/RNG operations, filesystem mutation or archival. Quota visibility is
reported honestly; an unavailable quota command does not veto authorized work.
"""
import hashlib
import json
from pathlib import Path
import shutil
import socket
import os
import subprocess
import time

GiB = 1 << 30


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    sha = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            sha.update(block)
    return sha.hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def validate_args(args):
    if args.capture_updates is None:
        require(args.capture_storage_budget is None, 'Storage capture budget requires --capture-updates')
        return
    require(args.scope == 'profile' and args.pilot_origin_run and args.pilot_origin_tag and
            not args.debug_scenes and not args.benchmark_no_end_save and not args.resume_origin_run,
            'Capture observations require a full-population native-origin profile; never formal/benchmark')
    require(args.capture_storage_budget is not None, 'Explicit bounded storage budget required')


def observation_plan(text, origin_completed, updates):
    if text is None:
        return None
    require(isinstance(text, str) and text and all(x.strip().isdigit() for x in text.split(',')), 'Comma-separated positive capture offsets required')
    offsets = [int(x.strip()) for x in text.split(',')]
    require(offsets == sorted(set(offsets)) and all(0 < x <= updates for x in offsets), 'Capture offsets must be unique, ordered and within the fixed continuation')
    require(origin_completed >= 0 and updates > 0, 'Finite immutable origin required')
    return {'relative_updates': offsets, 'origin_completed': origin_completed,
            'absolute_updates': [origin_completed+x for x in offsets],
            'end_update': origin_completed+updates, 'stop_at_capture': False,
            'save_path': 'existing native save callback, unchanged tensor layout',
            'retain_observations': 'pilot milestone tags; original rolling-latest retention unchanged'}


def allocated_bytes(root):
    """Count only explicit owned roots, without following symlinks."""
    root = Path(root)
    if not root.exists():
        return 0
    require(not root.is_symlink(), 'Owned storage root cannot be a symlink')
    total = 0
    for item in root.rglob('*'):
        require(not item.is_symlink(), 'Symlink inside owned storage tree')
        if item.is_file():
            stat = item.stat()
            total += getattr(stat, 'st_blocks', (stat.st_size+511)//512)*512
    return total


def quota_observation(mount):
    """Read only this uid's quota if exposed; never infer unlimited from df."""
    lfs, quota = shutil.which('lfs'), shutil.which('quota')
    command = ([lfs, 'quota', '-u', str(os.getuid()), str(mount)] if lfs else
               [quota, '-w'] if quota else None)
    if command is None:
        return {'status': 'quota_not_exposed', 'command': None,
                'risk': 'Physical free space is not a verified per-user quota or exclusive reservation'}
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=3, check=False)
        return {'status': 'quota_command_observed' if result.returncode == 0 else 'quota_not_exposed',
                'command': command, 'returncode': result.returncode,
                'output': (result.stdout+result.stderr)[:4096],
                'enforced_by_admission': False, 'unlimited_claimed': False}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {'status': 'quota_not_exposed', 'command': command, 'error': repr(exc),
                'unlimited_claimed': False}


def existing_parent(path):
    path = Path(path)
    while not path.exists():
        require(path != path.parent, 'No existing output filesystem ancestor')
        path = path.parent
    return path


def storage_admission(budget_path, output, *, now=None, expected_digest=None):
    """Before EACH save: actual free space and explicitly bounded OWN caps.

    Caps are cooperative accounting, not an exclusive filesystem reservation.
    Missing quota visibility is a recorded risk. EDQUOT/ENOSPC from the existing
    native save stops the owned pilot without removing preceding checkpoints.
    """
    started = time.monotonic()
    now = time.time() if now is None else now
    budget_sha = digest(budget_path)
    require(expected_digest is None or budget_sha == expected_digest, 'Bound storage plan changed')
    budget = read(budget_path)
    require(budget['schema'] == 'native_capture_storage_budget_v1' and budget['approved_for_pilot'] is True, 'Storage plan is not admitted for this pilot')
    host = socket.gethostname().removesuffix('-worker-0')
    require(host == budget['host'], 'Storage attestation belongs to another host')
    reserve, pending = budget['reserve_bytes'], budget['checkpoint_upper_bytes']
    require(type(reserve) is int and reserve >= 100*GiB and type(pending) is int and pending > 0, 'At least 100 GiB plus one full checkpoint staging reserve required')
    owners = budget['owned_runs']
    paths = [Path(x['path']).resolve() for x in owners]
    require(paths and len(paths) == len(set(paths)), 'Explicit unique owned run roots required')
    require(all(not a.is_relative_to(b) for a in paths for b in paths if a != b), 'Owned storage roots cannot overlap')
    out = Path(output).resolve()
    require(out in paths, 'Output is not an explicitly owned numerical run')
    require(not Path(output).is_symlink(), 'Output cannot be a symlink')
    usage = []
    for owner, path in zip(owners, paths):
        require(owner['kind'] == 'owned_numerical_pilot' and type(owner['cap_bytes']) is int and owner['cap_bytes'] > 0, 'Formal/origin/rollback assets are excluded from writable ownership')
        used = allocated_bytes(path)
        require(used+(pending if path == out else 0) <= owner['cap_bytes'], 'Owned run checkpoint cap exceeded')
        usage.append({'path': str(path), 'allocated_bytes': used, 'cap_bytes': owner['cap_bytes']})
    other = budget.get('other_owned_reservation_bytes', 0)
    require(type(other) is int and other >= 0 and type(budget['pool_cap_bytes']) is int and
            sum(x['cap_bytes'] for x in owners)+other <= budget['pool_cap_bytes'], 'Reservations exceed explicit storage pool cap')
    mount = Path(budget['filesystem_probe_path']).resolve()
    require(mount.exists() and mount.stat().st_dev == existing_parent(out).stat().st_dev, 'Output filesystem differs from reserved pool')
    physical = shutil.disk_usage(mount)
    require(physical.free >= reserve+pending, 'Physical free space cannot preserve reserve after next save')
    quota = quota_observation(mount)
    require(digest(budget_path) == budget_sha, 'Storage plan changed during admission')
    return {'ok': True, 'budget_sha256': budget_sha, 'checked_unix': now,
            'physical_free_bytes': physical.free, 'reserve_bytes': reserve, 'next_checkpoint_upper_bytes': pending,
            'owned_usage': usage, 'quota': quota, 'exclusive_filesystem_reservation': False,
            'admission_wall_seconds': time.monotonic()-started}

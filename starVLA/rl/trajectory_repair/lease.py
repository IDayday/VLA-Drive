"""Absolute lease time contract. This is not a training watchdog/dispatcher.

New training remains held until its independent owned-process deadline guard
and native checkpoint/resume are qualified. Importing this helper cannot launch
or terminate a process, expand a global allowlist, or signal another task.
"""
from dataclasses import dataclass
import hashlib
import json
import math


def _seal(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


@dataclass(frozen=True)
class Lease:
    fixed: tuple
    temporary: tuple
    starts: float
    stop_admission: float
    checkpoint: float
    migrate: float
    exit_by: float
    expires: float

    @classmethod
    def read(cls, value):
        body = dict(value)
        if body.pop('sha256') != _seal(body):
            raise ValueError('Changed resource authorization')
        d = value['deadlines']
        lease = cls(tuple(value['fixed_hosts']), tuple(value['temporary_candidate_hosts']),
                    value['starts_unix'], d['stop_new_admission_unix'],
                    d['request_checkpoint_unix'], d['migration_target_unix'],
                    d['temporary_workers_exit_by_unix'], value['expires_unix'])
        if set(lease.fixed) & set(lease.temporary):
            raise ValueError('Temporary hosts cannot be permanent migration destinations')
        if not (lease.starts < lease.stop_admission < lease.checkpoint < lease.migrate < lease.exit_by < lease.expires):
            raise ValueError('Invalid absolute drain/migration deadlines')
        if lease.expires - lease.starts != 9 * 3600:
            raise ValueError('This user authorization is exactly nine hours')
        return lease

    def phase(self, host, now):
        if not math.isfinite(now):
            raise ValueError('Absolute lease clock must be finite')
        if host in self.fixed:
            return 'FIXED_HOST_CAPACITY_AND_LIVE_TASK_AUTHORIZATION_STILL_REQUIRED'
        if host not in self.temporary:
            raise PermissionError('Host outside this task resource contract')
        if now < self.starts:
            return 'NOT_YET_AUTHORIZED'
        if now >= self.expires:
            return 'EXPIRED_NO_USE'
        if now >= self.exit_by:
            return 'TEMPORARY_WORKERS_MUST_HAVE_EXITED'
        if now >= self.migrate:
            return 'MIGRATE_TO_FIXED_OR_CHECKPOINT_AND_WAIT'
        if now >= self.checkpoint:
            return 'COMMIT_NATIVE_CHECKPOINT_AND_EXIT'
        if now >= self.stop_admission:
            return 'NO_NEW_ADMISSION_DRAIN_ONLY'
        return 'TIME_WINDOW_OPEN_GUARD_AND_NATIVE_RESUME_STILL_REQUIRED'

    def require_training_time(self, host, now, maximum_noninterruptible_seconds=0):
        if not math.isfinite(maximum_noninterruptible_seconds) or maximum_noninterruptible_seconds < 0:
            raise ValueError('Negative checkpoint latency is invalid')
        phase = self.phase(host, now)
        if host in self.fixed:
            return  # Time only. Caller must still check live scope/UUID/capacity.
        if phase != 'TIME_WINDOW_OPEN_GUARD_AND_NATIVE_RESUME_STILL_REQUIRED':
            raise PermissionError(phase)
        if now + maximum_noninterruptible_seconds >= self.checkpoint:
            raise PermissionError('No time to finish an atomic training boundary before checkpoint drain')

    def require_migration_destination(self, host):
        if host not in self.fixed:
            raise PermissionError('Cannot migrate to another temporary server')

#!/usr/bin/python3
"""Performance pilot lease for one inspected synthetic pressure entrypoint.

Uses the existing pidfd/capture/restore machinery without changing the Navtest
helper's trusted entrypoint. No other script is accepted by this task adapter.
"""
import json
from pathlib import Path
import navtest_pressure_lease as lease

SCRIPT = Path('/var/tmp/zt-gpu-pressure/zt_gpu_pressure.py')
SHA256 = 'e9d14e8245d6a284720e07fb00f5c4626615c59d1d1502355251625c6759a555'


def main():
    args = lease.build_parser().parse_args()
    if lease.digest(SCRIPT) != SHA256:
        raise lease.LeaseError('inspected_pressure_script_changed')
    helper = lease.PressureLease(args.state_root, lease.LocalHost(SCRIPT), args.timeout)
    selected = [v.strip() for v in (args.gpu_uuids or '').split(',') if v.strip()]
    if args.action in ('inspect', 'acquire') and not selected:
        raise lease.LeaseError('gpu_uuids_required')
    if args.action == 'inspect':
        result = helper.inspect(selected, args.worker_pid)
    elif args.action == 'acquire':
        result = helper.acquire(args.lease_id, selected, args.worker_pid)
    elif args.action == 'restore':
        result = helper.restore(args.lease_id, args.worker_pid, args.no_workers_started,
                                selected or None)
    else:
        result = helper.status(args.lease_id, selected or None)
    print(json.dumps(result, sort_keys=True))
    return 3 if result.get('deferred') else 0


if __name__ == '__main__':
    raise SystemExit(main())

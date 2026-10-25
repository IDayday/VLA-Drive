"""Read a sealed absolute deadline contract; never starts GPU work."""
import argparse
import json
from pathlib import Path
import time
from starVLA.rl.trajectory_repair.lease import Lease


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--authorization', required=True)
    parser.add_argument('--now-unix', type=float)
    args = parser.parse_args()
    value = json.loads(Path(args.authorization).read_text())
    lease = Lease.read(value)
    now = time.time() if args.now_unix is None else args.now_unix
    print(json.dumps(dict(now_unix=now, expires_utc=value['expires_utc'],
                         phases={host: lease.phase(host, now) for host in lease.fixed + lease.temporary},
                         no_dispatch=True, not_a_qualified_training_watchdog=True), indent=2))


if __name__ == '__main__':
    main()

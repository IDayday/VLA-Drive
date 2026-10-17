"""Bounded read-only observer of two pre-identified metadata-builder processes.

No task signals, child launch, directory traversal, or ledger/cache mutation.
The only target artifacts read are COMPLETE.json and CONTROL_BUILD_RESULT.json.
"""
import hashlib
import json
import os
from pathlib import Path
import resource
import stat
import time

HERE = Path(__file__).resolve().parent
ROOT = Path('/mnt/project/st-world-policy-20261016/caches/full_future_metadata_train_v1')
EXPECTED = {480124: 1366936766, 480140: 1366936849}
FIRST_OBSERVATION = 1792241903.0442202
DEADLINE = FIRST_OBSERVATION + 600
READ_LIMIT = 32 * 1024**2
OUTPUT_LIMIT = 1024**2
PER_FILE = 256 * 1024
os.sched_setaffinity(0, {8, 9})
resource.setrlimit(resource.RLIMIT_AS, (256 * 1024**2, 256 * 1024**2))
resource.setrlimit(resource.RLIMIT_CPU, (60, 60))
resource.setrlimit(resource.RLIMIT_FSIZE, (OUTPUT_LIMIT, OUTPUT_LIMIT))
read_bytes = 0
written_bytes = 0


def bounded_read(path, limit=PER_FILE):
    global read_bytes
    allowance = min(limit + 1, READ_LIMIT - read_bytes)
    if allowance <= 0:
        raise RuntimeError('observer read allowance exhausted')
    fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NONBLOCK)
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            raise RuntimeError('nonregular input rejected')
        data = os.read(fd, allowance)
        read_bytes += len(data)
        if len(data) > limit:
            raise RuntimeError('input too large for bounded administrative read')
        return data
    finally:
        os.close(fd)


def write_json(path, value):
    global written_bytes
    data = (json.dumps(value, sort_keys=True, indent=2) + '\n').encode()
    if written_bytes + len(data) > OUTPUT_LIMIT:
        raise RuntimeError('observer cumulative output allowance exhausted')
    with open(path, 'xb') as out:
        out.write(data)
    written_bytes += len(data)


def proc_stat(pid):
    data = bounded_read(Path('/proc') / str(pid) / 'stat', 16384)
    parts = data.decode().rsplit(') ', 1)[1].split()
    return dict(pid=pid, start_ticks=int(parts[19]), state=parts[0], ppid=int(parts[1]))


def process_snapshot(pid, start):
    result = dict(pid=pid, expected_start_ticks=start)
    try:
        before = proc_stat(pid)
        result['before'] = before
        if before['start_ticks'] != start:
            result['identity'] = 'MISMATCH_NO_FURTHER_ACCESS'
            return result
        text = bounded_read(Path('/proc') / str(pid) / 'io', 16384).decode()
        io = {k: int(v.strip()) for k, v in (line.split(':', 1) for line in text.splitlines())}
        status = bounded_read(Path('/proc') / str(pid) / 'status', 65536).decode()
        result.update(io=io, memory_lines=[line for line in status.splitlines()
                                          if line.startswith(('VmRSS:', 'VmHWM:'))])
        after = proc_stat(pid)
        result['after'] = after
        result['identity'] = 'MATCH' if after['start_ticks'] == start else 'RACED_IDENTITY_REJECTED'
    except (FileNotFoundError, ProcessLookupError):
        result['identity'] = 'ABSENT_OR_EXITED_DURING_SNAPSHOT'
    except PermissionError as error:
        result['identity'] = 'PERMISSION_DENIED'
        result['error'] = str(error)
    return result


def receipt_snapshot(name, serial):
    try:
        data = bounded_read(ROOT / name)
    except FileNotFoundError:
        return {'exists': False}
    digest = hashlib.sha256(data).hexdigest()
    try:
        parsed = json.loads(data)
    except (ValueError, UnicodeDecodeError) as error:
        return {'exists': True, 'sha256': digest, 'error': str(error)}
    saved = HERE / ('observed_' + str(serial).zfill(2) + '_' + name)
    write_json(saved, {'source': str(ROOT / name), 'sha256': digest,
                      'observed_at': time.time(), 'content': parsed})
    return {'exists': True, 'sha256': digest, 'saved': saved.name, 'content': parsed}


def main():
    start = time.monotonic()
    snapshots = []
    final_state = 'SNAPSHOT_ONLY_DEADLINE'
    serial = 0
    while time.time() <= DEADLINE:
        now = time.time()
        item = {'time_unix': now, 'time_utc': time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime(now)),
                'processes': [process_snapshot(pid, identity) for pid, identity in EXPECTED.items()]}
        for name in ('COMPLETE.json', 'CONTROL_BUILD_RESULT.json'):
            item[name] = receipt_snapshot(name, serial)
        snapshots.append(item)
        print(json.dumps({'sample': serial, 'time': item['time_utc'],
                          'identities': [p['identity'] for p in item['processes']],
                          'complete': item['COMPLETE.json']['exists'],
                          'result': item['CONTROL_BUILD_RESULT.json']['exists'],
                          'observer_bytes_read': read_bytes}), flush=True)
        terminal_result = item['CONTROL_BUILD_RESULT.json'].get('content')
        still_live = any(p['identity'] == 'MATCH' and p.get('after', {}).get('state') != 'Z'
                         for p in item['processes'])
        if terminal_result is not None and not still_live:
            final_state = 'TERMINAL_RESULT_OBSERVED_NO_STATUS_OVERRIDE'
            break
        serial += 1
        remaining = DEADLINE - time.time()
        if remaining < 30:
            break
        time.sleep(30)
    summary = {'status': final_state, 'reviewer': '/root/st_owned_descendants_review_v2',
               'task_root': str(ROOT), 'first_tool_observation_unix': FIRST_OBSERVATION,
               'absolute_deadline_unix': DEADLINE, 'elapsed_collector_s': time.monotonic() - start,
               'cpu_affinity': sorted(os.sched_getaffinity(0)), 'read_bytes': read_bytes,
               'observer_peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
               'output_bytes_before_summary': written_bytes,
               'limits': {'elapsed_from_first_observation_s': 600, 'read_bytes': READ_LIMIT,
                          'rss_bytes': 256 * 1024**2, 'output_bytes': OUTPUT_LIMIT},
               'actual_task_signals': 0, 'task_mutations': 0, 'gpu_calls': 0,
               'snapshots': snapshots}
    write_json(HERE / 'OBSERVATION.json', summary)
    print(json.dumps({'status': final_state, 'read_bytes': read_bytes,
                      'output_bytes': written_bytes}), flush=True)


if __name__ == '__main__':
    main()

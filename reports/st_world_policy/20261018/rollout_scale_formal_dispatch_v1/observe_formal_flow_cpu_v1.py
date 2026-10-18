"""Read-only bounded formal invocation CPU census; sends no process signals."""
import argparse
import json
import os
from pathlib import Path
import socket
import time

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--invocation', required=True)
a = p.parse_args()
r = Path(a.invocation).resolve()
assert r.is_relative_to(Path('/mnt/project/st-world-policy-20261016/formal/consequence/registrations'))
launch = json.loads((r/'LAUNCH.json').read_text())
assert launch['scope'] == 'FORMAL' and launch['host'] == socket.gethostname()
policy = json.loads(Path('/mnt/project/server_dispatch_policy.json').read_text())
assert socket.gethostname() in policy['allowed_aliases']
reg = json.loads(Path(launch['registration']).read_text())
assert reg['execution']['max_updates'] == reg['execution']['stop_after'] == 4096
root = int(launch['pid'])
hz = os.sysconf('SC_CLK_TCK')
page = os.sysconf('SC_PAGE_SIZE')

def stat(pid):
    s = (Path('/proc')/str(pid)/'stat').read_text()
    v = s[s.rfind(')')+2:].split()
    return dict(pid=pid,ppid=int(v[1]),start_ticks=int(v[19]),
        user_ticks=int(v[11]),system_ticks=int(v[12]),
        waited_user_ticks=int(v[13]),waited_system_ticks=int(v[14]),
        rss_bytes=int(v[21])*page,state=v[0])

started = time.time()
started_monotonic = time.monotonic()
limit = launch['invocation_seconds'] + launch['save_cleanup_grace_seconds']
assert 0 < limit <= 29580
try:
    original = stat(root)['start_ticks']
except OSError:
    original = None
with (r/'PROCESS_TREE_SAMPLES.jsonl').open('x') as stream:
    stream.write(json.dumps({'schema':'st_formal_process_tree_v1',
        'observer_host':socket.gethostname(),'observer_pid':os.getpid(),
        'root_pid':root,'root_start_ticks':original,'started_unix':started,
        'clock_ticks_per_second':hz,'interval_seconds':10,'max_bytes':256*2**20,
        'max_seconds':limit,'deadline_mode':'cooperative_monotonic_between_proc_samples',
        'scope':'post-launch live subtree self+waited CPU; never sum cumulative samples; '
                '10-second boundary error; RSS sum is upper bound, not PSS; no signals'})+'\n')
    stream.flush()
    reason = 'BOUND_EXHAUSTED'
    # Keep a separate footer allowance as well as the maximum one-record size.
    while original is not None and time.monotonic()-started_monotonic < limit and stream.tell() < 256*2**20-1024*1024-4096:
        now = time.time()
        sample_started = time.monotonic()
        try:
            if stat(root)['start_ticks'] != original:
                reason = 'ROOT_IDENTITY_CHANGED'
                break
        except OSError:
            reason = 'ROOT_EXITED'
            break
        table = {}
        for q in Path('/proc').iterdir():
            if q.name.isdigit():
                try:
                    table[int(q.name)] = stat(int(q.name))
                except (OSError,ValueError,IndexError):
                    continue
        live = {root}
        while True:
            more = {pid for pid,v in table.items() if v['ppid'] in live}-live
            if not more:
                break
            live.update(more)
        vals = [table[x] for x in live if x in table]
        row = {'unix':now,'processes':vals,
            'tree_cpu_seconds':sum(v['user_ticks']+v['system_ticks']+
                v['waited_user_ticks']+v['waited_system_ticks'] for v in vals)/hz,
            'rss_sum_bytes':sum(v['rss_bytes'] for v in vals)}
        encoded = json.dumps(row,separators=(',',':'))+'\n'
        if len(encoded.encode()) > 1024*1024:
            reason = 'RECORD_SIZE_BOUND'
            break
        stream.write(encoded)
        stream.flush()
        remaining = limit-(time.monotonic()-started_monotonic)
        time.sleep(max(0,min(10-(time.monotonic()-sample_started),remaining)))
    if original is None:
        reason = 'ROOT_ALREADY_EXITED'
    footer = json.dumps({'status':'OBSERVATION_ENDED','unix':time.time(),
        'reason':reason,'scope':'no signals; process exit or bounded observation only'})+'\n'
    assert len(footer.encode()) <= 4096
    stream.write(footer)

"""Bounded resource monitor for one immutable, locally launched diagnostic.

No process signals. On excess or monitoring error requests this run's existing
optimizer-boundary save. The launcher's own outer timeout remains the backstop.
"""
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import time

GIB = 2**30

def stat(pid):
    values = (Path('/proc')/str(pid)/'stat').read_text().rsplit(')', 1)[1].split()
    return {'pid':pid, 'ppid':int(values[1]), 'start_ticks':int(values[19]),
            'rss_bytes':int(values[21])*os.sysconf('SC_PAGE_SIZE')}


def subtree(root, start_ticks):
    if stat(root)['start_ticks'] != start_ticks:
        raise RuntimeError('Owned root process identity changed')
    table={}
    for path in Path('/proc').iterdir():
        if path.name.isdigit():
            try: table[int(path.name)]=stat(int(path.name))
            except (OSError, ValueError, IndexError): continue
    live={root}
    while True:
        more={pid for pid,value in table.items() if value['ppid'] in live}-live
        if not more: break
        live.update(more)
    return live, sum(table[pid]['rss_bytes'] for pid in live if pid in table)


def directory_bytes(path):
    total=0
    for root, _, names in os.walk(path):
        for name in names:
            try: total+=(Path(root)/name).stat().st_size
            except FileNotFoundError: continue
    return total


def request_stop(run, evidence):
    # Do not create/modify any other run or overwrite an operator's stop reason.
    if not run.is_dir():
        return False
    try:
        with (run/'STOP_REQUESTED').open('x') as stream:
            stream.write(json.dumps(evidence,sort_keys=True)+'\n')
    except FileExistsError:
        return True
    return True


def wait_owned(child, out, run, uuids):
    launch=json.loads((out/'LAUNCH.json').read_text())
    assert launch['host']==socket.gethostname() and launch['pid']==child.pid
    assert Path(launch['output']).resolve()==run.resolve() and out.resolve() in run.resolve().parents
    initial=stat(child.pid)['start_ticks']
    started=time.monotonic(); triggered=None
    monitor=out/'RESOURCE_MONITOR.jsonl'
    with monitor.open('x') as stream:
        while child.poll() is None:
            row={'unix':time.time(),'pid':child.pid,'start_ticks':initial}
            violations=[]
            try:
                live,rss=subtree(child.pid,initial)
                raw=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_gpu_memory',
                    '--format=csv,noheader,nounits'],text=True,timeout=5)
                used={uuid:0 for uuid in uuids}
                for line in raw.splitlines():
                    uuid,pid,mib=[part.strip() for part in line.split(',')]
                    if int(pid) in live and uuid in used: used[uuid]+=int(mib)*2**20
                free=shutil.disk_usage(out).free; written=directory_bytes(out)
                logs=sum(p.stat().st_size for p in out.glob('*.log'))+monitor.stat().st_size
                row.update(rss_sum_bytes=rss,owned_gpu_used_bytes=used,free_bytes=free,
                           study_output_bytes=written,log_bytes=logs)
                if rss>256*GIB: violations.append('OWNED_TREE_RSS_OVER_256_GIB')
                if max(used.values(),default=0)>48*GIB: violations.append('OWNED_GPU_USED_OVER_48_GIB')
                if free<1024*GIB: violations.append('SHARED_FREE_BELOW_1_TIB')
                if written>96*GIB: violations.append('STUDY_OUTPUT_OVER_96_GIB')
                if logs>256*2**20: violations.append('LOGS_OVER_256_MIB')
            except Exception as exc:
                if child.poll() is not None: break
                row['monitor_error']=type(exc).__name__+': '+str(exc)
                violations.append('RESOURCE_MONITOR_ERROR')
            if violations and triggered is None:
                triggered={'scope':'owned diagnostic only','reasons':violations,'observed':row,
                    'action':'request optimizer-boundary native save; no process signals',
                    'runtime_source':launch['source_sha']}
            if triggered is not None:
                row['stop_requested']=request_stop(run,triggered)
                row['stop_reasons']=triggered['reasons']
            # A bounded audit stream, even after requesting a safe stop.
            if stream.tell()<16*2**20:
                stream.write(json.dumps(row,sort_keys=True)+'\n');stream.flush()
            time.sleep(5)
        result={'status':'MONITOR_ENDED','exit_code':child.wait(),'elapsed_seconds':time.monotonic()-started,
                'stop_request':triggered,'sample_period_seconds':5,
                'scope':'sampled upper-bound RSS and owned driver GPU use; safe-boundary request, no hard instantaneous memory quota'}
        stream.write(json.dumps(result,sort_keys=True)+'\n');stream.flush()
    return result

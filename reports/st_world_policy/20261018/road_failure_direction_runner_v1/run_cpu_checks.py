"""Bound this implementation's CPU fixtures; no real trajectory/map assets."""
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import time

ROOT=Path(__file__).resolve().parents[4]
OUT=Path(__file__).resolve().parent
COMMAND=['/root/miniconda3/envs/ddp/bin/python','-m','pytest','-q',
         'tests/st_world/test_road_failure_selection.py','tests/st_world/test_run_road_direction.py',
         'tests/st_world/test_road_margin_proxy.py']


def tree(pid):
    result=[];todo=[pid]
    while todo:
        current=todo.pop()
        try:
            stat=Path(f'/proc/{current}/stat').read_text().rsplit(')',1)[1].split()
            rss=int(stat[21])*os.sysconf('SC_PAGE_SIZE')
            cpu=sum(int(stat[i]) for i in (11,12,13,14))/os.sysconf('SC_CLK_TCK')
            result.append((rss,cpu))
            todo.extend(int(p) for p in Path(f'/proc/{current}/task/{current}/children').read_text().split())
        except (FileNotFoundError,ProcessLookupError):pass
    return sum(r[0] for r in result),sum(r[1] for r in result)


def main():
    environment={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1',
                 'OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
    start=time.monotonic();peak=0;why=None;cpu=0.
    with (OUT/'TEST_OUTPUT.txt').open('w') as stream:
        child=subprocess.Popen(COMMAND,cwd=ROOT,env=environment,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        while child.poll() is None:
            rss,cpu=tree(child.pid);peak=max(peak,rss)
            if rss>2*2**30:why='ACTUAL_AGGREGATE_RSS_CAP'
            elif cpu>120:why='ACTUAL_CPU_CAP'
            elif time.monotonic()-start>120:why='WALL_CAP'
            if why:
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=5)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                break
            time.sleep(.05)
        child.wait()
    waited=resource.getrusage(resource.RUSAGE_CHILDREN)
    result={'command':COMMAND,'cwd':str(ROOT),'exit_code':child.returncode,'stop_reason':why,
        'wall_seconds':time.monotonic()-start,'aggregate_sampled_peak_rss_bytes':peak,
        'waited_child_cpu_seconds':waited.ru_utime+waited.ru_stime,
        'limits':{'actual_aggregate_rss_bytes':2*2**30,'cpu_seconds':120,'wall_seconds':120},
        'real_asset_reads':0,'real_official_calls':0,'GPU_calls':0,
        'scope':'synthetic fixtures and original small preparation metadata only; no real map/score/model/image payloads'}
    (OUT/'TEST_EXECUTION.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
    return child.returncode


if __name__=='__main__':raise SystemExit(main())

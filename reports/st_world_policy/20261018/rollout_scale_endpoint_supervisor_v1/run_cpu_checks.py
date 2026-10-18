"""Actual bounded CPU fixtures for the external endpoint controller."""
import json,os,resource,signal,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
OUT=Path(__file__).resolve().parent
COMMAND=['/root/miniconda3/envs/ddp/bin/python','-m','pytest','-q','tests/st_world/test_endpoint_supervisor.py']

def tree(pid):
    rows=[];pending=[pid]
    while pending:
        current=pending.pop()
        try:
            stat=Path(f'/proc/{current}/stat').read_text().rsplit(')',1)[1].split()
            rows.append((int(stat[21])*os.sysconf('SC_PAGE_SIZE'),sum(int(stat[i]) for i in (11,12,13,14))/os.sysconf('SC_CLK_TCK')))
            pending.extend(map(int,Path(f'/proc/{current}/task/{current}/children').read_text().split()))
        except (FileNotFoundError,ProcessLookupError):pass
    return sum(row[0] for row in rows),sum(row[1] for row in rows)

def main():
    env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
    start=time.monotonic();peak=0;why=None
    with (OUT/'TEST_OUTPUT.txt').open('w') as log:
        child=subprocess.Popen(COMMAND,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        while child.poll() is None:
            rss,cpu=tree(child.pid);peak=max(peak,rss)
            size=sum(path.stat().st_size for path in OUT.rglob('*') if path.is_file())
            if rss>2*2**30:why='AGGREGATE_RSS_CAP'
            elif cpu>120:why='CPU_CAP'
            elif size>32*2**20:why='OUTPUT_CAP'
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
        'wall_seconds':time.monotonic()-start,'waited_child_CPU_seconds':waited.ru_utime+waited.ru_stime,
        'sampled_aggregate_peak_RSS_bytes':peak,'limits':{'CPU_seconds':120,'RSS_bytes':2*2**30,'output_bytes':32*2**20},
        'real_checkpoint_images_metric_cache_reads':0,'real_GPU_calls':0,'real_scoring_calls':0,
        'scope':'Synthetic CPU fixtures and small actual frozen source/command metadata only; not deployment equivalence or execution approval'}
    (OUT/'TEST_EXECUTION.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
    return child.returncode
if __name__=='__main__':raise SystemExit(main())

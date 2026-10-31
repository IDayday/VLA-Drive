"""Read committed execution receipts and a bounded, aligned native log window.

No model loading, scoring, signals, resource admission or training mutations.
Historical/current timing ratios are descriptive when cohosts differ.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics


def read(path):
    return json.loads(Path(path).read_text())


def tail(path, size=2*1024**2):
    with path.open('rb') as stream:
        start=max(0,path.stat().st_size-size)
        stream.seek(start)
        if start:stream.readline()
        lines=stream.read().splitlines()
    rows=[]
    for i,line in enumerate(lines):
        try:rows.append(json.loads(line))
        except json.JSONDecodeError:
            if i!=len(lines)-1:raise
    updates=[r['updates']for r in rows]
    if updates!=sorted(set(updates)):raise ValueError('Duplicate or disordered effective updates')
    return {r['updates']:r for r in rows}


def window(run, boundary, count):
    ranks=[tail(run/f'metrics_rank{i}.jsonl')for i in range(8)]
    common=sorted(set.intersection(*(set(r)for r in ranks)))
    selected=[u for u in common if u>boundary+2][-count:]
    if len(selected)!=count or selected!=list(range(selected[0],selected[-1]+1)):
        raise ValueError('Not enough continuous post-handoff updates for the requested window')
    rows=[]
    for u in selected:
        rs=[r[u]for r in ranks]
        for r in rs:
            if r['cursor']!=32*u or r['optimizer_global_steps']!=u or r['optimizer_skipped_steps']!=0:
                raise ValueError('Optimizer/exposure mismatch')
        rows.append(dict(update=u,max_rank_wall=max(r['seconds']for r in rs),
                         max_rank_data=max(r['data_seconds']for r in rs),
                         max_peak_allocated_GiB=max(r['peak_allocated']for r in rs)/2**30))
    seconds=[r['max_rank_wall']for r in rows]
    return dict(first_update=selected[0],last_update=selected[-1],n=count,
        wall_median=statistics.median(seconds),wall_p90=sorted(seconds)[int(.9*(count-1))],
        data_median_max_rank=statistics.median(r['max_rank_data']for r in rows),
        memory_max_GiB=max(r['max_peak_allocated_GiB']for r in rows),raw_per_update=rows)


def remote_worker(job):
    from starVLA.rl.qwen_world_compare.resources import remote
    live=read(job/'RUNNING.json')
    code='''import json,pathlib,sys
p=pathlib.Path('/proc')/sys.argv[1];s=(p/'stat').read_text().split()
assert s[2]not in('Z','X')and int(s[21])==int(sys.argv[2])
c=pathlib.Path('/proc')/sys.argv[3];t=(c/'stat').read_text().split()
assert t[2]not in('Z','X')and int(t[21])==int(sys.argv[4])
children=(c/'task'/sys.argv[3]/'children').read_text().split();r=[]
for child in children:
 q=pathlib.Path('/proc')/child
 try:
  args=(q/'cmdline').read_bytes().decode().split('\\0');state=(q/'stat').read_text().split()[2]
  if 'tools.qwen_world_compare.train_il'in args:r.append(dict(pid=int(child),state=state))
 except FileNotFoundError:pass
assert len(r)==8 and all(x['state']not in('Z','X')for x in r)
print(json.dumps(dict(worker_verified=True,torchrun_verified=True,actual_live_ranks=r)))
'''
    args=[str(live[k])for k in('worker_pid','worker_start_ticks','child_pid','child_start_ticks')]
    return json.loads(remote(live['host'],['/usr/bin/python3','-c',code,*args],timeout=45))


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--performance-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--window',type=int,default=64)
    p.add_argument('--verify-remote',action='store_true');a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    handoff=read(a.performance_root/'migration/HANDOFF_OWNER/COMPLETE.json')
    if handoff['status']!='PASS':raise ValueError('No complete four-arm handoff')
    before=read(a.performance_root/'LIVE64.json');owner=read(a.root/'campaign/OWNER.json')
    s=Path('/proc',str(owner['pid']),'stat').read_text().split()
    if s[2]in('Z','X')or int(s[21])!=owner['start_ticks']:raise ValueError('Steady campaign owner is not live')
    rows={}
    for arch in'ABCD':
        r=handoff['architectures'][arch];job=Path(r['new_job']);run=a.root/'runs'/('ARCH_'+arch)/'IL_seed42'
        marker=read(run/'EXECUTION_HANDOFF_COMPLETE.json')
        if marker!=r['committed']or marker['updates']!=r['old_boundary']+2:
            raise ValueError('Native commit does not match the adopted handoff')
        cp=Path(marker['checkpoint'])
        if not(cp/'COMPLETE').is_file():raise ValueError('Published native handoff checkpoint is absent')
        if(job/'RESULT.json').exists():raise ValueError('Adopted invocation is no longer running')
        current=window(run,r['old_boundary'],a.window)
        rows[arch]=dict(host=r['host'],native_slot=r['lease_slot'],handoff=marker,
            old_boundary=r['old_boundary'],native_checkpoint_sealed=True,job=str(job),
            before=before['rows'][arch],after=current,
            observed_window_ratio=before['rows'][arch]['wall_median']/current['wall_median'])
    if a.verify_remote:
        with ThreadPoolExecutor(max_workers=4)as pool:
            values=list(pool.map(remote_worker,[Path(rows[k]['job'])for k in'ABCD']))
        for arch,value in zip('ABCD',values):rows[arch]['actual_process_check']=value
    result=dict(status='PASS',generated_utc=datetime.now(timezone.utc).isoformat(),rows=rows,
        scientific_source='fdaff809652370328e45c0d996ad706e1af2fc0c',execution_source=handoff['source'],
        steady_owner=owner,optimizer_updates_added_by_analysis=0,scorer_calls=0,
        caveat='Different windows, data and GPU cohosts: observed ratios are not a controlled causal speedup. Step timers exclude checkpoint saves and initialization. Actual eight-rank state equivalence is a separate qualification.')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:dict(last=v['after']['last_update'],old_s=v['before']['wall_median'],new_s=v['after']['wall_median'],ratio=v['observed_window_ratio'])for k,v in rows.items()}))


if __name__=='__main__':main()

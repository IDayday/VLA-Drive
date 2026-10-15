"""Read-only provenance audit for overlapping timestamps and duplicate images.

All raw frames of each selected recording fragment are inspected, including
future teacher frames. Candidate image duplicates across roles are confirmed
by file SHA256 (only equal-size files can be duplicates). No image is exported.
"""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import gzip
import json
import multiprocessing
from pathlib import Path
import pickle
import resource
import sys
import time
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.decision_local.common import read,atomic_json,digest,check_host,source_sha
from tools.decision_local.audit_base_training_sources import recording


def scan(task):
    log,role,root,sensors=task
    cpu=time.process_time()
    with (Path(root)/(log+'.pkl')).open('rb') as stream:frames=pickle.load(stream)
    images=[];missing=[];times=[]
    for frame in frames:
        times.append((frame['vehicle_name'],int(frame['timestamp']),frame['token']))
        for camera in ('CAM_F0','CAM_L0','CAM_R0'):
            relative=frame['cams'][camera]['data_path'];path=Path(sensors)/relative
            try:stat=path.stat();images.append((relative,stat.st_size,stat.st_ino,stat.st_dev))
            except FileNotFoundError:missing.append(relative)
    return dict(log=log,role=role,times=times,images=images,missing=missing,cpu=time.process_time()-cpu)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('source-audit','raw-logs','sensors','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--workers',type=int,default=8);p.add_argument('--hash-workers',type=int,default=32)
    p.add_argument('--max-seconds',type=float,default=1800)
    a=p.parse_args();check_host();start=time.monotonic();cpu=time.process_time();before=resource.getrusage(resource.RUSAGE_CHILDREN)
    source=read(a.source_audit);roles=source['roles'];tasks=[]
    for log in sorted(p.stem for p in a.raw_logs.glob('*.pkl')):
        r=recording(log)
        if r in roles:tasks.append((log,roles[r],str(a.raw_logs),str(a.sensors)))
    a.output.mkdir(parents=True,exist_ok=True)
    atomic_json(a.output/'COMMAND.json',dict(argv=sys.argv,source_sha=source_sha(ROOT),source_audit_sha256=digest(a.source_audit)))
    images=defaultdict(list);time_roles=defaultdict(set);token_roles=defaultdict(set);missing=[];worker_cpu=0.
    with ProcessPoolExecutor(a.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        for i,result in enumerate(pool.map(scan,tasks,chunksize=2)):
            if time.monotonic()-start>a.max_seconds:raise TimeoutError('Finite provenance audit budget')
            role=result['role'];log=result['log'];worker_cpu+=result['cpu'];missing+=result['missing']
            for car,timestamp,token in result['times']:
                time_roles[(car,timestamp)].add((role,log));token_roles[token].add((role,log))
            for relative,size,inode,dev in result['images']:images[size].append((role,log,relative,inode,dev))
            if i%50==0:atomic_json(a.output/'STATUS.json',dict(logs=i,total_logs=len(tasks),stage='metadata',seconds=time.monotonic()-start))
    conflicts=lambda values:[sorted(v) for v in values if len({x[0] for x in v})>1]
    time_conflicts=conflicts(time_roles.values());token_conflicts=conflicts(token_roles.values())
    candidates=[x for values in images.values() if len({x[0] for x in values})>1 for x in values]
    image_roles=defaultdict(set);bytes_hashed=0
    def hashed(item):
        role,log,path,_,_=item
        return digest(a.sensors/path),role,log,path
    # Bound the number of queued futures to chunks, not millions of images.
    # Atomic chunk receipts make an interrupted finite audit recoverable.
    bank=a.output/'hash_chunks';bank.mkdir(exist_ok=True)
    def hash_chunk(task):
        at,items=task;path=bank/f'{at:06d}.json'
        key=__import__('hashlib').sha256(json.dumps(items,sort_keys=True).encode()).hexdigest()
        if path.exists():
            saved=read(path)
            if saved['candidate_hash']!=key:raise ValueError('Changed duplicate audit chunk')
            return saved['rows']
        rows=[hashed(item) for item in items]
        atomic_json(path,dict(candidate_hash=key,rows=rows))
        return rows
    chunks=((i//512,candidates[i:i+512]) for i in range(0,len(candidates),512))
    completed=0;timed_out=False
    with ThreadPoolExecutor(a.hash_workers) as pool:
        # At most two windows in flight. Do not enqueue work past the deadline.
        from concurrent.futures import wait,FIRST_COMPLETED
        pending=set()
        def fill():
            while len(pending)<2*a.hash_workers and time.monotonic()-start<a.max_seconds:
                item=next(chunks,None)
                if item is None:break
                pending.add(pool.submit(hash_chunk,item))
        fill()
        while pending:
            ready,pending=wait(pending,return_when=FIRST_COMPLETED)
            for future in ready:
                rows=future.result();completed+=len(rows)
                for sha,role,log,path in rows:image_roles[sha].add((role,log,path))
            atomic_json(a.output/'STATUS.json',dict(images=completed,candidates=len(candidates),stage='SHA256',seconds=time.monotonic()-start))
            fill()
        timed_out=completed!=len(candidates)
    image_conflicts=conflicts(image_roles.values())
    after=resource.getrusage(resource.RUSAGE_CHILDREN)
    report=dict(schema='recording_frame_provenance_audit_v1',source_sha=source_sha(ROOT),source_audit_identity=source['identity'],
        source_audit_sha256=digest(a.source_audit),raw_logs=len(tasks),raw_frames=len(token_roles),
        all_image_references=sum(len(v) for v in images.values()),hashed_equal_size_cross_role_candidates=len(candidates),
        exact_timestamp_cross_role_conflicts=time_conflicts,exact_token_cross_role_conflicts=token_conflicts,
        exact_image_content_cross_role_conflicts=image_conflicts,missing_images=missing,
        population_ready=not (time_conflicts or token_conflicts or image_conflicts or missing or timed_out),
        hash_complete=not timed_out,hashed_images_this_invocation=completed,
        CPU_parent_seconds=time.process_time()-cpu,CPU_waited_children_seconds=after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime,
        worker_CPU_seconds=worker_cpu,wall_seconds=time.monotonic()-start,
        image_scope='All three forward camera frames in selected parent recordings; equal-size pruning is exact for byte duplicate detection',
        limitation='Exact duplicate audit is not a claim that visually similar nearby roads are independent')
    atomic_json(a.output/('INCOMPLETE.json' if timed_out else 'COMPLETE.json'),report)
    print(json.dumps({k:v for k,v in report.items() if 'conflicts' not in k and k!='missing_images'}),flush=True)
    if timed_out:raise SystemExit(1)


if __name__=='__main__':main()

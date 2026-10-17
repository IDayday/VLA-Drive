"""Read-only 32-scene future/current metadata reuse diagnostic, not training IO.

Keeps MetadataIndex's complete startup hashes and per-record checksum validation.
No image, tensor, raw log, source cache or ledger is written or opened as payload.
Only immutable <=1 MiB PLAN/RESULT controls are produced under caller report root.
"""
import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import time

from starVLA.dataloader import st_world_metadata_index as mi
from starVLA.dataloader.st_world_future_provenance import LogCoverage, validate_ledger
from starVLA.dataloader.st_world_schema import TeacherFeatureContract, identity_hash
from starVLA.dataloader.st_world_targets import payload_key, validate_target_manifest

MIB=1 << 20
LIMITS=dict(max_metadata_read_bytes=1 << 30,max_rss_bytes=2 << 30,max_seconds=180,
            cpu_threads=2,max_sqlite_cache_bytes=16*MIB,max_output_bytes=MIB)
TIMES=[0.,.5,1.,2.,4.]


def small(path, maximum=MIB):
    path=Path(path)
    with path.open('rb') as stream:
        before=os.fstat(stream.fileno());raw=stream.read(maximum+1);after=os.fstat(stream.fileno())
    stamp=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
    if len(raw)>maximum or stamp(before)!=stamp(after) or stamp(after)!=stamp(path.stat()):
        raise ValueError('Small diagnostic control changed/oversized')
    return json.loads(raw),hashlib.sha256(raw).hexdigest()


def immutable(path,value):
    raw=(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
    if len(raw)>MIB:raise ValueError('Diagnostic output exceeds 1 MiB')
    path=Path(path)
    if path.exists():
        if path.read_bytes()!=raw:raise ValueError('Immutable report conflict')
        return
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as stream:stream.write(raw)


def seal(doc):
    return {**doc,'identity':identity_hash(doc)}


def source_pins():
    return {str(Path(p).resolve()):hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in
        (__file__,mi.__file__,Path(mi.__file__).with_name('st_world_targets.py'),
         Path(mi.__file__).with_name('st_world_future_provenance.py'))}


def plan(future_root,current_index):
    f=Path(future_root).resolve();i=Path(current_index).resolve()
    complete,complete_sha=small(f/'COMPLETE.json')
    index,index_sha=small(i/'COMPLETE.json')
    request,request_sha=small(f/'TEACHER_REQUESTS.json')
    validate_target_manifest(request)
    if (complete.get('schema')!='st_future_metadata_complete_v1' or complete.get('status')!='COMPLETE'
            or complete.get('scenes')!=32 or complete.get('logs')!=2 or complete.get('scope')!='qualification'
            or complete.get('request_identity')!=request['identity']
            or complete['files'].get('TEACHER_REQUESTS.json')!=request_sha
            or request['times_s']!=TIMES or len(request['scenes'])!=32
            or len({s['log'] for s in request['scenes']})!=2):
        raise ValueError('Only the completed fixed future32/2log diagnostic is permitted')
    if index.get('status')!='COMPLETE' or index.get('cache_header') is None:
        raise ValueError('A COMPLETE current-cache metadata index is required')
    contract=TeacherFeatureContract.from_dict(index['contract'])
    refs=[ref for s in request['scenes'] for frame in s['frames'] for ref in frame['views'] if ref is not None]
    unique=len({payload_key(ref,contract) for ref in refs})
    controls={str(f/'COMPLETE.json'):complete_sha,str(i/'COMPLETE.json'):index_sha}
    for name,sha in complete['files'].items():
        if Path(name).name!=name:raise ValueError('Unsafe future completion control')
        doc,actual=small(f/name)
        if actual!=sha:raise ValueError('Future completed control changed')
        controls[str(f/name)]=sha
    coverage=LogCoverage()
    for s in request['scenes']:coverage.add(s['token'],s['log'])
    validate_ledger(request,coverage=coverage.records())
    paths=[i/'metadata.sqlite']+[Path(d['path']) for d in index['dependencies'].values()]
    startup=sum(p.stat().st_size for p in paths)+3*(i/'COMPLETE.json').stat().st_size
    # Conservative allowance, not predicted/observed IO: one MiB per distinct
    # lookup plus 16 MiB controls/proc/import slack. Runtime counts actual rchar.
    estimate=startup+unique*MIB+16*MIB+sum(Path(p).stat().st_size for p in controls)
    if estimate>LIMITS['max_metadata_read_bytes']:raise ValueError('BLOCKED: predeclared read envelope exceeds 1 GiB')
    return seal(dict(schema='st_future_current_reuse_plan_v1',future_root=str(f),current_index=str(i),
        controls=controls,index_identity=index['identity'],current_request_identity=index['request_identity'],
        future_request_identity=request['identity'],teacher_contract_identity=contract.identity,
        source_pins=source_pins(),source_base_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        limits=LIMITS,scene_count=32,log_count=2,view_time_slots=480,distinct_present_keys=unique,
        metadata_only_startup_hash_bytes=startup,conservative_total_read_envelope_bytes=estimate,
        current_population=dict(scenes=index['scene_count'],records=index['record_count']),
        no_image_tensor_rawlog_reads=True,no_teacher_extraction=True,no_ledger_mutation=True,
        limitation='Fixed32/2logs only; metadata membership is not tensor availability, teacher throughput or full-population hit rate'))


def proc_io():
    with open('/proc/self/io') as stream:rows=dict(line.strip().split(': ') for line in stream)
    return {k:int(rows[k]) for k in ('rchar','read_bytes')}


class Budget:
    def __init__(self):
        self.started=time.monotonic();self.initial=proc_io();self.checks=0
    def stats(self):
        now=proc_io()
        return dict(elapsed_s=time.monotonic()-self.started,
            logical_read_bytes=now['rchar']-self.initial['rchar'],
            physical_read_bytes=now['read_bytes']-self.initial['read_bytes'],
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            checks=self.checks)
    def check(self,reserve=0):
        self.checks+=1;s=self.stats()
        if (s['elapsed_s']>LIMITS['max_seconds'] or s['peak_rss_bytes']>LIMITS['max_rss_bytes']
                or s['logical_read_bytes']+reserve>LIMITS['max_metadata_read_bytes']):
            raise RuntimeError('BLOCKED_RESOURCE: diagnostic wall/RSS/metadata read limit')
        return s


@contextmanager
def read_guard(plan,budget):
    allowed=set(plan['controls'])
    d,_=small(Path(plan['current_index'])/'COMPLETE.json')
    allowed.update(ref['path'] for ref in d['dependencies'].values())
    allowed.add(str(Path(plan['current_index'])/'metadata.sqlite'))
    original_open=Path.open;original_check=mi._storage_check;original_sha=mi.checked_sha
    hashes=[];opened=set()
    def checked_open(path,*args,**kwargs):
        name=str(path.resolve())
        mode=args[0] if args else kwargs.get('mode','r')
        if name not in allowed or any(x in mode for x in 'wa+'):
            raise ValueError('Read-only metadata allowlist rejected: '+name)
        opened.add(name)
        return original_open(path,*args,**kwargs)
    def check(*args,**kwargs):
        budget.check(MIB);return original_check(*args,**kwargs)
    def checked_sha(path):
        start=time.perf_counter();result=original_sha(path)
        hashes.append(dict(path=str(path),bytes=Path(path).stat().st_size,seconds=time.perf_counter()-start,sha256=result[0]))
        return result
    Path.open=checked_open;mi._storage_check=check;mi.checked_sha=checked_sha
    try:yield hashes,opened
    finally:Path.open=original_open;mi._storage_check=original_check;mi.checked_sha=original_sha


def analyze(manifest,index,contract,check):
    """Each distinct key uses original indexed row checksum validation once."""
    for key in ('split','source_student_identity','source_index_sha256','partition_sha256'):
        if manifest[key]!=index.manifest[key]:raise ValueError('Future/current source contract differs: '+key)
    if tuple(manifest['view_names'])!=contract.view_names:raise ValueError('View contract differs')
    by_time={str(t):dict(slots=0,present=0,missing=0,current_cache_hit=0,current_cache_miss=0) for t in TIMES}
    keys={};image_shas=set();query_s=0.;query_count=0
    for scene in manifest['scenes']:
        for frame in scene['frames']:
            stats=by_time[str(frame['requested_s'])]
            for ref in frame['views']:
                stats['slots']+=1
                if ref is None:stats['missing']+=1;continue
                stats['present']+=1;image_shas.add(ref['sha256'])
                key=payload_key(ref,contract)
                if key not in keys:
                    check(MIB);start=time.perf_counter()
                    try:
                        row=index.row('records','key',key)
                    except KeyError:keys[key]=False
                    else:
                        if row['key']!=key or row['image']['sha256']!=ref['sha256'] or payload_key(row['image'],contract)!=key:
                            raise ValueError('Indexed payload key/source corruption')
                        keys[key]=True
                    query_s+=time.perf_counter()-start;query_count+=1;check()
                stats['current_cache_hit' if keys[key] else 'current_cache_miss']+=1
    total={name:sum(row[name] for row in by_time.values()) for name in next(iter(by_time.values()))}
    return dict(by_time=by_time,total=total,distinct_present_images=len(image_shas),distinct_payload_keys=len(keys),
        distinct_hit_keys=sum(keys.values()),distinct_miss_keys=len(keys)-sum(keys.values()),
        repeated_present_slots=total['present']-len(keys),sqlite_queries=query_count,sqlite_query_seconds=query_s,
        hit_definition='image SHA256 + complete teacher contract; original checksummed indexed record exists')


def run(p):
    budget=Budget();start=time.perf_counter()
    if p.get('schema')!='st_future_current_reuse_plan_v1' or p['limits']!=LIMITS or p['identity']!=identity_hash({k:v for k,v in p.items() if k!='identity'}):
        raise ValueError('Diagnostic PLAN changed')
    if p['source_pins']!=source_pins():raise ValueError('Diagnostic source changed after PLAN')
    with read_guard(p,budget) as (hashes,opened):
        controls={}
        for path,expected in p['controls'].items():
            controls[path],actual=small(path)
            if actual!=expected:raise ValueError('Pinned diagnostic control changed')
        manifest=validate_target_manifest(controls[str(Path(p['future_root'])/'TEACHER_REQUESTS.json')])
        coverage=LogCoverage()
        for s in manifest['scenes']:coverage.add(s['token'],s['log'])
        validate_ledger(manifest,coverage=coverage.records())
        if manifest['identity']!=p['future_request_identity']:raise ValueError('Future identity changed')
        startup=time.perf_counter()
        with mi.MetadataIndex(p['current_index'],expected_request_identity=p['current_request_identity'],
            expected_index_identity=p['index_identity'],max_metadata_cache_bytes=LIMITS['max_sqlite_cache_bytes']) as index:
            startup_s=time.perf_counter()-startup
            contract=TeacherFeatureContract.from_dict(index.receipt['contract'])
            if contract.identity!=p['teacher_contract_identity']:raise ValueError('Teacher contract changed')
            result=analyze(manifest,index,contract,budget.check)
            index.check();budget.check()
    return dict(schema='st_future_current_reuse_result_v1',status='COMPLETE_METADATA_DIAGNOSTIC',plan_identity=p['identity'],
        **result,index_startup_seconds=startup_s,startup_hash_calls=hashes,
        startup_hash_seconds=sum(h['seconds'] for h in hashes),startup_hashed_bytes=sum(h['bytes'] for h in hashes),
        runtime=budget.stats(),metadata_opened_paths=sorted(opened),wall_seconds=time.perf_counter()-start,
        current_population=p['current_population'],future_population=dict(scenes=32,logs=2,slots=480),
        tensor_files_opened=0,image_files_opened=0,raw_logs_opened=0,tensor_disk_bytes=0,
        scientific_result=False,training_admission=False,limitation=p['limitation'])


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    q=sub.add_parser('plan');q.add_argument('--future-root',required=True);q.add_argument('--current-index',required=True);q.add_argument('--output',required=True)
    q=sub.add_parser('run');q.add_argument('--plan',required=True);q.add_argument('--plan-sha256',required=True);q.add_argument('--output',required=True)
    a=parser.parse_args()
    def timeout(*_):raise TimeoutError('BLOCKED_RESOURCE: 175 second internal wall stop')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(175)
    try:
        if a.command=='plan':result=plan(a.future_root,a.current_index)
        else:
            p,sha=small(a.plan)
            if sha!=a.plan_sha256:raise ValueError('PLAN raw SHA mismatch')
            result=run(p)
        immutable(a.output,result)
        print(json.dumps({'status':result.get('status','PLANNED_READ_ONLY'),'output':a.output,'identity':result.get('identity')}))
        return 0
    except Exception as exc:
        result=dict(status='BLOCKED_METADATA_DIAGNOSTIC',error=repr(exc),training_admission=False,tensor_disk_bytes=0)
        immutable(a.output,result);print(json.dumps(result));return 2
    finally:signal.alarm(0)


if __name__=='__main__':raise SystemExit(main())

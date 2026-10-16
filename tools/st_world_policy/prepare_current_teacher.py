"""Registered all-current teacher metadata, without history/GT decoding or GPU work.

Preserves the original train/dev index order. Two bounded CPU workers hash real
t0 image bytes and original current/ego records into recoverable per-log shards.
This prepares metadata only; cache computation remains a separate authorization.
"""
import argparse
from collections import defaultdict,deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sys
import threading
import time

from PIL import Image

from starVLA.dataloader.st_world_history import file_sha256,VIEWS
from starVLA.dataloader.st_world_schema import identity_hash
from starVLA.dataloader.st_world_targets import atomic_json,file_lock,validate_target_manifest


class ReadBudget:
    def __init__(self,mib_per_second,max_seconds):
        self.started=time.monotonic();self.deadline=self.started+max_seconds
        self.rate=mib_per_second*1024**2;self.next_read=self.started
        self.bytes=0;self.lock=threading.Lock()

    def check(self):
        if time.monotonic()>=self.deadline:raise TimeoutError('Metadata preparation reached its registered time budget')

    def consume(self,n):
        self.check()
        with self.lock:
            now=time.monotonic();scheduled=max(self.next_read,now)
            self.next_read=scheduled+n/self.rate;self.bytes+=n
        delay=scheduled-time.monotonic()
        if delay>0:time.sleep(delay)
        self.check()


def bounded_sha(path,budget):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        while chunk:=f.read(1024**2):budget.consume(len(chunk));h.update(chunk)
    return h.hexdigest()


def current_image_reference(path,budget):
    path=Path(path).resolve();before=path.stat();digest=bounded_sha(path,budget)
    with Image.open(path) as image:wh=list(image.size)
    after=path.stat()
    if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):
        raise ValueError('Current image changed during metadata hashing')
    if wh[0]<1 or wh[1]<1 or abs(wh[0]/wh[1]-16/9)>.001:
        raise ValueError('Current image is outside the original complete 16:9 field')
    return {'path':str(path),'sha256':digest,'bytes':after.st_size,'mtime_ns':after.st_mtime_ns,'original_wh':wh}


def source_population(student_root,partition_path,split):
    if split not in ('train','dev'):raise ValueError('Only original train/dev roles are allowed')
    root=Path(student_root).resolve();partition_path=Path(partition_path).resolve()
    source=json.loads((root/'identity.json').read_text());index=json.loads((root/'index.json').read_text())
    partition=json.loads(partition_path.read_text())
    if (source['schema']!='ddpolicy_current_cameras_v1' or source['split']!=split or
            source['identity']!=identity_hash({k:v for k,v in source.items() if k!='identity'}) or
            source['index_sha256']!=identity_hash(index) or source['partition_sha256']!=file_sha256(partition_path)):
        raise ValueError('Original source population/partition/role identity mismatch')
    tokens=[r['token'] for r in index]
    if len(tokens)!=len(set(tokens)) or tokens!=partition[split+'_tokens']:
        raise ValueError('Source population is not the exact original ordered role')
    if any(partition['token_logs'][r['token']]!=r['log'] for r in index):raise ValueError('Source log mapping changed')
    a={partition['token_logs'][t] for t in partition['train_tokens']}
    b={partition['token_logs'][t] for t in partition['dev_tokens']}
    if a&b:raise ValueError('Train/dev source-log leakage')
    ego=json.loads((root/'ego_identity.json').read_text())
    if (ego['schema']!='foresight_ego_labels_v1' or ego['index_sha256']!=source['index_sha256'] or
            ego['identity']!=identity_hash({k:v for k,v in ego.items() if k!='identity'})):
        raise ValueError('Original ego target identity mismatch')
    identity={'student_root':str(root),'source_identity':source['identity'],
        'source_index_identity':source['index_sha256'],'source_index_file_sha256':file_sha256(root/'index.json'),
        'partition_path':str(partition_path),'partition_sha256':source['partition_sha256'],
        'ego_identity_sha256':file_sha256(root/'ego_identity.json'),'split':split,
        'scenes':len(index),'logs':len({r['log'] for r in index}),'ordered_scene_ids_hash':identity_hash(tokens)}
    return source,index,identity


def plan(args):
    _,index,source=source_population(args.student_root,args.partition,args.split)
    if not 1<=args.workers<=2 or not 0<args.read_mib_per_second<=128 or not 0<args.max_seconds<=14400:
        raise ValueError('Metadata requires workers<=2, read bandwidth<=128MiB/s and wall<=4h')
    registration={'schema':'st_current_population_registration_v1','phase':'data_preparation',
        'source':source,'max_scenes':len(index),'target_times_s':[0.],
        'workers':args.workers,'read_mib_per_second':args.read_mib_per_second,'max_seconds':args.max_seconds,
        'writer_sha256':file_sha256(__file__),'created_utc':datetime.now(timezone.utc).isoformat(),
        'scope':'all_original_current_observations','raw_history_read':False,'future_read':False,
        'gt_values_decoded':False,'gpu_work_authorized':False,'formal_training_approved':False}
    registration['identity']=identity_hash(registration)
    out=Path(args.output)
    with file_lock(out/'PREPARATION.lock',blocking=False):
        path=out/'REGISTRATION.json'
        if path.exists():
            old=json.loads(path.read_text())
            comparable=lambda x:{k:v for k,v in x.items() if k not in ('created_utc','identity')}
            if comparable(old)!=comparable(registration):raise ValueError('Population preparation already registered with another identity')
            registration=old
        atomic_json(path,registration,immutable=True)
        population={'schema':'st_unpinned_source_population_v1','source':source,'scenes':index,
                    'purpose':'Metadata preparation only; not a pinned training manifest'}
        atomic_json(out/'SOURCE_POPULATION.json',population,immutable=True)
    return registration


def build_log(root,source,log,rows,registration,output,budget):
    budget.check();name=identity_hash({'log':log})+'.json';path=output/'log_shards'/name
    if path.exists():
        shard=json.loads(path.read_text())
        if (shard['identity']!=identity_hash({k:v for k,v in shard.items() if k!='identity'}) or
                shard['registration_identity']!=registration['identity'] or shard['log']!=log or
                [r['token'] for r in shard['selected']]!=[r['token'] for r in rows]):
            raise ValueError('Stale/conflicting metadata shard')
        for row,scene in zip(shard['selected'],shard['scenes']):
            for key,folder,ext in [('current_record_sha256','current','.json'),('ego_record_sha256','ego','.pt')]:
                if bounded_sha(root/folder/(row['token']+ext),budget)!=row[key]:raise ValueError('Previously pinned source record changed')
            for ref in scene['frames'][0]['views']:
                stat=Path(ref['path']).stat()
                if (stat.st_size,stat.st_mtime_ns)!=(ref['bytes'],ref['mtime_ns']):
                    raise ValueError('Previously hashed image version changed; preserve shard and create a new registration')
        return path,True
    selected=[];scenes=[]
    for row in rows:
        budget.check();token=row['token'];current=root/'current'/(token+'.json')
        data=current.read_bytes();budget.consume(len(data));record=json.loads(data)
        if record['identity']!=source['identity'] or record['token']!=token or record['log']!=log:
            raise ValueError('Foreign current record in fixed source role')
        if len(record['image_paths'])!=len(VIEWS) or type(record['timestamp']) is not int:
            raise ValueError('Current view count or decision timestamp is illegal')
        refs=[current_image_reference(p,budget) for p in record['image_paths']]
        pinned=dict(row,current_record_sha256=hashlib.sha256(data).hexdigest(),
                    ego_record_sha256=bounded_sha(root/'ego'/(token+'.pt'),budget))
        selected.append(pinned)
        scenes.append({'token':token,'log':log,'split':registration['source']['split'],
            'timestamp_us':record['timestamp'],'frames':[{'requested_s':0.,'timestamp_us':record['timestamp'],
            'frame_token':token,'views':refs}]})
    shard={'schema':'st_current_metadata_log_shard_v1','registration_identity':registration['identity'],
           'log':log,'selected':selected,'scenes':scenes}
    shard['identity']=identity_hash(shard)
    atomic_json(path,shard,immutable=True)
    return path,False


def build(args):
    output=Path(args.output);reg=json.loads((output/'REGISTRATION.json').read_text())
    if (reg.get('schema')!='st_current_population_registration_v1' or
            reg['identity']!=identity_hash({k:v for k,v in reg.items() if k!='identity'}) or
            reg['writer_sha256']!=file_sha256(__file__) or reg['gpu_work_authorized'] is not False):
        raise ValueError('Metadata registration or writer changed')
    source,index,actual=source_population(args.student_root,args.partition,args.split)
    if actual!=reg['source']:raise ValueError('Build source differs from registered complete population')
    groups=defaultdict(list)
    for row in index:groups[row['log']].append(row)
    budget=ReadBudget(reg['read_mib_per_second'],reg['max_seconds']);completed=[];reused=0
    with file_lock(output/'PREPARATION.lock',blocking=False):
        if (output/'COMPLETE.json').exists():
            done=json.loads((output/'COMPLETE.json').read_text())
            if (done['registration_identity']!=reg['identity'] or
                    done['selected_sha256']!=file_sha256(output/'SELECTED_SCENES.json') or
                    done['requests_sha256']!=file_sha256(output/'TEACHER_REQUESTS.json')):
                raise ValueError('Completed population artifacts changed')
            return {**done,'reused':True}
        with ThreadPoolExecutor(max_workers=reg['workers']) as pool:
            iterator=iter(groups.items());pending=deque()
            def submit():
                log,rows=next(iterator)
                return pool.submit(build_log,Path(args.student_root),source,log,rows,reg,output,budget)
            for _ in range(min(reg['workers'],len(groups))):pending.append(submit())
            try:
                while pending:
                    path,old=pending.popleft().result();completed.append(path);reused+=int(old)
                    atomic_json(output/'PROGRESS.json',{'status':'PREPARING','registration_identity':reg['identity'],
                        'logs_complete':len(completed),'logs_total':len(groups),'logs_reused':reused,
                        'source_bytes_hashed_this_invocation':budget.bytes,'elapsed_s':time.monotonic()-budget.started})
                    try:pending.append(submit())
                    except StopIteration:pass
            except TimeoutError:
                return {'status':'PARTIAL','reason':'TIME_BUDGET','registration_identity':reg['identity'],
                        'complete_shards_preserved':True}
        rows={};scenes={};receipts=[]
        for path in completed:
            shard=json.loads(path.read_text());rows.update({r['token']:r for r in shard['selected']})
            scenes.update({r['token']:r for r in shard['scenes']})
            receipts.append({'path':str(path.relative_to(output)),'sha256':file_sha256(path),'identity':shard['identity']})
        if set(rows)!={r['token'] for r in index} or set(scenes)!=set(rows):raise ValueError('Incomplete full population')
        selected=[rows[r['token']] for r in index]
        manifest={'schema':'st_fixed_scene_list_v1','split':args.split,'seed':None,
            'source_index_hash':source['index_sha256'],'source_student_identity':source['identity'],
            'ego_identity_sha256':reg['source']['ego_identity_sha256'],'registration_identity':reg['identity'],
            'selection_rule':'entire_original_index_in_original_order','scenes':selected}
        requests={'schema':'st_teacher_requests_v1','split':args.split,'source_student_identity':source['identity'],
            'source_index_sha256':source['index_sha256'],'partition_sha256':source['partition_sha256'],
            'selection_identity':identity_hash(selected),'view_names':list(VIEWS),'timestamp_tolerance_s':0.,
            'times_s':[0.],'source':'original_current_json_exact_t0','registration_identity':reg['identity'],
            'scenes':[scenes[r['token']] for r in index]}
        requests['identity']=identity_hash(requests);validate_target_manifest(requests)
        atomic_json(output/'SELECTED_SCENES.json',manifest,immutable=True)
        atomic_json(output/'TEACHER_REQUESTS.json',requests,immutable=True)
        atomic_json(output/'SHARD_RECEIPTS.json',{'registration_identity':reg['identity'],'shards':receipts},immutable=True)
        image_hashes={r['sha256']:r['bytes'] for s in requests['scenes'] for r in s['frames'][0]['views']}
        done={'status':'COMPLETE','scope':'current_only_metadata','registration_identity':reg['identity'],
            'scenes':len(selected),'logs':len(groups),'unique_image_contents':len(image_hashes),
            'unique_source_image_bytes':sum(image_hashes.values()),'selected_sha256':file_sha256(output/'SELECTED_SCENES.json'),
            'requests_sha256':file_sha256(output/'TEACHER_REQUESTS.json'),'request_identity':requests['identity'],
            'source_bytes_hashed_this_invocation':budget.bytes,'elapsed_s':time.monotonic()-budget.started,
            'gt_values_decoded':False,'raw_history_read':False,'future_read':False,'gpu_used':False,
            'teacher_cache_built':False,'formal_training_approved':False}
        atomic_json(output/'COMPLETE.json',done,immutable=True)
        return done


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    commands=p.add_subparsers(dest='command',required=True)
    for name in ['plan','build']:
        child=commands.add_parser(name)
        for field in ['student-root','partition','output']:child.add_argument('--'+field,required=True)
        child.add_argument('--split',choices=['train','dev'],required=True)
        if name=='plan':
            child.add_argument('--workers',type=int,default=2)
            child.add_argument('--read-mib-per-second',type=float,default=64.)
            child.add_argument('--max-seconds',type=float,default=7200.)
    return p


def main():
    try:
        a=parser().parse_args();result=plan(a) if a.command=='plan' else build(a)
        print(json.dumps(result,sort_keys=True));return 3 if result.get('status')=='PARTIAL' else 0
    except Exception as exc:
        print(json.dumps({'status':'FAILED','error':repr(exc)}),file=sys.stderr);return 2


if __name__=='__main__':raise SystemExit(main())

"""Bounded nonrecycled CPU Dataset/PCA benchmark; no model/optimizer/GPU work.

Optional ranks partition the fixed manifest by stride, never repeat observations.
A coordinator must aggregate real concurrent wall time, not multiply one rate.
"""
import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import resource
import time

import numpy as np
import torch

from starVLA.dataloader.st_world_dataset import STWorldTrainingDataset,projected_teacher_contract
from starVLA.dataloader.st_world_schema import TeacherFeatureContract,identity_hash
from starVLA.dataloader.st_world_history import file_sha256
from starVLA.dataloader.st_world_targets import atomic_json


def snapshot():
    usage=resource.getrusage(resource.RUSAGE_SELF)
    io={}
    path=Path('/proc/self/io')
    if path.is_file():
        io={key:int(value) for key,value in (line.split(':',1) for line in path.read_text().splitlines())}
    return {'time_ns':time.perf_counter_ns(),'utime_s':usage.ru_utime,'stime_s':usage.ru_stime,
            'maxrss_kib':usage.ru_maxrss,'process_io':io}


def interval(before,after):
    return {'start_ns':before['time_ns'],'end_ns':after['time_ns'],
        'elapsed_s':(after['time_ns']-before['time_ns'])/1e9,
        'cpu_utime_s':after['utime_s']-before['utime_s'],'cpu_stime_s':after['stime_s']-before['stime_s'],
        'maxrss_kib':after['maxrss_kib'],'process_io_start':before['process_io'],'process_io_end':after['process_io'],
        'process_io_delta':{key:after['process_io'][key]-value for key,value in before['process_io'].items()}}


def run(a):
    init_start=snapshot()
    if (not 0<=a.shard_index<a.shard_count or a.shard_count>8 or not 1<=a.workers<=2 or
            not a.workers<=a.prefetch_records<=4 or a.max_scenes<1):
        raise ValueError('Unregistered/unbounded CPU IO benchmark')
    torch.set_num_threads(1)
    cache=Path(a.teacher_root);pca=Path(a.pca_root);out=Path(a.output)
    ident=json.loads((cache/'identity.json').read_text());p=json.loads((pca/'PCA.json').read_text())
    selection=json.loads(Path(a.manifest).read_text())
    if len(selection['scenes'])>a.max_scenes:raise ValueError('Fixed manifest exceeds benchmark sample budget')
    scope=json.loads((pca/'CHANNEL_SCOPE.json').read_text())
    raw=TeacherFeatureContract.from_dict(ident['contract'])
    target=projected_teacher_contract(raw,pca_identity=p['identity'],dimension=384)
    ds=STWorldTrainingDataset(a.data_root,a.manifest,cache,ident['identity'],target,pca_path=pca,need_future=False)
    indices=list(range(a.shard_index,len(ds),a.shard_count))
    if not indices:raise ValueError('Empty CPU benchmark shard')
    registration={'source_sha256':file_sha256(__file__),'manifest_sha256':file_sha256(a.manifest),
        'cache_identity':ident['identity'],'pca_identity':p['identity'],'scope_identity':scope['identity'],
        'dataset_identity':ds.identity['identity'],'scenes':len(ds),'cycles':1,'workers':a.workers,
        'prefetch_records':a.prefetch_records,'torch_threads':1,'shard_count':a.shard_count,
        'shard_index':a.shard_index,'selected_indices':indices,'command':vars(a),
        'cache_state':'May be OS warm; no physical cold-read claim','gpu_used':False}
    atomic_json(out/'REGISTRATION.json',registration,immutable=True)
    init_cost=interval(init_start,snapshot())
    def consume(fn):
        start=snapshot();times=[];tokens=[]
        def timed(i):
            t=time.monotonic();token=fn(i);return token,time.monotonic()-t
        with ThreadPoolExecutor(max_workers=a.workers) as pool:
            iterator=iter(indices);pending=deque()
            for _ in range(min(a.prefetch_records,len(indices))):pending.append(pool.submit(timed,next(iterator)))
            while pending:
                token,elapsed=pending.popleft().result();tokens.append(token);times.append(elapsed)
                try:pending.append(pool.submit(timed,next(iterator)))
                except StopIteration:pass
        cost=interval(start,snapshot());total=cost['elapsed_s']
        if tokens!=[ds.rows[i]['token'] for i in indices] or len(set(tokens))!=len(indices):
            raise ValueError('CPU benchmark repeated or reordered scenes')
        return {'scenes':len(tokens),'unique_scenes':len(set(tokens)),'scene_ids':tokens,
            'consumption_order_sha256':identity_hash(tokens),'wall_s':total,'scenes_per_s':len(tokens)/total,
            'individual_task_mean_s':float(np.mean(times)),'individual_task_p50_s':float(np.percentile(times,50)),
            'individual_task_p95_s':float(np.percentile(times,95)),'worker_seconds_sum':sum(times),
            'resource_interval':cost}
    def heads(i):
        item=ds.teacher.read_scene(ds.rows[i]['token'],expected_split='train',need_future=False)
        value=ds.pca.transform(item['current_dense_features'])
        if not torch.isfinite(value).all() or value.shape[-1]!=384:raise ValueError('Invalid cached/PCA teacher')
        return ds.rows[i]['token']
    def full(i):
        obs,targets=ds[i]
        if len(obs.current['image'])!=3 or targets['future_dense_features'].shape[0]!=0:
            raise ValueError('Unexpected full current-only loader path')
        if not torch.isfinite(targets['current_dense_features']).all():raise ValueError('Nonfinite PCA target')
        return obs.current['token']
    # Each separately reported stage sees each assigned observation exactly once.
    result={'registration':registration,'initialization':init_cost,
            'reader_pca':consume(heads) if a.stage in ('reader','both') else None,
            'image_ego_reader_pca':consume(full) if a.stage in ('full','both') else None,
            'process_maxrss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'not_training_throughput':True,'gpu_used':False}
    atomic_json(out/'RESULT.json',result,immutable=True)
    return {key:value for key,value in result.items() if key!='registration'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['data-root','manifest','teacher-root','pca-root','output']:p.add_argument('--'+name,required=True)
    p.add_argument('--max-scenes',type=int,required=True)
    p.add_argument('--workers',type=int,default=2);p.add_argument('--prefetch-records',type=int,default=4)
    p.add_argument('--shard-count',type=int,default=1);p.add_argument('--shard-index',type=int,default=0)
    p.add_argument('--stage',choices=['reader','full','both'],default='both')
    print(json.dumps(run(p.parse_args()),sort_keys=True))


if __name__=='__main__':main()

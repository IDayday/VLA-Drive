"""Finite real-image on-demand teacher qualification, never a training launcher.

plan: CPU metadata only; no weights/tensors/CUDA. run: an explicitly dispatched
single-GPU foreground SMOKE. No tensor output, cache build, approval or worker.
"""
import argparse
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import random
import resource
import time

import numpy as np
import torch

from starVLA.dataloader.st_world_history import file_sha256
from starVLA.dataloader.st_world_schema import identity_hash
from starVLA.dataloader.st_world_targets import (
    TeacherCacheReader, atomic_json, payload_key, tensor_digest, validate_target_manifest)
from starVLA.dataloader.st_world_ondemand import BoundedTeacherProvider
from starVLA.model.modules.st_world.requests import sample_requests, gather_current, gather_future

GiB = 1 << 30
MiB = 1 << 20


def read_json(path, max_bytes=32*MiB):
    path = Path(path)
    if path.stat().st_size > max_bytes:
        raise ValueError('Metadata exceeds bounded qualification input size: '+str(path))
    return json.loads(path.read_text())


def pin(path):
    path = Path(path).resolve()
    return {'path':str(path),'sha256':file_sha256(path),'bytes':path.stat().st_size}


def open_reader(root):
    root = Path(root)
    # Explicitly refuses the full101k identity: small existing readers only.
    metadata = read_json(root/'identity.json')
    read_json(root/'requests.json')
    return TeacherCacheReader(root, expected_identity=metadata['identity'])


def _keys(manifest, contract, current):
    return list(dict.fromkeys(payload_key(ref,contract)
        for s in manifest['scenes'] for frame in (s['frames'][:1] if current else s['frames'][1:])
        for ref in frame['views'] if ref is not None))


def make_plan(requests_path, reference_root, current_root, teacher_config, *, max_seconds=900):
    if type(max_seconds) is not int or not 1 <= max_seconds <= 900:
        raise ValueError('SMOKE runtime is bounded to at most 900 seconds')
    manifest = validate_target_manifest(read_json(requests_path))
    if manifest['split'] != 'train' or not 1 <= len(manifest['scenes']) <= 64:
        raise ValueError('Qualification is fixed legal train only, at most64 scenes')
    if manifest['times_s'] != [0.,.5,1.,2.,4.]:
        raise ValueError('Qualification needs the exact registered future table')
    ref, current = open_reader(reference_root), open_reader(current_root)
    ref.verify_population(manifest)
    c = ref.contract
    if (c.identity != current.contract.identity or c.storage_dtype != 'float32' or
            c.compute_dtype != 'float32' or c.pooling['kind'] != 'none' or
            c.native_grid_hw != c.output_grid_hw or c.source_cache_identity is not None):
        raise ValueError('Both references must bind the same actual native FP32 teacher')
    if current.manifest['times_s'] != [0.]:
        raise ValueError('Reuse source must be a COMPLETE current-only cache')
    for key in ('split','source_student_identity','source_index_sha256','partition_sha256'):
        if manifest.get(key) != current.manifest.get(key):
            raise ValueError('Current source and qualification role/lineage differ: '+key)
    cfg = read_json(teacher_config)
    if (cfg.get('command') != 'build' or cfg.get('variant') not in ('SP2','SP3') or
            cfg.get('storage_dtype') != 'float32' or cfg.get('batch_size') != 8 or
            Path(cfg['output']).resolve() != Path(reference_root).resolve() or not cfg.get('teacher_root')):
        raise ValueError('Teacher path must come from the actual reference build resolved_config')
    for r in (ref,current):
        e = r.identity['execution']
        if e.get('batch_size') != 8 or e.get('teacher') != ref.identity['execution']['teacher'] or e.get('transform') != ref.identity['execution']['transform']:
            raise ValueError('Reference compute/transform/batch identities differ')
    current_keys = _keys(manifest,c,True)
    future_keys = _keys(manifest,c,False)
    if not set(current_keys) <= set(current.locations):
        raise ValueError('Current-only reference does not cover all requested current images')
    if set(future_keys) & set(current.locations):
        raise ValueError('This preregistered smoke expects a current-only192 source with zero future matches')
    record_bytes = math.prod(c.output_grid_hw)*(c.channels*4+1)
    if len(ref.locations)*record_bytes > 2*GiB:
        raise ValueError('Frozen online comparison tensors exceed the2 GiB reference budget')
    input_pins = [pin(requests_path),pin(teacher_config)]
    for r in (ref,current):
        input_pins += [pin(r.root/n) for n in ('identity.json','requests.json','COMPLETE.json')]
    plan = {'schema':'st_ondemand_qualification_plan_v1','mode':'SMOKE','formal_approved':False,
        'requests_path':str(Path(requests_path).resolve()),'reference_root':str(Path(reference_root).resolve()),
        'current_root':str(Path(current_root).resolve()),'teacher_config':str(Path(teacher_config).resolve()),
        'teacher_root':cfg['teacher_root'],'variant':cfg['variant'],'contract':c.to_dict(),
        'reference_identity':ref.identity['identity'],'current_identity':current.identity['identity'],
        'reference_execution':ref.identity['execution'],'current_execution':current.identity['execution'],
        'inputs':input_pins,'scenes':len(manifest['scenes']), 'logs':len({s['log'] for s in manifest['scenes']}),
        'current_unique_images':len(current_keys),'future_unique_images':len(future_keys),
        'unique_images':len(ref.locations),'true_absent_views':sum(v is None for s in manifest['scenes'] for f in s['frames'] for v in f['views']),
        'historical_full_current_reuse':'441/324 was a different metadata join; it is NOT this smoke reuse fraction',
        'cases':['production8_all','current_batch1','current_reversed8','current_tail8_plus_duplicate',
                 'provider_cold','provider_warm','provider_restored_cold'],
        'gate':{'bitwise_targets':True,'bitwise_masks':True,'rng_unchanged':True,'teacher_frozen':True,
                'nonzero_floating_difference':'NUMERICAL_REVIEW_REQUIRED; no automatic tolerance widening',
                'relative_floor':1e-12,'numeric_tolerance':None,
                'cross_batch_envelope':'descriptive observed error, not a post-hoc acceptance threshold'},
        'backend':{'tf32':False,'autocast':False,'deterministic_algorithms':False,
                   'cudnn_benchmark':False,'cudnn_deterministic':False,
                   'flash_sdp_enabled':True,'mem_efficient_sdp_enabled':True,'math_sdp_enabled':True,
                   'historical_limit':'Original cache metadata did not record every GPU/backend flag; no bitwise promise'},
        'budget':{'max_seconds':max_seconds,'cpu_threads':2,'visible_gpus':1,'ram_peak_bytes':8*GiB,
                  'resident_provider_bytes':2*GiB,'online_reference_bytes':2*GiB,
                  'metadata_bytes':32*MiB,'tensor_disk_bytes':0,'batch_images':8,
                  'max_image_forwards':len(ref.locations)+3*len(current_keys)+1+2*len(future_keys),
                  'max_batch_calls':2048},
        'runner_source_sha256':file_sha256(__file__),
        'source_files':[pin(Path(__file__).with_name('ondemand_teacher.py')),
                       pin(Path(__file__).parents[2]/'starVLA/dataloader/st_world_ondemand.py')]}
    plan['identity'] = identity_hash(plan)
    return plan


def difference(actual, expected):
    if actual.shape != expected.shape or actual.dtype != expected.dtype or not torch.isfinite(actual).all() or not torch.isfinite(expected).all():
        raise ValueError('Qualification cannot compare invalid shape/dtype/finite targets')
    a, b = actual.double(), expected.double()
    delta = a-b
    abs_delta = delta.abs()
    ref_sq = float(b.square().sum())
    err_sq = float(delta.square().sum())
    return {'elements':actual.numel(),'bitwise_equal':tensor_digest(actual)==tensor_digest(expected),
            'max_abs':float(abs_delta.max()),'sum_abs':float(abs_delta.sum()),'sum_squared':err_sq,
            'reference_squared':ref_sq,'rms':math.sqrt(err_sq/actual.numel()),
            'l2_relative':math.sqrt(err_sq/max(ref_sq,1e-24)),
            'max_elementwise_relative':float((abs_delta/b.abs().clamp_min(1e-12)).max())}


def summarize(rows):
    if not rows:return {'comparisons':0,'elements':0,'bitwise_equal':None}
    n=sum(r['elements'] for r in rows);sq=sum(r['sum_squared'] for r in rows);ref=sum(r['reference_squared'] for r in rows)
    return {'comparisons':len(rows),'elements':n,'bitwise_equal':all(r['bitwise_equal'] for r in rows),
            'max_abs':max(r['max_abs'] for r in rows),'mean_abs':sum(r['sum_abs'] for r in rows)/n,
            'rms':math.sqrt(sq/n),'l2_relative':math.sqrt(sq/max(ref,1e-24)),
            'max_elementwise_relative':max(r['max_elementwise_relative'] for r in rows)}


def rng_digest(device):
    np_state=np.random.get_state()
    return {'python':identity_hash(repr(random.getstate())),
            'numpy':identity_hash([np_state[0],np_state[1].tolist(),np_state[2],np_state[3],np_state[4]]),
            'cpu':tensor_digest(torch.get_rng_state()),'cuda':tensor_digest(torch.cuda.get_rng_state(device))}


class Budget:
    def __init__(self,plan):self.plan=plan;self.start=time.monotonic();self.calls=0;self.images=0
    def check(self):
        if time.monotonic()-self.start>self.plan['max_seconds']:raise TimeoutError('Qualification runtime exhausted')
        peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        if peak>self.plan['ram_peak_bytes']:raise MemoryError('Qualification peak RSS exceeded fixed8 GiB cap')
    def admit(self,count):
        self.check()
        if self.images+count>self.plan['max_image_forwards'] or self.calls+1>self.plan['max_batch_calls']:
            raise RuntimeError('Qualification forward budget exhausted')
        self.images+=count;self.calls+=1


class GuardedExecutor:
    def __init__(self,executor,budget,device):
        self.executor=executor;self.budget=budget;self.device=device
        self.execution_identity=executor.execution_identity;self.contract_identity=executor.contract_identity
        self.rng_checks=0
    def compute_batch(self,*args,**kwargs):
        self.budget.admit(len(args[0]));before=rng_digest(self.device)
        try:return self.executor.compute_batch(*args,**kwargs)
        finally:
            if rng_digest(self.device)!=before:raise RuntimeError('Teacher forward changed an external RNG stream')
            self.rng_checks+=1;self.budget.check()


def bounded_json(path,value,cap):
    data=json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n'
    path=Path(path)
    other=sum(p.stat().st_size for p in path.parent.iterdir() if p.is_file() and p!=path)
    # Atomic replacement may temporarily coexist with the previous version.
    peak=other+len(data.encode())+(path.stat().st_size if path.exists() else 0)
    if peak>cap:raise RuntimeError('Qualification metadata + atomic staging budget exhausted')
    atomic_json(path,value)


def run(plan_path,output):
    plan=read_json(plan_path)
    if (plan.get('schema')!='st_ondemand_qualification_plan_v1' or plan['identity']!=identity_hash({k:v for k,v in plan.items() if k!='identity'}) or
            plan['runner_source_sha256']!=file_sha256(__file__)):
        raise ValueError('Qualification plan/source changed')
    for entry in plan['inputs']+plan['source_files']:
        if file_sha256(entry['path'])!=entry['sha256']:raise ValueError('Pinned qualification input changed: '+entry['path'])
    # Rebuild only the small metadata plan; a self-consistent JSON checksum is
    # not permission to expand RAM, candidates, outputs or alter the protocol.
    expected=make_plan(plan['requests_path'],plan['reference_root'],plan['current_root'],
                       plan['teacher_config'],max_seconds=plan['budget']['max_seconds'])
    if plan!=expected:raise ValueError('Qualification protocol or budget differs from fixed runner')
    output=Path(output)
    if output.exists():raise FileExistsError('Use a fresh isolated qualification output; preserve earlier evidence')
    output.mkdir(parents=True)
    bounded_json(output/'RESOLVED_PLAN.json',plan,plan['budget']['metadata_bytes'])
    budget=Budget(plan['budget'])
    result={'schema':'st_ondemand_teacher_qualification_v1','plan_identity':plan['identity'],
            'status':'STARTING','cases':{},'errors':[],'scientific_gain':'UNTESTED','tensor_disk_bytes':0}
    code=1
    try:
        torch.set_num_threads(2)
        if not torch.cuda.is_available() or torch.cuda.device_count()!=1:
            raise ValueError('Explicitly expose exactly one authorized GPU')
        device=torch.device('cuda:0')
        torch.cuda.init();torch.cuda.get_rng_state(device)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=False
        torch.use_deterministic_algorithms(False)
        torch.backends.cuda.enable_flash_sdp(True);torch.backends.cuda.enable_mem_efficient_sdp(True);torch.backends.cuda.enable_math_sdp(True)
        ref,current=open_reader(plan['reference_root']),open_reader(plan['current_root'])
        manifest=validate_target_manifest(read_json(plan['requests_path']));c=ref.contract
        from tools.st_world_policy.ondemand_teacher import FrozenDINOExecutor
        before=rng_digest(device)
        executor=FrozenDINOExecutor(plan['teacher_root'],contract=c,variant=plan['variant'],device=str(device),max_batch_images=8)
        if rng_digest(device)!=before:raise RuntimeError('Teacher construction changed an external RNG stream')
        checked=GuardedExecutor(executor,budget,device)
        result.update(construction_rng_unchanged=True,execution=executor.execution,
                      device=torch.cuda.get_device_name(device),frozen_initial_versions_hash=identity_hash(executor._versions))
        records=ref.identity['records'];by_key={r['key']:r for r in records}
        current_keys=_keys(manifest,c,True)
        baseline={}
        def compare_case(name,groups,keep=False):
            tick=time.monotonic();rows=[];mask_equal=True
            for group in groups:
                produced=checked.compute_batch(tuple(group),contract=c,execution_identity=checked.execution_identity)
                for i,row in enumerate(group):
                    key=row['key'];stored,mask=ref.read_image(key)
                    if not torch.equal(mask,produced.valid[i]):raise ValueError('Teacher mask differs from immutable reference')
                    metrics=difference(produced.features[i],stored)
                    online=difference(produced.features[i],baseline[key][0]) if key in baseline else None
                    rows.append({'key':key,'reference':metrics,'online_production8':online,
                                 'computed_digest':tensor_digest(produced.features[i]),'reference_digest':tensor_digest(stored)})
                    if keep:
                        with torch.inference_mode(False):baseline[key]=(produced.features[i].clone(),produced.valid[i].clone())
                budget.check()
            result['cases'][name]={'seconds':time.monotonic()-tick,'all_masks_equal':mask_equal,
                'reference':summarize([r['reference'] for r in rows]),
                'online_production8':summarize([r['online_production8'] for r in rows if r['online_production8'] is not None]),'records':rows}
            bounded_json(output/'PROGRESS.json',result,plan['budget']['metadata_bytes'])
        def chunks(keys,n):
            return [[by_key[k] for k in keys[i:i+n]] for i in range(0,len(keys),n)]
        # Replay original shard boundaries and batch8, including the same tail.
        groups=[]
        for start in range(0,len(records),ref.identity['images_per_shard']):
            shard=records[start:start+ref.identity['images_per_shard']]
            groups.extend(shard[i:i+8] for i in range(0,len(shard),8))
        compare_case('production8_all',groups,True)
        compare_case('current_batch1',chunks(current_keys,1))
        compare_case('current_reversed8',chunks(list(reversed(current_keys)),8))
        compare_case('current_tail8_plus_duplicate',chunks(current_keys+[current_keys[0]],8))
        provider=BoundedTeacherProvider(manifest,c,[current],executor=checked,
            execution_identity=checked.execution_identity,max_cached_bytes=plan['budget']['resident_provider_bytes'],
            max_scene_bytes=64*MiB,max_batch_images=8,max_manifest_bytes=8*MiB)
        patch=torch.Generator().manual_seed(20261017);horizon=torch.Generator().manual_seed(20261018)
        initial_streams=(patch.get_state(),horizon.get_state());sampled={};cold_hashes={}
        for case in ('provider_cold','provider_warm','provider_restored_cold'):
            if case=='provider_restored_cold':provider.load_state_dict(provider.state_dict())
            patch.set_state(initial_streams[0]);horizon.set_state(initial_streams[1])
            images_before=budget.images;tick=time.monotonic();rows=[]
            for scene in manifest['scenes']:
                budget.check();token=scene['token'];rng_before=rng_digest(device)
                item=provider.read_scene(token,expected_split='train')
                if rng_digest(device)!=rng_before:raise RuntimeError('Provider read changed external RNG')
                ref_item=ref.read_scene(token,expected_split='train')
                for name in ('current_valid','future_valid'):
                    if not torch.equal(item[name],ref_item[name]):raise ValueError('Physical missing-source masks changed')
                for name in ('current_dense_features','future_dense_features'):
                    metric=difference(item[name],ref_item[name]);digest=tensor_digest(item[name])
                    key=token+'/'+name
                    if case=='provider_cold':cold_hashes[key]=digest
                    elif digest!=cold_hashes[key]:raise ValueError('Warm/cold-restored provider changed target bytes')
                    rows.append({'scene':token,'target':name,'reference':metric,'computed_digest':digest})
                request=sample_requests(1,len(c.view_names),c.output_grid_hw,min(96,math.prod(c.output_grid_hw)),
                    future=True,patch_generator=patch,horizon_generator=horizon)
                ids=(tensor_digest(request.patch_indices),tensor_digest(request.future_pair))
                if case=='provider_cold':sampled[token]=ids
                elif ids!=sampled[token]:raise RuntimeError('Provider changed patch/pair sampling streams')
                # Gather the real labels with existing APIs; no replacement request.
                gather_current(item['current_dense_features'][None],item['current_valid'][None],request)
                gather_future(item['future_dense_features'][None],item['future_valid'][None],request)
            result['cases'][case]={'seconds':time.monotonic()-tick,'reference':summarize([r['reference'] for r in rows]),
                'records':rows,'all_masks_equal':True,'sampler_equal':True,'new_image_forwards':budget.images-images_before,
                'provider_memory':provider.memory_report()}
            if case=='provider_warm' and budget.images!=images_before:raise RuntimeError('Warm qualification unexpectedly recomputed records')
            bounded_json(output/'PROGRESS.json',result,plan['budget']['metadata_bytes'])
        if executor._frozen_versions()!=executor._versions:raise RuntimeError('Teacher frozen state changed')
        exact=all(x['reference']['bitwise_equal'] for x in result['cases'].values())
        result['status']='BITWISE_TARGET_QUALIFIED' if exact else 'NUMERICAL_REVIEW_REQUIRED'
        result['bitwise_target_gate_passed']=exact
        result['online_cross_batch_envelope']=summarize([r['online_production8'] for n,x in result['cases'].items()
             if n.startswith('current_') for r in x['records'] if r.get('online_production8') is not None])
        result['frozen_final_versions_hash']=identity_hash(executor._frozen_versions())
        result['rng_forward_checks']=checked.rng_checks
        code=0 if exact else 2
    except Exception as exc:
        result['status']='PARTIAL_BUDGET' if isinstance(exc,(TimeoutError,MemoryError)) else 'FAILED'
        result['errors'].append({'type':type(exc).__name__,'message':str(exc)})
        code=2 if result['status']=='PARTIAL_BUDGET' else 1
    finally:
        result.update(elapsed_seconds=time.monotonic()-budget.start,image_forwards=budget.images,batch_calls=budget.calls,
                      peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
        if torch.cuda.is_initialized():
            result.update(cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved())
        bounded_json(output/'RESULT.json',result,plan['budget']['metadata_bytes'])
    return result,code


def parser():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    plan=sub.add_parser('plan',help='CPU metadata only; no teacher/weights/CUDA')
    for name in ('requests','reference-root','current-cache-root','teacher-config','output'):plan.add_argument('--'+name,required=True)
    plan.add_argument('--max-seconds',type=int,default=900)
    execute=sub.add_parser('run',help='Foreground single-GPU finite SMOKE; requires explicit Lead dispatch')
    execute.add_argument('--plan',required=True);execute.add_argument('--output',required=True)
    return p


def main():
    args=parser().parse_args()
    if args.command=='plan':
        plan=make_plan(args.requests,args.reference_root,args.current_cache_root,args.teacher_config,max_seconds=args.max_seconds)
        output=Path(args.output)
        if output.exists():raise FileExistsError('Do not replace a preregistered qualification plan')
        output.mkdir(parents=True)
        bounded_json(output/'PLAN.json',plan,plan['budget']['metadata_bytes'])
        print(json.dumps({'status':'PLANNED_NOT_RUN','identity':plan['identity'],'plan':str(output/'PLAN.json'),
                          'budget':plan['budget'],'scenes':plan['scenes'],'current_hits':plan['current_unique_images'],
                          'future_misses':plan['future_unique_images']}))
        return 0
    result,code=run(args.plan,args.output)
    print(json.dumps({k:result[k] for k in ('status','image_forwards','batch_calls','elapsed_seconds','errors')}));return code


if __name__=='__main__':
    try:
        exit_code=main()
    except Exception as error:
        print(json.dumps({'status':'FAILED','error':type(error).__name__,'message':str(error)}))
        exit_code=1
    raise SystemExit(exit_code)

"""Bounded independent frozen DINOv3 cache: plan/build/verify, no optimizer.

Native384x288 means18x24 *stored* patches. SP1 pools the real native output by3
into6x8. SP0 only reuses the exact byte-verified historical C1 bank. All numerical
and storage choices are explicit; no teacher or student weights are trained.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import importlib.metadata
import inspect
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np
from PIL import Image
import torch

from starVLA.dataloader.st_world_history import file_sha256
from starVLA.dataloader.st_world_schema import TeacherFeatureContract, identity_hash
from starVLA.dataloader.st_world_targets import (
    atomic_json, TeacherCacheReader, TeacherCacheWriter, LegacyC1Reader,
    unique_image_records, validate_target_manifest, payload_key)
from starVLA.model.modules.foresight import dinov3_target_encoder as dino


def teacher_descriptor(root):
    root = Path(root)
    identity = json.loads((root / 'IDENTITY.json').read_text())
    if identity['repository'] != dino.REPOSITORY or identity['revision'] != dino.REVISION:
        raise ValueError('Unregistered generic teacher')
    for name, expected in [('model.safetensors',dino.WEIGHT_SHA256),('config.json',dino.CONFIG_SHA256)]:
        if file_sha256(root / name) != expected:
            raise ValueError('Teacher weight/config checksum mismatch')
    return {'repository':dino.REPOSITORY,'revision':dino.REVISION,'weight_sha256':dino.WEIGHT_SHA256,
            'config_sha256':dino.CONFIG_SHA256,'encoder_source_sha256':file_sha256(dino.__file__),
            'timm':importlib.metadata.version('timm'),'torch':torch.__version__,
            'compute_dtype':'float32','tf32':False,'autocast':False,'extra_norm':'none',
            'prefix_tokens_excluded':5,'feature':'last_block_post_norm_dense_patch'}


def make_contract(root, variant, storage_dtype, *, legacy_root=None):
    descriptor = teacher_descriptor(root)
    image_hw, pool = ((192,256),2) if variant == 'SP0' else ((288,384),3 if variant == 'SP1' else 1)
    transform = {'source':'complete original16:9 RGB','image_hw':list(image_hw),'crop':'none',
                 'resize':'PIL_BICUBIC','normalization':'ImageNet_once','mean':[.485,.456,.406],
                 'std':[.229,.224,.225],'padding':'none','augmentation':'none',
                 'preprocess_source_sha256':identity_hash(inspect.getsource(preprocess_image))}
    source = None
    if variant == 'SP0':
        if not legacy_root or storage_dtype != 'float16':
            raise ValueError('SP0 exact reuse requires --legacy-c1-root and original float16 storage')
        old = json.loads((Path(legacy_root) / 'identity.json').read_text())
        if old['identity'] != LegacyC1Reader.IDENTITY or identity_hash({k:v for k,v in old.items() if k != 'identity'}) != LegacyC1Reader.IDENTITY:
            raise ValueError('Unregistered legacy source')
        descriptor = old['recipe']
        transform = {'legacy_recipe':old['recipe'],'candidate':old['candidate']}
        source = old['identity']
    native = tuple(x // 16 for x in image_hw)
    return TeacherFeatureContract(image_hw=image_hw,patch_hw=(16,16),native_grid_hw=native,
        output_grid_hw=tuple(x // pool for x in native),channels=1024,storage_dtype=storage_dtype,
        compute_dtype='float32',view_names=('CAM_F0','CAM_L0','CAM_R0'),
        teacher_identity=identity_hash(descriptor),transform_identity=identity_hash(transform),
        pooling={'kind':'none' if pool==1 else 'avg','kernel_hw':(pool,pool),'stride_hw':(pool,pool)},
        source_cache_identity=source), descriptor, transform


def preprocess_image(ref, contract):
    path = Path(ref['path'])
    if file_sha256(path) != ref['sha256']:
        raise ValueError('Teacher source image changed')
    with Image.open(path) as source:
        if list(source.size) != ref['original_wh'] or abs(source.width/source.height - 16/9) > .001:
            raise ValueError('Teacher source field differs from registered full16:9 input')
        rgb = source.convert('RGB').resize((contract.image_hw[1],contract.image_hw[0]),Image.Resampling.BICUBIC)
        pixels = torch.from_numpy(np.array(rgb, copy=True)).permute(2,0,1).float() / 255.
    return (pixels - pixels.new_tensor([.485,.456,.406])[:,None,None]) / pixels.new_tensor([.229,.224,.225])[:,None,None]


def validate_budget(args, manifest):
    if min(args.max_scenes,args.images_per_shard,args.batch_size,args.workers,args.cpu_threads,args.max_seconds) <= 0:
        raise ValueError('All resource/sample budgets must be explicit and positive')
    if args.workers > 8 or args.batch_size > 64 or args.images_per_shard > 1024:
        raise ValueError('Unbounded teacher process memory/IO settings')
    if len(manifest['scenes']) > args.max_scenes:
        raise ValueError('Manifest exceeds max-scenes; never silently truncate a fixed population')
    if args.mode == 'smoke' and args.max_scenes > 64:
        raise ValueError('Native teacher smoke limited to64 scenes')
    if args.mode == 'registered':
        if not args.registration:
            raise ValueError('Full registered cache requires explicit immutable budget registration')
        registration = json.loads(Path(args.registration).read_text())
        if (registration['phase'] != 'teacher_cache' or registration['request_manifest_identity'] != manifest['identity'] or
                registration['max_scenes'] < args.max_scenes or registration['storage_dtype'] != args.storage_dtype or
                registration['variant'] != args.variant):
            raise ValueError('Teacher cache scope differs from registration')
        if args.command in ('build', 'derive-pooled') and bool(registration.get('defer_finalize', False)) != args.defer_finalize:
            raise ValueError('Deferred cache publication differs from registration')
    if args.defer_finalize and (args.mode != 'registered' or args.command not in ('build', 'derive-pooled')):
        raise ValueError('Deferred publication requires an explicitly registered builder')
    if args.command == 'finalize' and (args.rank != 0 or args.world_size != 1 or args.device != 'cpu'):
        raise ValueError('Complete publication requires one CPU finalizer')
    if not 0 <= args.rank < args.world_size or args.world_size < 1:
        raise ValueError('Invalid deterministic shard assignment')
    if args.min_free_gib < 0:
        raise ValueError('Disk safety reserve must be nonnegative')


def finish_assigned(writer, args):
    """A rank commits only its shards; a single later CPU job publishes all.

    Retain full checksum verification, but avoid every GPU worker serially
    hashing the entire full-population cache under the finalization lock.
    """
    if not args.defer_finalize:
        return writer.finalize()
    assigned = list(range(args.rank, writer.shard_count, args.world_size))
    receipts = []
    for shard in assigned:
        if not writer.committed(shard):
            return {'status': 'PARTIAL', 'identity': writer.identity['identity'],
                    'rank': args.rank, 'missing_assigned_shard': shard}
        receipts.append({'shard': shard, 'receipt_sha256': file_sha256(writer.paths(shard)[1])})
    result = {'status': 'SHARDS_COMMITTED', 'identity': writer.identity['identity'],
              'rank': args.rank, 'world_size': args.world_size, 'shards': receipts,
              'global_cache_complete': False, 'requires_separate_full_checksum_finalizer': True}
    atomic_json(writer.root / f'RANK_COMMIT_world{args.world_size}_rank{args.rank}.json', result, immutable=True)
    return result


def run(args):
    manifest = validate_target_manifest(json.loads(Path(args.requests).read_text()))
    validate_budget(args,manifest)
    contract, teacher, transform = make_contract(args.teacher_root,args.variant,args.storage_dtype,legacy_root=args.legacy_c1_root)
    source_reader=None
    if args.source_cache:
        if args.command not in ('plan','derive-pooled','verify','finalize') or args.variant!='SP1' or args.storage_dtype!='float32' or not args.source_cache_identity:
            raise ValueError('Source-cache derivation only supports explicit FP32 SP1 pool3')
        source_reader=TeacherCacheReader(args.source_cache,expected_identity=args.source_cache_identity)
        source_reader.verify_population(manifest)
        source_contract=source_reader.contract
        if (source_contract.image_hw!=(288,384) or source_contract.output_grid_hw!=(18,24) or
                source_contract.storage_dtype!='float32' or source_contract.pooling['kind']!='none' or
                source_reader.identity['execution']['teacher']!=teacher or
                source_reader.identity['execution']['transform']!=transform):
            raise ValueError('SP1 derivation requires the identical registered native FP32 source')
        transform={**transform,'pool_derivation':{'source_cache_identity':source_reader.identity['identity'],
            'source_contract_identity':source_contract.identity,'kernel_hw':[3,3],'stride_hw':[3,3],
            'compute_device':'cpu','compute_dtype':'float32','mask_rule':'all_source_patches_valid'}}
        value=contract.to_dict();value.update(transform_identity=identity_hash(transform),source_cache_identity=source_reader.identity['identity'])
        contract=TeacherFeatureContract.from_dict(value)
    elif args.command=='derive-pooled':
        raise ValueError('derive-pooled requires source-cache and source-cache-identity')
    execution = {'teacher':teacher,'transform':transform,'batch_size':args.batch_size,
                 'writer_sha256':file_sha256(__file__),'format':'safetensors_uncompressed','optimizer_updates':0}
    records = unique_image_records(manifest,contract)
    pixel_values = int(np.prod(contract.output_grid_hw))*contract.channels
    bytes_per = pixel_values * (4 if args.storage_dtype=='float32' else 2) + int(np.prod(contract.output_grid_hw))
    requested_bytes = len(records) * bytes_per
    out = Path(args.output)
    out.mkdir(parents=True,exist_ok=True)
    plan = {'status':'PLAN_ONLY','request_identity':manifest['identity'],'contract':contract.to_dict(),
            'contract_identity':contract.identity,'scenes':len(manifest['scenes']),'unique_images':len(records),
            'logical_tensor_bytes':requested_bytes,'bytes_per_unique_image':bytes_per,
            'shards':(len(records)+args.images_per_shard-1)//args.images_per_shard,
            'temporary_shard_bytes':args.images_per_shard*bytes_per,'disk_free_bytes':shutil.disk_usage(out).free,
            'teacher':teacher,'transform':transform,'mode':args.mode,'gpu_work_started':False}
    atomic_json(out / ('PLAN_rank%d.json'%args.rank),plan)
    resolved = vars(args).copy()
    resolved.update(contract=contract.to_dict(),request_identity=manifest['identity'])
    atomic_json(out / ('resolved_config_%s_rank%d.json'%(args.command,args.rank)),resolved)
    if args.command == 'plan':
        return plan
    if args.command == 'verify':
        identity = json.loads((out/'identity.json').read_text())['identity']
        reader = TeacherCacheReader(out,expected_identity=identity,expected_contract=contract)
        reader.verify_population(manifest)
        count = 0
        for key in reader.locations:
            reader.read_image(key)
            count += 1
        result = {'status':'VERIFIED','identity':identity,'images_verified':count,'scenes':len(reader.scenes),
                  'request_manifest_identity':manifest['identity'],
                  'request_scene_tokens_hash':identity_hash(sorted(reader.scenes)),
                  'contract_identity':contract.identity,'all_record_hashes_verified':True,
                  'exact_requested_population':True}
        atomic_json(out/'VERIFICATION.json',result)
        return result
    if args.command=='derive-pooled':
        if shutil.disk_usage(out).free < requested_bytes + args.min_free_gib*1024**3:
            raise ValueError('Insufficient declared derived-cache disk reserve')
        torch.set_num_threads(args.cpu_threads)
        writer=TeacherCacheWriter(out,contract,manifest,images_per_shard=args.images_per_shard,execution=execution)
        started=time.monotonic()
        for shard in range(args.rank,writer.shard_count,args.world_size):
            if time.monotonic()-started>=args.max_seconds:
                return {'status':'PARTIAL','reason':'TIME_BUDGET','identity':writer.identity['identity']}
            with writer.shard_lock(shard):
                if writer.committed(shard):continue
                features=[];masks=[]
                for row in writer.rows(shard):
                    raw,valid=source_reader.read_image(payload_key(row['image'],source_reader.contract))
                    pooled=torch.nn.functional.avg_pool2d(raw.permute(2,0,1)[None],3,3)[0].permute(1,2,0).contiguous()
                    mask=torch.nn.functional.avg_pool2d(valid.float()[None,None],3,3)[0,0]==1
                    features.append(torch.where(mask[...,None],pooled,torch.zeros_like(pooled)));masks.append(mask)
                writer.commit_shard(shard,torch.stack(features),torch.stack(masks),metrics={'teacher_forward_calls':0})
        return {**finish_assigned(writer,args),'teacher_forward_calls':0,'elapsed_s_this_invocation':time.monotonic()-started}
    writer = TeacherCacheWriter(out,contract,manifest,images_per_shard=args.images_per_shard,execution=execution)
    if args.command == 'finalize':
        # Original full coverage + full shard checksum logic, unchanged. Missing
        # ranks cannot create COMPLETE and corrupt shards remain hard failures.
        return writer.finalize()
    pending = [i for i in range(args.rank,writer.shard_count,args.world_size) if not writer.committed(i)]
    reserve = int(args.min_free_gib * (1<<30)) + args.images_per_shard*bytes_per
    remaining_bytes = sum(len(writer.rows(i))*bytes_per for i in pending)
    if shutil.disk_usage(out).free < reserve + remaining_bytes:
        raise ValueError('BLOCKED_STORAGE: remaining cache + atomic shard + requested reserve do not fit')
    if not pending:
        return finish_assigned(writer,args)
    started = time.monotonic()
    torch.set_num_threads(args.cpu_threads)
    if args.variant == 'SP0':
        model = None
        legacy = LegacyC1Reader(args.legacy_c1_root)
    else:
        legacy = None
        if args.device.startswith('cuda') and not torch.cuda.is_available():
            raise ValueError('BLOCKED_RESOURCE: requested CUDA device unavailable')
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        model = dino.DINOv3TargetEncoder(args.teacher_root,device=args.device)
        if model.patch_size != contract.patch_hw or model.feature_dim != contract.channels or model.prefix != 5:
            raise ValueError('Strict teacher architecture differs from planned native output')
        if any(p.requires_grad for p in model.parameters()) or model.training:
            raise ValueError('Teacher must remain frozen/eval')
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for shard in pending:
            if time.monotonic()-started >= args.max_seconds:
                return {'status':'PARTIAL','reason':'BUDGET_EXHAUSTED','identity':writer.identity['identity']}
            with writer.shard_lock(shard):
                if writer.committed(shard):
                    continue
                features, masks, timings = [], [], {'read_s':0.,'teacher_s':0.,'stored_quantization_sse':0.,'elements':0,'max_abs_storage_error':0.}
                for begin in range(0,len(writer.rows(shard)),args.batch_size):
                    batch = writer.rows(shard)[begin:begin+args.batch_size]
                    tick = time.monotonic()
                    if legacy:
                        values = [legacy.read_image(row['image']['sha256']) for row in batch]
                        z = torch.stack([r[0] for r in values]); mask=torch.stack([r[1] for r in values])
                        timings['read_s'] += time.monotonic()-tick
                    else:
                        pixels = list(pool.map(lambda row:preprocess_image(row['image'],contract),batch))
                        timings['read_s'] += time.monotonic()-tick
                        if args.device.startswith('cuda'):torch.cuda.synchronize()
                        tick=time.monotonic()
                        with torch.inference_mode():
                            native = model(torch.stack(pixels).to(args.device))
                            factor=contract.pooling['kernel_hw']
                            if contract.pooling['kind']=='avg':native=torch.nn.functional.avg_pool2d(native,factor,factor)
                            z=native.permute(0,2,3,1).contiguous().cpu()
                        if args.device.startswith('cuda'):torch.cuda.synchronize()
                        timings['teacher_s']+=time.monotonic()-tick
                        mask=torch.ones(z.shape[:-1],dtype=torch.bool)
                    stored=z.to(getattr(torch,args.storage_dtype))
                    if not torch.isfinite(stored).all():raise FloatingPointError('Teacher storage overflow')
                    error=stored.float()-z.float()
                    timings['stored_quantization_sse']+=float(error.square().sum())
                    timings['elements']+=z.numel()
                    timings['max_abs_storage_error']=max(timings['max_abs_storage_error'],float(error.abs().max()))
                    features.append(stored);masks.append(mask)
                writer.commit_shard(shard,torch.cat(features),torch.cat(masks),metrics=timings)
                atomic_json(out/('progress_rank%d.json'%args.rank),{'status':'RUNNING','last_committed_shard':shard,
                    'elapsed_s':time.monotonic()-started,'identity':writer.identity['identity'],'metrics':timings})
    result=finish_assigned(writer,args)
    result={**result,'elapsed_s_this_invocation':time.monotonic()-started,'rank':args.rank}
    atomic_json(out/('RESULT_rank%d.json'%args.rank),result)
    return result


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['plan','build','verify','derive-pooled','finalize'])
    for key in ('requests','teacher-root','output'):p.add_argument('--'+key,required=True)
    p.add_argument('--variant',choices=['SP0','SP1','SP2','SP3'],required=True)
    p.add_argument('--storage-dtype',choices=['float16','float32'],required=True)
    p.add_argument('--legacy-c1-root')
    p.add_argument('--source-cache');p.add_argument('--source-cache-identity')
    p.add_argument('--mode',choices=['smoke','registered'],default='smoke')
    p.add_argument('--registration')
    p.add_argument('--max-scenes',type=int,required=True)
    p.add_argument('--images-per-shard',type=int,default=32)
    p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--workers',type=int,default=2)
    p.add_argument('--cpu-threads',type=int,default=2)
    p.add_argument('--max-seconds',type=float,default=900.)
    p.add_argument('--min-free-gib',type=float,default=20.)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--rank',type=int,default=0)
    p.add_argument('--world-size',type=int,default=1)
    p.add_argument('--defer-finalize',action='store_true',
                   help='Registered builders commit assigned shards; a separate CPU finalize verifies and publishes the full cache')
    return p


def main():
    try:
        result=run(parser().parse_args())
        print(json.dumps(result,sort_keys=True))
        return 3 if result['status']=='PARTIAL' else 0
    except Exception as exc:
        print(json.dumps({'status':'FAILED','error':repr(exc)}),file=sys.stderr)
        return 2


if __name__=='__main__':raise SystemExit(main())

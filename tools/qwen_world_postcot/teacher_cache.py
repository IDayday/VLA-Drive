"""Build bounded, immutable DINO/current and WA/future target shards.

The input manifest is a fixed legal population, never a directory scan. This
offline builder has no optimizer, student, actor, scoring code or GT dependency.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import pickle
import random
import signal
import time

import numpy as np
import torch

from starVLA.model.modules.qwen_world_postcot.teacher_targets import (
    CAMERAS, IMAGE_MEAN, IMAGE_STD, BinaryTeacherCacheWriter, TeacherCacheReader,
    atomic_json, file_sha256, finalize_teacher_cache, identity_hash, tensor_layout,
    validate_clip_timestamps,clip_timing_valid, _read_sealed, _cache_index)


def load_population(manifest_path):
    path = Path(manifest_path).resolve()
    manifest = json.loads(path.read_text())
    records_path = Path(manifest['records_file'])
    if not records_path.is_absolute():
        records_path = path.parent/records_path
    expected_hash = manifest['records_sha256']
    if file_sha256(records_path) != expected_hash:
        raise ValueError('Population records differ from the immutable input manifest')
    if records_path.suffix == '.jsonl':
        with records_path.open() as stream:
            records = [json.loads(line) for line in stream if line.strip()]
    elif records_path.suffix in ('.pkl','.pickle'):
        # User-controlled local population artifact, not a downloaded model.
        with records_path.open('rb') as stream:
            records = pickle.load(stream)
    else:
        records = json.loads(records_path.read_text())
    if isinstance(records, dict):
        order=manifest.get('tokens',list(records))
        if len(order)!=len(records) or set(order)!=set(records):
            raise ValueError('Population token order and records differ')
        records = [dict(records[token], token=str(token)) for token in order]
    if not isinstance(records,list) or len(records) != manifest['count']:
        raise ValueError('Population count/container does not match manifest')
    tokens = [str(row.get('token',row.get('scene_id',''))) for row in records]
    if any(not token for token in tokens) or len(set(tokens)) != len(tokens):
        raise ValueError('Every scene must have a unique real token')
    for row,token in zip(records,tokens):
        row['token'] = token
        role = row.get('cohort',row.get('split',manifest['cohort']))
        if role != manifest['cohort']:
            raise ValueError('Population mixes data roles')
    return manifest,records


def source_record_identity(row):
    # Target coordinates, object labels and GT do not identify frozen teachers.
    return row.get('source_record_hash') or identity_hash(dict(token=row['token'],
                                                               teacher_frames=row['teacher_frames']))


def partition_producer_records(records,rank,world_size,priority_tokens=()):
    """Contiguous temporal producer chunks; original training order untouched.

    Adjacent scenes share most12-frame clips. Striding every eighth scene
    needlessly decodes those images again. Only offline production order is
    changed; consumer lookup remains token-addressed and globally accessible.
    """
    if world_size<1 or not 0<=rank<world_size:
        raise ValueError('Invalid builder partition')
    ordered=sorted(records,key=lambda row:(str(row.get('log','')),int(row['timestamp']),row['token']))
    selected=ordered[len(ordered)*rank//world_size:len(ordered)*(rank+1)//world_size]
    if priority_tokens:
        if len(set(priority_tokens))!=len(priority_tokens) or not set(priority_tokens)<={row['token'] for row in records}:
            raise ValueError('Priority scene list crosses the fixed legal population')
        priority={token:i for i,token in enumerate(priority_tokens)}
        selected.sort(key=lambda row:priority.get(row['token'],len(priority)))
    return selected


class BoundedImageDecoder:
    """Bounded decoded-image LRU; content hash calculated only on first read.

    The key also includes an immutable file version. A change during a build is
    rejected rather than silently mixing image revisions in a record.
    """
    def __init__(self, max_images=64):
        if max_images < 1:
            raise ValueError('Positive bounded image LRU required')
        self.max_images=max_images
        self.cache=OrderedDict()
        self.bytes_read=0
        self.hits=0
        self.decodes=0
        self.dino_decodes=0

    def dino_from_payload(self,payload,path):
        from PIL import Image
        try:
            rgb=np.asarray(Image.open(io.BytesIO(payload)).convert('RGB'))
        except (OSError,ValueError) as error:
            raise ValueError('Cannot decode registered real current image: '+str(path)+
                '; sha256='+hashlib.sha256(payload).hexdigest()) from error
        image=np.asarray(Image.fromarray(rgb).resize((512,288),Image.Resampling.BICUBIC)).copy()
        image=torch.from_numpy(image).permute(2,0,1).float().div_(255)
        self.dino_decodes+=1
        return (image-torch.tensor(IMAGE_MEAN)[:,None,None])/torch.tensor(IMAGE_STD)[:,None,None]

    def get(self,path,*,dino_required=True):
        import cv2
        path=Path(path).resolve()
        stat=path.stat()
        version=(stat.st_size,stat.st_mtime_ns)
        if str(path) in self.cache:
            self.hits+=1
            value=self.cache.pop(str(path)); self.cache[str(path)]=value
            if (value[2]['bytes'],value[2]['mtime_ns']) != version:
                raise ValueError('Source image changed during cache build: '+str(path))
            if dino_required and value[0] is None:
                value=(self.dino_from_payload(value[3],path),value[1],value[2],value[3])
                self.cache[str(path)]=value
            return value[:3]
        payload=path.read_bytes()
        if path.stat().st_mtime_ns != stat.st_mtime_ns or len(payload) != stat.st_size:
            raise ValueError('Source image changed while reading')
        dino=self.dino_from_payload(payload,path) if dino_required else None
        # Match the released WA dataset's OpenCV decoding and INTER_AREA path.
        bgr=cv2.imdecode(np.frombuffer(payload,dtype=np.uint8),cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError('OpenCV could not decode a registered teacher image')
        wa=cv2.resize(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB),(512,256),interpolation=cv2.INTER_AREA)
        wa=torch.from_numpy(wa.copy()).permute(2,0,1).float().div_(255)
        self.bytes_read+=len(payload); self.decodes+=1
        value=(dino,wa,dict(path=str(path),bytes=len(payload),sha256=hashlib.sha256(payload).hexdigest(),
                            mtime_ns=stat.st_mtime_ns),payload)
        self.cache[str(path)]=value
        while len(self.cache)>self.max_images:
            self.cache.popitem(last=False)
        return value[:3]


class CanonicalClipResolver:
    """Resolve physical camera files from immutable logs, not a partial mirror.

    Current RGB uses the identical registered path as Qwen. Other teacher-only
    frames are recovered from the real log's camera paths and real timestamps;
    missing or nonregistered frames remain explicitly absent.
    """
    def __init__(self,raw_log_root='/mnt/project/DriveDreamer-Policy/navsim_raw/navsim_logs/trainval',
                 sensor_root='/mnt/navsim/trainval_all/trainval_sensor_blobs/trainval',max_open_logs=2,
                 sensor_fallback_root='/mnt/navsim/trainval_sensor_blobs/trainval'):
        self.raw_log_root=Path(raw_log_root).resolve()
        self.sensor_root=Path(sensor_root).resolve()
        self.sensor_fallback_root=Path(sensor_fallback_root).resolve() if sensor_fallback_root else None
        self.max_open_logs=max_open_logs
        if max_open_logs<1:raise ValueError('Positive bounded raw-log LRU required')
        self.logs=OrderedDict()
        self.last_receipt=None
        self.mirror_checks=[]

    def resolve(self,row):
        log=str(row['log'])
        path=(self.raw_log_root/(log+'.pkl')).resolve()
        if not path.is_relative_to(self.raw_log_root):
            raise ValueError('Raw log escaped registered source root')
        stat=path.stat();version=(stat.st_size,stat.st_mtime_ns)
        if log in self.logs:
            frames,indices,receipt=self.logs.pop(log)
            if (receipt['bytes'],receipt['mtime_ns'])!=version:
                raise ValueError('Raw teacher log changed during extraction')
        else:
            payload=path.read_bytes()
            if len(payload)!=stat.st_size or path.stat().st_mtime_ns!=stat.st_mtime_ns:
                raise ValueError('Raw teacher log changed while reading')
            frames=pickle.loads(payload)
            indices={frame['token']:i for i,frame in enumerate(frames)}
            receipt=dict(path=str(path),sha256=hashlib.sha256(payload).hexdigest(),
                         bytes=stat.st_size,mtime_ns=stat.st_mtime_ns)
        self.logs[log]=(frames,indices,receipt)
        while len(self.logs)>self.max_open_logs:self.logs.popitem(last=False)
        index=indices[row['token']]
        if int(frames[index]['timestamp'])!=int(row['timestamp']):
            raise ValueError('Raw log and registered current input timestamp differ')
        resolved=[]
        for sealed in row['teacher_frames']:
            offset=int(sealed['offset']);j=index+offset
            frame=frames[j] if 0<=j<len(frames) else None
            timestamp=None if frame is None else int(frame['timestamp'])
            if timestamp!=sealed['timestamp']:
                raise ValueError('Raw teacher timestamp differs from sealed observation identity')
            timing_valid=frame is not None and abs(timestamp-int(row['timestamp'])-offset*500000)<60000
            images={}
            for camera in CAMERAS:
                if offset==0:
                    candidate=Path(row['images'][camera])
                elif timing_valid:
                    relative=Path(frame['cams'][camera.upper()]['data_path'])
                    if relative.is_absolute() or '..' in relative.parts:
                        raise ValueError('Raw camera path must be a registered relative path')
                    candidate=self.sensor_root/relative
                    alternate=None if self.sensor_fallback_root is None else self.sensor_fallback_root/relative
                    if candidate.is_file() and alternate is not None and alternate.is_file() and len(self.mirror_checks)<16:
                        a,b=candidate.resolve(),alternate.resolve()
                        same_physical_file=a==b
                        # A finite content audit establishes mirror agreement;
                        # actual chosen image hashes are still saved per record.
                        ah=file_sha256(a)
                        bh=ah if same_physical_file else file_sha256(b)
                        check=dict(primary=str(a),fallback=str(b),primary_sha256=ah,
                                   fallback_sha256=bh,equal=ah==bh,same_physical_file=same_physical_file)
                        self.mirror_checks.append(check)
                        if ah!=bh:raise ValueError('Available sensor mirrors disagree for the same registered raw observation')
                    if not candidate.is_file() and alternate is not None:
                        candidate=alternate
                else:
                    images[camera]=None;continue
                images[camera]=str(candidate) if candidate.is_file() else None
            resolved.append(dict(offset=offset,timestamp=timestamp,images=images,timing_valid=timing_valid))
        self.last_receipt=receipt
        return resolved


def load_teacher_inputs(row,decoder,resolver=None):
    resolver=resolver or CanonicalClipResolver()
    frames=resolver.resolve(row)
    if len(frames)!=12 or [frame['offset'] for frame in frames]!=list(range(-3,9)):
        raise ValueError('Require exactly4history+8future frames, offsets-3..8')
    timestamps=torch.tensor([[-1 if frame['timestamp'] is None else int(frame['timestamp']) for frame in frames]],dtype=torch.int64)
    validate_clip_timestamps(timestamps,allow_incomplete=True)
    if 'timestamp' in row and int(row['timestamp'])!=int(timestamps[0,3]):
        raise ValueError('Current actor and teacher timestamp differ')
    dino=torch.zeros(1,3,3,288,512)
    clip=torch.zeros(1,12,3,3,256,512)
    valid=torch.zeros(1,12,3,dtype=torch.bool)
    image_receipts=[]
    complete_cameras=[bool(clip_timing_valid(timestamps).all()) and
        all(frame['images'].get(camera) is not None for frame in frames) for camera in CAMERAS]
    for t,frame in enumerate(frames):
        camera_receipts=[]
        for c,camera in enumerate(CAMERAS):
            path=frame['images'].get(camera)
            if path is None or (t!=3 and not complete_cameras[c]):
                camera_receipts.append(None); continue
            dino_image,wa_image,receipt=decoder.get(path,dino_required=(t==3))
            if t==3:
                dino[0,c]=dino_image
            clip[0,t,c]=wa_image
            valid[0,t,c]=True
            camera_receipts.append(receipt)
        image_receipts.append(camera_receipts)
    return dino,clip,valid,timestamps,image_receipts


def target_tensors(dino,future):
    # Keep the declared cache representation exact: no lossy casts/compression.
    return dict(dino=dino.features[0].cpu().contiguous(),dino_valid=dino.valid[0].cpu().contiguous(),
        future=future.features[0].cpu().contiguous(),future_valid=future.valid[0].cpu().contiguous(),
        frame_timestamps_us=future.frame_timestamps_us[0].cpu().contiguous())


def build(args):
    from starVLA.model.modules.qwen_world_postcot.teachers import DINOCurrentTeacher,WAJEPATargetTeacher
    from starVLA.rl.qwen_world_postcot.contracts import boundary_stop_due,require_dispatch
    if str(args.device).startswith('cuda'):
        require_dispatch(admit_long_job=True)
    manifest,records=load_population(args.population)
    if args.max_scenes<1 or args.max_seconds<=0 or args.shard_records<1:
        raise ValueError('Explicit positive scene/time/shard budgets required')
    if args.world_size<1 or not 0<=args.rank<args.world_size or args.cpu_threads<1 or args.throttle_seconds<0:
        raise ValueError('Builder rank outside fixed worker topology')
    root=Path(args.root); root.mkdir(parents=True,exist_ok=True)
    lock=(root/f'builder-rank{args.rank:04d}.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    torch.set_num_threads(args.cpu_threads)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    dino=DINOCurrentTeacher(args.dino_root,device=args.device,compute_dtype=torch.float32)
    wa=WAJEPATargetTeacher(args.wa_source,args.wa_checkpoint,device=args.device,compute_dtype=torch.bfloat16)
    example=dict(dino=torch.empty(3,9,16,1024,dtype=torch.float32),
        dino_valid=torch.empty(3,9,16,dtype=torch.bool),
        future=torch.empty(3,4,8,16,512,dtype=torch.bfloat16),
        future_valid=torch.empty(3,4,8,16,dtype=torch.bool),
        frame_timestamps_us=torch.empty(12,dtype=torch.int64))
    identity=dict(cohort=manifest['cohort'],population_sha256=file_sha256(args.population),
        records_sha256=manifest['records_sha256'],population_count=len(records),
        token_sha256=identity_hash(sorted(row['token'] for row in records)),
        camera_names=list(CAMERAS),dino=dino.teacher_identity,wa=wa.teacher_identity,
        layout=tensor_layout(example),code={name:file_sha256(Path(__file__).parents[2]/name) for name in
          ['tools/qwen_world_postcot/teacher_cache.py',
           'starVLA/model/modules/qwen_world_postcot/teachers.py',
           'starVLA/model/modules/qwen_world_postcot/teacher_targets.py']},
        student_weights_cached=False,teacher_role='supervision_only',
        frame_resolution=dict(schema='resolved_raw_log_offsets_v3',current='identical row.images as Qwen; PIL DINO',
            future_sensor_root='/mnt/navsim/trainval_all/trainval_sensor_blobs/trainval',
            explicit_complementary_fallback='/mnt/navsim/trainval_sensor_blobs/trainval',
            mirror_check='first16 both-available image SHA comparisons per producer; selected image SHA in every record',
            future_decode='official cv2 only; no unused PIL preprocessing for future frames',
            raw_log_root='/mnt/project/DriveDreamer-Policy/navsim_raw/navsim_logs/trainval',
            actual_image_hashes_in_each_record=True),
        storage='FP32 DINO/native BF16 WA after official trained projector; lossless raw record bytes')
    writer=BinaryTeacherCacheWriter(root,identity,rank=args.rank,shard_records=args.shard_records)
    decoder=BoundedImageDecoder(args.image_lru)
    resolver=CanonicalClipResolver()
    # Producer topology never determines consumer access or training order.
    priority_tokens=[]
    if args.priority_scenes:
        value=json.loads(Path(args.priority_scenes).read_text())
        priority_tokens=value['tokens'] if isinstance(value,dict) else value
    selected=partition_producer_records(records,args.rank,args.world_size,priority_tokens)
    priority_set=set(priority_tokens)
    priority_since_commit=0
    stop={'requested':False}
    def request_stop(signum,frame):
        stop['requested']=True
    for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGUSR1):
        signal.signal(sig,request_stop)
    started=time.monotonic(); added=0; skipped=0; failure=None;current_token=None
    try:
        for row in selected:
            current_token=row['token']
            if row['token'] in writer.existing:
                previous=writer.existing[row['token']]['metadata']
                if previous.get('source_record_hash')!=source_record_identity(row):
                    raise ValueError('A committed token changed source observation')
                skipped+=1; continue
            if os.environ.get('WORLD_POSTCOT_STOP_FILE') and boundary_stop_due():
                stop['requested']=True
            if stop['requested'] or added>=args.max_scenes or time.monotonic()-started>=args.max_seconds:
                break
            dino_input,clip,valid,timestamps,images=load_teacher_inputs(row,decoder,resolver)
            dino_targets=dino(dino_input,valid[:,3],timestamps[:,3])
            future_targets=wa(clip,valid,timestamps)
            tensors=target_tensors(dino_targets,future_targets)
            # Current3-view supervision is required. Future missing views are
            # explicit masked targets, not silently accepted random features.
            if not tensors['dino_valid'].all():
                raise ValueError('Current3-camera image absent from a legal training scene')
            writer.add(row['token'],tensors,dict(cohort=manifest['cohort'],
                source_record_hash=source_record_identity(row),images=images,
                raw_log=resolver.last_receipt,
                current_timestamp_us=int(timestamps[0,3]),
                valid_future_views=int(future_targets.valid.flatten(2).any(2).sum()),
                missing_frame_timestamps=int((timestamps<0).sum()),
                teacher_clip_timing_valid=bool(clip_timing_valid(timestamps).all())))
            added+=1
            if row['token'] in priority_set:
                priority_since_commit+=1
                # Expose the original training cursor promptly, then return
                # to the default ~856MB shards for bulk target production.
                if priority_since_commit>=16:
                    writer.commit();priority_since_commit=0
            elif priority_since_commit:
                writer.commit();priority_since_commit=0
            if args.throttle_seconds:
                time.sleep(args.throttle_seconds)
            if added%args.shard_records==0:
                print(json.dumps(dict(event='shard_committed',rank=args.rank,added=added,
                    seconds=time.monotonic()-started,images_decoded=decoder.decodes,
                    image_lru_hits=decoder.hits,bytes_read=decoder.bytes_read)),flush=True)
    except BaseException as error:
        failure=repr(error)
        raise
    finally:
        writer.close()
        elapsed=time.monotonic()-started
        receipt=dict(status='FAIL' if failure else ('STOPPED' if stop['requested'] else 'BUDGET_COMPLETE'),
            rank=args.rank,world_size=args.world_size,added=added,already_complete=skipped,
            seconds=elapsed,scenes_per_second=added/elapsed if elapsed else 0.,
            source_bytes_read=decoder.bytes_read,image_lru_hits=decoder.hits,
            image_decodes=decoder.decodes,dino_decodes=decoder.dino_decodes,
            mirror_SHA_checks=resolver.mirror_checks,
            failure=failure,last_token=current_token,identity=writer.identity['sha256'],
            CUDA_peak_allocated=torch.cuda.max_memory_allocated() if torch.cuda.is_available() and str(args.device).startswith('cuda') else 0,
            cache_complete=(root/'MANIFEST.json').exists(),
            no_optimizer=True,no_actor_forward=True,no_student_feature_cache=True)
        atomic_json(root/f'BUILD_RANK{args.rank:04d}.json',receipt)
        print(json.dumps(receipt),flush=True)


def verify(args):
    manifest,records=load_population(args.population)
    if args.finalize:
        result=finalize_teacher_cache(args.root,[row['token'] for row in records],
                                      verify_full_shards=args.full_shards)
    else:
        identity=_read_sealed(Path(args.root)/'IDENTITY.json')
        entries,shards=_cache_index(Path(args.root),identity)
        if args.full_shards:
            for name,info in shards.items():
                if file_sha256(Path(args.root)/name)!=info['sha256']:
                    raise ValueError('Whole teacher shard checksum mismatch')
        result=dict(status='COMPLETE' if (Path(args.root)/'MANIFEST.json').exists() else 'PARTIAL',
                    committed=len(entries),expected=len(records),shards=len(shards))
    reader=TeacherCacheReader(args.root,expected_cohort=manifest['cohort'],allow_building=not args.finalize)
    expected={row['token']:row for row in records}
    tokens=sorted(reader.records)
    random.Random(args.seed).shuffle(tokens)
    checked=0
    for token in tokens[:args.sample_records]:
        record=reader.get(token,source_record_hash=source_record_identity(expected[token]))
        if record['dino'].dtype!=torch.float32 or record['future'].dtype!=torch.bfloat16:
            raise ValueError('Teacher cache precision differs from primary recipe')
        if not torch.isfinite(record['dino'][record['dino_valid']]).all() or not torch.isfinite(record['future'][record['future_valid']]).all():
            raise ValueError('Nonfinite valid teacher target')
        times=record['frame_timestamps_us'][None]
        validate_clip_timestamps(times,allow_incomplete=True)
        if record['future_valid'].any() and not clip_timing_valid(times).all():
            raise ValueError('Non2Hz/missing teacher clip was incorrectly marked supervised')
        checked+=1
    reader.close()
    print(json.dumps(dict(status=result['status'],sample_records_checked=checked,
                          expected=len(records),committed=len(tokens))),flush=True)


def main():
    parser=argparse.ArgumentParser(__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    builder=sub.add_parser('build',help='Generate explicit bounded offline teacher targets')
    builder.add_argument('--population',required=True)
    builder.add_argument('--root',required=True)
    builder.add_argument('--dino-root',required=True)
    builder.add_argument('--wa-source',required=True)
    builder.add_argument('--wa-checkpoint',required=True)
    builder.add_argument('--device',default='cuda:0')
    builder.add_argument('--rank',type=int,default=0)
    builder.add_argument('--world-size',type=int,default=1)
    builder.add_argument('--max-scenes',type=int,required=True)
    builder.add_argument('--max-seconds',type=float,default=3600)
    builder.add_argument('--shard-records',type=int,default=256)
    builder.add_argument('--image-lru',type=int,default=64)
    builder.add_argument('--cpu-threads',type=int,default=2)
    builder.add_argument('--throttle-seconds',type=float,default=0.)
    builder.add_argument('--priority-scenes',help='Optional fixed legal token list produced first; consumption order is unchanged')
    checker=sub.add_parser('verify',help='Validate committed shards; optionally complete exact population')
    checker.add_argument('--population',required=True)
    checker.add_argument('--root',required=True)
    checker.add_argument('--full-shards',action='store_true')
    checker.add_argument('--finalize',action='store_true')
    checker.add_argument('--sample-records',type=int,default=64)
    checker.add_argument('--seed',type=int,default=42)
    args=parser.parse_args()
    if args.command=='build':
        build(args)
    else:
        verify(args)


if __name__=='__main__':
    main()

"""Inspect exact frozen teacher assets; optionally obtain the public WA release."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import tarfile
import time

import torch

from starVLA.model.modules.qwen_world_postcot.teachers import (
    DINOCurrentTeacher, WAJEPATargetTeacher, WA_REVISION, WA_SOURCE_COMMIT, WA_WEIGHT_SHA256)
from starVLA.model.modules.qwen_world_postcot.teacher_targets import atomic_json, identity_hash, file_sha256


def download_wa_assets(root):
    import requests
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    source = root/'WA-JEPA-source'
    if not source.exists():
        response = requests.get(f'https://codeload.github.com/AFARI-Research/WA-JEPA/tar.gz/{WA_SOURCE_COMMIT}', timeout=120)
        response.raise_for_status()
        archive = response.content
        with tarfile.open(fileobj=io.BytesIO(archive), mode='r:gz') as tar:
            for member in tar.getmembers():
                parts = Path(member.name).parts[1:]
                if not parts:
                    continue
                if member.issym() or member.islnk() or '..' in parts:
                    raise ValueError('Unsafe source archive entry')
                path = source.joinpath(*parts)
                if member.isdir():
                    path.mkdir(parents=True, exist_ok=True)
                elif member.isfile():
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(tar.extractfile(member).read())
        atomic_json(root/'WA_SOURCE_DOWNLOAD.json', dict(repository='AFARI-Research/WA-JEPA',
            commit=WA_SOURCE_COMMIT, archive_sha256=hashlib.sha256(archive).hexdigest()))
    checkpoint = root/'wa-jepa-model_state_dict.pt'
    expected_bytes = 1575763741
    if checkpoint.exists():
        if checkpoint.stat().st_size != expected_bytes or file_sha256(checkpoint) != WA_WEIGHT_SHA256:
            raise ValueError('Existing WA checkpoint is not the pinned official file')
        return source, checkpoint
    partial = checkpoint.with_suffix('.partial')
    # Bounded explicit attempts also recover a proxy terminating an otherwise
    # successful HTTP200 body early. Never publish a partial as a checkpoint.
    for attempt in range(5):
        offset = partial.stat().st_size if partial.exists() else 0
        if offset == expected_bytes:
            break
        if offset > expected_bytes:
            raise ValueError('Oversized partial WA checkpoint; preserve for diagnosis')
        url = (f'https://huggingface.co/AFARI-Research/WA-JEPA/resolve/{WA_REVISION}/model_state_dict.pt'
               f'?download=true&attempt={time.time_ns()}')
        headers = {'Accept-Encoding':'identity', 'Cache-Control':'no-cache'}
        if offset:
            headers['Range'] = f'bytes={offset}-'
        with requests.get(url, headers=headers, stream=True, timeout=(30,120)) as response:
            response.raise_for_status()
            if offset and (response.status_code != 206 or not response.headers.get('Content-Range','').startswith(f'bytes {offset}-')):
                raise ValueError('Resume Range not honored; retained original partial')
            with partial.open('ab' if offset else 'wb') as stream:
                for block in response.iter_content(8*1024*1024):
                    if block:
                        stream.write(block)
                stream.flush()
                os.fsync(stream.fileno())
    if partial.stat().st_size != expected_bytes or file_sha256(partial) != WA_WEIGHT_SHA256:
        raise ValueError('Official WA download incomplete or checksum mismatch; partial preserved')
    os.replace(partial, checkpoint)
    return source, checkpoint


def inspect_assets(dino_root, wa_source, wa_checkpoint, *, dino_compute='float32', wa_compute='bfloat16'):
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    dino = DINOCurrentTeacher(dino_root, compute_dtype=getattr(torch,dino_compute))
    dino_record = dict(identity=dino.teacher_identity, parameters=sum(p.numel() for p in dino.parameters()),
                       strict_load=True, all_frozen=all(not p.requires_grad for p in dino.parameters()))
    del dino
    wa = WAJEPATargetTeacher(wa_source, wa_checkpoint, compute_dtype=getattr(torch,wa_compute))
    wa_record = dict(identity=wa.teacher_identity, parameters=sum(p.numel() for p in wa.parameters()),
                    strict_load=True, all_frozen=all(not p.requires_grad for p in wa.parameters()))
    return dict(schema='world_postcot_teacher_assets_v1', status='VERIFIED_CPU_LOAD',
        dino=dino_record, wa_jepa=wa_record, GPU_forward_tested=False,
        warning='Public WA stage2 was trained on navtrain; internal DEV physical-log overlap may exist. No claim of teacher-unseen DEV.')


def audit_image_extraction(dino_root,wa_source,wa_checkpoint,population,cache_root,*,device):
    """Real images: stored exact values, repeatability, regrouping and RNG.

    Numerical differences between different batch shapes are measured in full,
    never hidden by sample-only feature comparisons or an expanded tolerance.
    """
    from tools.qwen_world_postcot.teacher_cache import load_population,load_teacher_inputs,BoundedImageDecoder,source_record_identity
    from starVLA.model.modules.qwen_world_postcot.teacher_targets import TeacherCacheReader
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    manifest,rows=load_population(population)
    reader=TeacherCacheReader(cache_root,expected_cohort=manifest['cohort'])
    selected=[]
    for row in rows:
        target=reader.get(row['token'],source_record_hash=source_record_identity(row))
        if target['future_valid'].all():
            selected.append((row,target))
        if len(selected)==2:break
    if len(selected)!=2:
        raise ValueError('Image extraction audit needs two real fully observed clips')
    dino=DINOCurrentTeacher(dino_root,device=device,compute_dtype=torch.float32)
    wa=WAJEPATargetTeacher(wa_source,wa_checkpoint,device=device,compute_dtype=torch.bfloat16)
    decoder=BoundedImageDecoder(64)
    inputs=[load_teacher_inputs(row,decoder) for row,_ in selected]
    rng_cpu=torch.get_rng_state().clone()
    rng_cuda=torch.cuda.get_rng_state() if str(device).startswith('cuda') else None
    def run(items):
        pixels=torch.cat([item[0] for item in items])
        clips=torch.cat([item[1] for item in items])
        valid=torch.cat([item[2] for item in items])
        times=torch.cat([item[3] for item in items])
        return dino(pixels,valid[:,3],times[:,3]),wa(clips,valid,times)
    one=[run([item]) for item in inputs]
    repeat=run([inputs[0]])
    grouped=run(inputs)
    shuffled=run(list(reversed(inputs)))
    duplicate=run([inputs[0],inputs[0]])
    def difference(left,right):
        a,b=left.float().cpu(),right.float().cpu()
        error=(a-b)
        return dict(bitwise_equal=torch.equal(left.cpu().view(torch.uint8),right.cpu().view(torch.uint8)),
            max_abs=float(error.abs().max()),rmse=float(error.square().mean().sqrt()),
            relative_l2=float(error.norm()/a.norm().clamp_min(1e-12)),elements=a.numel())
    results={}
    for name,index in [('dino',0),('future',1)]:
        raw=one[0][index].features[0]
        stored=selected[0][1][name]
        results[name]=dict(shape=list(raw.shape),dtype=str(raw.dtype),
            exact_storage_replay=difference(stored,raw),
            repeat=difference(raw,repeat[index].features[0]),
            regrouped=difference(raw,grouped[index].features[0]),
            shuffled=difference(raw,shuffled[index].features[1]),
            repeated_sample=difference(raw,duplicate[index].features[1]))
    unchanged=torch.equal(rng_cpu,torch.get_rng_state())
    if rng_cuda is not None:unchanged=unchanged and torch.equal(rng_cuda,torch.cuda.get_rng_state())
    exact=all(result['exact_storage_replay']['bitwise_equal'] and result['repeat']['bitwise_equal'] for result in results.values())
    return dict(schema='world_postcot_real_teacher_audit_v1',status='PASS' if exact and unchanged else 'INCONCLUSIVE',
        actual_tokens=[row['token'] for row,_ in selected],device=str(device),
        GPU_name=torch.cuda.get_device_name() if str(device).startswith('cuda') else None,
        dino_identity=dino.teacher_identity,wa_identity=wa.teacher_identity,comparisons=results,
        frozen_eval=all(not parameter.requires_grad for parameter in list(dino.parameters())+list(wa.parameters()))
            and not dino.training and not wa.training,
        RNG_unchanged=bool(unchanged),
        batch_shape_claim='Measured separately; exact storage replay uses original per-scene3-view production shape.',
        peak_GPU_allocated=torch.cuda.max_memory_allocated() if str(device).startswith('cuda') else 0,
        no_student_forward=True,no_GT_or_score_read=True)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--dino-root', required=True)
    parser.add_argument('--wa-source')
    parser.add_argument('--wa-checkpoint')
    parser.add_argument('--download-wa-root')
    parser.add_argument('--output', required=True)
    parser.add_argument('--dino-compute', choices=['float32','bfloat16'], default='float32')
    parser.add_argument('--wa-compute', choices=['float32','bfloat16'], default='bfloat16')
    parser.add_argument('--image-audit-population')
    parser.add_argument('--image-audit-cache')
    parser.add_argument('--device',default='cpu')
    args = parser.parse_args()
    if args.download_wa_root:
        if args.wa_source or args.wa_checkpoint:
            parser.error('Use existing source/checkpoint or a download root, not both')
        args.wa_source,args.wa_checkpoint = download_wa_assets(args.download_wa_root)
    if not args.wa_source or not args.wa_checkpoint:
        parser.error('A complete official WA source and checkpoint are required')
    if bool(args.image_audit_population)!=bool(args.image_audit_cache):
        parser.error('Real-image audit requires both population and complete teacher cache')
    if args.image_audit_population:
        if args.dino_compute!='float32' or args.wa_compute!='bfloat16':
            parser.error('The main real-image audit is fixed to FP32 DINO and official BF16 WA')
        if str(args.device).startswith('cuda'):
            from starVLA.rl.qwen_world_postcot.contracts import require_dispatch
            require_dispatch(admit_long_job=True)
        result=audit_image_extraction(args.dino_root,args.wa_source,args.wa_checkpoint,
            args.image_audit_population,args.image_audit_cache,device=args.device)
    else:
        if args.device!='cpu':
            parser.error('Asset inspection loads on CPU; --device applies to real-image audit only')
        result = inspect_assets(args.dino_root,args.wa_source,args.wa_checkpoint,
                               dino_compute=args.dino_compute,wa_compute=args.wa_compute)
    result['sha256'] = identity_hash(result)
    path = Path(args.output)
    if path.exists():
        if json.loads(path.read_text()) != result:
            raise ValueError('Asset audit already binds different weights/runtime')
    else:
        atomic_json(path,result)
    print(json.dumps(result))
    if result['status']=='INCONCLUSIVE':
        raise SystemExit(2)


if __name__ == '__main__':
    main()

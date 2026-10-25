"""Train a separate deterministic teacher-feature RGB decoder; no actor loss."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import signal
import time

import numpy as np
from PIL import Image
import torch

from starVLA.model.modules.qwen_world_postcot.rgb_decoder import DINORGBDecoder,rgb_reconstruction_loss
from starVLA.model.modules.qwen_world_postcot.teacher_targets import (
    TeacherCacheReader, atomic_json, file_sha256, identity_hash, CAMERAS)
from .teacher_cache import load_population,source_record_identity


def rgb_targets(row):
    images=[]
    for camera in CAMERAS:
        path=row['images'][camera]
        if path is None:
            raise ValueError('RGB decoder requires real current3-view images')
        with Image.open(path) as opened:
            value=np.asarray(opened.convert('RGB').resize((512,288),Image.Resampling.BICUBIC)).copy()
        images.append(torch.from_numpy(value).permute(2,0,1).float().div_(255))
    return torch.stack(images)


def rng_state():
    return dict(python=random.getstate(),numpy=np.random.get_state(),cpu=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state):
    random.setstate(state['python']); np.random.set_state(state['numpy']); torch.set_rng_state(state['cpu'])
    if state['cuda']:
        torch.cuda.set_rng_state_all(state['cuda'])


def save_checkpoint(path,state):
    temporary=path.with_name(path.name+'.partial')
    torch.save(state,temporary)
    with temporary.open('rb') as stream:
        os.fsync(stream.fileno())
    os.replace(temporary,path)
    atomic_json(path.with_suffix('.receipt.json'),dict(path=path.name,sha256=file_sha256(path),
        step=state['step'],identity=state['identity'],complete=True))


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--population',required=True)
    parser.add_argument('--teacher-cache',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--updates',type=int,default=10000)
    parser.add_argument('--batch-scenes',type=int,default=2)
    parser.add_argument('--lr',type=float,default=1e-4)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--save-every',type=int,default=250)
    parser.add_argument('--resume')
    parser.add_argument('--allow-building-cache',action='store_true')
    parser.add_argument('--max-seconds',type=float,default=7200)
    args=parser.parse_args()
    if min(args.updates,args.batch_scenes,args.save_every)<1 or args.lr<=0 or args.max_seconds<=0:
        parser.error('Positive explicit training/save/time budgets required')
    manifest,rows=load_population(args.population)
    if str(manifest['cohort']).upper()!='TRAIN':
        raise ValueError('RGB visualization decoder training must use TRAIN only')
    reader=TeacherCacheReader(args.teacher_cache,expected_cohort=manifest['cohort'],
                              allow_building=args.allow_building_cache)
    # A partial pilot is an explicitly separate decoder training population;
    # bind that token list, never silently refresh it after initialization.
    eligible=[row for row in rows if row['token'] in reader.records]
    if not eligible:
        raise ValueError('No committed real teacher records available for decoder')
    root=Path(args.output); root.mkdir(parents=True,exist_ok=True)
    if not args.resume and (root/'resolved.json').exists():
        raise ValueError('Existing decoder run requires explicit --resume')
    if args.resume:
        fixed_tokens=json.loads((root/'TRAINING_TOKENS.json').read_text())
        by_token={row['token']:row for row in rows}
        if any(token not in reader.records for token in fixed_tokens):
            raise ValueError('Decoder resume lost a bound teacher record')
        eligible=[by_token[token] for token in fixed_tokens]
    else:
        atomic_json(root/'TRAINING_TOKENS.json',[row['token'] for row in eligible])
    config=dict(vars(args),teacher_identity=reader.identity['sha256'],
        population_sha256=file_sha256(args.population),
        training_tokens_sha256=identity_hash([row['token'] for row in eligible]),
        training_scenes=len(eligible),loss='FP32 L1 teacher features→RGB only',
        planning_loss=False,model=dict(feature_dim=1024,channels=[256,128,96,64,32,16]))
    identity=identity_hash({k:v for k,v in config.items() if k not in ('resume','max_seconds')})
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    model=DINORGBDecoder().to(args.device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=.01)
    sampler=torch.Generator().manual_seed(args.seed+1)
    order=torch.randperm(len(eligible),generator=sampler).tolist(); cursor=0; epoch=0; step=0
    if args.resume:
        receipt=json.loads(Path(args.resume).with_suffix('.receipt.json').read_text())
        if not receipt['complete'] or file_sha256(args.resume)!=receipt['sha256']:
            raise ValueError('Decoder checkpoint receipt does not match file')
        state=torch.load(args.resume,map_location='cpu',weights_only=False)
        if state['identity']!=identity:
            raise ValueError('Decoder experiment/data/teacher identity differs')
        model.load_state_dict(state['model'],strict=True);optimizer.load_state_dict(state['optimizer'])
        sampler.set_state(state['sampler_rng']); order=state['order'];cursor=state['cursor'];epoch=state['epoch'];step=state['step']
        restore_rng(state['rng'])
    atomic_json(root/'resolved.json',config)
    stop={'requested':False}
    def stopping(signum,frame): stop['requested']=True
    for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGUSR1): signal.signal(sig,stopping)
    started=time.monotonic();last=None
    while step<args.updates and not stop['requested'] and time.monotonic()-started<args.max_seconds:
        if os.environ.get('WORLD_POSTCOT_STOP_FILE'):
            from starVLA.rl.qwen_world_postcot.contracts import boundary_stop_due
            if boundary_stop_due():
                stop['requested']=True
                break
        selected=[]
        for _ in range(args.batch_scenes):
            if cursor==len(order):
                epoch+=1;cursor=0;order=torch.randperm(len(eligible),generator=sampler).tolist()
            selected.append(eligible[order[cursor]]);cursor+=1
        targets=[reader.get(row['token'],source_record_hash=source_record_identity(row)) for row in selected]
        features=torch.stack([target['dino'] for target in targets]).to(args.device)
        masks=torch.stack([target['dino_valid'].all((-2,-1)) for target in targets]).to(args.device)
        rgb=torch.stack([rgb_targets(row) for row in selected]).to(args.device)
        optimizer.zero_grad(set_to_none=True)
        use_amp=str(args.device).startswith('cuda')
        with torch.autocast(device_type='cuda' if use_amp else 'cpu',dtype=torch.bfloat16,enabled=use_amp):
            predicted=model(features)
        loss=rgb_reconstruction_loss(predicted,rgb,masks)
        if not torch.isfinite(loss): raise FloatingPointError('Nonfinite RGB decoder loss')
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();step+=1
        last=float(loss.detach())
        if step%10==0:
            print(json.dumps(dict(step=step,loss_l1=last,seconds=time.monotonic()-started)),flush=True)
        if step%args.save_every==0 or step==args.updates:
            save_checkpoint(root/f'step_{step:06d}.pt',dict(model=model.state_dict(),optimizer=optimizer.state_dict(),
                rng=rng_state(),sampler_rng=sampler.get_state(),order=order,cursor=cursor,epoch=epoch,
                step=step,identity=identity,config=config))
    if step and not (root/f'step_{step:06d}.pt').exists():
        save_checkpoint(root/f'step_{step:06d}.pt',dict(model=model.state_dict(),optimizer=optimizer.state_dict(),
            rng=rng_state(),sampler_rng=sampler.get_state(),order=order,cursor=cursor,epoch=epoch,
            step=step,identity=identity,config=config))
    atomic_json(root/'RUN_STATUS.json',dict(status='COMPLETE' if step==args.updates else 'STOPPED',
        step=step,epoch=epoch,cursor=cursor,last_l1=last,seconds=time.monotonic()-started,
        teacher_identity=reader.identity['sha256'],planning_training_blocked_by_decoder=False))
    reader.close()


if __name__=='__main__': main()

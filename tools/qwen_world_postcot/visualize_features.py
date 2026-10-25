"""One frozen deterministic decoder for aligned teacher/student comparisons."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from starVLA.model.modules.qwen_world_postcot.rgb_decoder import DINORGBDecoder
from starVLA.model.modules.qwen_world_postcot.teacher_targets import TeacherCacheReader,file_sha256,atomic_json
from .teacher_cache import load_population,source_record_identity
from .train_rgb_decoder import rgb_targets


def shared_pca(teacher,student):
    """Fit once on teacher; use the exact same centre/basis/range for student."""
    x=teacher.reshape(-1,teacher.shape[-1]).float()
    mean=x.mean(0)
    _,_,v=torch.linalg.svd(x-mean,full_matrices=False)
    basis=v[:3].T
    t=(x-mean)@basis
    s=(student.reshape(-1,student.shape[-1]).float()-mean)@basis
    low,high=t.amin(0),t.amax(0)
    return ((t-low)/(high-low).clamp_min(1e-6)).clamp(0,1).reshape(*teacher.shape[:-1],3), \
           ((s-low)/(high-low).clamp_min(1e-6)).clamp(0,1).reshape(*student.shape[:-1],3)


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--population',required=True)
    parser.add_argument('--teacher-cache',required=True)
    parser.add_argument('--decoder',required=True)
    parser.add_argument('--student-features',required=True,help='Bound dict with teacher_identity and features{token:tensor[3,9,16,1024]}')
    parser.add_argument('--token',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--shared-pca',action='store_true')
    args=parser.parse_args()
    manifest,rows=load_population(args.population)
    row=next(row for row in rows if row['token']==args.token)
    reader=TeacherCacheReader(args.teacher_cache,expected_cohort=manifest['cohort'])
    record=reader.get(args.token,source_record_hash=source_record_identity(row))
    state=torch.load(args.decoder,map_location='cpu',weights_only=False)
    receipt=json.loads(Path(args.decoder).with_suffix('.receipt.json').read_text())
    if not receipt['complete'] or receipt['sha256']!=file_sha256(args.decoder):
        raise ValueError('Decoder checkpoint is not complete')
    decoder=DINORGBDecoder().to(args.device).eval().requires_grad_(False)
    decoder.load_state_dict(state['model'],strict=True)
    teacher=record['dino']
    student_record=torch.load(args.student_features,map_location='cpu',weights_only=True)
    if not isinstance(student_record,dict) or student_record.get('teacher_identity')!=reader.identity['sha256']:
        raise ValueError('Student feature visualization requires bound teacher_identity and features[token]')
    student=student_record['features'][args.token]
    if student.shape!=teacher.shape or not torch.isfinite(student).all():
        raise ValueError('Student feature contract mismatch/nonfinite')
    with torch.no_grad():
        teacher_rgb=decoder(teacher.to(args.device)).cpu()
        student_rgb=decoder(student.to(args.device)).cpu()
    original=rgb_targets(row)
    error=(student_rgb-original).abs()
    # Columns are original / teacher reconstruction / student / absolute error.
    panels=[original,teacher_rgb,student_rgb,error]
    image=torch.cat([torch.cat(list(panel),dim=-1) for panel in panels],dim=-2)
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    Image.fromarray((image.permute(1,2,0).clamp(0,1).numpy()*255).round().astype(np.uint8)).save(output)
    if args.shared_pca:
        t,s=shared_pca(teacher,student)
        pca=torch.cat([torch.cat(list(t),dim=1),torch.cat(list(s),dim=1)],dim=0)
        Image.fromarray((pca.numpy()*255).round().astype(np.uint8)).resize((1536,576),Image.Resampling.NEAREST).save(output.with_name(output.stem+'_shared_pca.png'))
    atomic_json(output.with_suffix('.json'),dict(token=args.token,teacher_identity=reader.identity['sha256'],
        decoder_sha256=receipt['sha256'],student_features_sha256=file_sha256(args.student_features),
        teacher_l1=float((teacher_rgb-original).abs().mean()),student_l1=float(error.mean()),
        layout='rows=original,teacher,student,error; columns=cam_l0,cam_f0,cam_r0',
        decoder_frozen=True,shared_pca=args.shared_pca))
    reader.close()


if __name__=='__main__': main()

#!/usr/bin/env python3
"""Portable asset validation and training-side frozen-Base inference smoke.

This is NOT the pending 11-arm full GRPO training executor.
"""
import argparse,hashlib,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
os.environ['PYTHONDONTWRITEBYTECODE']='1';sys.dont_write_bytecode=True

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()

def config(path):
    path=Path(path).resolve();c=json.loads(path.read_text())
    for key in ['checkpoint','model_config','source_root','tokenizer_root','navsim_root','train_pairs','test_pairs','raw_meta_root','output_root']:
        p=Path(c[key]).expanduser();c[key]=str(p if p.is_absolute() else (path.parent/p).resolve())
    return c

def mapped(path,c):
    p=str(path)
    for a,b in sorted(c.get('path_prefix_map',{}).items(),key=lambda x:-len(x[0])):
        if p==a or p.startswith(a.rstrip('/')+'/'):return b.rstrip('/')+p[len(a.rstrip('/')):]
    return p

def rows(c,split):return [json.loads(line) for line in Path(c[split+'_pairs']).read_text().splitlines() if line]

def scene(row,c):
    sys.path.insert(0,str(ROOT/'vendor/mining_v3r2'))
    from common import raw_scene
    s=raw_scene({'raw_meta':str(Path(c['raw_meta_root'])/(row['token']+'.pkl'))})
    s['image_paths']=[mapped(p,c) for p in s['image_paths']]
    return s

def validate(c,full_hash=False,deep=False):
    import numpy as np
    expected=json.loads((ROOT/'data_interface/manifest_identity.json').read_text());groups={};tokens={};counts={};failures=[]
    for key in ['checkpoint','model_config','source_root','tokenizer_root','navsim_root','train_pairs','test_pairs','raw_meta_root']:
        if not Path(c[key]).exists():failures.append('Missing '+key+': '+c[key])
    for name in ['model.safetensors','tokenizer.json','tokenizer_config.json','config.json']:
        if not (Path(c['tokenizer_root'])/name).is_file():failures.append('Missing tokenizer/init file: '+name)
    if failures:return {'status':'FAIL','errors':failures}
    if full_hash and digest(c['checkpoint'])!=c['checkpoint_sha256']:failures.append('checkpoint SHA256 mismatch')
    score=json.loads((ROOT/'data_interface/scoring_identity.json').read_text())
    for rel,h in score['source_hashes'].items():
        path=Path(c['navsim_root'])/rel
        if not path.is_file() or digest(path)!=h:failures.append('scorer source mismatch: '+rel)
    for split in ['train','test']:
        data=rows(c,split);counts[split]=len(data)
        if len(data)!=expected[split]['count'] or digest(c[split+'_pairs'])!=expected[split]['sha256']:failures.append(split+' manifest identity mismatch')
        tokens[split]={r['token'] for r in data};groups[split]={r['log'].rsplit('_',2)[0] for r in data}
        if len(tokens[split])!=len(data):failures.append(split+' duplicate tokens')
        for r in data:
            t=r['token'];paths={'raw_meta':Path(c['raw_meta_root'])/(t+'.pkl'),'metric_cache':Path(mapped(r['source_paths']['metric'],c))}
            for k,p in paths.items():
                if not p.is_file():failures.append(t+' missing '+k)
                elif deep and digest(p)!=r[k+'_sha256']:failures.append(t+' changed '+k)
            for field in ['input_trajectory','gt_trajectory']:
                a=np.asarray(r[field]);
                if a.shape!=(8,3) or not np.isfinite(a).all() or np.any(np.abs(a[:,2])>np.pi+1e-5):failures.append(t+' invalid '+field)
            if r['checkpoint_sha256']!=c['checkpoint_sha256']:failures.append(t+' wrong checkpoint provenance')
            if r['score_protocol_sha256']!=score['sha256']:failures.append(t+' wrong scorer provenance')
            if not r['training_ready'] or not r['guidance']['guidance_text']:failures.append(t+' invalid eligibility/guidance')
            if deep and paths['raw_meta'].is_file():
                s=scene(r,c)
                if not np.allclose(s['gt'],r['gt_trajectory'],rtol=0,atol=1e-8):failures.append(t+' GT frame mismatch')
                for p in s['image_paths']:
                    if not Path(p).is_file():failures.append(t+' missing image: '+p)
    if tokens['train']&tokens['test']:failures.append('Token leakage')
    if groups['train']&groups['test']:failures.append('Drive leakage')
    return {'status':'PASS' if not failures else 'FAIL','counts':counts,'drive_counts':{k:len(v) for k,v in groups.items()},'checkpoint_hashed':full_hash,'raw_assets_hashed_and_images_checked':deep,'errors':failures}

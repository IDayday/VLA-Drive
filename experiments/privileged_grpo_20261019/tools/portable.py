#!/usr/bin/env python3
"""Portable asset validation and training-side frozen-Base inference smoke.

This is NOT the pending 11-arm full GRPO training executor.
"""
import argparse,hashlib,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
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
    sys.path.insert(0,str(ROOT/'reference/mining_v3r2'))
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

def infer(c,args):
    # Fail before importing model code if runtime differs or identity is wrong.
    import numpy as np
    check=validate(c,full_hash=True)
    if check['status']!='PASS':raise RuntimeError(json.dumps(check))
    out=Path(c['output_root'])/args.run_name
    out.mkdir(parents=True,exist_ok=False)
    for key,name in [('HF_HOME','hf'),('TORCH_HOME','torch'),('TRITON_CACHE_DIR','triton'),('XDG_CACHE_HOME','cache')]:
        os.environ[key]=str(out/name)
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    import torch
    if not torch.cuda.is_available():raise RuntimeError('CUDA-compatible accelerator unavailable')
    sys.path.insert(0,c['source_root']);sys.path.insert(0,str(ROOT/'reference/grpo_smoke_v1'))
    import smoke as core
    from common import decode
    from pipeline import load_images
    from score_trace import score_one
    from omegaconf import OmegaConf
    from starVLA.model.framework.QwenOFT import Qwenvl_OFT
    cfg=OmegaConf.load(c['model_config']);cfg.framework.qwenvl.base_vlm=c['tokenizer_root'];cfg.framework.qwenvl.attn_implementation='sdpa'
    model=Qwenvl_OFT(cfg,infer_not_load_wan=1)
    state=torch.load(c['checkpoint'],map_location='cpu',weights_only=True,mmap=True)
    model.load_state_dict(state,strict=True);del state
    device='cuda:0';model=model.to(device).float().eval().requires_grad_(False)
    # Deliberately only a training scene. No new held-out evaluation or hyperparameter search.
    row=sorted(rows(c,'train'),key=lambda r:r['token'])[args.scene_index];s=scene(row,c);images=load_images(s)
    request=json.loads((ROOT/'protocol/prompt_contract.json').read_text())['request']
    text=s['instruction']
    if args.arm!='A':text+='\n'+request
    if args.arm in ['C','E']:
        text+='\nInitial trajectory: '+'; '.join(f'w{i+1}: x={p[0]:+.4f}m, y={p[1]:+.4f}m, yaw={p[2]:+.5f}rad' for i,p in enumerate(row['input_trajectory']))
    if args.arm=='E':text+='\n'+row['guidance']['guidance_text']
    seed=int(hashlib.sha256(('portable-smoke:'+row['token']).encode()).hexdigest()[:8],16)
    with torch.no_grad():
        batch=core.build_inputs(model,s,text,images);cond=core.context(model,batch,s['state'])
        raw,_=core.sample(model,cond,seed,group_size=8,eta=.1)
    metric=mapped(row['source_paths']['metric'],c)
    replay=score_one(row['input_trajectory'],metric,c)
    if replay['metrics']!=row['base_score']['metrics']:raise RuntimeError('Frozen Base PDMS replay mismatch')
    result={'token':row['token'],'arm':args.arm,'seed':seed,'group_size':8,'checkpoint_sha256':c['checkpoint_sha256'],'training':False,'base_score_replay_exact':True,'candidates':[]}
    for i,a in enumerate(raw.cpu().numpy()):
        trajectory=decode(a);score=score_one(trajectory,metric,c)
        result['candidates'].append({'candidate':i,'trajectory':trajectory.tolist(),'metrics':score['metrics']})
    (out/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
    (out/'CONFIG.json').write_text(json.dumps(c,indent=2)+'\n')
    return {'status':'PASS','output':str(out),'candidates':8,'training':False,'note':'New portability smoke; not a full paired experiment or historical rollout replay.'}

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);s=p.add_subparsers(dest='command',required=True)
    a=s.add_parser('validate');a.add_argument('--full-hash',action='store_true');a.add_argument('--deep',action='store_true')
    a=s.add_parser('infer-smoke');a.add_argument('--arm',choices=['A','B','C','E'],default='A');a.add_argument('--scene-index',type=int,default=0);a.add_argument('--run-name',required=True)
    s.add_parser('train-full')
    args=p.parse_args()
    if args.command=='train-full':p.error('11-arm full training is not implemented; epochs/budget and full F donors are not locked. No job started.')
    c=config(args.config);result=validate(c,args.full_hash,args.deep) if args.command=='validate' else infer(c,args)
    print(json.dumps(result,ensure_ascii=False,indent=2))
    if result['status']!='PASS':sys.exit(1)
if __name__=='__main__':main()

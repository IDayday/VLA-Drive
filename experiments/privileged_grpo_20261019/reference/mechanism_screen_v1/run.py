"""Frozen Base, five input conditions, common random numbers, three replicates."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
import numpy as np
import torch

ROOT=Path(__file__).resolve().parent
TASK=ROOT.parent
sys.path.insert(0,str(TASK/'grpo_smoke_v1'))
import smoke as core
from grpo_math import transition
from diversity import summarize
from common import read,write,file_hash,raw_scene,decode
from pipeline import load_images
from score_trace import score_one,protocol


def append(path,value):
    with path.open('a') as f:f.write(json.dumps(value,ensure_ascii=False,allow_nan=False)+'\n')
def hash_tensor(x):return hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
def seed(token,rep):
    text=f'probe-v2:evaluation:{token}' if rep==0 else f'mechanism-v1:{rep}:{token}'
    return int(hashlib.sha256(text.encode()).hexdigest()[:8],16)
def delta(a,b):
    a=a.float().flatten();b=b.float().flatten();d=a-b
    return dict(rms=float(d.square().mean().sqrt()),relative_l2=float(d.norm()/(b.norm()+1e-12)),cosine=float(torch.nn.functional.cosine_similarity(a[None],b[None])))


@torch.no_grad()
def sample(model,cond,noise):
    x=noise[0];v0=None
    for step in range(10):
        velocity=core.velocity(model,cond,x,step)
        if step==0:v0=velocity.clone()
        mean,std=transition(x,velocity,step)
        x=mean+std*noise[step+1]
        assert torch.isfinite(x).all()
    return x,v0


def main(out):
    start=time.monotonic();plan=read(ROOT/'PLAN.json');data=read(ROOT/'DATA.json');cfg=read(TASK/'mining_v3r2/CONFIG.json')
    def status(stage,**extra):
        elapsed=time.monotonic()-start
        if elapsed>plan['budget']['wall_seconds']:raise TimeoutError('Fixed screening budget exceeded')
        write(out/'STATUS.json',dict(stage=stage,pid=os.getpid(),elapsed_s=elapsed,updated_utc=datetime.now(timezone.utc).isoformat(),**extra),replace=True)
    status('loading');assert torch.cuda.is_available() and torch.cuda.device_count()==1
    assert file_hash(data['source'])==data['source_sha256'] and file_hash(cfg['checkpoint'])==cfg['checkpoint_sha256']
    assert (ROOT/'DATA.json').read_bytes()==(TASK/'grpo_probe_v2/DATA.json').read_bytes()
    scenes=read(ROOT/'PROMPTS.json')['scenes'];donors=read(ROOT/'DONORS.json')
    assert len(scenes)==32 and donors['exact_instruction_token_count_matches']==32
    all_rows={r['token']:r for r in map(json.loads,Path(data['source']).open())}
    assert protocol(cfg)['sha256']==all_rows[scenes[0]['token']]['score_protocol_sha256']
    sys.path.insert(0,cfg['source_root'])
    from omegaconf import OmegaConf
    from starVLA.model.framework.QwenOFT import Qwenvl_OFT
    model_cfg=OmegaConf.load(Path(cfg['run_root'])/'config.yaml')
    model_cfg.framework.qwenvl.base_vlm=cfg['tokenizer_root'];model_cfg.framework.qwenvl.attn_implementation='sdpa'
    model=Qwenvl_OFT(model_cfg,infer_not_load_wan=1)
    state=torch.load(cfg['checkpoint'],map_location='cpu',weights_only=True,mmap=True)
    model.load_state_dict(state,strict=True);del state
    model=model.to('cuda:0').float().eval().requires_grad_(False)
    assert not any(p.requires_grad for p in model.parameters())
    names=core.witnesses(model);witnesses={k:core.witness_value(model,v) for k,v in names.items()}
    old={(r['token'],r['candidate']):r for r in map(json.loads,(TASK/'grpo_probe_v2/runs/attempt01/rollouts.jsonl').open()) if r['stage']=='initial' and r['arm']=='plain'}
    files=[ROOT/n for n in ['run.py','prepare.py','diversity.py','PLAN.json','DATA.json','DONORS.json','PROMPTS.json']]+[TASK/'grpo_smoke_v1'/n for n in ['smoke.py','grpo_math.py']]
    write(out/'CONFIG.json',dict(plan=plan,data=data,checkpoint_sha256=cfg['checkpoint_sha256'],device=torch.cuda.get_device_name(0),parameter_count=sum(p.numel() for p in model.parameters()),trainable_parameters=0,source_sha256={str(p):file_hash(p) for p in files},start_utc=datetime.now(timezone.utc).isoformat()))
    counts={'groups':0,'candidates':0,'scenes':0};replay={'candidates':0,'trajectory_max_abs_error':0.,'metric_max_abs_error':0.};matched=0
    (out/'features').mkdir()
    with torch.no_grad():
        for index,prompt in enumerate(scenes):
            token=prompt['token'];row=all_rows[token];scene=raw_scene(row['source_paths']);images=load_images(scene)
            assert prompt['arms']['A']['text']==scene['instruction']
            status('scene',scene_index=index,token=token,counts=counts)
            inputs={};queries={};conditions={};arrays={};input_rows={}
            for arm in 'ABCDE':
                batch=core.build_inputs(model,scene,prompt['arms'][arm]['text'],images)
                ids=batch['input_ids'];positions={k:model._find_token_positions(ids,model._special_token_ids[k]).cpu().tolist() for k in ['history','action']}
                with torch.autocast('cuda',enabled=False):q=core.queries(model,batch,scene['state']);cond=model.action_model.qwen_proj(q.float())
                assert torch.isfinite(q).all() and torch.isfinite(cond).all()
                queries[arm]=q;conditions[arm]=cond;arrays['query_'+arm]=q.float().cpu().numpy();arrays['context_'+arm]=cond.cpu().numpy()
                input_rows[arm]=dict(token=token,arm=arm,text_sha256=prompt['arms'][arm]['sha256'],input_ids_sha256=hash_tensor(ids),input_token_count=int(ids.numel()),special_positions=positions,pixel_values_sha256=hash_tensor(batch['pixel_values']),state_sha256=hashlib.sha256(np.asarray(scene['state']).tobytes()).hexdigest())
                inputs[arm]=ids.cpu().numpy();del batch
            assert input_rows['C']['input_token_count']==input_rows['D']['input_token_count']
            assert input_rows['C']['special_positions']==input_rows['D']['special_positions']
            cids=inputs['C'].flatten();dids=inputs['D'].flatten();changed=np.where(cids!=dids)[0]
            assert len(changed)>0
            assert np.array_equal(cids[:changed[0]],dids[:changed[0]]) and np.array_equal(cids[changed[-1]+1:],dids[changed[-1]+1:])
            matched+=1
            for arm in 'ABCDE':append(out/'inputs.jsonl',input_rows[arm])
            append(out/'token_matching.jsonl',dict(token=token,full_token_count=input_rows['C']['input_token_count'],C_D_special_positions_equal=True,changed_token_count=len(changed),first_changed_token=int(changed[0]),last_changed_token=int(changed[-1])))
            representation=[]
            for left,right in [('B','A'),('C','B'),('C','D'),('D','A'),('E','C')]:
                representation.append(dict(contrast=left+'-'+right,query=delta(queries[left],queries[right]),projected_context=delta(conditions[left],conditions[right])))
            append(out/'representations.jsonl',dict(token=token,contrasts=representation))
            for rep in range(3):
                rng=torch.Generator(device='cuda').manual_seed(seed(token,rep))
                noise=[torch.randn((8,8,4),device='cuda',dtype=torch.float32,generator=rng) for _ in range(11)]
                noise_hash=hash_tensor(torch.stack(noise));v0={}
                for arm in 'ABCDE':
                    status('sampling',scene_index=index,token=token,replicate=rep,arm=arm,counts=counts)
                    raw,velocity=sample(model,conditions[arm],noise);v0[arm]=velocity
                    if index==0 and rep==0 and arm=='A':
                        original,_=core.sample(model,conditions[arm],seed(token,rep),group_size=8)
                        assert torch.equal(raw,original),'Shared-noise sampler differs from accepted sampler'
                    poses=[];metrics=[];gid=f'{token}:{rep}:{arm}'
                    for candidate,x in enumerate(raw.cpu().numpy()):
                        p=decode(x);score=score_one(p,row['source_paths']['metric'],cfg);m=score['metrics']
                        assert all(np.isfinite(v) and 0<=v<=1 for v in m.values())
                        if arm=='A' and rep==0:
                            original=old[token,candidate];te=float(np.max(np.abs(p-np.asarray(original['trajectory']))));me=max(abs(m[k]-v) for k,v in original['metrics'].items())
                            assert te<1e-6 and me<1e-9,(token,te,me)
                            replay['candidates']+=1;replay['trajectory_max_abs_error']=max(te,replay['trajectory_max_abs_error']);replay['metric_max_abs_error']=max(me,replay['metric_max_abs_error'])
                        append(out/'rollouts.jsonl',dict(group_id=gid,token=token,log=row['log'],arm=arm,replicate=rep,seed=seed(token,rep),noise_sha256=noise_hash,candidate=candidate,trajectory=p.tolist(),metrics=m))
                        poses.append(p);metrics.append(m);counts['candidates']+=1
                    append(out/'groups.jsonl',dict(group_id=gid,token=token,log=row['log'],arm=arm,replicate=rep,seed=seed(token,rep),noise_sha256=noise_hash,rewards=[m['PDMS'] for m in metrics],**summarize(poses,metrics)))
                    counts['groups']+=1;del raw
                for arm in 'ABCDE':arrays[f'velocity_rep{rep}_{arm}']=v0[arm].cpu().numpy()
                append(out/'velocity_contrasts.jsonl',dict(token=token,replicate=rep,noise_sha256=noise_hash,contrasts={left+'-'+right:delta(v0[left],v0[right]) for left,right in [('B','A'),('C','B'),('C','D'),('D','A'),('E','C')]}))
            np.savez_compressed(out/'features'/f'{token}.npz',**arrays)
            counts['scenes']+=1
            print(json.dumps(dict(event='scene_complete',token=token,counts=counts,elapsed_s=time.monotonic()-start)),flush=True)
            if index==0:write(out/'SMOKE.json',dict(status='PASS',first_scene=token,candidates=120,sampler_exact_replay=True,C_D_full_token_count_and_special_positions_equal=True,baseline_replay=replay.copy()))
            del inputs,queries,conditions,arrays,images,noise,v0
    assert counts=={'groups':480,'candidates':3840,'scenes':32} and replay['candidates']==256 and matched==32
    assert all(torch.equal(core.witness_value(model,n),witnesses[k]) for k,n in names.items())
    result=dict(status='COMPLETE',counts=counts,elapsed_s=time.monotonic()-start,peak_gpu_allocated_gib=torch.cuda.max_memory_allocated()/2**30,baseline_replay=replay,C_D_matched_scenes=matched,model_witnesses_unchanged=True,training_performed=False,final_test_used=False)
    write(out/'RESULT.json',result);status('complete',result=result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True);args=parser.parse_args()
    out=ROOT/'runs'/args.run;out.mkdir(parents=True,exist_ok=False)
    try:main(out)
    except BaseException as e:write(out/'ERROR.json',dict(type=type(e).__name__,message=str(e),traceback=traceback.format_exc()));raise

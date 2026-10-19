"""Frozen F condition with cached paired controls and exact C replay."""
import importlib.util
import argparse
from pathlib import Path
import sys,os,time,json,hashlib,traceback
from datetime import datetime,timezone
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent; TASK=ROOT.parent; PARENT=TASK/'mechanism_screen_v1'; OLD=PARENT/'runs/attempt01'
spec=importlib.util.spec_from_file_location('paired_parent',PARENT/'run.py'); base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
from common import read,write,file_hash,raw_scene,decode
core=base.core; append=base.append; delta=base.delta; ht=base.hash_tensor

def main(out):
    start=time.monotonic();plan=read(ROOT/'PLAN.json');data=read(ROOT/'DATA.json');cfg=read(TASK/'mining_v3r2/CONFIG.json')
    def status(stage,**extra):
        elapsed=time.monotonic()-start
        if elapsed>plan['budget']['wall_seconds']:raise TimeoutError('Fixed 900s budget exceeded')
        write(out/'STATUS.json',dict(stage=stage,pid=os.getpid(),elapsed_s=elapsed,updated_utc=datetime.now(timezone.utc).isoformat(),**extra),replace=True)
    status('loading');assert torch.cuda.is_available() and torch.cuda.device_count()==1
    assert read(ROOT/'PREPARATION.json')['status']=='PASS'
    assert read(OLD/'RESULT.json')['status']=='COMPLETE'
    for p,h in read(OLD/'CONFIG.json')['source_sha256'].items():assert file_hash(p)==h,p
    assert file_hash(data['source'])==data['source_sha256'] and file_hash(cfg['checkpoint'])==cfg['checkpoint_sha256']
    assert (ROOT/'DATA.json').read_bytes()==(PARENT/'DATA.json').read_bytes()
    scenes=read(ROOT/'PROMPTS.json')['scenes']; all_rows={r['token']:r for r in map(json.loads,Path(data['source']).open())}
    old_inputs={(r['token'],r['arm']):r for r in map(json.loads,(OLD/'inputs.jsonl').open())}
    old_groups={(r['token'],r['replicate'],r['arm']):r for r in map(json.loads,(OLD/'groups.jsonl').open())}
    old_roll={(r['token'],r['replicate'],r['candidate']):r for r in map(json.loads,(OLD/'rollouts.jsonl').open()) if r['arm']=='C'}
    assert base.protocol(cfg)['sha256']==all_rows[scenes[0]['token']]['score_protocol_sha256']
    sys.path.insert(0,cfg['source_root'])
    from omegaconf import OmegaConf
    from starVLA.model.framework.QwenOFT import Qwenvl_OFT
    mc=OmegaConf.load(Path(cfg['run_root'])/'config.yaml');mc.framework.qwenvl.base_vlm=cfg['tokenizer_root'];mc.framework.qwenvl.attn_implementation='sdpa'
    model=Qwenvl_OFT(mc,infer_not_load_wan=1);state=torch.load(cfg['checkpoint'],map_location='cpu',weights_only=True,mmap=True)
    model.load_state_dict(state,strict=True);del state
    model=model.to('cuda:0').float().eval().requires_grad_(False)
    names=core.witnesses(model);witness={k:core.witness_value(model,v) for k,v in names.items()}
    files=[ROOT/n for n in ['run.py','prepare.py','diversity.py','PLAN.json','DATA.json','DONORS.json','PROMPTS.json']]+[PARENT/'run.py',PARENT/'VALIDATION.json',OLD/'rollouts.jsonl',OLD/'groups.jsonl',OLD/'inputs.jsonl']+[TASK/'grpo_smoke_v1'/n for n in ['smoke.py','grpo_math.py']]+list((OLD/'features').glob('*.npz'))
    write(out/'CONFIG.json',dict(plan=plan,checkpoint_sha256=cfg['checkpoint_sha256'],device=torch.cuda.get_device_name(0),trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),source_sha256={str(p):file_hash(p) for p in files}))
    counts=dict(groups=0,candidates=0,scenes=0);replay=dict(candidates=0,trajectory_max_abs_error=0.,metric_max_abs_error=0.,feature_scenes=0);(out/'features').mkdir()
    with torch.no_grad():
        for index,prompt in enumerate(scenes):
            token=prompt['token'];row=all_rows[token];scene=raw_scene(row['source_paths']);images=base.load_images(scene);cached=np.load(OLD/'features'/f'{token}.npz')
            qs={};conds={};receipts={};arrays={}
            status('scene',scene_index=index,counts=counts)
            for arm in ['C','F']:
                batch=core.build_inputs(model,scene,prompt['arms'][arm]['text'],images);ids=batch['input_ids']
                positions={k:model._find_token_positions(ids,model._special_token_ids[k]).cpu().tolist() for k in ['history','action']}
                with torch.autocast('cuda',enabled=False):q=core.queries(model,batch,scene['state']);cond=model.action_model.qwen_proj(q.float())
                assert torch.isfinite(q).all() and torch.isfinite(cond).all()
                qs[arm]=q;conds[arm]=cond
                receipt=dict(token=token,arm=arm,text_sha256=prompt['arms'][arm]['sha256'],input_ids_sha256=ht(ids),input_token_count=int(ids.numel()),special_positions=positions,pixel_values_sha256=ht(batch['pixel_values']),state_sha256=hashlib.sha256(np.asarray(scene['state']).tobytes()).hexdigest())
                if arm=='C':
                    assert receipt==old_inputs[token,'C'],token
                    assert np.array_equal(q.float().cpu().numpy(),cached['query_C']),token
                    assert np.array_equal(cond.cpu().numpy(),cached['context_C']),token
                    replay['feature_scenes']+=1
                else:arrays.update(query_F=q.float().cpu().numpy(),context_F=cond.cpu().numpy())
                receipts[arm]=receipt;append(out/'inputs.jsonl',receipt);del batch
            assert receipts['C']['input_token_count']==receipts['F']['input_token_count']
            assert receipts['C']['special_positions']==receipts['F']['special_positions']
            assert receipts['C']['pixel_values_sha256']==receipts['F']['pixel_values_sha256'] and receipts['C']['state_sha256']==receipts['F']['state_sha256']
            append(out/'representations.jsonl',dict(token=token,contrasts={f'F-{arm}':dict(query=delta(qs['F'],torch.from_numpy(cached['query_'+arm]).cuda()),projected_context=delta(conds['F'],torch.from_numpy(cached['context_'+arm]).cuda())) for arm in ['A','C','D']}))
            for rep in range(3):
                rng=torch.Generator(device='cuda').manual_seed(base.seed(token,rep));noise=[torch.randn((8,8,4),device='cuda',dtype=torch.float32,generator=rng) for _ in range(11)];nh=ht(torch.stack(noise))
                assert nh==old_groups[token,rep,'C']['noise_sha256']
                for arm in (['C','F'] if index==0 else ['F']):
                    raw,v=base.sample(model,conds[arm],noise);poses=[];metrics=[];gid=f'{token}:{rep}:{arm}'
                    for candidate,x in enumerate(raw.cpu().numpy()):
                        p=decode(x);m=base.score_one(p,row['source_paths']['metric'],cfg)['metrics'];assert all(np.isfinite(z) and 0<=z<=1 for z in m.values())
                        record=dict(group_id=gid,token=token,log=row['log'],arm=arm,replicate=rep,seed=base.seed(token,rep),noise_sha256=nh,candidate=candidate,trajectory=p.tolist(),metrics=m)
                        if arm=='C':
                            old=old_roll[token,rep,candidate];te=float(np.max(np.abs(p-np.asarray(old['trajectory']))));me=max(abs(m[k]-v) for k,v in old['metrics'].items());assert te<1e-6 and me<1e-9,(te,me)
                            replay['candidates']+=1;replay['trajectory_max_abs_error']=max(te,replay['trajectory_max_abs_error']);replay['metric_max_abs_error']=max(me,replay['metric_max_abs_error']);append(out/'replay.jsonl',record)
                        else:append(out/'rollouts.jsonl',record);counts['candidates']+=1
                        poses.append(p);metrics.append(m)
                    if arm=='F':
                        arrays[f'velocity_rep{rep}_F']=v.cpu().numpy();append(out/'groups.jsonl',dict(group_id=gid,token=token,log=row['log'],arm=arm,replicate=rep,seed=base.seed(token,rep),noise_sha256=nh,rewards=[m['PDMS'] for m in metrics],**base.summarize(poses,metrics)));counts['groups']+=1
                        append(out/'velocity_contrasts.jsonl',dict(token=token,replicate=rep,contrasts={f'F-{a}':delta(v,torch.from_numpy(cached[f'velocity_rep{rep}_{a}']).cuda()) for a in ['A','C','D']}))
                    del raw
            np.savez_compressed(out/'features'/f'{token}.npz',**arrays);cached.close();counts['scenes']+=1
            print(json.dumps(dict(event='scene_complete',counts=counts,elapsed_s=time.monotonic()-start)),flush=True)
            if index==0:write(out/'SMOKE.json',dict(status='PASS',F_candidates=24,C_replay=replay.copy(),C_F_token_and_special_positions_equal=True))
            del arrays,qs,conds,images,noise
    assert counts==dict(groups=96,candidates=768,scenes=32) and replay['candidates']==24 and replay['feature_scenes']==32
    assert all(torch.equal(core.witness_value(model,n),witness[k]) for k,n in names.items())
    result=dict(status='COMPLETE',counts=counts,C_replay=replay,elapsed_s=time.monotonic()-start,peak_gpu_allocated_gib=torch.cuda.max_memory_allocated()/2**30,model_witnesses_unchanged=True,training_performed=False,final_test_used=False)
    write(out/'RESULT.json',result);status('complete',result=result);print(json.dumps(result),flush=True)
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True);args=parser.parse_args();out=ROOT/'runs'/args.run;out.mkdir(parents=True,exist_ok=False)
    try:main(out)
    except BaseException as e:write(out/'ERROR.json',dict(type=type(e).__name__,message=str(e),traceback=traceback.format_exc()));raise

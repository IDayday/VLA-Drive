"""Original SFT inputs only: matched third GRPO arm, G8, four-scene accumulation."""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import torch
from torch.utils.checkpoint import checkpoint

ROOT=Path(__file__).resolve().parent
TASK=ROOT.parent
sys.path.insert(0,str(TASK/'grpo_smoke_v1'))
import smoke as core
from grpo_math import transition,log_prob,advantage,objective
from diversity import summarize
from common import read,write,file_hash,raw_scene,decode
from pipeline import load_images
from score_trace import score_one,protocol


def prompt(scene,row,arm):
    assert arm=='plain'
    return scene['instruction']


def seed(role,token):return int(hashlib.sha256(f'probe-v2:{role}:{token}'.encode()).hexdigest()[:8],16)


def append(path,value):
    with path.open('a') as f:f.write(json.dumps(value,ensure_ascii=False,allow_nan=False)+'\n')


def main(out):
    started=time.monotonic();plan=read(ROOT/'PLAN.json');data=read(ROOT/'DATA.json')
    def status(stage,**extra):
        elapsed=time.monotonic()-started
        if elapsed>plan['budget']['wall_seconds']:raise TimeoutError('Probe wall budget exceeded')
        write(out/'STATUS.json',dict(stage=stage,pid=os.getpid(),elapsed_s=elapsed,updated_utc=datetime.now(timezone.utc).isoformat(),**extra),replace=True)
    status('loading')
    cfg=read(TASK/'mining_v3r2/CONFIG.json')
    assert file_hash(ROOT/'DATA.json')==file_hash(TASK/'grpo_probe_v2/DATA.json')
    assert file_hash(data['source'])==data['source_sha256']
    all_rows={r['token']:r for r in map(json.loads,Path(data['source']).open())}
    train=[all_rows[r['token']] for r in data['train']]
    evaluation=[(role,all_rows[r['token']]) for role,key in [('dev','dev'),('train_monitor','train_monitor')] for r in data[key]]
    assert not {r['log'].rsplit('_',2)[0] for r in train}&{r['source_group'] for r in data['dev']}
    assignments=read(TASK/'splits/log_reason_90_10_v1/ASSIGNMENTS.json')
    assert all(assignments['logs'][r['log']]=='train' for r in train+[r for _,r in evaluation])
    assert torch.cuda.is_available() and torch.cuda.device_count()==1
    assert file_hash(cfg['checkpoint'])==cfg['checkpoint_sha256']
    assert protocol(cfg)['sha256']==train[0]['score_protocol_sha256']
    sys.path.insert(0,cfg['source_root'])
    from omegaconf import OmegaConf
    from starVLA.model.framework.QwenOFT import Qwenvl_OFT
    model_cfg=OmegaConf.load(Path(cfg['run_root'])/'config.yaml')
    model_cfg.framework.qwenvl.base_vlm=cfg['tokenizer_root'];model_cfg.framework.qwenvl.attn_implementation='sdpa'
    model=Qwenvl_OFT(model_cfg,infer_not_load_wan=1)
    state=torch.load(cfg['checkpoint'],map_location='cpu',weights_only=True,mmap=True)
    model.load_state_dict(state,strict=True);del state
    model=model.to('cuda:0').float().eval().requires_grad_(True)
    reference=copy.deepcopy(model).eval().requires_grad_(False)
    names=core.witnesses(model);reference_witnesses={k:core.witness_value(reference,n) for k,n in names.items()}
    locked_files=list(ROOT.glob('*.py'))+[TASK/'grpo_smoke_v1/smoke.py',TASK/'grpo_smoke_v1/grpo_math.py']
    write(out/'CONFIG.json',dict(plan=plan,data=data,code_sha256={str(p):file_hash(p) for p in locked_files},device=torch.cuda.get_device_name(0),checkpoint_sha256=cfg['checkpoint_sha256'],strict_load=True,trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),start_utc=datetime.now(timezone.utc).isoformat()))
    counters={'training_groups':0,'evaluation_groups':0,'candidate_scores':0}
    previous={(r['token'],r['candidate']):r for r in map(json.loads,(TASK/'grpo_probe_v2/runs/attempt01/rollouts.jsonl').open()) if r['stage']=='initial' and r['arm']=='plain'}
    replay={'candidate_count':0,'trajectory_max_abs_error':0.,'metric_max_abs_error':0.}
    def input_receipt(scene,batch,row,stage):
        instruction=prompt(scene,row,'plain')
        assert instruction==scene['instruction']
        ids=batch['input_ids'].detach().cpu().contiguous().numpy()
        append(out/'inputs.jsonl',dict(stage=stage,token=row['token'],instruction=instruction,instruction_sha256=hashlib.sha256(instruction.encode()).hexdigest(),input_ids_sha256=hashlib.sha256(ids.tobytes()).hexdigest(),input_token_count=int(ids.size),pixel_shape=list(batch['pixel_values'].shape),state_shape=list(np.asarray(scene['state']).shape),privileged_input=False))

    def score_group(row,raw,role,stage,arm,window=None):
        poses=[];metrics=[];gid=f'{stage}:{arm}:{row["token"]}'
        for candidate,x in enumerate(raw.detach().cpu().numpy()):
            p=decode(x);s=score_one(p,row['source_paths']['metric'],cfg)
            if stage=='initial':
                original=previous[row['token'],candidate]
                trajectory_error=float(np.max(np.abs(p-np.asarray(original['trajectory']))))
                metric_error=max(abs(s['metrics'][k]-v) for k,v in original['metrics'].items())
                assert trajectory_error<1e-6 and metric_error<1e-9,(row['token'],candidate,trajectory_error,metric_error)
                replay['candidate_count']+=1
                replay['trajectory_max_abs_error']=max(replay['trajectory_max_abs_error'],trajectory_error)
                replay['metric_max_abs_error']=max(replay['metric_max_abs_error'],metric_error)
            poses.append(p);metrics.append(s['metrics']);counters['candidate_scores']+=1
            append(out/'rollouts.jsonl',dict(group_id=gid,stage=stage,arm=arm,role=role,window=window,token=row['token'],log=row['log'],candidate=candidate,trajectory=p.tolist(),metrics=s['metrics']))
        summary=summarize(poses,metrics)
        record=dict(group_id=gid,stage=stage,arm=arm,role=role,window=window,token=row['token'],log=row['log'],seed=seed('training' if stage=='train' else 'evaluation',row['token']),rewards=[m['PDMS'] for m in metrics],base_cached_pdms=row['base_score']['metrics']['PDMS'],**summary)
        append(out/'groups.jsonl',record)
        counters['training_groups' if stage=='train' else 'evaluation_groups']+=1
        return record

    def evaluate(policy,stage,arm):
        for i,(role,row) in enumerate(evaluation):
            status('evaluation',evaluation_stage=stage,arm=arm,completed=i,total=len(evaluation),counts=counters)
            scene=raw_scene(row['source_paths']);images=load_images(scene)
            batch=core.build_inputs(policy,scene,prompt(scene,row,arm),images)
            input_receipt(scene,batch,row,stage)
            with torch.no_grad():
                cond=core.context(policy,batch,scene['state'])
                raw,_=core.sample(policy,cond,seed('evaluation',row['token']),group_size=8)
            score_group(row,raw,role,stage,arm)
            del batch,cond,raw,images
            if (i+1)%8==0:print(json.dumps(dict(event='evaluation',stage=stage,arm=arm,completed=i+1,total=len(evaluation),elapsed_s=time.monotonic()-started)),flush=True)

    evaluate(reference,'initial','plain')
    assert replay['candidate_count']==384
    write(out/'BASE_REPLAY.json',dict(status='PASS',source='grpo_probe_v2/runs/attempt01',**replay))
    receipts={}
    for arm in ['plain']:
        model.load_state_dict(reference.state_dict(),strict=True);model.eval();model.zero_grad(set_to_none=True)
        assert all(torch.equal(core.witness_value(model,n),reference_witnesses[k]) for k,n in names.items())
        vlm=[p for n,p in model.named_parameters() if n.startswith('qwen_vl_interface.')]
        action=[p for n,p in model.named_parameters() if not n.startswith('qwen_vl_interface.')]
        assert len({id(p) for p in vlm+action})==len(list(model.parameters()))
        optimizer=torch.optim.AdamW([{'params':vlm,'lr':2e-7,'name':'vlm'},{'params':action,'lr':1e-6,'name':'action_and_history'}],betas=(.9,.999),eps=1e-8,weight_decay=0,foreach=False)
        updates=0
        for window in range(16):
            chunk=train[window*4:(window+1)*4];assert len(chunk)==4
            optimizer.zero_grad(set_to_none=True)
            before={k:core.witness_value(model,n) for k,n in names.items()}
            active=0;losses=[];kls=[];max_lp_error=0.;group_stats=[]
            for j,row in enumerate(chunk):
                status('training',arm=arm,window=window,scene_in_window=j,updates=updates,counts=counters)
                scene=raw_scene(row['source_paths']);images=load_images(scene)
                batch=core.build_inputs(model,scene,prompt(scene,row,arm),images)
                input_receipt(scene,batch,row,'train')
                with torch.no_grad():
                    cond=core.context(model,batch,scene['state']);ref_cond=core.context(reference,batch,scene['state'])
                    raw,transitions=core.sample(model,cond,seed('training',row['token']),group_size=8)
                    ref_means=[transition(t['x'],core.velocity(reference,ref_cond,t['x'],t['step']),t['step'])[0] for t in transitions]
                stats=score_group(row,raw,'train','train',arm,window);group_stats.append(stats)
                rewards=torch.tensor(stats['rewards'],device='cuda',dtype=torch.float32);adv,reward_std=advantage(rewards)
                if stats['zero_reward_variance']:adv=torch.zeros_like(adv)
                else:active+=1
                graph_cond=checkpoint(lambda:core.context(model,batch,scene['state']),use_reentrant=False)
                assert torch.allclose(graph_cond.detach(),cond,rtol=1e-5,atol=1e-5)
                leaf=graph_cond.detach().requires_grad_()
                for t,ref_mean in zip(transitions,ref_means):
                    mean,std=transition(t['x'],core.velocity(model,leaf,t['x'],t['step']),t['step'])
                    lp=log_prob(t['next'],mean,std)
                    diff=float((lp.detach()-t['old_logp']).abs().max());max_lp_error=max(max_lp_error,diff)
                    assert diff<1e-3,('old/new likelihood mismatch',diff)
                    loss,ratio,kl=objective(lp,t['old_logp'],adv,mean,ref_mean,std)
                    assert torch.isfinite(loss) and torch.isfinite(ratio).all()
                    # 10 denoising transitions and 4 equally weighted scenes.
                    (loss/(10*4)).backward();losses.append(float(loss.detach()));kls.extend(kl.detach().tolist())
                graph_cond.backward(leaf.grad)
                del graph_cond,leaf,batch,images,cond,ref_cond,raw,transitions,ref_means
            # No optimizer update is allowed between the four prompts above.
            assert all(torch.equal(core.witness_value(model,n),before[k]) for k,n in names.items())
            grads=core.gradient_stats(model,names)
            norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True))
            optimizer.step();updates+=1
            changes={k:float((core.witness_value(model,n)-before[k]).abs().max()) for k,n in names.items()}
            assert all(v>0 for v in changes.values()),changes
            receipt=dict(arm=arm,window=window,optimizer_update=updates,tokens=[r['token'] for r in chunk],scenes=4,candidates=32,active_reward_groups=active,zero_variance_groups=4-active,loss_mean=float(np.mean(losses)),reference_kl_mean=float(np.mean(kls)),old_new_logprob_max_abs_error=max_lp_error,gradient_norms=grads,pre_clip_gradient_norm=norm,max_parameter_changes=changes,lr_vlm=2e-7,lr_action=1e-6,pdms_mean=float(np.mean([s['pdms_mean'] for s in group_stats])),pairwise_ADE_m=float(np.mean([s['pairwise_ADE_m'] for s in group_stats])),peak_gpu_allocated_gib=torch.cuda.max_memory_allocated()/2**30,elapsed_s=time.monotonic()-started)
            append(out/'updates.jsonl',receipt);print(json.dumps(dict(event='update',**receipt)),flush=True)
        model.zero_grad(set_to_none=True);del optimizer
        evaluate(model,'final',arm)
        status('saving_checkpoint',arm=arm,counts=counters)
        checkpoint_dir=out/'checkpoints'/arm;checkpoint_dir.mkdir(parents=True)
        checkpoint_path=checkpoint_dir/'pytorch_model.pt'
        torch.save(model.state_dict(),checkpoint_path)
        receipts[arm]=dict(checkpoint=str(checkpoint_path),sha256=file_hash(checkpoint_path),optimizer_updates=updates,training_scenes=64,seed_schema='probe-v2:training:token',weights_only=True,resumable_optimizer_state=False)
        write(checkpoint_dir/'METADATA.json',receipts[arm])
    assert counters=={'training_groups':64,'evaluation_groups':96,'candidate_scores':1280},counters
    assert all(torch.equal(core.witness_value(reference,n),reference_witnesses[k]) for k,n in names.items())
    result=dict(status='COMPLETE',counts=counters,arms=receipts,elapsed_s=time.monotonic()-started,peak_gpu_allocated_gib=torch.cuda.max_memory_allocated()/2**30,final_test_used=False,base_replay=replay,privileged_model_inputs=False,reference_witnesses_unchanged=True,formal_long_training_started=False)
    write(out/'RESULT.json',result);status('complete',result=result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True);args=parser.parse_args()
    out=ROOT/'runs'/args.run;out.mkdir(parents=True,exist_ok=False)
    try:main(out)
    except BaseException as error:
        write(out/'ERROR.json',dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc()));raise

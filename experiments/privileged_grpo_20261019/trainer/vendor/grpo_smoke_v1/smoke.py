"""Two-update, training-only Flow-GRPO smoke with jointly trainable VLM.

Original source and accepted data remain read-only. No held-out test rollouts.
This is an engineering acceptance run, not an expert efficacy experiment.
"""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import os
import sys
import time
import traceback

import numpy as np
import torch
from torch.utils.checkpoint import checkpoint
from grpo_math import transition, log_prob, advantage, objective

ROOT = Path(__file__).resolve().parent
TASK = ROOT.parent
MINING = TASK / 'mining_v3r2'
SPLIT = TASK / 'splits/log_reason_90_10_v1'
sys.path.insert(0, str(MINING))
from common import raw_scene, decode, file_hash, read, write
from pipeline import load_images
from score_trace import score_one, protocol
from input_protocol import INPUT_PROTOCOL_VERSION, suffix, validate_inputs


def prompt(scene, row):
    poses='; '.join(f'w{i+1}: x={p[0]:+.4f}m, y={p[1]:+.4f}m, yaw={p[2]:+.5f}rad' for i,p in enumerate(row['input_trajectory']))
    return (scene['instruction']+'\nRefine the following failed trajectory using the privileged feedback. '
            'Output the complete corrected future trajectory in the original ego frame.\nInitial trajectory: '
            +poses+'\n'+row['guidance']['guidance_text'])


def build_inputs(model, scene, instruction, images):
    batch = model.qwen_vl_interface.build_qwenvl_inputs(images=[images], instructions=[instruction+suffix(model)])
    validate_inputs(model, batch)
    return batch


def queries(model, batch, state):
    ids=batch['input_ids']; mask=batch['attention_mask']
    with torch.no_grad():
        pos,_=model.qwen_vl_interface.model.model.get_rope_index(input_ids=ids,image_grid_thw=batch['image_grid_thw'],video_grid_thw=None,attention_mask=mask)
    with torch.autocast('cuda',dtype=torch.bfloat16):
        embeds=model.qwen_vl_interface.model.get_input_embeddings()(ids)
        # Keep image features in the graph; original training helper used no_grad here.
        parts,deep=model.qwen_vl_interface.model.model.get_image_features(batch['pixel_values'],batch['image_grid_thw'])
        image_embeds=torch.cat(parts,dim=0)
    states=torch.as_tensor(state,device=ids.device,dtype=torch.float32)
    hist=model.action_input_model(states).to(embeds.dtype)
    history_pos=model._find_token_positions(ids,model._special_token_ids['history'])[:,0]
    embeds=embeds.clone(); embeds[torch.arange(ids.shape[0],device=ids.device),history_pos]=hist
    with torch.autocast('cuda',dtype=torch.bfloat16):
        hidden=model._qwen_language_forward(ids,embeds,mask,pos,image_embeds,deep)
    action_pos=model._find_token_positions(ids,model._special_token_ids['action'])
    for tid in model._special_token_ids['action']:
        assert int((ids==tid).sum())==1
    return hidden.gather(1,action_pos[:,:,None].expand(-1,-1,hidden.shape[-1]))


def context(model, batch, state):
    q=queries(model,batch,state)
    return model.action_model.qwen_proj(q.float())


def velocity(model, cond, x, step):
    t=torch.full((x.shape[0],),int(step/10*model.action_model.num_timestep_buckets),device=x.device,dtype=torch.long)
    return model.action_model.predict_action_onestep(cond.expand(x.shape[0],-1,-1),x.float(),t).float()


@torch.no_grad()
def sample(model, cond, seed, group_size=4, eta=.1):
    generator=torch.Generator(device=cond.device).manual_seed(seed)
    x=torch.randn((group_size,8,4),device=cond.device,dtype=torch.float32,generator=generator)
    transitions=[]
    for step in range(10):
        mean,std=transition(x,velocity(model,cond,x,step),step,eta=eta)
        nxt=mean+std*torch.randn(x.shape,device=x.device,dtype=torch.float32,generator=generator)
        lp=log_prob(nxt,mean,std)
        assert torch.isfinite(nxt).all() and torch.isfinite(lp).all()
        transitions.append({'step':step,'x':x.detach(),'next':nxt.detach(),'old_logp':lp.detach()})
        x=nxt
    return x,transitions


def witnesses(model):
    names={
        'visual':'qwen_vl_interface.model.model.visual.patch_embed.proj.weight',
        'language':'qwen_vl_interface.model.model.language_model.layers.0.self_attn.q_proj.weight',
        'action_query':'qwen_vl_interface.model.model.language_model.embed_tokens.weight',
        'dit':'action_model.action_decoder.layer2.weight',
    }
    parameters=dict(model.named_parameters())
    for key,name in list(names.items()):
        if name not in parameters:
            candidates=[n for n,p in parameters.items() if ('.visual.' in n if key=='visual' else '.language_model.layers.0.' in n if key=='language' else 'embed_tokens.weight' in n if key=='action_query' else n.startswith('action_model.'))]
            assert candidates,(key,name)
            names[key]=candidates[0]
    return names


def witness_value(model,name):
    p=dict(model.named_parameters())[name]
    if 'embed_tokens' in name:
        return p[list(model._special_token_ids['action'])].detach().float().cpu().clone()
    return p.detach().float().cpu().clone()


def gradient_stats(model,names):
    groups={'visual':[], 'language':[], 'dit':[], 'state':[]}
    for name,p in model.named_parameters():
        if p.grad is None: continue
        assert torch.isfinite(p.grad).all(),name
        k='visual' if '.visual.' in name else 'language' if name.startswith('qwen_vl_interface.') else 'dit' if name.startswith('action_model.') else 'state'
        groups[k].append(p.grad.float().square().sum())
    out={k:float(torch.stack(vals).sum().sqrt()) if vals else 0. for k,vals in groups.items()}
    p=dict(model.named_parameters())[names['action_query']]
    out['action_query']=float(p.grad[list(model._special_token_ids['action'])].float().norm())
    assert all(out[k]>0 for k in ['visual','language','dit','action_query']),out
    return out


def main(out):
    started=time.monotonic(); status=lambda stage,**kw:write(out/'STATUS.json',dict(stage=stage,pid=os.getpid(),elapsed_s=time.monotonic()-started,updated_utc=datetime.now(timezone.utc).isoformat(),**kw),replace=True)
    status('loading')
    cfg=read(MINING/'CONFIG.json'); accepted=read(MINING/'FINAL_ACCEPTANCE.json')
    assert read(SPLIT/'VALIDATION.json')['status']=='ACCEPTED'
    assert file_hash(SPLIT/'train/TRAINING_PAIRS.jsonl')==read(SPLIT/'SUMMARY.json')['exports']['TRAINING_PAIRS.jsonl']['sha256']['train']
    rows=[json.loads(line) for line in (SPLIT/'train/TRAINING_PAIRS.jsonl').open()]
    priority=['NC','DAC','TTC','DDC','EP','CF']
    buckets={k:[] for k in priority}
    for row in sorted(rows,key=lambda r:hashlib.sha256(('grpo-smoke-20261019:'+r['token']).encode()).hexdigest()):
        buckets[next(k for k in priority if row['base_score']['metrics'][k]<1)].append(row)
    selected=[buckets[k][j] for j in range(2) for k in priority if len(buckets[k])>j][:10]
    assignments=read(SPLIT/'ASSIGNMENTS.json')
    assert all(assignments['logs'][r['log']]=='train' for r in selected)
    assert torch.cuda.is_available() and torch.cuda.device_count()==1
    assert file_hash(cfg['checkpoint'])==cfg['checkpoint_sha256']
    assert protocol(cfg)['sha256']==selected[0]['score_protocol_sha256']
    sys.path.insert(0,cfg['source_root'])
    from omegaconf import OmegaConf
    from starVLA.model.framework.QwenOFT import Qwenvl_OFT
    model_cfg=OmegaConf.load(Path(cfg['run_root'])/'config.yaml')
    model_cfg.framework.qwenvl.base_vlm=cfg['tokenizer_root'];model_cfg.framework.qwenvl.attn_implementation='sdpa'
    model=Qwenvl_OFT(model_cfg,infer_not_load_wan=1)
    weights=torch.load(cfg['checkpoint'],map_location='cpu',weights_only=True,mmap=True)
    model.load_state_dict(weights,strict=True); del weights
    model=model.to('cuda:0').eval(); model._inference_qwen_forward_mode='optimized'
    write(out/'CONFIG.json',dict(plan=read(ROOT/'PLAN.json'),input_identity=accepted['identity'],selected_train_tokens=[r['token'] for r in selected],device=torch.cuda.get_device_name(0),checkpoint_sha256=cfg['checkpoint_sha256'],code_hashes={p.name:file_hash(p) for p in ROOT.glob('*.py')},started_utc=datetime.now(timezone.utc).isoformat()))
    # Before any update, verify that custom visual-language routing preserves Base.
    row=selected[0];scene=raw_scene(row['source_paths']);images=load_images(scene)
    batch=build_inputs(model,scene,scene['instruction'],images)
    with torch.inference_mode():
        torch.manual_seed(719);torch.cuda.manual_seed_all(719)
        orig=model.predict_action_infer_1d([dict(image=images,state=scene['state'],lang=scene['instruction'],token=row['token'])])['normalized_actions']
        torch.manual_seed(719);torch.cuda.manual_seed_all(719)
        q=queries(model,batch,scene['state'])
        with torch.autocast('cuda',dtype=torch.float32):
            local=model.action_model.predict_action(q).cpu().numpy()
    base_diff=float(np.max(np.abs(orig-local)))
    assert np.allclose(orig,local,rtol=1e-5,atol=1e-5),base_diff
    score_replay=score_one(row['input_trajectory'],row['source_paths']['metric'],cfg)
    assert score_replay['metrics']==row['base_score']['metrics']
    write(out/'BASE_REPLAY.json',dict(max_abs_action_difference=base_diff,score_replay_exact=True,strict_load=True))
    del orig,local,q,batch,images
    # FP32 master weights prevent BF16 rounding from hiding small Adam updates.
    model.float();model.requires_grad_(True);model.eval()
    reference=copy.deepcopy(model).eval().requires_grad_(False)
    names=witnesses(model);ref_before={k:witness_value(reference,n) for k,n in names.items()}
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-6,betas=(.9,.999),eps=1e-8,weight_decay=0,foreach=False)
    counts={'updates':0,'scenes':0,'zero_variance_groups':0,'rollouts':0}
    write(out/'PARAMETERS.json',dict(total=sum(p.numel() for p in model.parameters()),trainable=sum(p.numel() for p in model.parameters() if p.requires_grad),witness_names=names,precision='FP32 parameters and optimizer; VLM autocast BF16; DiT FP32',dropout='all modules eval; gradients enabled'))
    for group_index,row in enumerate(selected):
        if counts['updates']>=2 or time.monotonic()-started>1200: break
        status('rollout',counts=counts,token=row['token'])
        scene=raw_scene(row['source_paths']);images=load_images(scene);text=prompt(scene,row)
        batch=build_inputs(model,scene,text,images)
        with torch.no_grad():
            cond=context(model,batch,scene['state'])
            ref_cond=context(reference,batch,scene['state'])
            raw,transitions=sample(model,cond,seed=20261019+group_index)
            # Reference is frozen at initialization and uses exactly the same condition.
            ref_means=[transition(t['x'],velocity(reference,ref_cond,t['x'],t['step']),t['step'])[0].detach() for t in transitions]
        scores=[]
        for candidate,normalized in enumerate(raw.cpu().numpy()):
            trajectory=decode(normalized); score=score_one(trajectory,row['source_paths']['metric'],cfg)
            scores.append(score['metrics']['PDMS'])
            with (out/'rollouts.jsonl').open('a') as f:
                f.write(json.dumps(dict(token=row['token'],log=row['log'],group=group_index,candidate=candidate,policy_update=counts['updates'],trajectory=trajectory.tolist(),score=score,privilege='initial_failure_trajectory_and_original_guidance',seed=20261019+group_index),allow_nan=False)+'\n')
        rewards=torch.tensor(scores,device='cuda',dtype=torch.float32); adv,std_reward=advantage(rewards)
        counts['scenes']+=1;counts['rollouts']+=4
        print(json.dumps(dict(event='rollout',token=row['token'],rewards=scores,std=float(std_reward),counts=counts)),flush=True)
        if float(std_reward)<1e-6:
            counts['zero_variance_groups']+=1
            del cond,ref_cond,raw,transitions,ref_means,batch,images
            continue
        status('backward',counts=counts,token=row['token'])
        optimizer.zero_grad(set_to_none=True)
        before={k:witness_value(model,n) for k,n in names.items()}
        graph_cond=checkpoint(lambda:context(model,batch,scene['state']),use_reentrant=False)
        assert torch.allclose(graph_cond.detach(),cond,rtol=1e-5,atol=1e-5)
        leaf=graph_cond.detach().requires_grad_()
        losses=[];ratios=[];kls=[]
        for t,ref_mean in zip(transitions,ref_means):
            mean,std=transition(t['x'],velocity(model,leaf,t['x'],t['step']),t['step'])
            lp=log_prob(t['next'],mean,std)
            max_logp_diff=float((lp.detach()-t['old_logp']).abs().max())
            assert max_logp_diff<1e-3,('old/new replay mismatch',max_logp_diff)
            loss,ratio,kl=objective(lp,t['old_logp'],adv,mean,ref_mean,std)
            assert torch.isfinite(loss) and torch.isfinite(ratio).all()
            (loss/10).backward()
            losses.append(float(loss.detach()));ratios.extend(ratio.detach().tolist());kls.extend(kl.detach().tolist())
        assert leaf.grad is not None and torch.isfinite(leaf.grad).all()
        graph_cond.backward(leaf.grad)
        grads=gradient_stats(model,names)
        grad_norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.0,error_if_nonfinite=True))
        status('optimizer',counts=counts,token=row['token'])
        optimizer.step()
        changes={k:float((witness_value(model,n)-before[k]).abs().max()) for k,n in names.items()}
        assert all(v>0 for v in changes.values()),changes
        counts['updates']+=1
        step_record=dict(update=counts['updates'],token=row['token'],rewards=scores,reward_std=float(std_reward),loss=float(np.mean(losses)),ratio_min=min(ratios),ratio_max=max(ratios),reference_kl_mean=float(np.mean(kls)),gradient_norms=grads,pre_clip_gradient_norm=grad_norm,max_parameter_changes=changes,peak_gpu_allocated_gib=torch.cuda.max_memory_allocated()/2**30,elapsed_s=time.monotonic()-started)
        with (out/'steps.jsonl').open('a') as f:f.write(json.dumps(step_record,allow_nan=False)+'\n')
        print(json.dumps(dict(event='optimizer_update',**step_record)),flush=True)
        del cond,ref_cond,raw,transitions,ref_means,graph_cond,leaf,batch,images,before
        optimizer.zero_grad(set_to_none=True)
        status('updated',counts=counts)
    reference_unchanged=all(torch.equal(witness_value(reference,n),ref_before[k]) for k,n in names.items())
    assert reference_unchanged
    result=dict(status='PASS' if counts['updates']==2 else 'INCOMPLETE_NO_VALID_UPDATE_BUDGET',counts=counts,reference_witnesses_unchanged=reference_unchanged,heldout_test_used=False,base_replay=read(out/'BASE_REPLAY.json'),elapsed_s=time.monotonic()-started,peak_gpu_allocated_gib=torch.cuda.max_memory_allocated()/2**30,formal_training_started=False,note='Engineering smoke only; no repair efficacy conclusion; no full checkpoint or optimizer checkpoint saved.')
    # Small parameter witnesses demonstrate actual updates without producing a misleading full expert checkpoint.
    torch.save({k:witness_value(model,n) for k,n in names.items()},out/'UPDATED_PARAMETER_WITNESSES.pt')
    write(out/'RESULT.json',result);status('complete',result=result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True);args=parser.parse_args()
    out=ROOT/'runs'/args.run;out.mkdir(parents=True,exist_ok=False)
    try:
        main(out)
    except BaseException as error:
        write(out/'ERROR.json',dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc()))
        raise

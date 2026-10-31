"""Frozen ABCD IL readouts on one shared legal TRAIN panel; no updates/scoring.

Baselines use 64 disjoint TRAIN scenes and retain camera/time/spatial position.
Counterfactual candidates have sensitivity diagnostics, never factual WA labels.
"""
import argparse
import copy
import gc
import json
import os
from pathlib import Path
import time
import torch
from torch.nn import functional as F
from starVLA.model.modules.qwen_world_compare.factory import build_model
from starVLA.model.modules.qwen_world_postcot.teacher_targets import TeacherCacheReader
from starVLA.model.modules.qwen_world_postcot.readouts import valid_scene_mse
from starVLA.model.modules.qwen_world_postcot.candidate_bank import CandidateBank
from starVLA.model.modules.qwen_sourceflow.qwen_encoder import QwenObservationProcessor
from starVLA.rl.qwen_world_postcot.data import WorldSceneData
from starVLA.rl.qwen_world_compare.batching import BatchBuilder,to_device,update_denominators
from starVLA.rl.qwen_world_compare.track_targets import FutureTrackStore
from starVLA.rl.qwen_world_compare.evidence import EvidenceBank
from starVLA.rl.flow_grpo.reproducibility import configure_numerics


def write(path,value):
    p=Path(path);t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(value,indent=2,allow_nan=False));os.replace(t,p)


def feature_rows(pred,target,valid,baseline,queries=None):
    rows=[]
    for i in range(len(pred)):
        m=valid[i].bool();y=target[i].float()
        if not m.any():continue
        mean=baseline
        if queries is not None:
            q=queries[i];mean=baseline[q[:,0],q[:,1],q[:,2],q[:,3]]
        pp=pred[i].float()[m];yy=y[m];mm=mean[m].float()
        rows.append(dict(MSE=float((pp-yy).square().mean()),position_mean_MSE=float((mm-yy).square().mean()),
            zero_MSE=float(yy.square().mean()),cosine=float(F.cosine_similarity(pp,yy,dim=-1).mean()),
            valid_queries=int(m.sum())))
    return rows


def summary(rows):
    if not rows:return dict(scenes=0)
    d={k:sum(r[k]for r in rows)/len(rows)for k in rows[0]}
    d.update(scenes=len(rows),skill_vs_position_mean=1-d['MSE']/max(d['position_mean_MSE'],1e-30));return d


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--plan',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    if(out/'COMPLETE.json').exists():raise FileExistsError('Completed probe is immutable')
    plan=json.loads(Path(a.plan).read_text());configure_numerics();torch.set_num_threads(4);torch.cuda.set_device(0)
    cfg=plan['models']['A']['identity']['config'];data=WorldSceneData(Path(cfg['data_root'])/'TRAIN',role='TRAIN')
    teachers=TeacherCacheReader(cfg['teacher_cache'],expected_cohort='TRAIN')
    tokens=plan['probe_tokens'];calib=plan['calibration_tokens']
    if set(tokens)&set(calib)or any(t not in data.tokens for t in tokens+calib):raise ValueError('Probe/calibration population invalid')
    means={};counts={}
    for token in calib:
        v=teachers.get(token)
        for key,mask in [('dino','dino_valid'),('future','future_valid')]:
            y=v[key].float();m=v[mask].bool()
            if key not in means:means[key]=torch.zeros_like(y);counts[key]=torch.zeros_like(m,dtype=torch.float32)
            means[key]+=torch.where(m[...,None],y,0.);counts[key]+=m
    means={k:v/counts[k].clamp_min(1)[...,None]for k,v in means.items()}
    processor=QwenObservationProcessor(cfg['qwen_path']);bank=CandidateBank(cfg['candidate_bank'],allowed_splits=('TRAIN',))
    builder=BatchBuilder(data,processor,teachers,bank,FutureTrackStore(Path(cfg['target_root'])/'TRAIN'),EvidenceBank(cfg['evidence_root']))
    results={}
    for arch,item in plan['models'].items():
        tick=time.monotonic();identity=item['identity'];cfg=identity['config'];micro=cfg['execution']['micro_batch']
        model=build_model(cfg,checkpoint_config=copy.deepcopy(identity['qwen_config']),train_heads=True)
        raw=torch.load(item['model'],map_location='cpu',weights_only=False,mmap=True)
        model.load_state_dict(raw['module'],strict=True);del raw
        model=model.eval().requires_grad_(False).to('cuda');torch.cuda.reset_peak_memory_stats()
        before={n:p._version for n,p in model.named_parameters()}
        batches=[builder.build(tokens[i:i+micro],list(range(i,i+micro)),plan['probe_update'])for i in range(0,len(tokens),micro)]
        den=update_denominators(batches,torch.device('cpu'));den={k:torch.as_tensor(v,device='cuda')for k,v in den.items()}
        collected=dict(dino=[],wa_pre=[],wa_post=[]);losses={};all_factor=[];all_target=[];all_valid=[];all_event=[];all_event_y=[];all_event_v=[];sens=[]
        for cpu in batches:
            b=to_device(cpu,torch.device('cuda'));saved={};hooks=[]
            def capture(key):
                def hook(module,args,result):saved.setdefault(key,[]).append(result)
                return hook
            hooks.append(model.readouts.dino.register_forward_hook(capture('dino')))
            hooks.append(model.readouts.wa.register_forward_hook(capture('wa')))
            def evaluation_hook(module,args,result):saved['R']=args[0];saved['evaluation']=result
            hooks.append(model.readouts.evaluation.register_forward_hook(evaluation_hook))
            with torch.autocast('cuda',dtype=torch.bfloat16):r=model(b,denominators=den,weights=cfg['il']['loss_weights'])
            for h in hooks:h.remove()
            for k,v in r['losses'].items():losses[k]=losses.get(k,0)+float(v)
            collected['dino']+=feature_rows(saved['dino'][0].cpu(),cpu['dino_target'],cpu['dino_valid'],means['dino'])
            collected['wa_pre']+=feature_rows(saved['wa'][0].cpu(),cpu['future_target'],cpu['future_valid'],means['future'],cpu['future_queries'])
            selected=cpu['aux_scene_indices'];ns=len(selected)
            if ns:
                valid=cpu['future_valid'][selected]&cpu['action_valid'][selected].all(1)[:,None]
                collected['wa_post']+=feature_rows(saved['wa'][1].cpu(),cpu['future_target'][selected],valid,means['future'],cpu['future_queries'][selected])
                pred=saved['evaluation'];all_factor.append(pred['factor_logits'].float().sigmoid().cpu().reshape(ns,3,6));all_target.append(cpu['post_labels']);all_valid.append(cpu['post_valid'])
                all_event.append(pred['event_logits'].float().cpu().reshape(ns,3,3,9));all_event_y.append(cpu['post_events']);all_event_v.append(cpu['post_event_valid'])
                query=b['future_queries'][b['aux_scene_indices']].repeat_interleave(3,0)
                with torch.autocast('cuda',dtype=torch.bfloat16):future=model.readouts.wa(saved['R'],query).float().reshape(ns,3,128,512)
                R=saved['R'].float().reshape(ns,3,16,512)
                for i in range(ns):
                    traj=cpu['post_trajectories'][i]
                    sens.append(dict(R_rms_difference=float((R[i,1:]-R[i,:1]).square().mean().sqrt()),
                        predicted_future_rms_difference=float((future[i,1:]-future[i,:1]).square().mean().sqrt()),
                        candidate_normalized_per_channel_MAE=(traj[1:]-traj[:1]).abs().mean((0,1)).tolist()))
            del b,saved,r
        factors={};events={};p=torch.cat(all_factor);y=torch.cat(all_target);v=torch.cat(all_valid)
        for i,name in enumerate(['NC','DAC','TTC','EP','Comfort','DDC']):
            m=v[:,:,i];pp=p[:,:,i][m];yy=y[:,:,i][m]
            if not len(pp):
                factors[name]=dict(labels=0,metrics=None);continue
            entropy=-(yy*yy.clamp_min(1e-12).log()+(1-yy)*(1-yy).clamp_min(1e-12).log())
            bce=F.binary_cross_entropy(pp,yy,reduction='none')
            factors[name]=dict(labels=int(m.sum()),not_full_pass_labels=int((yy<1).sum()),
                Brier=float((pp-yy).square().mean()),always_one_Brier=float((1-yy).square().mean()),
                BCE=float(bce.mean()),soft_target_entropy_floor=float(entropy.mean()),excess_BCE=float((bce-entropy).mean()))
        p=torch.cat(all_event);y=torch.cat(all_event_y);v=torch.cat(all_event_v)
        for i,name in enumerate(['NC','DAC','TTC']):
            m=v[:,:,i];pp=p[:,:,i].argmax(-1)[m];yy=y[:,:,i][m]
            if not len(pp):
                events[name]=dict(labels=0,metrics=None);continue
            events[name]=dict(labels=int(m.sum()),events=int((yy!=8).sum()),accuracy=float((pp==yy).float().mean()),always_none_accuracy=float((yy==8).float().mean()))
        if any(p._version!=before[n]for n,p in model.named_parameters()):raise ValueError('Frozen probe changed a parameter')
        result=dict(status='PASS',architecture=arch,IL_update=item['updates'],features={k:summary(v)for k,v in collected.items()},
            common_panel_losses=losses,factors=factors,events=events,candidate_sensitivity=sens,
            wall_seconds=time.monotonic()-tick,peak_allocated_GiB=torch.cuda.max_memory_allocated()/2**30,
            optimizer_updates=0,scorer_calls=0,parameters_unchanged=True,
            limitations='32 TRAIN scenes and 8 factual post scenes, no independent generalization/rare-event accuracy claim; different training budgets; counterfactual future used only for sensitivity, never paired with recorded future as ground truth.')
        write(out/(arch+'_RESULT.json'),result);results[arch]=result
        print(json.dumps(dict(arch=arch,update=item['updates'],features=result['features'],wall=result['wall_seconds'])),flush=True)
        del model,before,batches,den
        gc.collect();torch.cuda.empty_cache()
    bank.close();teachers.close();write(out/'COMPLETE.json',dict(status='PASS',results=results,logical_probe_scenes=len(tokens),calibration_scenes=len(calib),new_training_updates=0,new_scores=0))


if __name__=='__main__':main()

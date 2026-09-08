"""Bounded real InternVL/PDM acceleration audit. Not a Navtest or final-score claim."""
import argparse
import copy
import gc
import json
from pathlib import Path
import subprocess
import time
import torch
from navsim.agents.EpisodeDrive.planreg_v2.agent import PlanRegV2Agent, file_sha256
from navsim.agents.EpisodeDrive.planreg_v2.data import InputOnlyV2Dataset,v2_collate
from navsim.agents.EpisodeDrive.planreg_v2.runtime import load_config,source_fingerprint


def difference(a,b):
    a,b=a.detach().float().flatten(),b.detach().float().flatten()
    return dict(max_abs=float((a-b).abs().max()),relative_l2=float((a-b).norm()/a.norm().clamp_min(1e-20)))


def set_backend(agent,backend):
    agent.backbone.planning_register_adapter.read_only_attention_backend=backend
    agent.ema_teacher.adapter.read_only_attention_backend=backend


def main():
    p=argparse.ArgumentParser(__doc__)
    for key in ('config','manifest','output'):p.add_argument('--'+key,required=True)
    a=p.parse_args();output=Path(a.output)
    if output.exists():raise FileExistsError('Never overwrite numerical evidence')
    output.parent.mkdir(parents=True,exist_ok=True)
    report=dict(status='RUNNING',code_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        source_fingerprint=source_fingerprint(),manifest_sha256=file_sha256(a.manifest),
        input_type='Real cached train scenes, full pretrained InternVL3-2B, actual NAVSIM CPU labels',
        gates=dict(fp32_block_relative_l2=1e-4,fp32_gradient_relative_l2=1e-3,bf16_visual_relative_l2=.02,
                   overlap_loss_gradient_atol=0,checkpoint_loss_gradient_atol=0),
        blocks=[],scenes=[],final_pdms_equivalence='NOT_ESTABLISHED')
    def save():output.write_text(json.dumps(report,indent=2))
    save();torch.manual_seed(0);torch.cuda.manual_seed_all(0);torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    cfg=load_config(a.config);cfg.update(read_only_attention_backend='eager',overlap_metric_target_with_ema=False)
    agent=PlanRegV2Agent(cfg,'cuda').eval()
    dataset=InputOnlyV2Dataset(a.manifest,cfg['vlm_path'],True)
    # Full-range maximum input is record 0; include distinct stop/turn logs as well.
    indices=[0]+[next(i for i,r in enumerate(dataset.records) if r.get('category')==category) for category in ('stop','turn')]
    captured={};hooks=[]
    for i,block in enumerate(agent.backbone.model.vision_model.encoder.layers):
        def capture(module,inputs,index=i):captured[index]=inputs[0][-1:].detach().cpu()
        hooks.append(block.attn.register_forward_pre_hook(capture))
    try:
        for index in indices:
            features,targets=v2_collate([dataset[index]])
            results={}
            for backend in ('eager','split_sdpa'):
                set_backend(agent,backend)
                with torch.no_grad():
                    pred=agent(features);teacher=agent.encode_teacher(features,pred)
                results[backend]={k:pred[k].cpu() for k in ('proposals','trajectory','visual_content','semantic_queries')}
                results[backend]['teacher']=teacher.cpu()
                del pred,teacher
            row=dict(token=dataset.records[index]['token'],category=dataset.records[index]['category'],
                     tiles=len(features['pixel_values'][0]),difference={k:difference(results['eager'][k],results['split_sdpa'][k]) for k in results['eager']})
            for key in ('visual_content','teacher'):
                assert row['difference'][key]['relative_l2']<=report['gates']['bf16_visual_relative_l2'],row
            report['scenes'].append(row);save()
            if hooks:
                for handle in hooks:handle.remove()
                hooks=[]
            del features,targets,results
        # Every real block, actual captured activations, pretrained Q/K norms and fresh Q/V adapters.
        for i,block in enumerate(agent.backbone.model.vision_model.encoder.layers):
            eager=copy.deepcopy(block.attn).float().eval();split=copy.deepcopy(eager).eval()
            eager._planreg_read_only_backend='eager';split._planreg_read_only_backend='split_sdpa'
            x=captured[i].cuda().float().requires_grad_(True);y=x.detach().clone().requires_grad_(True)
            ref,new=eager(x),split(y);upstream=torch.randn_like(ref)
            (ref*upstream).mean().backward();(new*upstream).mean().backward()
            rg=torch.cat([q.grad.flatten() for q in eager.parameters() if q.grad is not None])
            ng=torch.cat([q.grad.flatten() for q in split.parameters() if q.grad is not None])
            row=dict(block=i,forward=difference(ref,new),input_gradient=difference(x.grad,y.grad),parameter_gradient=difference(rg,ng))
            assert row['forward']['relative_l2']<=report['gates']['fp32_block_relative_l2'],row
            for key in ('input_gradient','parameter_gradient'):
                assert row[key]['relative_l2']<=report['gates']['fp32_gradient_relative_l2'],row
            assert torch.isfinite(ng).all()
            report['blocks'].append(row);save()
            del eager,split,x,y,ref,new,upstream,rg,ng
        del captured;gc.collect();torch.cuda.empty_cache()
        set_backend(agent,'split_sdpa');agent.train()
        features,targets=v2_collate([dataset[0]])
        selected=[(n,q) for n,q in agent.named_parameters() if q.requires_grad]
        # Same current forward and exact coordinates: only CPU/GPU scheduling differs.
        pred=agent(features);objectives={};gradients={};labels={}
        for overlap in (False,True):
            agent.config['overlap_metric_target_with_ema']=overlap
            losses=agent.compute_loss(features,targets,pred)
            objectives[str(overlap)]={k:v.detach().cpu() for k,v in losses.items()}
            labels[str(overlap)]=agent.last_metric_targets.cpu()
            gradients[str(overlap)]=[g.detach().cpu() if g is not None else None for g in
                torch.autograd.grad(losses['loss'],[q for _,q in selected],retain_graph=True,allow_unused=True)]
            del losses
        torch.testing.assert_close(labels['False'],labels['True'],atol=0,rtol=0)
        for k in objectives['False']:torch.testing.assert_close(objectives['False'][k],objectives['True'][k],atol=0,rtol=0)
        compared=0
        for (name,_),left,right in zip(selected,gradients['False'],gradients['True']):
            if left is None:assert right is None,name
            else:torch.testing.assert_close(left,right,atol=0,rtol=0);compared+=1
        report['real_pdm_overlap']=dict(status='PASS',all_loss_max_abs_diff=0,all_gradient_max_abs_diff=0,
            compared_gradient_tensors=compared,official_label_max_abs_diff=0,teacher_calls_per_loss=1,
            ttc_label_values=labels['False'][...,3].unique().tolist())
        del pred,objectives,gradients,labels;gc.collect();torch.cuda.empty_cache();save()
        # Same real graph/weights and dropout state, with versus without activation recomputation.
        outputs={};gradients={}
        for checkpointing in (True,False):
            backbone=agent.backbone
            backbone.gradient_checkpointing_enabled=checkpointing
            backbone.model.vision_model.encoder.gradient_checkpointing=checkpointing
            if checkpointing:
                backbone.model.language_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
            else:backbone.model.language_model.gradient_checkpointing_disable()
            agent.train()
            pred=agent(features)
            outputs[str(checkpointing)]={k:pred[k].detach().cpu() for k in ('proposals','visual_content','semantic_queries')}
            loss=pred['visual_content'].square().mean()+pred['semantic_queries'].square().mean()
            gradients[str(checkpointing)]=[g.detach().cpu() if g is not None else None for g in
                torch.autograd.grad(loss,[q for _,q in selected],allow_unused=True)]
            del pred,loss;gc.collect();torch.cuda.empty_cache()
        for k in outputs['True']:torch.testing.assert_close(outputs['True'][k],outputs['False'][k],atol=0,rtol=0)
        for (name,_),left,right in zip(selected,gradients['True'],gradients['False']):
            if left is None:assert right is None,name
            else:torch.testing.assert_close(left,right,atol=0,rtol=0)
        report['real_checkpointing']=dict(status='PASS',forward_max_abs_diff=0,gradient_max_abs_diff=0)
        report['fp32_trainable']=all(q.dtype==torch.float32 for q in agent.parameters() if q.requires_grad)
        report['status']='PASS';save();print(json.dumps({k:v for k,v in report.items() if k not in ('source_fingerprint','blocks')},indent=2))
    except BaseException as exc:
        report.update(status='FAIL',error=type(exc).__name__+': '+str(exc));save();raise
    finally:
        for handle in hooks:handle.remove()
        if agent._score_pool is not None:agent._score_pool.shutdown(wait=True,cancel_futures=True)


if __name__=='__main__':main()

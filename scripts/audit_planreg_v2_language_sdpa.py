"""Bounded real pretrained InternVL language-kernel/gradient audit, not a PDMS evaluation."""
import argparse
import json
from pathlib import Path
import subprocess
import torch
from navsim.agents.EpisodeDrive.planreg_v2.agent import PlanRegV2Agent, file_sha256
from navsim.agents.EpisodeDrive.planreg_v2.data import InputOnlyV2Dataset, v2_collate
from navsim.agents.EpisodeDrive.planreg_v2.language import configure_language_attention
from navsim.agents.EpisodeDrive.planreg_v2.optimizer import build_optimizer, audit_adam_state
from navsim.agents.EpisodeDrive.planreg_v2.runtime import load_config, source_fingerprint


def main():
    parser=argparse.ArgumentParser(__doc__)
    for name in ('config','manifest','output'): parser.add_argument('--'+name,required=True)
    args=parser.parse_args(); path=Path(args.output)
    if path.exists(): raise FileExistsError('Never overwrite measured evidence')
    report=dict(status='RUNNING',source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        source_fingerprint=source_fingerprint()['sha256'],manifest_sha256=file_sha256(args.manifest),
        input_type='Real training image, pretrained InternVL, actual PDM and future images',
        bf16_bitwise_equivalence=False,final_pdms_equivalence='NOT_EVALUATED')
    def save(): path.write_text(json.dumps(report,indent=2))
    save(); torch.manual_seed(0); torch.cuda.manual_seed_all(0); torch.set_num_threads(2)
    cfg=load_config(args.config); agent=PlanRegV2Agent(cfg,'cuda').eval()
    dataset=InputOnlyV2Dataset(args.manifest,cfg['vlm_path'],True)
    features,targets=v2_collate([dataset[0]])
    language=agent.backbone.model.language_model
    report['language']=agent.backbone.language_attention_audit
    # Capture exactly what the real backbone supplies to its internal language queries.
    captured={}
    def capture(module,inputs):
        captured['prefix']=inputs[1].detach();captured['mask']=inputs[2].detach()
    handle=agent.backbone.task_queries.register_forward_pre_hook(capture)
    outputs={}
    try:
        for backend in ('eager','sdpa'):
            configure_language_attention(language,backend)
            with torch.no_grad(): pred=agent(features)
            outputs[backend]={k:pred[k].detach().float().cpu() for k in ('proposals','semantic_queries')}
            del pred
        handle.remove()
        report['bf16_full_forward_difference']={}
        for key,left in outputs['eager'].items():
            right=outputs['sdpa'][key]
            report['bf16_full_forward_difference'][key]=dict(max_abs=float((left-right).abs().max()),
                relative_l2=float((left-right).norm()/left.norm().clamp_min(1e-20)))
        # Profile the language-only production call; visual SDPA cannot supply a false positive.
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                agent.backbone.task_queries(language,captured['prefix'],captured['mask'])
            torch.cuda.synchronize()
        kernels={e.key:e.count for e in prof.key_averages() if 'scaled_dot_product' in e.key}
        report['language_only_kernels']=kernels
        if not any(('flash_attention' in k or 'efficient_attention' in k) and v>0 for k,v in kernels.items()):
            raise AssertionError('Native SDPA did not actually execute a fused CUDA language kernel')
        agent.train(); optimizer,scheduler,_=build_optimizer(agent,cfg['total_steps'],cfg.get('learning_rates'))
        selected={n:p for n,p in agent.named_parameters() if p.requires_grad}
        before={n:p.detach().cpu().clone() for n,p in selected.items()}
        report['steps']=[]
        for step in range(2):
            optimizer.zero_grad(set_to_none=True);pred=agent(features)
            losses=agent.compute_loss(features,targets,pred);losses['loss'].backward()
            assert torch.isfinite(losses['loss'])
            assert all(p.grad is None or torch.isfinite(p.grad).all() for p in selected.values())
            groups={}
            for name,match in (
                ('vision_lora',lambda n:'.vision_model.' in n and '_lora_' in n),
                ('language_lora',lambda n:'.language_model.' in n and '.lora_' in n),
                ('registers',lambda n:'.planning_register_adapter.' in n),
                ('semantic_queries',lambda n:'.task_queries.' in n)):
                grads=[p.grad.float().square().sum() for n,p in selected.items() if match(n) and p.grad is not None]
                groups[name]=float(torch.stack(grads).sum().sqrt())
                assert groups[name]>0, name
            torch.nn.utils.clip_grad_norm_(selected.values(),1.0);optimizer.step();scheduler.step()
            audit_adam_state(optimizer)
            report['steps'].append(dict(step=step,loss=float(losses['loss']),gradient_norms=groups))
            del pred,losses;save()
        report['changed_language_lora_a_tensors']=sum(not torch.equal(before[n],p.detach().cpu())
            for n,p in selected.items() if '.language_model.' in n and '.lora_a.' in n)
        assert report['changed_language_lora_a_tensors']==len(agent.backbone.language_lora_modules)
        assert all(p.grad is None for p in agent.ema_teacher.parameters())
        report.update(status='PASS',fp32_trainable=True,fp32_adam_moments=True,
                      peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
        save(); print(json.dumps(report,indent=2))
    except BaseException as exc:
        report.update(status='FAIL',error=type(exc).__name__+': '+str(exc));save();raise
    finally:
        handle.remove()
        if agent._score_pool is not None:agent._score_pool.shutdown(wait=True,cancel_futures=True)


if __name__=='__main__':main()

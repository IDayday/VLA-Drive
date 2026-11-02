"""Real full IL70 GPU qualification of rescue, never a formal initialization.

Explicitly activates the auxiliary at IL update0 to inspect its gradient path;
the .5 value is a qualification gate injection, NOT claimed scene exposures.
"""
import argparse
from pathlib import Path
import time
import torch
from starVLA.rl.flow_grpo.transactions import atomic_json
from starVLA.rl.qwen_world_safety_exploration.group_runtime import collect_group
from starVLA.rl.qwen_world_performance.batch_runtime import PerformanceRuntime
from starVLA.rl.qwen_world_performance.nominal_improvement import attach_material,new_calibration_state,validate_material
from starVLA.rl.qwen_world_performance.ode_target import predict_clean_x


def main():
    p=argparse.ArgumentParser(__doc__)
    for name in ('config','manifest','output','il-checkpoint','parent-selection','data'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--maximum-scenes',type=int,default=24);a=p.parse_args();a.qualification=True
    output=Path(a.output);output.mkdir(parents=True,exist_ok=True)
    runtime=PerformanceRuntime();cfg=runtime.load_recipe(a)
    torch.cuda.set_device(0);actor,_,_=runtime.make_actor(cfg,a,torch.device('cuda',0))
    conditions=runtime.conditions;service=runtime.make_reward_service()
    actor._loss.auxiliary_state=new_calibration_state()
    generator=torch.Generator().manual_seed(84521)
    from starVLA.rl.qwen_world_performance.data_protocol import load_manifest
    manifest,logs=load_manifest(a.manifest)
    tokens=manifest['tokens'];started=time.monotonic();queries=0;tested=[]
    try:
        for token in tokens[:a.maximum_scenes]:
            version=dict(run='REAL_GPU_NOMINAL_QUALIFICATION_NOT_FORMAL',update=0,adapter_sha256=runtime.parent['model_sha256'])
            sample=collect_group(actor,conditions.get(token,actor.device),token,logs[token],service,generator,
                group_size=16,variant='R_CANONICAL',version=version)
            sample['valid_scene']=True
            attach_material(actor,sample,conditions,service,exposure_epoch=.5,rank=0)
            material=validate_material(sample,rank=0);queries+=16+material['actual_B1_queries']+material['actual_extra_teacher_queries']
            tested.append(dict(token=token,enabled=material['enabled'],reason=material['skip_reason']))
            if not material['enabled']:continue
            from starVLA.rl.qwen_world_safety_exploration.credit import canonical_components
            components=canonical_components(sample['records'],16,device=actor.device)['components']
            credit=runtime.credit.advantages(sample['rewards'],components)
            sample.update(advantages=credit['advantages'],adv_mask=credit['adv_mask'])
            nominal=material['nominal']
            with torch.no_grad():
                direct=predict_clean_x(actor,sample['cond_bg'],nominal['last_state'].to(actor.device),nominal['last_sigma'].to(actor.device))
            if not torch.equal(direct.cpu(),nominal['chain'][:,-1]):
                raise ValueError('Real deployed final clean-x and last-state prediction differ')
            actor.zero_grad(set_to_none=True);loss,metrics=actor(sample);loss.backward()
            named={n:p for n,p in actor.named_parameters() if p.requires_grad}
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in named.values()):
                raise ValueError('Missing/nonfinite real full-scope gradient')
            groups={prefix:sum(float(p.grad.float().square().sum()) for n,p in named.items() if n.startswith(prefix))**.5
                for prefix in ('action_expert.','scene_provider.','predictive_provider.')}
            if any(v<=0 for v in groups.values()):raise ValueError('One real trainable scope did not receive gradients')
            result=dict(status='PASS',real_full_IL70=True,formal_training=False,qualification_gate_injected=.5,
                actual_scene_exposures=0,initial_model_sha256=runtime.parent['model_sha256'],
                loss=float(loss),metrics={k:float(v) for k,v in metrics.items()},gradient_norms=groups,
                actual_PDM_queries=queries,scenes=tested,peak_GPU_bytes=torch.cuda.max_memory_allocated(),
                wall_seconds=time.monotonic()-started,teacher_gain=material['teacher']['gain'],
                exact_final_ODE_head_equal=True,PG_normalization='single-scene qualification only; not formal globalB')
            atomic_json(output/'COMPLETE.json',result);return
        raise RuntimeError('No verified improvement teacher in bounded TRAIN qualification panel; expand panel explicitly')
    finally:service.close()


if __name__=='__main__':main()

"""Synchronized measurements; file preprocessing, VLM and cached heads stay distinct."""
from __future__ import annotations
import time
import platform
import numpy as np
import torch
from ..contracts import require
from ..selector import SelectionRule


def benchmark(model, observation, scorer, router, expert_counts=(1,2,3,5), warmup=5, repetitions=20,
              device='cpu', *, observation_loader=None, rule=None, router_rule=None):
    require(warmup>=1 and repetitions>=2 and all(k>0 for k in expert_counts),'latency budgets')
    model.set_trainable_stage('inference');model.eval()
    def sync():
        if device.startswith('cuda'):torch.cuda.synchronize()
    def measure(fn):
        times=[]
        for i in range(warmup+repetitions):
            sync();start=time.perf_counter()
            with torch.no_grad():fn()
            sync()
            if i>=warmup:times.append((time.perf_counter()-start)*1000)
        return {'p50_ms':float(np.quantile(times,.5)),'p95_ms':float(np.quantile(times,.95)),'samples_ms':times}
    framework=model.adapter.framework
    reports={}
    shared={}
    if observation_loader:
        shared['image_numeric_preprocessing_and_validation']=measure(observation_loader)
    with torch.no_grad():
        encoded=framework.encode_current([observation])
        memory=framework.build_planner_condition(encoded)
        ego=framework.action_model.ego_state
        f=model.encode_scene([observation])
    shared['shared_VLM_including_tokenizer_and_visual']=measure(lambda:framework.encode_current([observation]))
    shared['QFormer_and_ego']=measure(lambda:framework.action_model.encode_features(memory,ego))
    for count in expert_counts:
        if count>len(model.experts):
            reports[str(count)]={'status':'BLOCKED_MISSING_REGISTERED_EXPERTS','registered':len(model.experts)}
            continue
        ids=tuple(model.experts)[:count]
        if device.startswith('cuda'):torch.cuda.reset_peak_memory_stats()
        def finish(c):
            if count==1:return c.trajectories[:,0].cpu()
            require(scorer is not None,'K>1 end-to-end latency needs actual trained Scorer')
            prediction=scorer(c.features,c.trajectories,c.valid)
            return (rule or SelectionRule(0,{},{}))(c,prediction)[0].cpu()
        def online():
            obs=observation_loader() if observation_loader else observation
            return finish(model.forward_candidates([obs],ids))
        reports[str(count)]={'status':'MEASURED','image_to_candidates':measure(lambda:model.forward_candidates([observation],ids)),
            'cached_all_decoders':measure(lambda:model.forward_candidates(None,ids,features=f)),
            'cached_each_decoder':{eid:measure(lambda eid=eid:model.adapter.expert_forward(model.experts[eid],f)) for eid in ids}}
        with torch.no_grad():c=model.forward_candidates(None,ids,features=f)
        if scorer is not None:
            reports[str(count)]['scorer']=measure(lambda:scorer(f,c.trajectories,c.valid))
            with torch.no_grad():p=scorer(f,c.trajectories,c.valid)
            reports[str(count)]['selection_and_CPU_copy']=measure(lambda:(rule or SelectionRule(0,{},{}))(c,p)[0].cpu())
        if count==1 or scorer is not None:
            reports[str(count)]['image_to_selected_trajectory']=measure(online)
        if router is not None:
            if tuple(router.expert_ids)==ids and ids==tuple(model.experts):
                reports[str(count)]['router_network_only']=measure(lambda:router(f))
                from ..selector import RouterRule
                actual_rule=router_rule or RouterRule()
                reports[str(count)]['true_prerouted_cached_trajectory']=measure(lambda:model.forward_prerouted(None,actual_rule,features=f)[0].cpu())
                reports[str(count)]['true_prerouted_image_to_trajectory']=measure(lambda:model.forward_prerouted(
                    [observation_loader() if observation_loader else observation],actual_rule)[0].cpu())
            else:
                reports[str(count)]['router_status']='UNMEASURED: trained Router registry differs from this K'
        reports[str(count)]['peak_gpu_bytes']=torch.cuda.max_memory_allocated() if device.startswith('cuda') else None
    return {'hardware':torch.cuda.get_device_name() if device.startswith('cuda') else platform.processor() or platform.machine(),
        'device':device,'batch':1,'parameter_dtype':str(next(model.parameters()).dtype),'VLM_autocast':'inherited_bfloat16',
        'warmup':warmup,'repetitions':repetitions,'synchronization':'CUDA synchronize before/after every measurement' if device.startswith('cuda') else 'wall clock',
        'image_file_preprocess_in_selected_total':observation_loader is not None,'shared_stages':shared,'results':reports,
        'LoRA_DiT_comparison':'UNMEASURED: no aligned runnable expert baseline bound'}

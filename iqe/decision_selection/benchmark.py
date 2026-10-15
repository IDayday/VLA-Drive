"""Same-device real-image latency; no cached-head timing is called end-to-end."""
from collections import Counter
from pathlib import Path
import time
import numpy as np
import torch
from iqe.contracts import require
from iqe.io import read_json,atomic_json,file_hash,digest
from iqe.data.sources import load_observation
from iqe.specialist_study.binding import scenes_by_role
from iqe.specialist_study.evaluation import ResearchSystem
from iqe.specialist_study.training import runtime
from .training import resource
from .online import systems
from .evaluation import _final_lock


@torch.no_grad()
def run(config,device='cuda:0',samples=100,warmup=10):
    require(samples==100 and warmup==10,'registered 100 real scenes and 10 warmups')
    resource(config,'benchmark-decision-systems');device=runtime(device)
    require(str(device).startswith('cuda'),'registered GPU end-to-end benchmark')
    frozen=_final_lock(config);margin=read_json(config.root/'MARGIN_CALIBRATION.json')
    scenes=sorted(scenes_by_role(config.source())['stage_val'],key=lambda s:digest([config.partition_seed,'latency',s.scene_id]))[:samples]
    require(len(scenes)==samples,'not enough fixed real benchmark scenes')
    identity=dict(final_lock=frozen['hash'],samples=samples,warmup=warmup,scene_ids=[s.scene_id for s in scenes],
        observation_hashes=[s.observation_hash for s in scenes],device=device,hardware=torch.cuda.get_device_name(),
        batch=1,precision='original frozen Base runtime, FP32 decision heads; no new autocast',
        includes='observation verification/read, image decode/preprocessing, VLA, selected/all branches, decision, CPU trajectory copy',
        sync='torch.cuda.synchronize before wall start and after CPU trajectory; CUDA events bracket head',
        implementation=file_hash(__file__),data_path='online real images; no VLA feature cache',
        shared_GPU_workloads_allowed=True,each_mode_uses_same_sequence=True)
    folder=config.root/'latency';atomic_json(folder/'PLAN.json',identity,immutable=True)
    models=systems(config,device);adapter=models['M_scene'].adapter
    shared=adapter.framework.qwen_vl_interface.model.model
    parameter_breakdown=dict(
        shared_visual=sum(p.numel() for p in shared.visual.parameters()),
        shared_language=sum(p.numel() for p in shared.language_model.parameters()),
        Base_planner_and_action=sum(p.numel() for p in adapter.framework.action_model.parameters()),
        entire_Base_adapter=sum(p.numel() for p in adapter.parameters()),
        generator_trainable=sum(p.numel() for p in adapter.parameters() if p.requires_grad))
    require(parameter_breakdown['generator_trainable']==0,'benchmark generator unexpectedly trainable')
    from iqe.robust_scorer.config import Config as ReferenceConfig
    from iqe.robust_scorer.evaluation import chosen_endpoint
    ref=ReferenceConfig(**read_json(Path(config.reference_run)/'resolved_config.json'))
    scorer,_=chosen_endpoint(ref,'R00',device)
    expected=read_json(Path(config.reference_run)/'MODEL_CHOICE.json')
    reference_row=next(x for x in expected['cells'] if x['cell']=='R00')
    require(file_hash(reference_row['checkpoint'])==verify_reference_hash(config),'R00 weight identity changed')
    r00=ResearchSystem(adapter,[models['M_set'].experts[eid] for eid in models['M_set'].expert_ids[1:]],scorer,models['M_set'].expert_ids).eval().requires_grad_(False)
    modes=['Base','K4_R00','M_scene','K4_M_set','M_scene_margin','K4_M_set_margin'];reports=[]
    for mode in modes:
        outpath=folder/(mode+'.json')
        if outpath.exists():
            old=read_json(outpath);require(old['identity_hash']==digest(identity),'latency identity changed');reports.append(old);continue
        base_only=mode=='Base';reference=mode=='K4_R00';kind='M_scene' if mode.startswith('M_scene') or base_only else 'M_set'
        active=r00 if reference else models[kind]
        head=scorer if reference else active.head
        calibrated=mode.endswith('_margin');rule=margin['models'][kind] if calibrated else None
        for candidate_kind,system in models.items():
            live=not base_only and not reference and candidate_kind==kind
            system.head.to(device if live else 'cpu')
            keep_experts=live or (reference and candidate_kind=='M_set')
            system.experts.to(device if keep_experts else 'cpu')
        scorer.to(device if reference else 'cpu')
        elapsed=[];head_ms=[];calls=[];fallbacks=0;selected_counts=Counter()
        torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
        for iteration in range(warmup+samples):
            scene=scenes[iteration%len(scenes) if iteration<warmup else iteration-warmup]
            events=[]
            def before(*_):
                e=torch.cuda.Event(enable_timing=True);e.record();events.append(e)
            def after(*_):
                e=torch.cuda.Event(enable_timing=True);e.record();events.append(e)
            handles=[] if base_only else [head.register_forward_pre_hook(before),head.register_forward_hook(after)]
            try:
                torch.cuda.synchronize();started=time.perf_counter()
                observations=[load_observation(scene,adapter.contract)]
                if reference:result=active.shared_forward(observations)
                else:result=active(observations,base_only=base_only,
                                  gamma=rule['gamma'] if rule else None,always_base=rule['always_Base'] if rule else False)
                cpu=result['trajectory'].detach().cpu()
                torch.cuda.synchronize();seconds=time.perf_counter()-started
            finally:
                for h in handles:h.remove()
            require(torch.isfinite(cpu).all(),'invalid benchmark output')
            require(result['calls']['visual']==result['calls']['language']==1,'benchmark second VLA pass')
            if iteration>=warmup:
                elapsed.append(seconds);head_ms.append(events[0].elapsed_time(events[1]) if events else 0.)
                calls.append(result['calls']|({'branches':result['branch_calls']} if reference else {}))
                selected_counts[int(result['selected'][0])]+=1
                if not reference:fallbacks+=int(result['invalid_fallback'].sum())
        report=dict(mode=mode,identity_hash=digest(identity),samples=len(elapsed),warmup=warmup,
            mean_seconds=float(np.mean(elapsed)),p50_seconds=float(np.median(elapsed)),p95_seconds=float(np.quantile(elapsed,.95)),
            decision_head_mean_ms=float(np.mean(head_ms)),decision_head_p50_ms=float(np.median(head_ms)),decision_head_p95_ms=float(np.quantile(head_ms,.95)),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            calls=calls,invalid_fallbacks=fallbacks,selected_counts=dict(selected_counts),
            wall_seconds=elapsed,head_cuda_ms=head_ms,inactive_heads_and_experts_on_CPU=True,
            Base_parameters=sum(p.numel() for p in adapter.parameters()),
            Base_parameter_breakdown=parameter_breakdown,
            expert_parameters={eid:sum(p.numel() for p in m.parameters()) for eid,m in models['M_set'].experts.items()},
            decision_parameters=0 if base_only else sum(p.numel() for p in head.parameters()),
            input_to_CPU_trajectory=True,score_labels_read_by_inference=False)
        atomic_json(outpath,report,immutable=True);reports.append(report)
    result=dict(status='COMPLETE',identity=identity,modes=reports,new_scoring_calls=0,new_navtest=False)
    atomic_json(folder/'BENCHMARK.json',result,immutable=True);return result


def verify_reference_hash(config):
    from .binding import verify_binding
    return verify_binding(config)['R00']['stage_val']['checkpoint_hash']

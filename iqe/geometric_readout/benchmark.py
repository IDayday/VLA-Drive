"""Measure the actual single-VLA geometric architecture, without quality claims."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from iqe.contracts import require,TrajectoryContract
from iqe.data.sources import load_observation
from iqe.io import atomic_json,read_json,file_hash
from iqe.internal_baseline.scorer import initialization_rng
from iqe.specialist_study.binding import scenes_by_role
from iqe.specialist_study.evaluation import ResearchSystem
from iqe.specialist_study.models import StudyScorer
from iqe.specialist_study.training import runtime
from .full import load_config,create,resource_check
from .calibration import CalibrationReader
from .bundle import payload_for,assemble


def run(config,checkpoint,device):
    resource_check(config);device=runtime(device);source=config.source()
    checkpoint=Path(checkpoint).resolve()
    require(checkpoint.parent==(config.root/'scorer').resolve(),'same experimental checkpoint only')
    receipt=read_json(str(checkpoint)+'.COMPLETE.json')
    require(file_hash(checkpoint)==receipt['checksum'],'checkpoint receipt checksum')
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    require(state['dependencies']==read_json(config.root/'scorer/RUN_IDENTITY.json')['dependencies'],
            'checkpoint/experiment mismatch')
    geometric,initial=create(config);geometric.load_state_dict(state['module'],strict=True)
    system=assemble(payload_for(config,geometric,initial),device)
    geometric=system.scorer
    contract=TrajectoryContract(**read_json(source.root/'OFFLINE_BASE_CONTRACT.json')['trajectory'])
    with initialization_rng(source.seed):
        control=StudyScorer(geometric.world_projection[1].in_features,
            geometric.action_projection[1].in_features,contract,spatial=True,
            visual_dim=geometric.spatial_reader.projection[1].in_features)
    control.eval().requires_grad_(False)
    global_system=ResearchSystem(system.adapter,list(system.experts),control,system.expert_ids)
    scenes=scenes_by_role(source)['stage_val'][:32];calibration=CalibrationReader(32)
    # The timing comparison uses the same warmed file-validation state in all
    # modes; images are still opened, decoded and processed inside every sample.
    for scene in scenes:
        load_observation(scene,system.adapter.contract);calibration.read(scene)
    modes=(('Base',0,None),('Base_plus_three',3,None),
           ('global_spatial_reference',3,'global'),('geometric',3,'geometric'))
    require(len(system.experts)==3,'registered K4 pool required for this benchmark')
    rows=[]
    sync=lambda:torch.cuda.synchronize(device)
    for name,count,scoring in modes:
        for i,expert in enumerate(system.experts):expert.to(device if i<count else 'cpu')
        geometric.to(device if scoring=='geometric' else 'cpu')
        control.to(device if scoring=='global' else 'cpu')
        active=global_system if scoring=='global' else system
        hooks=[];events=[]
        if scoring:
            def before(*_):
                pair=(torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True))
                pair[0].record();events.append(pair)
            def after(*_):events[-1][1].record()
            hooks=[active.scorer.register_forward_pre_hook(before),active.scorer.register_forward_hook(after)]
        torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats(device)
        times=[];calls=[];branches=[]
        try:
            for i in range(35):
                scene=scenes[i%len(scenes)];sync();start=time.perf_counter()
                observations=[load_observation(scene,system.adapter.contract)]
                if scoring=='geometric':
                    result=active.shared_forward(observations,expert_count=count,score=True,
                        calibration=calibration.batch([scene],device))
                else:result=active.shared_forward(observations,expert_count=count,score=scoring is not None)
                result['trajectory'].cpu();sync()
                elapsed=time.perf_counter()-start
                require(result['calls']=={'visual':1,'language':1},'extra VLA forward')
                expected=dict(Base_decoder=1,expert_decoders=[1]*count,
                    score_context_projections=int(scoring is not None),spatial_reader=int(scoring is not None))
                require(result['branch_calls']==expected,'branch/projection count changed')
                if i>=3:times.append(elapsed);calls.append(result['calls']);branches.append(result['branch_calls'])
        finally:
            for h in hooks:h.remove()
        scorer_ms=[a.elapsed_time(b) for a,b in events[3:]]
        rows.append(dict(mode=name,K=count+1,batch=1,warmup=3,repetitions=32,
            mean_seconds=float(np.mean(times)),p50_seconds=float(np.median(times)),p95_seconds=float(np.quantile(times,.95)),
            scorer_mean_seconds=float(np.mean(scorer_ms)/1000) if scorer_ms else 0.,
            scorer_p50_seconds=float(np.median(scorer_ms)/1000) if scorer_ms else 0.,
            scorer_p95_seconds=float(np.quantile(scorer_ms,.95)/1000) if scorer_ms else 0.,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
            verified_calls=calls[0],verified_branches=branches[0],all_calls_verified=True))
    report=dict(status='COMPLETE',checkpoint=str(checkpoint),checkpoint_hash=file_hash(checkpoint),
        checkpoint_optimizer_step=receipt['optimizer_step'],scientific_quality_evaluation=False,
        control_weights='same random initialization, architecture cost reference only',
        actual_images=True,cache_head_latency_substituted=False,filesystem_state='prevalidated warm file cache',
        input_load_decode_processor_and_output_CPU_copy_included=True,
        CUDA_synchronization='synchronize before/after full sample; scorer CUDA events in same forward',
        device=torch.cuda.get_device_name(device),VLM_runtime='unchanged locked Base inference runtime',
        source_Base_hash=read_json(source.root/'BASE_BINDING.json')['checkpoint_hash'],
        scorer_parameters=sum(p.numel() for p in geometric.parameters()),
        control_scorer_parameters=sum(p.numel() for p in control.parameters()),
        Base_parameters=sum(p.numel() for p in system.adapter.parameters()),
        expert_parameters=[sum(p.numel() for p in x.parameters()) for x in system.experts],rows=rows,
        raw_times_unit='seconds',new_optimizer_steps=0,external_scorer_weights_loaded=False)
    atomic_json(config.root/f'qualification/LATENCY_{receipt["optimizer_step"]:06d}.json',report,immutable=True)
    return report


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--config',required=True)
    p.add_argument('--checkpoint',required=True);p.add_argument('--device',default='cuda:0');a=p.parse_args()
    require(a.device.startswith('cuda'),'real GPU timing required')
    print(json.dumps(run(load_config(a.config),a.checkpoint,a.device),allow_nan=False))


if __name__=='__main__':main()

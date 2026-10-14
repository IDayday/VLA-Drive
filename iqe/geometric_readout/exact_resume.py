"""Eight disposable real-data updates across the two-flow phase boundary.

Run each mode in a new process. These weights cannot enter endpoint selection.
The production optimizer, schedule, sampler and budget remain untouched.
"""
import argparse
import json
import os
from pathlib import Path
import torch
import torch.distributed as dist
from iqe.contracts import require
from iqe.io import atomic_json, digest, file_hash, read_json
from iqe.specialist_study.binding import scenes_by_role
from iqe.specialist_study.bank import load_bank
from iqe.specialist_study.context_reader import VerifiedSpatialReader
from iqe.specialist_study.objectives import StudySampler, pair_population, terms, denominators
from iqe.specialist_study.scorer_training import batch_fetch
from iqe.specialist_study.training import preload, runtime
from iqe.specialist_study.qualifications import equal_tree
from iqe.training.trainer import train, init_distributed
from .calibration_bank import PreparedCalibrationReader
from .full import load_config, create, binding, resource_check, collective_run_lock, calibrated_fetch, together


def run(config,mode,device):
    require(mode in ('continuous','interrupted','resume','compare'),'resume diagnostic mode')
    source=config.source();root=config.root/'qualification/real_resume_geometry'
    if mode=='compare':
        checkpoints=[root/name/'step_000004.pt' for name in ('continuous','interrupted')]
        for p in checkpoints:
            require(file_hash(p)==read_json(str(p)+'.COMPLETE.json')['checksum'],'diagnostic checkpoint corrupt')
        a,b=[torch.load(p,map_location='cpu',weights_only=False) for p in checkpoints]
        keys=('module','optimizer','scheduler','sampler','rng_by_rank','consumed_sample_ids')
        equality={key:equal_tree(a[key],b[key]) for key in keys}
        logs=[[json.loads(line) for line in (root/name/'steps.jsonl').read_text().splitlines()]
              for name in ('continuous','interrupted')]
        equality['loss_sequence']=[r['loss'] for r in logs[0]]==[r['loss'] for r in logs[1]]
        equality['gradient_norm_sequence']=[r['gradient_norm'] for r in logs[0]]==[r['gradient_norm'] for r in logs[1]]
        equality['next_batch_after_resume']=read_json(root/'interrupted/RESUME_RESOLUTION.json')['next_sample_ids']==logs[0][2]['sample_ids']
        require(all(equality.values()),'new-process real-data exact resume mismatch')
        report=dict(status='PASS',equality=equality,actual_updates_total=8,phase_switch_at_diagnostic_step=2,
            production_phase_switch=3000,world_size=read_json(str(checkpoints[0])+'.COMPLETE.json')['world_size'],
            complete_checkpoint_hashes=[file_hash(p) for p in checkpoints],production_weights_modified=False,
            diagnostic_weights_used_for_selection=False)
        atomic_json(root/'REPORT.json',report,immutable=True);return report
    if int(os.environ.get('WORLD_SIZE','1'))>1 and device.startswith('cuda'):device='cuda:'+os.environ['LOCAL_RANK']
    device=runtime(device);rank,world=init_distributed(device);resource_check(config,True)
    folder=root/('interrupted' if mode=='resume' else mode)
    with collective_run_lock(folder/'RUN.lock',rank):
        bound=together(lambda:binding(config))
        dm,deploy=together(lambda:load_bank(source,'deploy','scorer_fit'))
        rm,rich=together(lambda:load_bank(source,'rich','scorer_fit'))
        pairs,_=pair_population(deploy['scores'],deploy['comparison_valid'])
        indices=list(dict.fromkeys(int(p[0]) for p in pairs))[:32]
        require(len(indices)==32,'32 real comparison scenes required')
        all_scenes=scenes_by_role(source)['scorer_fit']
        scenes=[all_scenes[i] for i in indices]
        deploy={k:v[indices] for k,v in deploy.items()};rich={k:v[indices] for k,v in rich.items()}
        model,initial=together(lambda:create(config))
        cache=together(lambda:preload(source,scenes));spatial=together(lambda:VerifiedSpatialReader(source,64))
        calibration=together(lambda:PreparedCalibrationReader(source,config.calibration_bank))
        sampler=StudySampler([s.scene_id for s in scenes],deploy['scores'],deploy['comparison_valid'],
            seed=source.seed+10,bank_hash=digest([dm['hash'],rm['hash'],indices]),phase_steps=2)
        dependencies=dict(production_binding_hash=digest(bound),source=file_hash(__file__),
            scene_ids=digest([s.scene_id for s in scenes]),sampler=sampler.identity,
            status='DISPOSABLE_DIAGNOSTIC',production_updates_added=0)
        cfg=dict(lr=1e-4,weight_decay=.01,global_batch_size=128,micro_batch_size=16,
            max_optimizer_steps=4,warmup_steps=1,lr_min_ratio=1.,grad_clip_norm=1.,schedule='warmup_constant',
            validation_steps=[],validation_every_steps=1000,checkpoint_every_steps=2,early_stop_patience=10**9)
        fetch=calibrated_fetch(batch_fetch(source,scenes,cache,spatial,deploy,rich,True),scenes,calibration)
        train(model,lambda m,b:terms(m(b['context'],b['trajectories']),b),fetch,sampler,cfg,dependencies,folder,
            device=device,resume='auto' if mode=='resume' else None,stop_after=2 if mode=='interrupted' else None,
            seed=source.seed+20,weights=dict(value=1.,components=.2,ranking=.2),denominator_function=denominators)
        return read_json(folder/'result.json')


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('mode',choices=('continuous','interrupted','resume','compare'))
    p.add_argument('--config',required=True);p.add_argument('--device',default='cuda:0');a=p.parse_args()
    result=run(load_config(a.config),a.mode,a.device)
    if not dist.is_initialized() or dist.get_rank()==0:print(json.dumps(result,allow_nan=False))
    if dist.is_initialized():dist.destroy_process_group()


if __name__=='__main__':main()

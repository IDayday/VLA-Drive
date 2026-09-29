"""Validated torchrun entry for complete-method diagnostics, profiles and training.

Run this inside run_allocated to borrow verified pressure-script GPUs. This
launcher never stops a process or allocates a device itself.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from tools.ddpolicy_vehicle.prepare_data import atomic_json
from starVLA.model.modules.vehicle_joint.initialization import file_sha256


def main():
    p=argparse.ArgumentParser(__doc__)
    for k in ('candidate','campaign-root','data','image-root','targets','index','interaction-root',
              'calibration','teacher-verification','qwen','sources','run-id'):
        p.add_argument('--'+k,required=True)
    p.add_argument('--gpus',type=int,required=True);p.add_argument('--micro-batch',type=int,required=True)
    p.add_argument('--master-port',type=int,required=True)
    p.add_argument('--scope',choices=('startup','small_fit','profile','formal'),required=True)
    p.add_argument('--updates',type=int,required=True);p.add_argument('--schedule-updates',type=int,required=True)
    p.add_argument('--warmup',type=int,default=5000);p.add_argument('--max-seconds',type=float,required=True)
    p.add_argument('--campaign-gpu-hours',type=float,required=True);p.add_argument('--limit',type=int,default=0)
    p.add_argument('--stop-after',type=int,default=0);p.add_argument('--save-every',type=int,default=200)
    p.add_argument('--milestones',default='0');p.add_argument('--seed',type=int,default=42)
    p.add_argument('--loader-workers',type=int,default=4);p.add_argument('--registration')
    p.add_argument('--resume',action='store_true');p.add_argument('--acknowledge-stop',action='store_true')
    p.add_argument('--deterministic',action='store_true');a=p.parse_args()
    if a.gpus not in (1,4,8) or 32%a.gpus or not 1024<=a.master_port<=65535:
        raise ValueError('Explicit single-host GPU/port allocation required')
    cal=json.loads(Path(a.calibration).read_text());teacher=json.loads(Path(a.teacher_verification).read_text())
    if not cal['passed'] or not cal['all_four_targets_real'] or not teacher['passed']:
        raise ValueError('Verified complete four-loss and teacher chains required')
    if cal['candidate']!='C3' or cal['lambda_cur']!=1 or min(cal['lambda_fut'],cal['lambda_int'])<=0:
        raise ValueError('Wrong common training-only calibration')
    targets=Path(a.targets)/a.candidate;di=json.loads((targets/'identity.json').read_text())
    ii=json.loads((Path(a.interaction_root)/'identity.json').read_text())
    if ii['identity']!=teacher['export_identities']['train'] or ii['identity']!=cal['target_identities']['interaction']:
        raise ValueError('All six students must share the verified frozen teacher')
    if a.scope=='formal':
        if not a.registration or a.limit:raise ValueError('Formal full-population registration required')
        reg=json.loads(Path(a.registration).read_text());spec=reg['runs'].get(a.run_id)
        expected={'candidate':a.candidate,'seed':a.seed,'updates':a.updates,'schedule_updates':a.schedule_updates,
                  'global_batch':32,'gpus':a.gpus,'micro_batch':a.micro_batch}
        if spec!=expected or reg['calibration_sha256']!=file_sha256(a.calibration) or reg['teacher_verification_sha256']!=file_sha256(a.teacher_verification):
            raise ValueError('Unregistered formal run/configuration')
        if reg['status']!='FROZEN_BEFORE_SCREENING' or a.campaign_gpu_hours>reg['gpu_hours_cap']:
            raise ValueError('Formal budget not frozen')
    env=dict(os.environ,FORESIGHT_QWEN=a.qwen,FORESIGHT_SOURCES=a.sources,
        FULL_LAMBDA_FUT=str(cal['lambda_fut']),FULL_LAMBDA_INT=str(cal['lambda_int']),
        TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1')
    if a.deterministic:env['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    from omegaconf import OmegaConf
    # Resolve a per-run immutable config outside the locked source worktree.
    for k in ('FORESIGHT_QWEN','FORESIGHT_SOURCES','FULL_LAMBDA_FUT','FULL_LAMBDA_INT'):os.environ[k]=env[k]
    cfg=OmegaConf.load(f'configs/foresight_resolution/{a.candidate.lower()}.yaml');cfg.seed=a.seed
    config=OmegaConf.to_container(cfg,resolve=True);directory=Path(a.campaign_root)/'run_configs';directory.mkdir(exist_ok=True)
    config_path=directory/(a.run_id+'.json')
    if config_path.exists():
        if json.loads(config_path.read_text())!=config:raise ValueError('Existing run config differs')
    else:atomic_json(config_path,config)
    cmd=[sys.executable,'-m','torch.distributed.run','--nproc_per_node',str(a.gpus),'--master_port',str(a.master_port),
         '-m','tools.foresight.train_student','--config',str(config_path),'--data',a.data,'--local-image-root',a.image_root,
         '--dino-root',str(targets),'--dino-index',a.index,'--dino-identity',di['identity'],
         '--interaction-root',a.interaction_root,'--interaction-identity',ii['identity'],
         '--campaign-root',a.campaign_root,'--run-id',a.run_id,'--global-batch','32']
    for key in ('micro_batch','scope','updates','schedule_updates','warmup','max_seconds','campaign_gpu_hours','limit','stop_after','save_every','milestones','loader_workers'):
        cmd.extend(['--'+key.replace('_','-'),str(getattr(a,key))])
    for flag in ('resume','acknowledge_stop','deterministic'):
        if getattr(a,flag):cmd.append('--'+flag.replace('_','-'))
    os.execvpe(cmd[0],cmd,env)

if __name__=='__main__':main()

"""Freeze explicit S0--S4 identities and launch existing full trainer, no old weights."""
import argparse,json,os,subprocess,sys
from pathlib import Path
from starVLA.model.modules.vehicle_joint.initialization import identity_hash,file_sha256
from starVLA.model.modules.foresight.config import ForesightConfig
from tools.ddpolicy_vehicle.prepare_data import atomic_json

ARMS={'S0':('dino_sequence','none','action_only'), 'S1':('dino_sequence','gt_ego','action_only'),
 'S2':('video_clip','none','action_only'),'S3':('video_clip','gt_ego','action_only'),
 'S4':('video_clip','gt_ego','action_plus_W'),'C_BASE_A':('video_clip','none','action_only'),
 'C_BASE_AW':('video_clip','none','action_plus_W'),'S4_NO_MAE':('video_clip','gt_ego','action_plus_W')}

PLANNING_ARMS = {'A_W': 'S0', 'A_NO_MAE': 'S0', 'A_ACTION': 'S0',
                 'V_NONE': 'S2', 'V_MEMORY': 'S3', 'V_QUERY': 'S3'}


def create_config(base,arm,clip,lambda_future,seed):
    if arm in PLANNING_ARMS:
        cfg = create_config(base, PLANNING_ARMS[arm], clip, lambda_future, seed)
        f = cfg['foresight']
        f.update(interaction_readout_source='action' if arm == 'A_ACTION' else 'world',
                 future_action_injection='memory_and_query' if arm == 'V_QUERY' else 'memory_only',
                 future_action_query_scale=1.0, interaction_functional_loss='disabled')
        if arm == 'A_NO_MAE':
            f.update(arm='NO_INTERACTION', ablation='NO_INTERACTION', enable_interaction=False, lambda_int=0.)
        ForesightConfig(**f).validate()
        return cfg
    import copy
    cfg=copy.deepcopy(base);target,condition,planner=ARMS[arm]
    cfg['seed']=seed;cfg['framework']['name']='DDPActionVideoForesight';cfg['framework']['qwenvl']['device_map']='cpu'
    f=cfg['foresight'];f.update(arm='C1',candidate='C1',ablation='FULL',future_target_type=target,
        future_action_condition=condition,planner_condition_mode=planner,clip_times_s=[.5*i for i in range(1,9)],
        future_feature_dim=1024,future_grid_height=9,future_grid_width=12,
        future_time_intervals_s=clip['time_intervals_s'],video_teacher_identity=identity_hash(clip['recipe']),
        future_normalization='nonaffine_layernorm_per_token_v1',lambda_cur=1.,lambda_fut=lambda_future,lambda_int=.8328945981862067)
    if arm in ('C_BASE_A','C_BASE_AW'):f.update(arm='NO_FUTURE',ablation='NO_FUTURE',enable_future_dino=False,lambda_fut=0.)
    if arm=='S4_NO_MAE':f.update(arm='NO_INTERACTION',ablation='NO_INTERACTION',enable_interaction=False,lambda_int=0.)
    ForesightConfig(**f).validate();return cfg


def main():
    p=argparse.ArgumentParser(__doc__)
    for k in ('base-config','data','dino-root','dino-index','interaction-root','clip-root','campaign-root','run-id'):p.add_argument('--'+k,required=True)
    p.add_argument('--arm',choices=(*ARMS, *PLANNING_ARMS),required=True);p.add_argument('--calibration');p.add_argument('--lambda-future',type=float)
    p.add_argument('--seed',type=int,default=42);p.add_argument('--gpus',type=int,default=8);p.add_argument('--micro-batch',type=int,default=4)
    p.add_argument('--updates',type=int,default=100000);p.add_argument('--schedule-updates',type=int,default=100000)
    p.add_argument('--scope',choices=('startup','profile','formal'),required=True);p.add_argument('--limit',type=int,default=0)
    p.add_argument('--stop-after',type=int,default=0);p.add_argument('--master-port',type=int,required=True)
    p.add_argument('--milestones',default='0,1,100,500,1000,2000,5000,10000,25000,50000,75000,100000')
    p.add_argument('--resume',action='store_true');p.add_argument('--acknowledge-stop',action='store_true');p.add_argument('--deterministic',action='store_true')
    p.add_argument('--local-image-root')
    p.add_argument('--campaign-gpu-hours', type=float)
    p.add_argument('--max-seconds', type=float)
    p.add_argument('--register-only',action='store_true');a=p.parse_args()
    source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    if subprocess.check_output(['git','status','--porcelain']).strip():raise ValueError('Freeze run source')
    base=json.loads(Path(a.base_config).read_text());base=base.get('config',base)
    clip=json.loads((Path(a.clip_root)/'identity.json').read_text())
    if a.calibration:
        calibration=json.loads(Path(a.calibration).read_text())
        if not calibration['passed']:raise ValueError('No real calibrated four-loss validation')
        weight=calibration['lambda_fut'];calibration_id=file_sha256(a.calibration)
    elif a.scope!='formal' and a.lambda_future and a.lambda_future>0:weight=a.lambda_future;calibration_id='engineering_unit_weight_only'
    else:raise ValueError('Formal experiment needs verified real training calibration')
    cfg=create_config(base,a.arm,clip,weight,a.seed)
    root=Path(a.campaign_root);directory=root/'registrations';directory.mkdir(parents=True,exist_ok=True)
    config_path=directory/(a.run_id+'_config.json');registration_path=directory/(a.run_id+'.json')
    index=json.loads((Path(a.data)/'index.json').read_text());di=json.loads((Path(a.dino_root)/'identity.json').read_text());ii=json.loads((Path(a.interaction_root)/'identity.json').read_text())
    milestones=sorted(int(x) for x in a.milestones.split(',') if x)
    registration={'schema':'action_video_formal_registration_v1','run_id':a.run_id,'arm':a.arm,'training_source_sha':source,
        'config_sha256':identity_hash(cfg),'scene_count':len(index),'scene_index':identity_hash(index),'seed':a.seed,
        'global_batch':32,'micro_batch':a.micro_batch,'world_size':a.gpus,'updates':a.updates,'schedule_updates':a.schedule_updates,
        'milestones':milestones,'calibration_sha256':calibration_id,'dino':di['identity'],'interaction':ii['identity'],
        'future_clip':clip['identity'] if cfg['foresight']['enable_future_dino'] else None,
        'initialization':'genericQwen plus independently seeded random driving, queries and heads; no checkpoint initialization',
        'GPU_hours_limit':a.campaign_gpu_hours,'time_limit':a.max_seconds,'stage_pause':a.stop_after,'scope':a.scope,
        'local_image_root':a.local_image_root}
    if a.resume and registration_path.exists():
        registration['stage_pause']=json.loads(registration_path.read_text())['stage_pause']
    for path,value in [(config_path,cfg),(registration_path,registration)]:
        if path.exists() and json.loads(path.read_text())!=value:raise ValueError('Frozen registration/config changed')
        if not path.exists():atomic_json(path,value)
    if a.scope=='formal':
        if a.limit or len(index)!=101592 or not (Path(a.data)/'COMPLETE.json').exists():raise ValueError('Formal common population must be complete')
        if cfg['foresight']['enable_future_dino']:
            complete=json.loads((Path(a.clip_root)/'COMPLETE.json').read_text())
            if complete['identity']!=clip['identity'] or complete['scenes']!=len(index):raise ValueError('No formal partial clip labels')
    if a.register_only:return
    command=[sys.executable,'-m','torch.distributed.run','--nproc_per_node',str(a.gpus),'--master_port',str(a.master_port),
        '-m','tools.foresight.train_student','--config',str(config_path),'--data',a.data,'--dino-root',a.dino_root,
        '--dino-index',a.dino_index,'--dino-identity',di['identity'],'--campaign-root',a.campaign_root,'--run-id',a.run_id,
        '--global-batch','32','--micro-batch',str(a.micro_batch),'--updates',str(a.updates),'--schedule-updates',str(a.schedule_updates),
        '--warmup',str(0 if a.scope=='startup' else 5000),'--scope',a.scope,'--limit',str(a.limit),'--stop-after',str(a.stop_after),
        '--save-every','200','--milestones',a.milestones,'--loader-workers','2']
    # The reusable trainer now explicitly supports unbounded wall time/GPUh;
    # the finite registered optimizer horizon remains the execution boundary.
    if a.scope=='startup':command+=['--max-seconds','1800','--campaign-gpu-hours','1000000']
    elif a.scope=='profile':command+=['--max-seconds','14400','--campaign-gpu-hours','1000000']
    if cfg['foresight']['enable_future_dino']:command+=['--clip-root',a.clip_root,'--clip-identity',clip['identity']]
    if cfg['foresight']['enable_interaction']:command+=['--interaction-root',a.interaction_root,'--interaction-identity',ii['identity']]
    if a.local_image_root:command+=['--local-image-root',a.local_image_root]
    if a.scope=='formal':command+=['--registration',str(registration_path)]
    if a.scope=='formal' and a.campaign_gpu_hours is not None:command+=['--campaign-gpu-hours',str(a.campaign_gpu_hours)]
    if a.scope=='formal' and a.max_seconds is not None:command+=['--max-seconds',str(a.max_seconds)]
    for key in ('resume','acknowledge_stop','deterministic'):
        if getattr(a,key):command+=['--'+key.replace('_','-')]
    os.execvpe(command[0],command,os.environ)

if __name__=='__main__':main()

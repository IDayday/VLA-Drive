"""Freeze an actual six-candidate, full-population schedule after measured topology selection."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
from omegaconf import OmegaConf
from tools.ddpolicy_vehicle.prepare_data import atomic_json
from starVLA.model.modules.vehicle_joint.initialization import identity_hash,file_sha256


def read(path):return json.loads(Path(path).read_text())


def main():
    p=argparse.ArgumentParser(__doc__)
    for k in ('campaign-root','data','interaction-root','qwen','sources','profiles','topology-decision','output'):
        p.add_argument('--'+k,required=True)
    p.add_argument('--gpus',type=int,choices=(4,8),required=True)
    p.add_argument('--updates',type=int,default=100000);p.add_argument('--screen',type=int,default=25000)
    p.add_argument('--gpu-hours-cap',type=float,required=True);a=p.parse_args()
    out=Path(a.output);root=Path(a.campaign_root)
    if out.exists():raise FileExistsError('Registration is immutable')
    if subprocess.check_output(['git','status','--porcelain']).strip():raise ValueError('Freeze source before registration')
    if not 5000<a.screen<a.updates or a.gpu_hours_cap<=0:raise ValueError('Invalid finite common schedule/budget')
    profiles=read(a.profiles);decision=read(a.topology_decision)
    if decision['gpus_per_student']!=a.gpus or decision['profile_sha256']!=file_sha256(a.profiles):raise ValueError('Measured topology identity mismatch')
    if not decision['cross_host_checked'] or not decision['actual_concurrency_checked']:raise ValueError('Topology comparison incomplete')
    if profiles['failures']:raise ValueError('Resolve profile accounting errors')
    measured=[r for r in profiles['profiles'] if r['world_size']==a.gpus and r['measurement_complete']]
    if not {'C0','C5'}<={r['candidate'] for r in measured}:raise ValueError('Both size extremes must be measured')
    data=read(Path(a.data)/'identity.json');index=read(Path(a.data)/'index.json');ego=read(Path(a.data)/'ego_identity.json')
    di=read(root/'dino_index_v1/identity.json');ti=read(root/'dino_index_v1/train_scenes.json');vi=read(root/'dino_index_v1/dev_scenes.json')
    if identity_hash(index)!=data['index_sha256'] or data['index_sha256']!=di['index_hashes']['train'] or len(index)!=len(ti):raise ValueError('Training population mismatch')
    if {r['log'] for r in ti}&{r['log'] for r in vi} or {r['token'] for r in ti}&{r['token'] for r in vi}:raise ValueError('Train/dev overlap')
    cal=read(root/'four_loss_calibration_v1.json');teacher=read(root/'teacher_reuse_verification.json');interaction=read(Path(a.interaction_root)/'identity.json')
    if not cal['passed'] or not teacher['passed'] or interaction['identity']!=teacher['export_identities']['train']:raise ValueError('Frozen teacher/calibration mismatch')
    os.environ.update(FORESIGHT_QWEN=a.qwen,FORESIGHT_SOURCES=a.sources,FULL_LAMBDA_FUT=str(cal['lambda_fut']),FULL_LAMBDA_INT=str(cal['lambda_int']))
    configs={};caches={};runs={}
    for c in [f'C{i}' for i in range(6)]:
        configs[c]=OmegaConf.to_container(OmegaConf.load(f'configs/foresight_resolution/{c.lower()}.yaml'),resolve=True)
        cache=read(root/'dino_targets_v1'/c/'identity.json');done=read(root/'dino_targets_v1'/c/'COMPLETE.json')
        if cache['index']!=di['identity'] or done['identity']!=cache['identity'] or done['images']!=cache['image_count']:raise ValueError('Incomplete/foreign candidate cache: '+c)
        caches[c]=cache['identity'];runs[f'formal_{c}_seed42_v1']={'candidate':c,'seed':42,'updates':a.updates,'schedule_updates':a.updates,'global_batch':32,'gpus':a.gpus,'micro_batch':32//a.gpus}
    # Conservative complete campaign bound: six screens, three finalists to end,
    # three full second seeds, three selected-size task ablations, plus overhead.
    worst=max(r['optimizer_gpu_hours_per_1000_updates']/1000 for r in measured)
    planned_updates=6*a.screen+3*(a.updates-a.screen)+6*a.updates
    estimate=worst*planned_updates*1.35+192
    if estimate>a.gpu_hours_cap:raise ValueError(f'Common plan estimated {estimate:.1f} GPUh exceeds registered cap; revise before ranking')
    milestones=sorted({0,1,100,500,1000,2000,5000,10000,a.screen,a.updates//2,3*a.updates//4,a.updates})
    record={'schema':'ddp_full_formal_registration_v1','status':'FROZEN_BEFORE_SCREENING','created_unix':time.time(),
        'training_source_sha':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'runs':runs,'configs':configs,'config_identities':{c:identity_hash(v) for c,v in configs.items()},
        'data_identity':data['identity'],'ego_identity':ego['identity'],'dino_index':di['identity'],'dino_identities':caches,
        'interaction_identity':interaction['identity'],'generic_sources_sha256':file_sha256(a.sources),
        'calibration_sha256':file_sha256(root/'four_loss_calibration_v1.json'),'teacher_verification_sha256':file_sha256(root/'teacher_reuse_verification.json'),
        'scene_count':len(index),'train_logs':len({r['log'] for r in ti}),'dev_scenes':len(vi),'dev_logs':len({r['log'] for r in vi}),
        'screen_updates':a.screen,'updates':a.updates,'warmup':5000,'save_every':200,'milestones':milestones,
        'diagnostic_updates':[1,100,500,1000,2000],'development_updates':[5000,10000,a.screen,a.updates//2,3*a.updates//4,a.updates],
        'loader_workers':4,'deterministic':True,'gpu_hours_cap':a.gpu_hours_cap,'planned_optimizer_updates':planned_updates,
        'estimated_campaign_gpu_hours':estimate,'cost_estimate':'worst measured complete-model optimizer rate times all reserved updates, 35% overhead, plus192GPUh preparation',
        'topology_decision':decision,'profile_sha256':file_sha256(a.profiles),
        'initialization':'verified generic Qwen; independent random driving/query/head streams; profile/startup weights forbidden',
        'scheduler':'common100000-reference cosine with5000warmup; screen is planned PAUSED, never COMPLETE',
        'selection':{'mandatory':'C0','others':['best complete-development PDMS','nondominated quality-cost compromise'],
                     'engineering_preference':{'PDMS_percentage_point_tolerance':0.2,'student_cost_saving_fraction':0.2},
                     'third_when_duplicate':'next nondominated candidate with plausible learning potential',
                     'ranking_metric':'official dev FP32 single sample seed42, full population; no Navtest selection'},
        'evaluation':{'precision':'FP32 masters, TF32 off','candidates':1,'steps':10,'screen_sampling_seeds':[42],'final_sampling_seeds':[42,43,44,45,46]},
        'secondary_budget':'three full finalist seed43 runs, then NO_CURRENT/NO_FUTURE/NO_INTERACTION at selected size; no profile-weight reuse',
        'authorized_hosts':['local','training-vla-zt2'],'automatic_external_expansion':False}
    record['identity']=identity_hash(record);atomic_json(out,record);print(json.dumps({'registration':str(out),'identity':record['identity'],'estimated_gpu_hours':estimate}))


if __name__=='__main__':main()

"""Audit reuse of this method's real trained MAE, including fresh masked export."""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import torch
from tools.ddpolicy_vehicle.prepare_data import atomic_json
from tools.ddpolicy_vehicle.run_meter import metered_run
from tools.foresight.teacher_runtime import TeacherDataset, inputs
from starVLA.model.modules.trajectory_mae.model import TrajectoryMAE
from starVLA.model.modules.vehicle_joint.initialization import identity_hash, file_sha256


def checked(path):
    data = json.loads(Path(path).read_text())
    if identity_hash({k:v for k,v in data.items() if k != 'identity'}) != data['identity']:
        raise ValueError('Changed identity: '+str(path))
    return data


def main():
    p=argparse.ArgumentParser(__doc__)
    for k in ('teacher-root','data','train-targets','dev-targets','frozen-teacher','student-train','student-dev','output','campaign-root','run-id'):
        p.add_argument('--'+k,required=True)
    p.add_argument('--device',default='cpu');p.add_argument('--samples',type=int,default=32)
    a=p.parse_args();torch.set_num_threads(4)
    with metered_run(a.campaign_root,a.run_id,int(a.device.startswith('cuda')),{'kind':'verify_reused_gt_mae','real_optimizer_updates':0}) as (meter,_,save):
        root=Path(a.teacher_root);frozen=checked(a.frozen_teacher);ident=checked(root/'identity.json')
        state=json.loads((root/'status.json').read_text())
        if state['status']!='COMPLETE' or state['completed']!=ident['updates'] or ident['config']['limit']!=0:
            raise ValueError('Teacher was not fully trained on the complete registered split')
        if frozen['teacher_run_identity']!=ident['identity'] or frozen['full_training_completed']!=state['completed']:
            raise ValueError('Wrong frozen training run')
        checkpoint=root/frozen['checkpoint']
        if file_sha256(checkpoint)!=frozen['checkpoint_sha256']:
            raise ValueError('Frozen checkpoint hash changed')
        # Reuse needs identical actual encoder/masking/data definitions, not names.
        files=['starVLA/model/modules/trajectory_mae/model.py','starVLA/model/modules/trajectory_mae/tokenizer.py',
               'starVLA/model/modules/trajectory_mae/masking.py','tools/foresight/prepare_vehicle_trajectories.py',
               'starVLA/model/modules/vehicle_joint/graphs.py','starVLA/model/modules/structured_world/geometry.py']
        source_files={}
        for file in files:
            original=subprocess.check_output(['git','show',ident['source_sha']+':'+file])
            if original!=Path(file).read_bytes():raise ValueError('Reuse definition changed: '+file)
            source_files[file]=file_sha256(file)
        ds={s:TeacherDataset(a.data,s) for s in ('train','dev')}
        if ds['train'].identity['identity']!=ident['data_identity'] or ds['train'].identity['identity']!=frozen['data_identity']:
            raise ValueError('Teacher data identity mismatch')
        if {r['log'] for r in ds['train'].index}&{r['log'] for r in ds['dev'].index}:
            raise ValueError('Teacher training/development log leakage')
        target_ids={}
        for split in ('train','dev'):
            student=checked(Path(getattr(a,'student_'+split))/'identity.json')
            target_path=Path(getattr(a,split+'_targets'));target=checked(target_path/'identity.json')
            done=json.loads((target_path/'COMPLETE.json').read_text())
            if ds[split].index_hash!=student['index_sha256'] or target['index_hash']!=ds[split].index_hash:
                raise ValueError('Teacher/student population mismatch')
            if target['frozen_teacher_identity']!=frozen['identity'] or target['teacher_checkpoint_sha256']!=frozen['checkpoint_sha256']:
                raise ValueError('Cached Z_T from a different teacher')
            if done['identity']!=target['identity'] or done['scenes']!=len(ds[split]) or done['failed']:
                raise ValueError('Incomplete target export')
            target_ids[split]=target['identity']
        saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
        if saved['identity']!=ident['identity']:raise ValueError('Checkpoint run identity mismatch')
        model=TrajectoryMAE(**ident['model']);model.load_state_dict(saved['model'],strict=True);del saved
        model.requires_grad_(False).eval().to(a.device)
        selected=list(range(min(a.samples,len(ds['train']))))
        batch=ds['train'].batch(selected,a.device);visible=batch['point_valid'].clone();visible[:,0]=False
        target=torch.zeros(len(selected),device=a.device,dtype=torch.long)
        with torch.inference_mode():
            first=model(inputs(batch,target,visible))
            poisoned={k:v.clone() for k,v in batch.items()};poisoned['future'][:,0]=float('nan')
            second=model(inputs(poisoned,target,visible))
            torch.testing.assert_close(first['latent'],second['latent'],rtol=0,atol=0)
            torch.testing.assert_close(first['xy'],second['xy'],rtol=0,atol=0)
        differences=[];valid_count=0
        for at,i in enumerate(selected):
            token=ds['train'].index[i]['token']
            label=torch.load(Path(a.train_targets)/'targets'/(token+'.pt'),map_location='cpu',weights_only=True)
            if label['identity']!=target_ids['train'] or label['token']!=token:
                raise ValueError('Foreign exported scene')
            if bool(visible[at].any())!=label['interaction_target_valid']:
                raise ValueError('Interaction validity mismatch')
            z=second['latent'][at].cpu();error=float((z-label['latent']).abs().max());differences.append(error)
            # CPU/CUDA kernels are not claimed bitwise identical; both are FP32.
            torch.testing.assert_close(z,label['latent'],rtol=2e-5,atol=3e-5)
            valid_count+=int(label['interaction_target_valid'])
        old_costs=[]
        for path in root.parent.glob('runs/*/status.json'):
            cost=json.loads(path.read_text())
            if cost.get('kind') in ('trajectory_mae_teacher','interaction_target_export') or cost.get('run_id','').startswith('teacher_'):
                old_costs.append({k:cost.get(k) for k in ('run_id','kind','status','gpu_hours','real_optimizer_updates')})
        summary_path=root.parent/'teacher_learning_final_v1/summary.json'
        metrics=json.loads(summary_path.read_text())
        report={'passed':True,'teacher_reused':True,'teacher_identity':ident['identity'],
                'checkpoint_sha256':frozen['checkpoint_sha256'],'frozen_identity':frozen['identity'],
                'source_file_hashes':source_files,'source_definition_identical':True,
                'teacher_updates':state['completed'],'teacher_exposure':state['exposure'],
                'populations':{s:len(d) for s,d in ds.items()},'train_dev_log_overlap':0,
                'export_identities':target_ids,'checked_export_scenes':len(selected),'valid_checked_scenes':valid_count,
                'hidden_ego_nan_forward_exact':True,'fresh_export_max_abs_error':max(differences),
                'fresh_export_tolerance':{'atol':3e-5,'rtol':2e-5},'device':a.device,
                'teacher_eval_metrics_sha256':file_sha256(summary_path),
                'selected_development_groups':metrics['milestones']['30'],
                'historical_shared_costs':old_costs,
                'limitation':'Conditional GT-peer reconstruction; stationary reference can outperform teacher on static vehicles; not student planning/PDMS'}
        atomic_json(a.output,report);meter['inference_scenes']=len(selected);save()

if __name__=='__main__':main()

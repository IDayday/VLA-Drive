"""Frozen MAE labels from a FRESH ego-hidden forward, never full-GT context."""
import argparse
import json
from pathlib import Path
import subprocess
import time
import torch
from tools.ddpolicy_vehicle.prepare_data import atomic_json
from tools.ddpolicy_vehicle.run_meter import metered_run
from starVLA.model.modules.vehicle_joint.initialization import identity_hash,file_sha256
from starVLA.model.modules.trajectory_mae.model import TrajectoryMAE
from .teacher_runtime import TeacherDataset,inputs


def save_target(path, payload):
    """Atomically save one latent without retaining its entire inference batch."""
    path = Path(path)
    compact = {**payload, 'latent': payload['latent'].detach().clone()}
    temporary = path.with_suffix('.tmp')
    torch.save(compact, temporary)
    temporary.replace(path)


def main():
    p=argparse.ArgumentParser(__doc__)
    for key in ('data','teacher-run','checkpoint','frozen-teacher','output','campaign-root','run-id'):p.add_argument('--'+key,required=True)
    p.add_argument('--split',choices=('train','dev'),required=True);p.add_argument('--batch',type=int,default=128)
    p.add_argument('--resume',action='store_true');p.add_argument('--max-seconds',type=float,default=7200)
    p.add_argument('--campaign-gpu-hours',type=float,default=6000)
    a=p.parse_args()
    if min(a.batch,a.max_seconds,a.campaign_gpu_hours)<=0:raise ValueError('Positive export batch and budgets required')
    attempt = a.run_id + '_attempt_' + str(time.time_ns())
    with metered_run(a.campaign_root,attempt,1,{'kind':'interaction_target_export','real_optimizer_updates':0,
                                           'run_id_parent':a.run_id}) as (record,_,save):
        source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
        if subprocess.check_output(['git','status','--porcelain']).strip():raise ValueError('Lock export source before producing teacher targets')
        teacher_id=json.loads((Path(a.teacher_run)/'identity.json').read_text())
        frozen=json.loads(Path(a.frozen_teacher).read_text())
        if frozen.get('schema')!='foresight_frozen_teacher_v1' or identity_hash({k:v for k,v in frozen.items() if k!='identity'})!=frozen['identity']:
            raise ValueError('Invalid frozen teacher identity')
        if frozen['teacher_run_identity']!=teacher_id['identity'] or frozen['checkpoint']!=a.checkpoint:
            raise ValueError('Export teacher differs from preregistered frozen selection')
        if teacher_id['config']['limit']!=0:raise ValueError('Small-fit teacher cannot label the full student dataset')
        data=TeacherDataset(a.data,a.split)
        if data.identity['identity']!=teacher_id['data_identity']:raise ValueError('Teacher training data definition changed')
        checkpoint=Path(a.teacher_run)/a.checkpoint
        if file_sha256(checkpoint)!=frozen['checkpoint_sha256']:raise ValueError('Frozen teacher checkpoint changed')
        from tools.ddpolicy_vehicle.campaign import charged_gpu_hours
        if charged_gpu_hours(Path(a.campaign_root))>=a.campaign_gpu_hours:raise RuntimeError('Campaign budget exhausted before export')
        state=torch.load(checkpoint,map_location='cpu',weights_only=False)
        if state['identity']!=teacher_id['identity']:raise ValueError('Wrong teacher checkpoint')
        identity={'schema':'foresight_interaction_latent_v1','source_sha':source,'teacher_identity':teacher_id['identity'],
                  'frozen_teacher_identity':frozen['identity'],
                  'teacher_checkpoint_sha256':file_sha256(checkpoint),'teacher_completed_updates':state['completed'],
                  'data_identity':data.identity['identity'],'index_hash':data.index_hash,'split':a.split,
                  'ego_mask':'all ego future removed before every encoding; fresh masked encoder forward',
                  'query_order_s':[.5*(i+1) for i in range(8)],'coordinate_frame':'ego_t0',
                  'normalization':'raw teacher Z; fixed nonaffine LayerNorm eps1e-5 on BOTH sides at loss',
                  'graph':data.identity['graph']}
        identity['identity']=identity_hash(identity);out=Path(a.output)
        if out.exists():
            if not a.resume or json.loads((out/'identity.json').read_text())!=identity:raise ValueError('Target cache identity changed')
        else:
            (out/'targets').mkdir(parents=True);atomic_json(out/'identity.json',identity);atomic_json(out/'index.json',data.index)
        model=TrajectoryMAE(**teacher_id['model']);model.load_state_dict(state['model'],strict=True);model.requires_grad_(False).eval().cuda()
        del state
        for start in range(0,len(data),a.batch):
            if time.time()-record['start_unix']>a.max_seconds or charged_gpu_hours(Path(a.campaign_root))>=a.campaign_gpu_hours:
                record['status']='PAUSED';save();return
            indices=list(range(start,min(start+a.batch,len(data))));batch=data.batch(indices,'cuda')
            visible=batch['point_valid'].clone();visible[:,0]=False
            target=torch.zeros(len(indices),device='cuda',dtype=torch.long)
            # Hidden ego future is deliberately poison, never available to encoder.
            batch['future'][:,0]=float('nan')
            with torch.inference_mode():z=model(inputs(batch,target,visible))['latent'].float().cpu()
            valid=visible.flatten(1).any(-1).cpu()
            for i,index in enumerate(indices):
                token=data.index[index]['token'];path=out/'targets'/(token+'.pt')
                payload={'identity':identity['identity'],'token':token,'latent':z[i],
                         'interaction_target_valid':bool(valid[i]),'known_peer_points':int(visible[i].sum())}
                if path.exists():
                    old=torch.load(path,weights_only=True)
                    if old['identity']!=identity['identity'] or not torch.equal(old['latent'],z[i]):raise ValueError('Non-deterministic/existing target mismatch')
                else:
                    save_target(path, payload)
            record['inference_scenes']=start+len(indices);save()
        atomic_json(out/'COMPLETE.json',{'scenes':len(data),'identity':identity['identity'],'failed':0})


if __name__=='__main__':main()

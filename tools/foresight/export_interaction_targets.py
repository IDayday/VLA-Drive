"""Frozen MAE labels from a FRESH ego-hidden forward, never full-GT context."""
import argparse
import json
from pathlib import Path
import subprocess
import torch
from tools.ddpolicy_vehicle.prepare_data import atomic_json
from tools.ddpolicy_vehicle.run_meter import metered_run
from starVLA.model.modules.vehicle_joint.initialization import identity_hash,file_sha256
from starVLA.model.modules.trajectory_mae.model import TrajectoryMAE
from .teacher_runtime import TeacherDataset,inputs


def main():
    p=argparse.ArgumentParser(__doc__)
    for key in ('data','teacher-run','checkpoint','output','campaign-root','run-id'):p.add_argument('--'+key,required=True)
    p.add_argument('--split',choices=('train','dev'),required=True);p.add_argument('--batch',type=int,default=128)
    p.add_argument('--resume',action='store_true');p.add_argument('--max-seconds',type=float,default=7200)
    a=p.parse_args()
    with metered_run(a.campaign_root,a.run_id,1,{'kind':'interaction_target_export','real_optimizer_updates':0}) as (record,_,save):
        import time
        source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
        teacher_id=json.loads((Path(a.teacher_run)/'identity.json').read_text())
        if teacher_id['config']['limit']!=0:raise ValueError('Small-fit teacher cannot label the full student dataset')
        data=TeacherDataset(a.data,a.split)
        if data.identity['identity']!=teacher_id['data_identity']:raise ValueError('Teacher training data definition changed')
        checkpoint=Path(a.teacher_run)/a.checkpoint
        state=torch.load(checkpoint,map_location='cpu',weights_only=False)
        if state['identity']!=teacher_id['identity']:raise ValueError('Wrong teacher checkpoint')
        identity={'schema':'foresight_interaction_latent_v1','source_sha':source,'teacher_identity':teacher_id['identity'],
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
            if time.time()-record['start_unix']>a.max_seconds:raise TimeoutError('Export allocation expired; completed shards preserved')
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
                    tmp=path.with_suffix('.tmp');torch.save(payload,tmp);tmp.replace(path)
            record['inference_scenes']=start+len(indices);save()
        atomic_json(out/'COMPLETE.json',{'scenes':len(data),'identity':identity['identity'],'failed':0})


if __name__=='__main__':main()

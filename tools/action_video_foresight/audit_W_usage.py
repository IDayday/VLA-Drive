"""Frozen planner intervention at fixed one/three and two/three decoder layers."""
import argparse,hashlib,json,subprocess
from pathlib import Path
import numpy as np
import torch
from tools.foresight.checkpoints import checkpoint_identity,load_student,scene_noise
from starVLA.dataloader.foresight_dataset import ForesightCurrentDataset,decode_ego
from starVLA.model.modules.foresight.world_intervention import permute_world_at_layer
from tools.ddpolicy_vehicle.prepare_data import atomic_json
from tools.ddpolicy_vehicle.run_meter import metered_run


def fixed_queries(index,per_log):
    groups={}
    for i,r in enumerate(index):groups.setdefault(r['log'],[]).append(i)
    return sorted(i for rows in groups.values() for i in sorted(rows,key=lambda i:hashlib.sha256(('planner-W-intervention-v1:'+index[i]['token']).encode()).hexdigest())[:per_log])


def main():
    p=argparse.ArgumentParser(__doc__)
    for k in ('training-run','checkpoint-tag','data','output','campaign-root','run-id'):p.add_argument('--'+k,required=True)
    p.add_argument('--per-log',type=int,default=8);a=p.parse_args()
    if subprocess.check_output(['git','status','--porcelain']).strip():raise ValueError('Freeze read-only diagnostic')
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=torch.backends.cudnn.allow_tf32=False
    ds=ForesightCurrentDataset(a.data)
    if ds.identity['split']!='dev':raise ValueError('Fixed dev diagnostics only')
    chosen=fixed_queries(ds.index,a.per_log)
    if len(chosen)%2:raise ValueError('Paired scene batches required; fix query manifest before evaluation')
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    atomic_json(out/'queries.json',[ds.index[i] for i in chosen])
    with metered_run(a.campaign_root,a.run_id,1,{'kind':'intermediate_W_planner_intervention','real_optimizer_updates':0}) as (meter,_,save):
        training,cp=checkpoint_identity(a.training_run,a.checkpoint_tag);model=load_student(a.training_run,a.checkpoint_tag,training)
        depth=len(model.qwen_vl_interface.model.model.language_model.layers);layers=[depth//3,2*depth//3]
        rows=[]
        for start in range(0,len(chosen),2):
            ids=chosen[start:start+2];observations=[ds[i] for i in ids]
            noise=torch.cat([scene_noise(ds.index[i]['token'],42,'cuda') for i in ids])
            with torch.inference_mode():base=model.predict_action(observations,initial_noise=noise)
            outputs={'base':base};records={}
            for layer in layers:
                with permute_world_at_layer(model,layer,[1,0]) as record,torch.inference_mode():output=model.predict_action(observations,initial_noise=noise)
                outputs['W_permuted_layer'+str(layer)]=output;records[str(layer)]=record
            for j,i in enumerate(ids):
                row={**ds.index[i],'swap_token':ds.index[ids[1-j]]['token'],'failure':None,'interventions':records}
                gt=decode_ego(torch.load(Path(a.data)/'ego'/(row['token']+'.pt'),weights_only=True)['ego']).cuda()
                for name,pred in outputs.items():
                    if not torch.isfinite(pred[j]).all():raise FloatingPointError('Invalid intervention ego')
                    xy=(pred[j,:,:2]-gt[:,:2]).norm(dim=-1);change=(pred[j,:,:2]-base[j,:,:2]).norm(dim=-1)
                    row[name]={'ADE':float(xy.mean()),'FDE':float(xy[-1]),'mean_xy_change_from_base':float(change.mean())}
                    folder=out/name;folder.mkdir(exist_ok=True);np.savez_compressed(folder/(row['token']+'.npz'),trajectory=pred[j].cpu().numpy())
                rows.append(row)
            meter['inference_scenes']=len(rows)*(1+len(layers));save()
        with (out/'queries.jsonl').open('w') as f:
            for r in rows:f.write(json.dumps(r)+'\n')
        summary={'checkpoint':cp,'scenes':len(rows),'layers':layers,'failures':0,'precision':'FP32 masters/compute TF32off',
            'scope':'mid-layer output scene permutation can be distributional; sensitivity is not causal evidence',
            'image_path_changed':False,'sequence_length_changed':False,'position_encoding_changed':False,'real_optimizer_updates':0}
        for name in outputs:summary[name]={k:sum(r[name][k] for r in rows)/len(rows) for k in ('ADE','FDE','mean_xy_change_from_base')}
        atomic_json(out/'SUMMARY.json',summary)

if __name__=='__main__':main()

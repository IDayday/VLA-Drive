"""GT-conditioned TRAIN64 mechanism diagnostic, never a deployment/PDMS score.

Current cameras alone determine queries and graph. Afterwards current GT loss
assignment attaches same-track GT future to existing neighbor slots. Ego stays
hidden. Each scene compares identical ego/noise/graph with and without that GT.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import time
from .prepare_data import atomic_json, load_trusted
from .run_meter import metered_run


def main():
    parser=argparse.ArgumentParser(__doc__)
    for name in ('training-run','checkpoint-tag','current-root','vehicle-root','processed-root',
                 'reference-predictions','output','campaign-root','run-id','local-checkpoint-cache'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--sampling-seed',type=int,default=42)
    args=parser.parse_args()
    with metered_run(args.campaign_root,args.run_id,1,{'kind':'train64_gt_completion_diagnostic',
            'real_optimizer_updates':0}) as (record,_,save):
        import numpy as np
        import torch
        from dataclasses import fields
        from types import SimpleNamespace
        from omegaconf import OmegaConf
        from starVLA.dataloader.ddpolicy_current import CurrentCameraDataset
        from starVLA.model.framework.DDPVehicle import DDPVehicle
        from starVLA.model.modules.structured_world.contracts import WorldTargets
        from starVLA.model.modules.structured_world.matching import match_current
        from starVLA.model.modules.vehicle_joint.action_head import modeled_mask
        from starVLA.model.modules.vehicle_joint.initialization import file_sha256
        from starVLA.dataloader.navsim_dataset import x_mean,x_std,y_mean,y_std
        from .checkpoints import checkpoint_identity,stage_checkpoint,scene_noise
        from .evaluate_ego import relative_ego_target,trajectory_errors
        from .paired_results import log_bootstrap
        if subprocess.check_output(['git','status','--porcelain']):raise ValueError('Commit diagnostic source first')
        source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
        dataset=CurrentCameraDataset(args.current_root)
        training,checkpoint=checkpoint_identity(args.training_run,args.checkpoint_tag)
        if not training.get('small_fit') or checkpoint['arm'] not in ('B','C') or len(dataset)!=64:
            raise ValueError('This diagnostic is restricted to the existing64-scene B/C small fits')
        if dataset.metadata.get('split')=='navtest':raise ValueError('Navtest not allowed')
        reference=Path(args.reference_predictions)
        reference_identity=json.loads((reference/'identity.json').read_text())
        if (reference_identity['checkpoint']['sha256']!=checkpoint['sha256'] or
                reference_identity['current_identity']!=dataset.identity or
                reference_identity['protocol']['sampling_seed']!=args.sampling_seed):
            raise ValueError('Reference checkpoint/current inputs/noise differ')
        target_root=Path(args.vehicle_root)
        target_identity=json.loads((target_root/'identity.json').read_text())['identity']
        if target_identity!=training['vehicle_identity']['identity']:raise ValueError('Training label identity changed')
        out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
        queries=[{**row,'target_actor':'ego_slot0','target_role':'ego'} for row in dataset.index]
        # All64 ego queries are fixed before loading the model or seeing errors.
        atomic_json(out/'queries.json',queries)
        identity={'source_sha':source,'checkpoint':checkpoint,'label_identity':target_identity,
            'current_identity':dataset.identity,'sampling_seed':args.sampling_seed,
            'query_manifest_sha256':file_sha256(out/'queries.json'),
            'scope':'TRAIN64 GT-future conditioned mechanism diagnostic; NOT camera-only deployment or PDMS',
            'graph':'unchanged camera-predicted graph within each pair; model graphs may differ',
            'assignment':'same current Hungarian loss assignment, no2m/class gate; no future-based graph selection',
            'reference_parity_tolerance':{'atol':3e-5,'rtol':1e-5},'known_coordinate_tolerance':0.}
        atomic_json(out/'identity.json',identity)
        cfg=OmegaConf.create(training['config']);cfg.framework.qwenvl.device_map='cpu'
        sources=json.loads(Path(cfg.from_scratch.source_manifest).read_text());os.environ['DEPTH_MODEL_CKPTS']=sources['ppd']['root']
        checkpoint_root=stage_checkpoint(args.training_run,args.checkpoint_tag,checkpoint,args.local_checkpoint_cache)
        model=DDPVehicle(cfg,accelerator=SimpleNamespace(process_index=0,device=torch.device('cpu')))
        from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
        state=get_fp32_state_dict_from_zero_checkpoint(str(checkpoint_root),tag=args.checkpoint_tag)
        model.float();model.load_state_dict(state,strict=True);del state
        model.to('cuda').eval();model.inference_fp32=True
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.backends.cudnn.benchmark=False
        rows=[]
        def decode(joint):
            ego=model.action_model.executed_ego(joint).float()
            xy=ego[...,:2]*ego.new_tensor([x_std,y_std])+ego.new_tensor([x_mean,y_mean])
            return torch.cat((xy,torch.atan2(ego[...,2],ego[...,3])[...,None]),-1)[0].cpu().numpy()
        for index,query in enumerate(queries):
            row={**query,'failure':None}
            try:
                if time.time()-record['start_unix']>1800:raise TimeoutError('Bounded diagnostic allocation exhausted')
                example=dataset[index]
                with torch.inference_mode():
                    # No labels have been loaded before either current function.
                    encoded=model.encode_current([example])
                    boxes,actor_queries,active,slot_queries,audits=model.current_graph(encoded,[example])
                    noise=scene_noise(query['token'],args.sampling_seed,9,'cuda')
                    baseline=model.action_model.sample(encoded['action'],actor_queries,boxes,active,noise)
                    base_ego=decode(baseline)
                    with np.load(reference/'predictions'/(query['token']+'.npz')) as previous:
                        parity=float(np.abs(base_ego-previous['trajectory']).max())
                        if not np.array_equal(slot_queries[0].cpu().numpy(),previous['selected_query_indices']):
                            raise ValueError('Camera graph changed from reference')
                        if not np.allclose(base_ego,previous['trajectory'],atol=3e-5,rtol=1e-5):
                            raise ValueError('All-hidden reference mismatch')
                    with torch.serialization.safe_globals([np.core.multiarray.scalar,np.dtype,type(np.dtype('U16')),np.str_]):
                        payload=torch.load(target_root/'targets'/(query['token']+'.pt'),weights_only=True,map_location='cpu')
                    if payload['identity']!=target_identity:raise ValueError('Changed target identity')
                    target=WorldTargets(**payload['targets'])
                    target=WorldTargets(**{f.name:getattr(target,f.name).to('cuda') if isinstance(getattr(target,f.name),torch.Tensor)
                        else getattr(target,f.name) for f in fields(target)})
                    assigned_queries,assigned_gt=match_current({k:v[0].float() for k,v in encoded['vehicle_prediction'].items()},target)
                    assignment=dict(zip(assigned_queries.tolist(),assigned_gt.tolist()))
                    known=torch.zeros_like(noise,dtype=torch.bool);values=torch.zeros_like(noise)
                    center_errors=[];known_actors=[]
                    for actor in torch.where(active[0,1:])[0].tolist():
                        actor+=1;slot=int(slot_queries[0,actor])
                        if slot not in assignment:continue
                        gt=assignment[slot];valid=target.future_valid_mask[gt]
                        known[0,actor,:,:2]=valid[:,None]
                        values[0,actor,:,:2]=(target.future_xy_in_ego_t0[gt]-boxes[0,actor,:2])/20.
                        if valid.any():
                            known_actors.append({'actor':actor,'query':slot,'gt_index':gt,
                                'track_id':str(target.track_ids[gt]),'points':int(valid.sum())})
                            center_errors.append(float((boxes[0,actor,:2]-target.current_boxes[gt,:2]).norm()))
                    assert not known[:,0].any() and not (known & ~modeled_mask(active)).any()
                    # No target ego future ever enters values. Hidden labels and
                    # unsupported coordinates are deliberately NaN-poisoned.
                    poisoned=torch.where(known,values,torch.full_like(values,float('nan')))
                    conditional,history=model.action_model.sample(encoded['action'],actor_queries,boxes,active,noise,
                        known_values=poisoned,known_mask=known,return_history=True)
                    assert all(torch.equal(h[known],values[known]) for h in history)
                    assert all(torch.count_nonzero(h[~modeled_mask(active)])==0 for h in history)
                    if not known.any() and not torch.equal(baseline,conditional):raise ValueError('No-condition pair differs')
                    cond_ego=decode(conditional)
                    # Metric-only ego labels are first opened AFTER generation.
                    raw=load_trusted(Path(args.processed_root)/(query['token']+'.pkl'))
                    ego_target=relative_ego_target(np.asarray(raw['glo_status']['global_poses'])[:12])
                    base_metrics=trajectory_errors(base_ego,ego_target);cond_metrics=trajectory_errors(cond_ego,ego_target)
                    row.update(known_other_xy_points=int(known[0,1:,:,:2].all(-1).sum()),
                        known_other_actors=len(known_actors),active_neighbors=int(active[0,1:].sum()),
                        context_vehicles=audits[0]['context_vehicles'],reference_max_abs_difference=parity,
                        assigned_current_center_error_mean=float(np.mean(center_errors)) if center_errors else None,
                        known_clamped_every_step=True,unmodeled_channels_zero=True)
                    for key in base_metrics:row.update({f'all_hidden_{key}':base_metrics[key],f'conditional_{key}':cond_metrics[key],
                        f'conditional_minus_all_hidden_{key}':cond_metrics[key]-base_metrics[key]})
                    atomic_json(out/(query['token']+'_assignment.json'),{'token':query['token'],'known_actors':known_actors,'graph':audits[0]})
            except Exception as error:row['failure']=repr(error)
            rows.append(row);atomic_json(out/'partial_rows.json',rows)
            record.update(inference_scenes=len(rows),failed=sum(x['failure'] is not None for x in rows));save()
        with (out/'queries.csv').open('w') as stream:
            writer=csv.DictWriter(stream,sorted(set().union(*(r.keys() for r in rows))));writer.writeheader();writer.writerows(rows)
        summary={'scenes':len(rows),'failed':sum(r['failure'] is not None for r in rows),'groups':{},'scope':identity['scope']}
        if not summary['failed']:
            for group,values in [('all',rows),('with_other_gt',[r for r in rows if r['known_other_xy_points']>0]),
                                 ('without_other_gt',[r for r in rows if r['known_other_xy_points']==0])]:
                result={'scenes':len(values)}
                for key in ('ADE','FDE','yaw_MAE_rad'):
                    result[key]={'all_hidden':float(np.mean([r['all_hidden_'+key] for r in values])) if values else None,
                        'conditional':float(np.mean([r['conditional_'+key] for r in values])) if values else None,
                        'conditional_minus_all_hidden':log_bootstrap([r['conditional_minus_all_hidden_'+key] for r in values],
                            [r['log'] for r in values]) if values else None}
                summary['groups'][group]=result
        atomic_json(out/'summary.json',summary)
        if summary['failed']:raise RuntimeError('Failed diagnostic rows retained; no valid full-population result')


if __name__=='__main__':main()

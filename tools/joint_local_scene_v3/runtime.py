"""Fixed same-target completion queries, separate from all-hidden scene results."""
import hashlib
import json
import math
from pathlib import Path
import torch
from starVLA.model.modules.joint_scene.contracts import stack_graphs
from starVLA.model.modules.joint_scene.flow import execution_from_joint
from starVLA.model.modules.joint_scene.masks import xy_point_valid


def batch_scenes(scenes,device='cuda'):
    return stack_graphs([s.graph for s in scenes],device),torch.cat([s.future for s in scenes]).to(device),torch.cat([s.feature_valid for s in scenes]).to(device)


def evaluation_noise(scenes,seed,device='cuda'):
    rows=[]
    for s in scenes:
        h=int.from_bytes(hashlib.sha256(f'{seed}:{s.token}'.encode()).digest()[:8],'little')%(2**63-1)
        rows.append(torch.randn(s.future.shape,generator=torch.Generator().manual_seed(h)))
    return torch.cat(rows).to(device)


def build_queries(corpus):
    """Enumerate all labeled local actors BEFORE model access; future masks are diagnostic-only."""
    rows=[];population={'scenes':len(corpus),'source_current_objects':0,'selected_neighbors':0,'neighbors_with_xy_labels':0,'source_population_available':True,
        'raw_log_population_available':True,'source_population_filtered':False}
    identities=[]
    for i in range(len(corpus)):
        s=corpus[i];s.validate();g=s.graph;v=xy_point_valid(s.feature_valid)[0]
        source=s.metadata.get('source_current_objects')
        population['raw_log_population_available'] &= s.metadata.get('raw_log_population_available',False)
        population['source_population_filtered'] |= s.metadata.get('source_population_filtered',False)
        if source is None:population['source_population_available']=False
        else:population['source_current_objects']+=source
        population['selected_neighbors']+=int(g.active_actor_mask[:,1:].sum());population['neighbors_with_xy_labels']+=int(v[1:].any(-1).sum())
        identities.append({'token':s.token,'source_indices':g.source_indices.tolist(),'modeled':g.modeled_state_mask.tolist(),'graph_schema':g.schema_version,'selection':g.selection_metadata})
        for slot in torch.where(g.active_actor_mask[0]&v.any(-1))[0].tolist():
            valid_steps=torch.where(v[slot])[0].tolist();source_index=int(g.source_indices[0,slot])
            rows.append({'query_id':f'{s.token}:{source_index}','scene_index':i,'token':s.token,'log':s.log,'slot':slot,'source_index':source_index,
                'target_class':int(g.classes[0,slot]),'current_distance_m':float(g.boxes[0,slot,:2].norm()),'valid_timesteps':valid_steps,
                'valid_xy_points':len(valid_steps),'horizon_steps':v.shape[-1],'complete_horizon':bool(v[slot].all()),
                'final_time_valid':bool(v[slot,-1]),'context_objects':int(g.context_mask.sum()),
                'known_other_xy_points':int(v.sum()-v[slot].sum()),'graph_origin':g.origin})
    if not rows:raise ValueError('No valid evaluation queries')
    payload={'schema_version':1,'query_rule':'enumerate_every_active_actor_with_at_least_one_complete_xy_label','queries':rows,'population':population,
        'corpus_identity':getattr(corpus,'manifest',{}).get('identity_sha256'),'graph_identity_sha256':hashlib.sha256(json.dumps(identities,sort_keys=True).encode()).hexdigest(),
        'scope':'GT-future-conditioned completion diagnostic; not deployment planning'}
    payload['sha256']=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()
    return payload


def validate_queries(corpus,manifest):
    rebuilt=build_queries(corpus)
    if manifest!=rebuilt:raise ValueError('Query list, graph identity, corpus, or valid-point population changed')
    return manifest


def target_metrics(pred,truth,valid,current_box,ego_velocity=None,dt=.5):
    points=valid[...,:2].all(-1)
    if not points.any():raise ValueError('No complete xy target points')
    if not torch.isfinite(pred).all() or not torch.isfinite(truth[valid]).all():raise ValueError('Nonfinite prediction/valid target')
    safe=torch.where(valid,truth,0.)
    error=(pred[...,:2]-safe[...,:2]).norm(dim=-1)
    stationary=(current_box[:2]-safe[...,:2]).norm(dim=-1)
    moving=bool((stationary[points]>=1.).any())
    result={'xy_ADE_m':float(error[points].mean()),'xy_error_sum_m':float(error[points].sum()),'valid_xy_points':int(points.sum()),
        'xy_FDE_m':float(error[-1]) if points[-1] else None,'final_time_valid':bool(points[-1]),
        'stationary_ADE_m':float(stationary[points].mean()),'stationary_FDE_m':float(stationary[-1]) if points[-1] else None,
        'dynamic':moving,'yaw_error_rad':None,'valid_yaw_points':0,'CV_ADE_m':None,'CV_available':ego_velocity is not None}
    yaw_valid=valid[...,2:].all(-1)
    if yaw_valid.any():
        if (pred[...,2:][yaw_valid].norm(dim=-1)<1e-8).any():raise ValueError('Degenerate modeled yaw prediction')
        py=torch.atan2(pred[...,2],pred[...,3]);ty=torch.atan2(safe[...,2],safe[...,3])
        angle=torch.atan2(torch.sin(py-ty),torch.cos(py-ty)).abs()
        result['yaw_error_rad']=float(angle[yaw_valid].mean());result['valid_yaw_points']=int(yaw_valid.sum())
    if ego_velocity is not None:
        time=torch.arange(1,len(truth)+1,device=pred.device)*dt
        cv=current_box[:2]+time[:,None]*ego_velocity
        result['CV_ADE_m']=float((cv-safe[...,:2]).norm(dim=-1)[points].mean())
    return result


def summarize(scene_rows,query_rows,manifest):
    failed=sum(r['status']!='ok' for r in query_rows);scene_failed=sum(r['status']!='ok' for r in scene_rows)
    summary={'scope':'structured-current-state checks; conditional rows use privileged future context','population':manifest['population'],
        'evaluated_queries':len(query_rows),'expected_queries':len(manifest['queries']),'failed_queries':failed,'failed_scenes':scene_failed,
        'aggregate_valid':not failed and not scene_failed and len(query_rows)==len(manifest['queries']),
        'query_manifest_sha256':manifest['sha256'],'groups':{}}
    for actor in ('ego','neighbor'):
        for motion in ('all','dynamic','static'):
            rows=[r for r in query_rows if (r['slot']==0)==(actor=='ego') and (motion=='all' or r.get('dynamic')==(motion=='dynamic'))]
            key=actor+'_'+motion;points=sum(r['valid_xy_points'] for r in rows)
            total_points=sum(r['horizon_steps'] for r in rows)
            row={'queries':len(rows),'complete_targets':sum(r['complete_horizon'] for r in rows),'partial_targets':sum(not r['complete_horizon'] for r in rows),
                'valid_xy_points':points,'expected_xy_points':total_points,'xy_label_coverage':points/total_points if total_points else None,
                'all_hidden_ADE_m':None,'conditional_ADE_m':None,'conditional_minus_all_hidden_m':None}
            if summary['aggregate_valid'] and points:
                row['all_hidden_ADE_m']=sum(r['all_hidden_xy_error_sum_m'] for r in rows)/points
                row['conditional_ADE_m']=sum(r['conditional_xy_error_sum_m'] for r in rows)/points
                row['conditional_minus_all_hidden_m']=row['conditional_ADE_m']-row['all_hidden_ADE_m']
            summary['groups'][key]=row
    return summary


@torch.no_grad()
def evaluate(model,corpus,query_manifest=None,seed=20260927,sampling_steps=20,device='cuda',output=None):
    # All identities/queries fixed before any model forward. This is not future-based graph selection.
    manifest=build_queries(corpus) if query_manifest is None else validate_queries(corpus,query_manifest)
    if output is not None:
        output=Path(output);output.mkdir(parents=True,exist_ok=True)
        frozen=output/'queries.json'
        if frozen.exists() and json.loads(frozen.read_text())!=manifest:raise ValueError('Refuse overwriting different queries')
        if not frozen.exists():frozen.write_text(json.dumps(manifest,indent=2)+'\n')
    prior=model.training;model.eval();scene_rows=[];query_rows=[]
    try:
        for i in range(len(corpus)):
            s=corpus[i];g,y,v=batch_scenes([s],device);noise=evaluation_noise([s],seed,device)
            queries=[q for q in manifest['queries'] if q['scene_index']==i]
            scene_row={'token':s.token,'log':s.log,'status':'ok','scope':'all-hidden structured GT current state; not camera planning',
                'source_current_objects':s.metadata.get('source_current_objects'),'selected_neighbors':int(g.active_actor_mask[:,1:].sum()),'queries':len(queries)}
            all_hidden=None
            try:
                all_hidden=model.sample(noise,g,sampling_steps=sampling_steps)
                if not torch.isfinite(all_hidden).all():raise ValueError('Nonfinite all-hidden joint sample')
                execution=execution_from_joint(all_hidden)
                scene_row['execution_ego_xy_difference_m']=float((execution['executed_ego_xyyaw'][...,:2]-all_hidden[:,0,:,:2]).abs().max())
                distance=(all_hidden[0,1:,:,:2]-all_hidden[0,0,None,:,:2]).norm(dim=-1)
                active=g.active_actor_mask[0,1:]
                scene_row['same_joint_sample_min_distance_m']=float(distance[active].min()) if active.any() else None
            except (ValueError,RuntimeError) as exc:
                scene_row.update(status='failed',error=str(exc));all_hidden=None
            scene_rows.append(scene_row)
            for query in queries:
                row=dict(query);row['status']='ok';slot=query['slot']
                try:
                    if all_hidden is None:raise ValueError('Scene all-hidden generation failed')
                    known=v.clone();known[:,slot]=False
                    conditional=model.sample_conditional(noise,g,known=y,known_mask=known,sampling_steps=sampling_steps)
                    if not torch.isfinite(conditional).all():raise ValueError('Nonfinite conditional joint sample')
                    kwargs={'current_box':g.boxes[0,slot],'ego_velocity':g.ego_state[0,1:3] if slot==0 else None,
                        'dt':getattr(corpus,'manifest',{}).get('time_step_s',.5)}
                    first=target_metrics(all_hidden[0,slot],y[0,slot],v[0,slot],**kwargs)
                    second=target_metrics(conditional[0,slot],y[0,slot],v[0,slot],**kwargs)
                    if first['valid_xy_points']!=query['valid_xy_points'] or second['valid_xy_points']!=first['valid_xy_points']:raise ValueError('Paired target denominator differs')
                    for prefix,metrics in [('all_hidden',first),('conditional',second)]:row.update({prefix+'_'+k:value for k,value in metrics.items()})
                    row['dynamic']=first['dynamic'];row['conditional_minus_all_hidden_ADE_m']=second['xy_ADE_m']-first['xy_ADE_m']
                except (ValueError,RuntimeError) as exc:row.update(status='failed',error=str(exc))
                query_rows.append(row)
    finally:model.train(prior)
    result={'scene_rows':scene_rows,'query_rows':query_rows,'query_manifest':manifest,'summary':summarize(scene_rows,query_rows,manifest)}
    # A separate target table prevents privileged-future diagnostics being mistaken
    # for deployment-format all-hidden scores. Query rows retain explicit paired deltas.
    all_hidden_rows=[]
    for row in query_rows:
        item={k:v for k,v in row.items() if not k.startswith(('conditional_','all_hidden_'))}
        item.update({k.removeprefix('all_hidden_'):v for k,v in row.items() if k.startswith('all_hidden_')})
        item['scope']='all-hidden structured GT current state; no other-actor future input; not camera planning'
        all_hidden_rows.append(item)
    result['all_hidden_target_rows']=all_hidden_rows
    if output is not None:
        import csv
        for name,rows in [('all_hidden_scenes',scene_rows),('all_hidden_targets',all_hidden_rows),('conditional_queries',query_rows)]:
            with (output/(name+'.csv')).open('w',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=sorted(set().union(*(r.keys() for r in rows))),lineterminator='\n');writer.writeheader();writer.writerows(rows)
        (output/'summary.json').write_text(json.dumps(result['summary'],indent=2,allow_nan=False)+'\n')
    return result

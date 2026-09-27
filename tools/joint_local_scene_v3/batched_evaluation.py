"""Batch independent fixed queries without changing current graphs, noise or target masks."""
import csv
import json
from pathlib import Path
import torch
from tools.joint_local_scene_v3.runtime import (build_queries,validate_queries,batch_scenes,evaluation_noise,target_metrics,summarize,EvaluationBudgetPause)
from starVLA.model.modules.joint_scene.flow import execution_from_joint


@torch.no_grad()
def evaluate_batched(model,corpus,query_manifest=None,seed=20260927,sampling_steps=20,device='cuda',output=None,budget_check=None,evaluation_identity=None,batch_size=16):
    if batch_size<1:raise ValueError('Positive evaluation batch required')
    manifest=build_queries(corpus) if query_manifest is None else validate_queries(corpus,query_manifest)
    output=Path(output) if output is not None else None
    if output is not None:
        output.mkdir(parents=True,exist_ok=True)
        for name,value in [('queries',manifest),('evaluation_identity',evaluation_identity)]:
            path=output/(name+'.json')
            if path.exists() and json.loads(path.read_text())!=value:raise ValueError('Existing evaluation identity differs')
            if not path.exists():path.write_text(json.dumps(value,indent=2)+'\n')
    def boundary():
        if budget_check is not None:
            try:budget_check()
            except RuntimeError as exc:raise EvaluationBudgetPause(str(exc)) from exc
    def record(kind,row):
        if output is not None:
            with (output/(kind+'_progress.jsonl')).open('a') as stream:stream.write(json.dumps(row,allow_nan=False)+'\n')
    prior=model.training;model.eval();scene_rows=[];query_rows=[];predictions={};scene_errors={};interrupted=None
    try:
        for start in range(0,len(corpus),batch_size):
            boundary();ids=list(range(start,min(start+batch_size,len(corpus))));scenes=[corpus[i] for i in ids];g,y,v=batch_scenes(scenes,device)
            try:joint=model.sample(evaluation_noise(scenes,seed,device),g,sampling_steps=sampling_steps);error=None
            except (ValueError,RuntimeError) as exc:joint=None;error=str(exc)
            for j,i in enumerate(ids):
                s=scenes[j];row={'token':s.token,'log':s.log,'status':'ok','scope':'all-hidden structured GT current state; not camera planning','source_current_objects':s.metadata.get('source_current_objects'),'selected_neighbors':int(g.active_actor_mask[j,1:].sum()),'queries':sum(q['scene_index']==i for q in manifest['queries'])}
                if joint is None or not torch.isfinite(joint[j]).all():
                    scene_errors[i]=error or 'Nonfinite all-hidden joint sample';row.update(status='failed',error=scene_errors[i])
                else:
                    predictions[i]=joint[j].cpu();executed=execution_from_joint(joint[j:j+1]);active=g.active_actor_mask[j,1:]
                    distance=(joint[j,1:,:,:2]-joint[j,0,None,:,:2]).norm(dim=-1)
                    row['execution_ego_xy_difference_m']=float((executed['executed_ego_xyyaw'][0,:,:2]-joint[j,0,:,:2]).abs().max())
                    row['same_joint_sample_min_distance_m']=float(distance[active].min()) if active.any() else None
                scene_rows.append(row);record('scene',row)
        queries=manifest['queries']
        for start in range(0,len(queries),batch_size):
            boundary();chunk=queries[start:start+batch_size];scenes=[corpus[q['scene_index']] for q in chunk];g,y,v=batch_scenes(scenes,device);known=v.clone()
            for j,q in enumerate(chunk):known[j,q['slot']]=False
            try:conditional=model.sample_conditional(evaluation_noise(scenes,seed,device),g,known=y,known_mask=known,sampling_steps=sampling_steps);error=None
            except (ValueError,RuntimeError) as exc:conditional=None;error=str(exc)
            for j,q in enumerate(chunk):
                row=dict(q,status='ok');slot=q['slot']
                try:
                    if q['scene_index'] in scene_errors:raise ValueError(scene_errors[q['scene_index']])
                    if conditional is None:raise ValueError(error)
                    if not torch.isfinite(conditional[j]).all():raise ValueError('Nonfinite conditional joint sample')
                    kwargs={'current_box':g.boxes[j,slot],'ego_velocity':g.ego_state[j,1:3] if slot==0 else None,'dt':getattr(corpus,'manifest',{}).get('time_step_s',.5)}
                    first=target_metrics(predictions[q['scene_index']][slot].to(device),y[j,slot],v[j,slot],**kwargs)
                    second=target_metrics(conditional[j,slot],y[j,slot],v[j,slot],**kwargs)
                    if first['valid_xy_points']!=q['valid_xy_points'] or second['valid_xy_points']!=first['valid_xy_points']:raise ValueError('Paired target denominator differs')
                    for prefix,metrics in [('all_hidden',first),('conditional',second)]:row.update({prefix+'_'+k:value for k,value in metrics.items()})
                    row['dynamic']=first['dynamic'];row['conditional_minus_all_hidden_ADE_m']=second['xy_ADE_m']-first['xy_ADE_m']
                except (RuntimeError,ValueError) as exc:row.update(status='failed',error=str(exc))
                query_rows.append(row);record('query',row)
    except EvaluationBudgetPause as exc:interrupted=str(exc)
    finally:model.train(prior)
    summary=summarize(scene_rows,query_rows,manifest);done={q['query_id'] for q in query_rows}
    summary.update(evaluation_identity=evaluation_identity,inference_batch_size=batch_size,interrupted=interrupted,missing_query_ids=[q['query_id'] for q in manifest['queries'] if q['query_id'] not in done])
    if interrupted:summary['aggregate_valid']=False
    all_hidden=[]
    for row in query_rows:
        item={k:v for k,v in row.items() if not k.startswith(('all_hidden_','conditional_'))}
        item.update({k.removeprefix('all_hidden_'):v for k,v in row.items() if k.startswith('all_hidden_')});item['scope']='all-hidden structured GT current state; no other-actor future input; not camera planning';all_hidden.append(item)
    if output is not None:
        for name,rows in [('all_hidden_scenes',scene_rows),('all_hidden_targets',all_hidden),('conditional_queries',query_rows)]:
            with (output/(name+'.csv')).open('w',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=sorted(set().union(*(r.keys() for r in rows))),lineterminator='\n');writer.writeheader();writer.writerows(rows)
        (output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    if interrupted:raise EvaluationBudgetPause(interrupted)
    return {'scene_rows':scene_rows,'query_rows':query_rows,'all_hidden_target_rows':all_hidden,'summary':summary,'query_manifest':manifest}

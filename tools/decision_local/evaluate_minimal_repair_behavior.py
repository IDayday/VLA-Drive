"""Post-export keep/repair behaviour with dense independent geometry checks.

Labels only enter this analysis, after the single executed trajectory is fixed.
All scenes, including unknown, failed and outside-scope inputs, remain present.
"""
import argparse,json,os,sys,time,multiprocessing,resource
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.decision_local.common import read,atomic_json,digest,identity,check_host
from tools.local_interaction_mask_v2.compare_pdms import read as read_scores,compare,METRICS
from starVLA.dataloader.minimal_repair_scene import load_annotated_scene
from starVLA.model.modules.decision_local.local_constraint_checker import LocalConstraintChecker,ConstraintConfig
from starVLA.model.modules.decision_local.minimal_repair_contract import CaseStatus
from starVLA.dataloader.foresight_dataset import decode_ego


def analyze(task):
    row,paths,anchor,config,original,final=task
    started=time.monotonic();cpu=time.process_time();token=row['token'];torch.set_num_threads(1)
    if anchor.get('action_codec'):
        from starVLA.dataloader.a_only_codec import install_legacy_codec
        codec=read(anchor['action_codec'])
        if codec['identity']!=anchor['action_codec_identity']:raise ValueError('Wrong analysis action codec')
        install_legacy_codec(codec)
    prediction=Path(paths['predictions'])/'predictions'/(token+'.npz')
    rec=read(Path(paths['predictions'])/'records'/(token+'.json'))
    if digest(prediction)!=rec['sha256']:raise ValueError('Prediction changed after canonical scoring')
    with np.load(prediction) as data:qin=data['q0'].copy();qf=data['q_final'].copy()
    labels=Path(paths['labels'])/'scenes'/(token+'.npz');lr=read(Path(paths['labels'])/'records'/(token+'.json'))
    if digest(labels)!=lr['sha256']:raise ValueError('Target labels changed')
    with np.load(labels) as data:
        i=rec['plan_index'];expected=data['plan_in_physical'][0,i]
        if not np.array_equal(expected,qin):raise ValueError('Analysis uses another actual input plan')
        gt=decode_ego(torch.from_numpy(data['ego'])).numpy();status=CaseStatus(int(data['case_status'][0,i])).name
        relation=data['relations'][0,i,:,0];valid=data['valid'][0,i,:,0];common=data['common_valid'][0,i,0]
    # Reuse legal history-derived state from the original current cache. It is
    # separate from original future GT and is never changed by plan disturbance.
    from safetensors import safe_open
    with safe_open(paths['feature_chunk'],framework='pt',device='cpu') as f:state=f.get_slice('state')[paths['feature_offset']]
    velocity=decode_ego(state[None])[0,:2].numpy()/.5
    scene=load_annotated_scene(anchor,'dev',row);checker=LocalConstraintChecker(scene,ConstraintConfig(**config))
    before=checker.check(qin,velocity);after=checker.check(qf,velocity,reference=qin)
    xy=np.linalg.norm(qf[:,:2]-qin[:,:2],axis=-1);heading=np.abs(np.arctan2(np.sin(qf[:,2]-qin[:,2]),np.cos(qf[:,2]-qin[:,2])))
    gt_in=np.linalg.norm(qin[:,:2]-gt[:,:2],axis=-1);gt_out=np.linalg.norm(qf[:,:2]-gt[:,:2],axis=-1)
    length=np.linalg.norm(qin[-1,:2]);direction=qin[-1,:2]/max(length,1e-6)
    current_occ=relation[0,:,2]>.5;future_occ=relation[2,:,2]>.5;pair=valid[0,:,2]&valid[2,:,2]
    import shapely
    from starVLA.model.modules.decision_local.local_constraint_checker import body_polygons
    footprint=body_polygons(qin,scene.body)
    near_static=any(obj['reliable'] and np.linalg.norm(obj['velocity'])<=.1 and
        (shapely.distance(footprint,shapely.polygons(obj['corners']))<=1.).any()
        for obj in (scene.frames[0] or {}).values())
    oscore=float(original.get('score') or 0);fscore=float(final.get('score') or 0)
    groups={'all':True,'KEEP':status=='KEEP','REPAIR':status=='REPAIR','unresolved':status not in ('KEEP','REPAIR'),
        'local_geometry_KEEP_non_GT_ADE_ge_0p5':bool(status=='KEEP' and gt_in.mean()>=.5),
        'original_high_ge_0p9':oscore>=.9,'original_low_le_0p5':oscore<=.5,
        'dynamic_enter':bool((pair&~current_occ&future_occ).any()),'space_release':bool((pair&current_occ&~future_occ).any()),
        'CV_future_clearance_difference_ge_0p5':bool((common[:,1]&(np.abs(relation[1,:,1]-relation[2,:,1])>=.5)).any()),
        'CV_future_overlap_difference_known':bool((common[:,2]&((relation[1,:,2]>.5)!=(relation[2,:,2]>.5))).any()),
        'road_boundary':bool((valid[0,:,0]&(relation[0,:,0]<=.5)).any()),
        'low_speed':bool(np.linalg.norm(velocity)<.5),'label_support_below_half':bool(valid[2,:,1:].mean()<.5),
        'static_annotated_obstacle_near_input':bool(near_static)}
    values={'token':token,'log':row['log'],'status':status,'groups':groups,'input_score':oscore,'final_score':fscore,
        'q0_ADE_m':float(gt_in.mean()),'q0_FDE_m':float(gt_in[-1]),'final_ADE_m':float(gt_out.mean()),'final_FDE_m':float(gt_out[-1]),
        'mean_change_m':float(xy.mean()),'max_change_m':float(xy.max()),'max_heading_change_rad':float(heading.max()),
        'endpoint_projected_change_m':float(np.dot(qf[-1,:2],direction)-length) if length>.5 else None,
        'KEEP_unnecessary_modification':bool(status=='KEEP' and (xy.max()>.25 or heading.max()>.1)),
        'REPAIR_success':bool(status=='REPAIR' and after['feasible']),
        'new_known_constraint_violation':bool(not before['known_violation'] and after['known_violation']),
        'new_out_of_range':bool(not before['out_of_range'] and after['out_of_range']),
        'new_stop':bool(length>=1 and np.linalg.norm(qf[-1,:2])<1),
        'input_constraint_check':before,'final_constraint_check':after,
        'future_relation_support':float(valid[2,:,1:].mean()),'common_q0_obstacle_support':float(common[:,1:].mean()),
        'CPU_process_seconds':time.process_time()-cpu,'worker_wall_seconds':time.monotonic()-started}
    return values


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('predictions','scores','labels','features','anchor','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--workers',type=int,default=8);a=p.parse_args();host_resource=check_host();started=time.monotonic();before_children=resource.getrusage(resource.RUSAGE_CHILDREN)
    pm=read(a.predictions/'identity.json');lm=read(a.labels/'identity.json');fm=read(a.features/'identity.json');sm=read(a.scores/'identity.json')
    if fm['split']!='dev' or pm['features']!=fm['identity'] or lm['feature_identity']!=fm['identity'] or sm['prediction_identity']!=pm['identity']:raise ValueError('Evaluation population/identity mismatch')
    original,final=[read_scores(a.scores/v/'scenes.csv') for v in ('q0','q_final')]
    index=pm['index']
    if set(original)!=set(final) or set(final)!={r['token'] for r in index}:raise ValueError('Full export population required')
    if not 1<=a.workers<=24:raise ValueError('Bounded workers required')
    declaration={'predictions':pm['identity'],'scores':sm['identity'],'labels':lm['identity'],
        'scope':'post-execution geometry diagnostics, distinct from canonical responsibility/safety metrics',
        'KEEP_group_scope':'accepted annotated local geometry and kinematics; not independent navigation/traffic-rule certification',
        'route_scope_implementation':'bounded relative position, heading and progress; no lane-graph route verification',
        'KEEP_modification_thresholds':{'max_xy_m':.25,'max_heading_rad':.1},
        'thresholds_origin':'fixed training-side physical scales; not fitted to development scores',
        'counterfactual_limit':'queries use the logged environment, no claimed response of other actors to changed ego actions'}
    ident=identity(declaration);a.output.mkdir(parents=True,exist_ok=True);(a.output/'records').mkdir(exist_ok=True)
    if (a.output/'identity.json').exists() and read(a.output/'identity.json')['identity']!=ident:raise ValueError('Preserve previous analysis')
    atomic_json(a.output/'identity.json',{'identity':ident,**declaration})
    locations={}
    for pth in sorted((a.features/'records').glob('*.json')):
        record=read(pth)
        for offset,t in enumerate(record['tokens']):locations[t]=(str(a.features/'chunks'/(pth.stem+'.safetensors')),offset)
    tasks=[];rows=[];anchor=read(a.anchor)
    for row in index:
        saved=a.output/'records'/(row['token']+'.json')
        if saved.exists():rows.append(read(saved));continue
        chunk,offset=locations[row['token']]
        tasks.append((row,{'predictions':str(a.predictions),'labels':str(a.labels),'feature_chunk':chunk,'feature_offset':offset},anchor,lm['constraints'],original[row['token']],final[row['token']]))
    with ProcessPoolExecutor(a.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        for row in pool.map(analyze,tasks,chunksize=4):
            atomic_json(a.output/'records'/(row['token']+'.json'),row);rows.append(row)
    paired,_=compare(final,original)
    transitions={name:sum((r['input_score']==0)==before and (r['final_score']==0)==after for r in rows)
        for name,before,after in (('zero_to_nonzero',True,False),('nonzero_to_zero',False,True),('both_zero',True,True),('both_nonzero',False,False))}
    grouped={}
    for group in rows[0]['groups']:
        subset=[r for r in rows if r['groups'][group]];tokens={r['token'] for r in subset}
        stats={'scenes':len(subset)}
        if subset:
            stats.update({key:float(np.mean([r[key] for r in subset])) for key in ('q0_ADE_m','final_ADE_m','final_FDE_m','mean_change_m','max_change_m','new_known_constraint_violation','new_stop')})
            stats['planning']=compare({t:final[t] for t in tokens},{t:original[t] for t in tokens})[0]
        grouped[group]=stats
    total={'identity':ident,'scenes':len(rows),'paired_planning':paired,'zero_transitions':transitions,'groups':grouped,
        'KEEP_unnecessary_modifications':sum(r['KEEP_unnecessary_modification'] for r in rows),
        'REPAIR_successes':sum(r['REPAIR_success'] for r in rows),'new_out_of_range':sum(r['new_out_of_range'] for r in rows),
        'CPU_process_seconds':sum(r['CPU_process_seconds'] for r in rows),'worker_wall_seconds':sum(r['worker_wall_seconds'] for r in rows),
        'elapsed_seconds':time.monotonic()-started,'resource_check':host_resource,
        'CPU_waited_children_seconds':sum((getattr(resource.getrusage(resource.RUSAGE_CHILDREN),key)-getattr(before_children,key)) for key in ('ru_utime','ru_stime')),
        'CPU_worker_seconds':sum(r['CPU_process_seconds'] for r in rows),
        'CPU_accounting':'worker CPU is contained within waited-child CPU; do not add them'}
    atomic_json(a.output/'COMPLETE.json',total);print(json.dumps({'scenes':len(rows),'zero_transitions':transitions,'PDMS_delta_pp':paired['delta_percentage_points']}))


if __name__=='__main__':main()

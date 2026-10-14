"""Fixed expert matrix, stage_val endpoint selection, and bounded extension."""
from dataclasses import asdict
from pathlib import Path
from itertools import combinations
import csv
import numpy as np
import torch
from iqe.contracts import require
from iqe.io import read_json,atomic_json,file_hash,digest
from iqe.evaluation.retention import module_hash
from .binding import scenes_by_role
from .data import SPECIALTIES,admissible_optimized_source,verified_base_scores
from .models import make_specialist


def endpoint(config,cell,step,kind):
    from .training import matrix_manifest
    _,spec=matrix_manifest(config,cell)
    path=config.root/'experts'/cell/f'{kind}_{step:06d}.pt'
    receipt=read_json(str(path)+'.COMPLETE.json')
    require(file_hash(path)==receipt['checksum'],'expert endpoint corruption')
    return {'cell':cell,'step':step,'endpoint':kind,'checkpoint':str(path),'checksum':receipt['checksum'],
            'model_hash':receipt['model_hash'],'private_reader':spec['private_reader'],'target':spec['target']}


def load_endpoint(config,item,device):
    require(file_hash(item['checkpoint'])==item['checksum'],'selected specialist checkpoint changed')
    module=make_specialist(config,item['private_reader']).to(device)
    module.load_state_dict(torch.load(item['checkpoint'],map_location=device,weights_only=True),strict=True)
    require(module_hash(module)==item['model_hash'],'selected specialist state identity')
    return module.eval().requires_grad_(False)


def rows_for_cell(config,cell):
    folder=config.root/'experts'/cell
    result=read_json(folder/'result.json')
    require(result['status']=='COMPLETE' and result['optimizer_steps']==5000,'expert fixed5000 endpoint not complete: '+cell)
    rows=[]
    for step in (1000,3000,5000):
        val=read_json(folder/f'validation_{step:06d}.json')
        for name in ('raw','ema'):
            row=val['endpoints'][name]
            rows.append(endpoint(config,cell,step,name)|{'validation':row,'ema_initial_base_weight':val['ema_initial_Base_weight'] if name=='ema' else 0.})
    return rows


def evaluate_experts(config,device='cpu'):
    matrix=read_json(config.root/'EXPERT_MATRIX_DATA.json');rows=[];blocked={}
    for cell,spec in matrix['cells'].items():
        if spec.get('status','READY')!='READY':blocked[cell]=spec['status'];continue
        rows+=rows_for_cell(config,cell)
    require(rows,'no trained expert endpoints')
    # Fixed endpoint/raw-EMA table is separate from selecting the system endpoint.
    best=max(rows,key=lambda r:(*r['validation']['selection_key'],-r['step'],r['endpoint']=='raw'))
    result={'schema':'stage_val_selected_expert_recipe_v1','pilot':matrix['pilot'],'selected':best,
            'all_endpoints':rows,'blocked_cells':blocked,'selected_on':'stage_val','dev_accessed':False,
            'matrix_manifest_hash':matrix['hash'],'scientific_status':'HELDOUT_ORACLE_GAIN'
            if best['validation']['oracle_gain_01']>1e-8 and best['validation']['new_oracle_scenes_gt_0_01']>0
            else 'NO_HELDOUT_ORACLE_GAIN'}
    result['hash']=digest(result);atomic_json(config.root/'EXPERT_RECIPE_SELECTION.json',result,immutable=True)
    with open(config.root/'EXPERT_2X2_RESULTS.csv','w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['cell','step','endpoint','target','private_reader','Base','Expert','Oracle','Oracle_gain','hard_val_gain','ordinary_gain','EMA_Base_weight'])
        writer.writeheader()
        for r in rows:
            v=r['validation'];writer.writerow({'cell':r['cell'],'step':r['step'],'endpoint':r['endpoint'],'target':r['target'],
                'private_reader':r['private_reader'],'Base':v['fixed_policy_01'][0],'Expert':v['fixed_policy_01'][1],
                'Oracle':v['oracle_01'],'Oracle_gain':v['oracle_gain_01'],'hard_val_gain':v['hard_val_direct_gain'],
                'ordinary_gain':v['ordinary_direct_gain'],'EMA_Base_weight':r['ema_initial_base_weight']})
    from .diagnostics import specialist_learning
    for cell,spec in matrix['cells'].items():
        if spec.get('status','READY')=='READY':specialist_learning(config,cell,device)
    return {'selected':best,'blocked':blocked,'scientific_status':result['scientific_status']}


def extension_manifest(config,specialty,recipe):
    roles=scenes_by_role(config);audit=read_json(config.root/'target_audit/TARGET_OPPORTUNITY_AUDIT.json')
    from .target_rules import core_audit
    core=core_audit(config)
    data={sid:read_json(config.root/'target_audit/scenes'/(sid+'.json')) | core['eligibility'][sid] for sid in audit['record_hashes']}
    de={s.scene_id for s in roles['expert_fit']};val={s.scene_id for s in roles['stage_val']}
    def eligible(row):
        if specialty not in row['specialties']:return False
        if recipe['target']=='gt':return row['gt_opportunity']
        return row['optimized_opportunity'] and admissible_optimized_source(row.get('optimized_provenance',{}).get('source',''))
    ids=sorted(i for i,r in data.items() if i in de and eligible(r));hv=sorted(i for i,r in data.items() if i in val and eligible(r))
    manifest={'schema':'specialty_extension_v1','specialty':specialty,'scene_ids':ids,'hard_val':hv,
              'cell':{'target':recipe['target'],'private_reader':recipe['private_reader']},'anchor_ratio':0,
              'recipe_selection_hash':file_hash(config.root/'EXPERT_RECIPE_SELECTION.json'),
              'record_hashes':{i:audit['record_hashes'][i] for i in ids+hv},'source':'full_eligible_D_E; extension is recipe validation, not paired target ablation'}
    manifest['hash']=digest(manifest)
    atomic_json(config.root/'extension_data'/('EXT_'+specialty+'.json'),manifest,immutable=True)
    return manifest


def extend(config,device='cuda:0'):
    from .training import train_expert
    from .recipes import load
    recipe=load(config,'specialist_extension');require(recipe['max_extension_runs']==2 and recipe['select_on']=='stage_val' and not recipe['force_three'],'bounded stage_val extension policy')
    selection=read_json(config.root/'EXPERT_RECIPE_SELECTION.json')
    if selection['scientific_status']=='NO_HELDOUT_ORACLE_GAIN':
        result={'status':'SKIPPED_NO_HELDOUT_ORACLE_GAIN','reason':'no automatic extra specialists without useful heldout candidate opportunity'}
        atomic_json(config.root/'SPECIALIST_EXTENSION.json',result,immutable=True);return result
    result={}
    for specialty in SPECIALTIES:
        if specialty==selection['pilot']:continue
        manifest=extension_manifest(config,specialty,selection['selected'])
        if not manifest['scene_ids']:
            result[specialty]={'status':'NO_ELIGIBLE_SPECIALTY'};continue
        cell='EXT_'+specialty;folder=config.root/'experts'/cell
        resume='auto' if list(folder.glob('step_*.pt.COMPLETE.json')) else None
        result[specialty]=train_expert(config,cell,device,resume)
    atomic_json(config.root/'SPECIALIST_EXTENSION.json',result,immutable=True);return result


def stage_val_columns(config,items,device):
    from .training import candidate_labels
    val=scenes_by_role(config)['stage_val'];base=verified_base_scores(config,val)
    names=('no_at_fault_collisions','drivable_area_compliance')
    values=[np.array([base[s.scene_id]['total_score_01'] for s in val])]
    safety=[np.array([[base[s.scene_id]['named_components'][c]
                       if base[s.scene_id]['component_valid_masks'].get(c,False) else np.nan
                       for c in names] for s in val])]
    for item in items:
        model=load_endpoint(config,item,device)
        _,labels=candidate_labels(config,model,val,item['cell']+'_'+item['endpoint'],device)
        values.append(np.array([x.total_score_01 for x in labels]));del model
        safety.append(np.array([[x.named_components[c] if x.component_valid_masks.get(c,False) else np.nan
                                 for c in names] for x in labels]))
    return val,np.stack(values,1),np.stack(safety,1)


def freeze_pool(config,device='cuda:0'):
    from .recipes import load
    policy=load(config,'specialist_extension')
    path=config.root/'SERVING_CANDIDATE_POOL.json'
    if path.exists():
        pool=read_json(path)
        require(pool['hash']==digest({k:v for k,v in pool.items() if k!='hash'}),'pool corruption')
        for p in pool['experts']:require(file_hash(p['checkpoint'])==p['checksum'],'pool endpoint changed')
        return pool
    selection=read_json(config.root/'EXPERT_RECIPE_SELECTION.json');items=[selection['selected']]
    for specialty in SPECIALTIES:
        cell='EXT_'+specialty
        if (config.root/'experts'/cell/'result.json').exists():
            choices=rows_for_cell(config,cell)
            items.append(max(choices,key=lambda r:(*r['validation']['selection_key'],-r['step'],r['endpoint']=='raw')))
    val,scores,safety=stage_val_columns(config,items,device)
    successful=(scores>=.8)&np.isfinite(safety).all(-1)&(safety>=1-1e-8).all(-1)
    subsets=[]
    for size in range(len(items)+1):
        for subset in combinations(range(1,len(items)+1),size):
            oracle=scores[:,[0,*subset]].max(1)
            subsets.append({'indices':[0,*subset],'cells':[items[i-1]['cell'] for i in subset],
                            'base_mean':float(scores[:,0].mean()),'oracle_mean':float(oracle.mean()),
                            'oracle_gain':float((oracle-scores[:,0]).mean()),
                            'safe_high_quality_scenes':int(successful[:,[0,*subset]].any(1).sum()),
                            'common_zero':int((scores[:,[0,*subset]]<=1e-8).all(1).sum())})
    # Greedy measurable marginal coverage; fully redundant specialists are omitted.
    remaining=list(range(1,len(items)+1));chosen=[];old=scores[:,0].copy()
    while remaining:
        index=max(remaining,key=lambda i:float((np.maximum(old,scores[:,i])-old).mean()))
        gain=float((np.maximum(old,scores[:,index])-old).mean())
        if gain<=policy['minimum_mean_oracle_gain'] or not (scores[:,index]-old>policy['minimum_scene_oracle_gain']).any():
            remaining.remove(index)
            continue
        chosen.append(index);remaining.remove(index);old=np.maximum(old,scores[:,index])
    research_only=not chosen
    # A failed-coverage research diagnostic retains the one actually trained pilot,
    # explicitly not an accepted useful deployment pool; no further expensive matrix.
    if research_only:chosen=[1]
    experts=[]
    for index in sorted(chosen):
        item=dict(items[index-1]);item.pop('all_endpoints',None);item['expert_id']=f'expert_{len(experts)+1}';experts.append(item)
    pool={'schema':'study_frozen_candidate_pool_v1','Base_binding_hash':file_hash(config.root/'BASE_BINDING.json'),
          'experts':experts,'expert_ids':['expert_0']+[p['expert_id'] for p in experts],
          'research_only_no_oracle_gain':research_only,'selection_role':'stage_val','selection_tolerance':policy['minimum_mean_oracle_gain'],'minimum_scene_oracle_gain':policy['minimum_scene_oracle_gain'],
          'stage_val_manifest_hash':digest([s.scene_id for s in val]),'subset_results':subsets,
          'expert_recipe_hash':file_hash(config.root/'EXPERT_RECIPE_SELECTION.json'),'ACTIVE_updated':False}
    pool['stage_val_expert_contribution']={items[i-1]['cell']:{
        'unique_safe_high_quality':int((successful[:,i]&~np.delete(successful,i,axis=1).any(1)).sum()),
        'leave_one_out_Oracle_gain':float((scores.max(1)-np.delete(scores,i,axis=1).max(1)).mean()),
        'scene_denominator':len(val)} for i in range(1,len(items)+1)}
    pool['hash']=digest(pool);atomic_json(path,pool,immutable=True)
    return pool


def run_study(config,device='cuda:0',resume=None):
    from .binding import bind
    from .historical import audit
    from .data import audit_opportunities,select_pilot,build_expert_data
    from .training import train_expert
    from .bank import build_deploy,build_rich
    from .scorer_training import train_scorer
    from .context import build
    from .evaluation import evaluate_scorers,calibrate,evaluate_final,benchmark
    bind(config);audit(config);audit_opportunities(config);select_pilot(config);build_expert_data(config)
    for cell,spec in read_json(config.root/'EXPERT_MATRIX_DATA.json')['cells'].items():
        if spec.get('status','READY')=='READY':
            folder=config.root/'experts'/cell
            train_expert(config,cell,device,'auto' if list(folder.glob('step_*.pt.COMPLETE.json')) else None)
    evaluate_experts(config,device);extend(config,device);pool=freeze_pool(config,device);build_deploy(config,device)
    if pool['research_only_no_oracle_gain']:
        return train_scorer(config,'S00_DIAGNOSTIC',device,None,300)
    build(config,device);build_rich(config,device)
    for cell in ('S00','S01','S10','S11'):
        folder=config.root/'scorers'/cell
        train_scorer(config,cell,device,'auto' if list(folder.glob('step_*.pt.COMPLETE.json')) else None)
    evaluate_scorers(config,device);calibrate(config,device);evaluate_final(config,device);return benchmark(config,device)

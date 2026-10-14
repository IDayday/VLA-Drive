"""Current-Base opportunities and paired original/optimized supervision.

Historical scores are never imported as current labels. Historical optimized
trajectories retain their generating-source disclosure and are rescored against
this campaign's official per-scene context.
"""
from collections import Counter, OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
from iqe.contracts import require, ScoreRecord, strict_record
from iqe.io import read_json, atomic_json, file_hash, digest
from iqe.internal_baseline.features import JointCache
from iqe.internal_baseline.config import Config
from iqe.internal_baseline.bank import save_candidate, score_candidates, protocol, COMPONENTS
from .binding import scenes_by_role

SPECIALTIES=('interaction','road_direction','safe_progress')
NC='no_at_fault_collisions'; DAC='drivable_area_compliance'; TTC='time_to_collision_within_bound'; DDC='driving_direction_compliance'


class ReusedContext:
    def __init__(self, config, capacity=256):
        self.config=config
        self.source_config=Config(**read_json(Path(config.reuse_root)/'resolved_config.json'))
        old=read_json(Path(config.reuse_root)/'BASE_BINDING.json'); current=read_json(config.root/'BASE_BINDING.json')
        require(all(old[k]==current[k] for k in ('checkpoint_hash','contract_hash','inference_runtime')), 'context Base/runtime mismatch')
        self.cache=JointCache(self.source_config,capacity=capacity)
    def read(self,scene): return self.cache.read(scene)
    def batch(self,scenes,device): return self.cache.batch(scenes,device)


def failure_specialties(base):
    if not base['score_valid']: return []
    c=base['named_components']; valid=base['component_valid_masks']
    if not all(valid.get(k,False) for k in (NC,DAC,TTC,DDC)): return []
    interaction=c[NC]<1-1e-8 or c[TTC]<1-1e-8
    road=c[DAC]<1-1e-8 or c[DDC]<1-1e-8
    out=[]
    if interaction: out.append('interaction')
    if road: out.append('road_direction')
    if not interaction and not road and base['total_score_01']<.9: out.append('safe_progress')
    return out


def admissible_optimized_source(source):
    # Explicit source whitelist; ambiguous ancestry and external action proposals
    # remain audit-only. No historical scoring network is loaded here.
    return source.startswith(('gt', 'parent_construction:gt', 'pdm_reference_blend',
                              'parent_construction:pdm_reference_blend', 'corridor_center', 'parent_construction:corridor_center', 'map_centerline', 'parent_construction:map_centerline'))


def safe_target(score, threshold):
    return (score['score_valid'] and score['total_score_01']>=threshold-1e-8 and
            all(score['component_valid_masks'].get(k,False) and score['named_components'][k]>=1-1e-8 for k in (NC,DAC,TTC,DDC,'comfort')))


def geometry(trajectory, dt, horizon=8):
    p=np.asarray(trajectory,dtype=float)
    if p.shape!=(horizon,3) or not np.isfinite(p).all(): return {'valid':False,'reason':'nonfinite_or_horizon'}
    # The original 8 positions are future samples at dt, not an extra t0 pose.
    if np.abs(p[:,2]).max()>np.pi+1e-5: return {'valid':False,'reason':'yaw_outside_radians'}
    xy=np.concatenate((np.zeros((1,2)),p[:,:2])); speed=np.linalg.norm(np.diff(xy,axis=0),axis=1)/dt
    valid=bool(speed.max()<70 and np.linalg.norm(np.diff(p[:,:2],axis=0),axis=1).max()<35)
    return {'valid':valid,'reason':None if valid else 'physical_jump','max_speed_mps':float(speed.max())}


def verified_base_scores(config, scenes):
    path=Path(config.reuse_root)/'BASE_FIT_VAL_SCORES.json'; labels=read_json(path)
    proto=protocol(config);binding=read_json(config.root/'BASE_BINDING.json')
    from iqe.contracts import CandidateRecord
    for s in scenes:
        r=strict_record(ScoreRecord,labels[s.scene_id])
        require(r.score_valid and r.metric_context_hash==s.metric_context_hash and r.score_protocol_hash==proto['protocol_hash'], 'stale Base official score')
        candidate_path=Path(config.reuse_root)/'candidates/expert_0'/binding['checkpoint_hash']/(s.scene_id+'.json')
        candidate=strict_record(CandidateRecord,read_json(candidate_path))
        require(candidate.expert_checkpoint_hash==binding['checkpoint_hash'] and candidate.feature_contract_hash==binding['contract_hash'] and r.candidate_key==candidate.key,'score belongs to another Base/trajectory')
    return {s.scene_id:labels[s.scene_id] for s in scenes}


def old_gt_labels(config):
    root=Path(config.reuse_root); result={}
    paths=[root/'EXPERT_MANIFEST.json']
    selected=root/'SELECTED_EXPERT_DATA.json'
    if selected.exists(): paths.append(Path(read_json(selected)['directory'])/'EXPERT_MANIFEST.json')
    for path in paths:
        if not path.exists():continue  # cold Base-only context source has no previous expert manifest
        for sid,row in read_json(path).get('gt_audit',{}).items():
            if 'GT' in row: result[sid]=row['GT']
    return result


def audit_opportunities(config, max_samples=None):
    roles=scenes_by_role(config)
    selected=roles['expert_fit']+roles['stage_val']
    base=verified_base_scores(config,selected)
    failures=[s for s in selected if failure_specialties(base[s.scene_id])]
    failures.sort(key=lambda s:digest([config.selection_seed,'target_audit',s.scene_id]))
    all_count=len(failures)
    if max_samples is not None:
        require(0<max_samples<=256,'pilot limited to256 scenes')
        failures=failures[:max_samples]
    out=config.root/'target_audit'; out.mkdir(parents=True,exist_ok=True)
    finished=out/('PILOT_RESULT.json' if max_samples else 'TARGET_OPPORTUNITY_AUDIT.json')
    if finished.exists():
        receipt=read_json(finished)
        require(set(receipt['record_hashes'])=={s.scene_id for s in failures},'target audit scene population changed')
        require(receipt['optimized_trajectory_hash']==file_hash(Path(config.optimized_root)/'optimized_trajectories.npz'),
                'optimized target asset changed since completed audit')
        for sid,checksum in receipt['record_hashes'].items():require(file_hash(out/'scenes'/(sid+'.json'))==checksum,'target audit changed')
        return receipt
    locked={'scene_ids':[s.scene_id for s in failures], 'complete_population':all_count, 'pilot':max_samples is not None,
            'base_binding_hash':file_hash(config.root/'BASE_BINDING.json'),'partition_hash':file_hash(config.root/'DATA_PARTITION.json')}
    atomic_json(out/('PILOT_INPUTS.json' if max_samples else 'INPUTS.json'),locked,immutable=True)
    root=Path(config.optimized_root)
    labels=pd.read_parquet(root/'optimized_labels.parquet').set_index('token')
    strict=pd.read_parquet(root/'strict_geometry.parquet').set_index('token')
    with np.load(root/'optimized_trajectories.npz',allow_pickle=False) as f:
        optimized={str(t):v for t,v in zip(f['tokens'],f['trajectories'])}
    optimized_hash=file_hash(root/'optimized_trajectories.npz')
    cache=ReusedContext(config); gt_prior=old_gt_labels(config); proto=protocol(config)
    trajectory_contract=read_json(config.root/'OFFLINE_BASE_CONTRACT.json')['trajectory']
    dt=trajectory_contract['dt'];horizon=trajectory_contract['horizon']
    pending=[]; records={}; all_start=time.perf_counter()
    def prepare(s):
        path=out/'scenes'/(s.scene_id+'.json')
        if path.exists():
            r=read_json(path)
            require(r['base_binding_hash']==locked['base_binding_hash'] and r['scene_context_hash']==s.metric_context_hash
                    and r['optimized_asset_hash']==optimized_hash,'stale target audit row')
            return s,r,False
        source_gt=Path(config.reuse_root)/'candidates/audit_gt'/s.target_id/(s.scene_id+'.json')
        if source_gt.exists():
            from iqe.contracts import CandidateRecord
            previous=strict_record(CandidateRecord,read_json(source_gt))
            require(previous.scene_id==s.scene_id and previous.expert_checkpoint_hash==s.target_id and previous.feature_contract_hash==read_json(config.root/'BASE_BINDING.json')['contract_hash'], 'GT identity changed')
            require(file_hash(previous.trajectory_ref)==previous.trajectory_hash,'GT candidate bytes changed')
            with np.load(previous.trajectory_ref) as f:g=f['trajectory'].copy()
        else:
            g=ReusedContext(config,capacity=0).read(s)['target_physical'].numpy()
        candidate=save_candidate(config,s,'audit_gt',s.target_id,g)
        old=gt_prior.get(s.scene_id)
        if old and old['candidate_key']==candidate.key and old['metric_context_hash']==s.metric_context_hash and old['score_protocol_hash']==proto['protocol_hash'] and old['score_valid']:
            gt=old
        else: gt=None
        opt=optimized.get(s.scene_id)
        r={'scene_id':s.scene_id,'source_log':s.source_log_id,'role':s.split_role,
           'base_binding_hash':locked['base_binding_hash'],'scene_context_hash':s.metric_context_hash,
           'optimized_asset_hash':optimized_hash,'base':base[s.scene_id], 'gt':gt,
           'gt_geometry':geometry(g,dt,horizon),'gt_candidate':asdict(candidate),
           'specialties':failure_specialties(base[s.scene_id]),'optimized':None}
        if opt is not None and s.scene_id in labels.index:
            info=labels.loc[s.scene_id]
            require(str(info['log_name'])==s.source_log_id,'optimized target source-log mismatch')
            r['optimized_provenance']={'kind':'offline_optimized','source':str(info['source']),
                  'snapshot_source':str(info['snapshot_source']),'run_identity':str(info['run_identity']),
                  'historical_context_hash':str(info['cache_sha256']),
                  'generation_scorer':'historical optimization provenance retained; current acceptance uses official environment scores only',
                  'external_source_may_have_seeded_search':not str(info['source']).startswith(('gt','quintic','geometry','local')),
                  'new_network_weights_loaded':False}
            r['optimized_geometry']=geometry(opt,dt,horizon)
            r['optimized_geometry']['historical_strict_pass']=bool(strict.loc[s.scene_id,'strict_geometry_pass'])
            opt_candidate=save_candidate(config,s,'audit_offline_optimized',optimized_hash,opt)
            r['optimized_candidate']=asdict(opt_candidate)
            r['target_max_abs_difference']=float(np.abs(g-opt).max())
            r['target_xy_ADE']=float(np.linalg.norm(g[:,:2]-opt[:,:2],axis=1).mean())
        return s,r,True
    with ThreadPoolExecutor(max_workers=4) as workers:
        for index,(s,r,is_pending) in enumerate(workers.map(prepare,failures)):
            records[s.scene_id]=r
            if is_pending:pending.append(s)
            if (index+1)%256==0:
                atomic_json(out/'PREPARE_STATUS.json',{'phase':'CANDIDATE_PREPARATION','prepared':index+1,'total':len(failures),'workers':4,'wall_seconds':time.perf_counter()-all_start})
    # Budgeted official work uses fresh per-candidate state and persistent retry receipts.
    for kind in ('gt','optimized'):
        subset=[s for s in pending if records[s.scene_id].get(kind) is None and kind+'_candidate' in records[s.scene_id]]
        if not subset: continue
        from iqe.contracts import CandidateRecord
        candidates=[strict_record(CandidateRecord,records[s.scene_id][kind+'_candidate']) for s in subset]
        values=score_candidates(config,subset,candidates,'study_target_'+kind)
        for s,v in zip(subset,values): records[s.scene_id][kind]=asdict(v)
    for s in pending:
        r=records[s.scene_id]; gt=r['gt']; opt=r.get('optimized')
        r['gt_eligible']=safe_target(gt,.9) and r['gt_geometry']['valid']
        r['paired_eligible']=bool(r['gt_eligible'] and opt and safe_target(opt,.95) and
             r['optimized_geometry']['valid'] and r['optimized_geometry']['historical_strict_pass'] and
             opt['total_score_01']>=gt['total_score_01']-1e-8 and opt['total_score_01']-r['base']['total_score_01']>=.02-1e-8)
        r['gt_opportunity']=bool(r['gt_eligible'] and gt['total_score_01']-r['base']['total_score_01']>=.02-1e-8)
        r['optimized_opportunity']=bool(opt and safe_target(opt,.95) and r['optimized_geometry']['valid'] and
                 r['optimized_geometry']['historical_strict_pass'] and opt['total_score_01']-r['base']['total_score_01']>=.02-1e-8)
        atomic_json(out/'scenes'/(s.scene_id+'.json'),r,immutable=True)
    for r in records.values():
        r['core_source_admissible']=admissible_optimized_source(r.get('optimized_provenance',{}).get('source',''))
        r['paired_eligible_for_core']=r['paired_eligible'] and r['core_source_admissible']
    train_ids={s.scene_id for s in roles['expert_fit']}
    summary={}
    for spec in SPECIALTIES:
        fit=[r for i,r in records.items() if i in train_ids and spec in r['specialties']]
        pairs=[r for r in fit if r['paired_eligible_for_core']]
        summary[spec]={'failures':len(fit),'logs':len({r['source_log'] for r in fit}),
             'gt_eligible':sum(r['gt_opportunity'] for r in fit),'paired':len(pairs),'paired_logs':len({r['source_log'] for r in pairs}),
             'optimized_available':sum(r.get('optimized') is not None for r in fit),
             'optimized_U':sum(max(0,r['optimized']['total_score_01']-r['base']['total_score_01']) for r in fit if r['optimized_opportunity'] and r['core_source_admissible'])/len(train_ids),
             'gt_U':sum(max(0,r['gt']['total_score_01']-r['base']['total_score_01']) for r in fit if r['gt_opportunity'])/len(train_ids),
             'pair_target_gain_ge_0_01':sum(r['optimized']['total_score_01']-r['gt']['total_score_01']>=.01 for r in pairs),
             'pair_nearly_identical':sum(r['target_max_abs_difference']<=1e-5 for r in pairs)}
    result={'status':'PILOT_COMPLETE' if max_samples else 'COMPLETE','requested':len(failures),'population':all_count,
            'summary':summary,'wall_seconds':time.perf_counter()-all_start,'DDC_PDMS_weight':0,
            'optimized_trajectory_hash':optimized_hash,'source_counts':dict(Counter(r.get('optimized_provenance',{}).get('source','absent') for r in records.values())),
            'record_hashes':{i:file_hash(out/'scenes'/(i+'.json')) for i in records}}
    atomic_json(out/('PILOT_RESULT.json' if max_samples else 'TARGET_OPPORTUNITY_AUDIT.json'),result,immutable=True)
    return result


def select_pilot(config):
    audit=read_json(config.root/'target_audit/TARGET_OPPORTUNITY_AUDIT.json')
    require(audit['status']=='COMPLETE','complete D_E opportunity audit required')
    from .target_rules import core_audit
    stats=core_audit(config)['summary']
    eligible=[s for s in SPECIALTIES if stats[s]['gt_eligible']]
    require(eligible,'NO_ELIGIBLE_SPECIALTY')
    # Prefer an identifiable paired target study, then aggregate supervised opportunity,
    # then cross-log support. This fixed rule never consumes heldout outcomes.
    pilot=max(eligible,key=lambda s:(stats[s]['pair_target_gain_ge_0_01']>0,stats[s]['optimized_U'],stats[s]['paired_logs'],stats[s]['gt_U'],s))
    status='PAIRED' if stats[pilot]['paired'] and stats[pilot]['pair_target_gain_ge_0_01'] else 'TARGET_EFFECT_NOT_IDENTIFIABLE'
    result={'specialty':pilot,'selection_rule':'paired_quality_support_then_U_then_logs_fixed_before_expert',
            'status':status,'uses_only_D_E':True,'audit_hash':file_hash(config.root/'target_audit/TARGET_OPPORTUNITY_AUDIT.json')}
    atomic_json(config.root/'PILOT_SELECTION.json',result,immutable=True)
    return result


def build_expert_data(config):
    pilot=read_json(config.root/'PILOT_SELECTION.json'); roles=scenes_by_role(config)
    audit=read_json(config.root/'target_audit/TARGET_OPPORTUNITY_AUDIT.json')
    from .target_rules import core_audit
    core=core_audit(config)
    rows={i:read_json(config.root/'target_audit/scenes'/(i+'.json')) | core['eligibility'][i] for i in audit['record_hashes']}
    de={s.scene_id for s in roles['expert_fit']}; val={s.scene_id for s in roles['stage_val']}
    paired=pilot['status']=='PAIRED'; spec=pilot['specialty']
    train=sorted([i for i,r in rows.items() if i in de and spec in r['specialties'] and (r['paired_eligible'] and admissible_optimized_source(r.get('optimized_provenance',{}).get('source','')) if paired else r['gt_opportunity'])])
    require(train,'NO_ELIGIBLE_PILOT_DATA')
    valid=sorted([i for i,r in rows.items() if i in val and spec in r['specialties'] and (r['paired_eligible'] and admissible_optimized_source(r.get('optimized_provenance',{}).get('source','')) if paired else r['gt_opportunity'])])
    from .recipes import load
    recipe=load(config,'pilot_expert_2x2');require(set(recipe['cells'])=={'E00','E10','E01','E11'} and recipe['anchor_ratio']==0,'fixed expert 2x2 recipe')
    out={'schema':'paired_expert_matrix_v1','pilot':spec,'paired_targets':paired,'scene_ids':train,'hard_val':valid,
         'core_rules_hash':core['rules_hash'],'anchor_ratio':0,'sample_order_seed':config.seed,'target_sources':['gt','offline_optimized'] if paired else ['gt'],
         'cells':{k:dict(v,status='READY' if paired or v['target']=='gt' else 'BLOCKED_TARGETS') for k,v in recipe['cells'].items()},
         'record_hashes':{i:audit['record_hashes'][i] for i in train+valid},'stage_val_scene_ids':[s.scene_id for s in roles['stage_val']],
         'paired_target_identifiable':sum(rows[i]['optimized']['total_score_01']-rows[i]['gt']['total_score_01']>=.01 for i in train) if paired else 0}
    out['hash']=digest(out); atomic_json(config.root/'EXPERT_MATRIX_DATA.json',out,immutable=True)
    return {k:v for k,v in out.items() if k not in ('scene_ids','record_hashes','stage_val_scene_ids','hard_val')}

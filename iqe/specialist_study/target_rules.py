"""Auditable core target rules, separate from immutable diagnostic score rows.

The inherited target safety checks are NC/DAC (configs/iqe/main.yaml). DDC has
zero official PDMS weight and remains a diagnostic, not an extra hidden target
gate. We retain any older stricter diagnostic rows rather than overwriting them.
"""
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
from iqe.io import read_json,atomic_json,file_hash,digest
from iqe.contracts import require
from .data import SPECIALTIES,admissible_optimized_source
from .binding import scenes_by_role

RULES={'schema':'specialist_paired_target_rules_v1','GT_min_PDMS':.9,'optimized_min_PDMS':.95,
       'target_over_Base_margin':.02,'optimized_not_worse_than_GT_tolerance':1e-8,
       'protected_components':['no_at_fault_collisions','drivable_area_compliance'],
       'protected_rule_source':'configs/iqe/main.yaml selection.protected_metrics',
       'geometry_required':True,'optimized_historical_strict_geometry_required':True,
       'optimized_source_prefix_whitelist':['gt','parent_construction:gt','pdm_reference_blend','parent_construction:pdm_reference_blend','corridor_center','parent_construction:corridor_center','map_centerline','parent_construction:map_centerline'],
       'Base_seen_D_S_disclosed':True,'no_dev_or_navtest_for_selection':True}


def verify_target_assets(config):
    names=('optimized_trajectories.npz','optimized_labels.parquet','strict_geometry.parquet')
    identity={name:file_hash(Path(config.optimized_root)/name) for name in names}
    audit=read_json(config.root/'target_audit/TARGET_OPPORTUNITY_AUDIT.json')
    require(identity['optimized_trajectories.npz']==audit['optimized_trajectory_hash'],'optimized trajectories changed')
    path=config.root/'TARGET_ASSET_BINDING.json'
    if path.exists():
        require(read_json(path)['assets']==identity,'target/provenance/geometry asset changed')
    else:
        atomic_json(path,{'root':config.optimized_root,'assets':identity,
            'source_audit_hash':file_hash(config.root/'target_audit/TARGET_OPPORTUNITY_AUDIT.json')},immutable=True)
    return identity


def eligibility(row):
    def safe(label,minimum):
        return bool(label and label['score_valid'] and label['total_score_01']>=minimum-1e-8 and
             all(label['component_valid_masks'].get(k,False) and label['named_components'][k]>=1-1e-8 for k in RULES['protected_components']))
    gt=row['gt'];opt=row.get('optimized');base=row['base'];source=admissible_optimized_source(row.get('optimized_provenance',{}).get('source',''))
    gt_valid=safe(gt,.9) and row['gt_geometry']['valid']
    opt_valid=safe(opt,.95) and row.get('optimized_geometry',{}).get('valid',False) and row.get('optimized_geometry',{}).get('historical_strict_pass',False) and source
    return {'gt_eligible':bool(gt_valid),'gt_opportunity':bool(gt_valid and gt['total_score_01']-base['total_score_01']>=.02-1e-8),
        'optimized_opportunity':bool(opt_valid and opt['total_score_01']-base['total_score_01']>=.02-1e-8),
        'paired_eligible':bool(gt_valid and opt_valid and opt['total_score_01']>=gt['total_score_01']-1e-8 and opt['total_score_01']-base['total_score_01']>=.02-1e-8)}


def core_audit(config):
    assets=verify_target_assets(config)
    path=config.root/'target_audit/CORE_TARGET_ELIGIBILITY.json';rules=config.root/'TARGET_RULES_BEFORE_EXPERT.json'
    if not rules.exists():atomic_json(rules,{'rules':RULES,'rules_hash':digest(RULES),'recorded_at_utc':datetime.now(timezone.utc).isoformat(),
        'pre_registration_claim':False,'phase':'before any specialist training','historical_stricter_diagnostic_rows_retained':True},immutable=True)
    require(read_json(rules)['rules_hash']==digest(RULES),'target rules changed within experiment')
    source=config.root/'target_audit/TARGET_OPPORTUNITY_AUDIT.json';audit=read_json(source)
    if path.exists():
        result=read_json(path);require(result['source_audit_hash']==file_hash(source) and result['rules_hash']==digest(RULES),'stale target rule application');return result
    ids={s.scene_id for s in scenes_by_role(config)['expert_fit']};rows={};summary={}
    for sid,h in audit['record_hashes'].items():
        p=config.root/'target_audit/scenes'/(sid+'.json');require(file_hash(p)==h,'target score evidence changed')
        row=read_json(p);rows[sid]=dict(row,**eligibility(row))
    import pandas as pd
    provenance=pd.read_parquet(Path(config.optimized_root)/'optimized_labels.parquet').set_index('token')
    geometry=pd.read_parquet(Path(config.optimized_root)/'strict_geometry.parquet').set_index('token')
    for sid,row in rows.items():
        if not row.get('optimized_provenance'):continue
        original=provenance.loc[sid]
        require(all(str(original[name])==row['optimized_provenance'][name]
                    for name in ('source','snapshot_source','run_identity')), 'audited optimized provenance changed')
        require(bool(geometry.loc[sid,'strict_geometry_pass'])==row['optimized_geometry']['historical_strict_pass'],
                'audited target geometry source changed')
    for specialty in SPECIALTIES:
        fit=[r for sid,r in rows.items() if sid in ids and specialty in r['specialties']];paired=[r for r in fit if r['paired_eligible']]
        summary[specialty]={'failures':len(fit),'logs':len({r['source_log'] for r in fit}),
            'gt_eligible':sum(r['gt_opportunity'] for r in fit),'paired':len(paired),'paired_logs':len({r['source_log'] for r in paired}),
            'pair_target_gain_ge_0_01':sum(r['optimized']['total_score_01']-r['gt']['total_score_01']>=.01 for r in paired),
            'pair_nearly_identical':sum(r['target_max_abs_difference']<=1e-5 for r in paired),
            'gt_U':sum(max(0,r['gt']['total_score_01']-r['base']['total_score_01']) for r in fit if r['gt_opportunity'])/len(ids),
            'optimized_U':sum(max(0,r['optimized']['total_score_01']-r['base']['total_score_01']) for r in fit if r['optimized_opportunity'])/len(ids)}
    result={'rules_hash':digest(RULES),'source_audit_hash':file_hash(source),'target_asset_hashes':assets,'summary':summary,
            'eligibility':{sid:eligibility(row) for sid,row in rows.items()},'status':'COMPLETE','source_scores_unchanged':True}
    atomic_json(path,result,immutable=True);return result

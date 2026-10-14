"""Matched fixed-5000 contrasts using completed official validation only."""
import argparse
from pathlib import Path
import numpy as np

from iqe.contracts import require
from iqe.io import atomic_json,file_hash,read_json
from iqe.internal_baseline.evaluation import paired_bootstrap


def contrasts(columns):
    require(set(columns)=={'E00','E10','E01','E11'},'four matched factorial cells required')
    a,b,c,d=(np.asarray(columns[k],dtype=np.float64) for k in ('E00','E10','E01','E11'))
    require(a.ndim==1 and a.shape==b.shape==c.shape==d.shape and all(np.isfinite(x).all() for x in (a,b,c,d)),
            'complete matched factorial populations required')
    return {'optimized_target_head_only':b-a,'optimized_target_private_reader':d-c,
            'private_reader_original_GT':c-a,'private_reader_optimized_target':d-b,
            'target_reader_interaction':(d-c)-(b-a)}


def run(config):
    from .binding import scenes_by_role
    from .data import verified_base_scores
    from .experiments import endpoint
    from .validation_reuse import completed_endpoint_records
    matrix=read_json(config.root/'EXPERT_MATRIX_DATA.json')
    require(all(matrix['cells'][k].get('status','READY')=='READY' for k in ('E00','E10','E01','E11')),
            'matched target/reader factorial not available; do not fill missing cells')
    scenes=scenes_by_role(config)['stage_val'];old=verified_base_scores(config,scenes)
    base=np.asarray([old[s.scene_id]['total_score_01'] for s in scenes]);groups=[s.source_log_id for s in scenes]
    rows=[];sources={}
    for kind in ('raw','ema'):
        values={}
        for cell in ('E00','E10','E01','E11'):
            item=endpoint(config,cell,5000,kind)
            records=completed_endpoint_records(config,item,scenes)
            values[cell]=np.asarray([label.total_score_01 for _,label in records])
            sources[cell+'_'+kind]={'checkpoint':item['checksum'],
                'validated_records':file_hash(config.root/'runtime/validation_reuse'/f'{cell}_005000_{kind}.json')}
        for objective,columns in [('expert_alone',values),('Oracle_with_same_Base',{k:np.maximum(base,v) for k,v in values.items()})]:
            for name,difference in contrasts(columns).items():
                rows.append({'endpoint':kind,'objective':objective,'contrast':name,
                             **paired_bootstrap(difference,groups,config.bootstrap_repetitions,config.bootstrap_seed)})
    result={'schema':'matched_fixed_endpoint_factorial_v1','step':5000,'stage_val_scenes':len(scenes),
            'stage_val_groups':len(set(groups)),'matched_hard_val_scenes':len(matrix['hard_val']),
            'Base_binding_hash':file_hash(config.root/'BASE_BINDING.json'),'matrix_manifest_hash':matrix['hash'],
            'source_receipts':sources,'rows':rows,'score_units':'[0,1]','new_model_updates':0,'new_official_calls':0,
            'scientific_limits':['single training seed','stage_val was used for system endpoint selection',
                'core specialty-specific hard validation has very few scenes',
                'fixed endpoint contrasts are not a matched old weak-finetune experiment'],
            'dev_accessed':False,'pool_or_recipe_changed':False}
    atomic_json(config.root/'EXPERT_FACTORIAL_CONTRASTS.json',result,immutable=True)
    lines=['# Matched target/reader contrasts','',
           'Fixed update5000, same paired training observations/order and same stage_val. Scores below are percentage-point differences; CIs use2000 paired log bootstrap resamples. No model or pool is selected by this extra analysis.','',
           '| Endpoint | Objective | Contrast | Difference (pp) | 95% interval (pp) |',
           '|---|---|---|---:|---:|']
    for row in rows:
        lines.append(f"| {row['endpoint']} | {row['objective']} | {row['contrast']} | {row['mean']*100:.4f} | [{row['lower_95']*100:.4f}, {row['upper_95']*100:.4f}] |")
    lines+=['','Only'+str(len(matrix['hard_val']))+' matched hard-validation scene(s) are available. These full-stage_val contrasts cannot establish a specialty-specific generalization claim. They do not isolate training duration relative to older experiments with different data, anchors, targets and schedules.']
    (config.root/'EXPERT_FACTORIAL_CONTRASTS.md').write_text('\n'.join(lines)+'\n')
    return result


def main():
    from .config import load_config
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True)
    a=p.parse_args();run(load_config(a.config))


if __name__=='__main__':main()

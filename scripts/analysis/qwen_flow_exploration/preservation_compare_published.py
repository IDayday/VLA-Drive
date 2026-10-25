"""P-route contrasts from sealed published scene-mean CSVs only; no aggregation.

Uses the shared log-cluster bootstrap. It never loads trajectories, models,
candidate banks, scorers, or a training runtime. Existing published summaries and
their CSV hashes are mandatory; partial test means cannot enter this analysis.
"""
import argparse
import csv
import json
from pathlib import Path
import tempfile
import numpy as np
from starVLA.rl.qwen_flow_exploration.common_analysis import bootstrap,checked_seal
from starVLA.rl.qwen_sourceflow.contracts import fingerprint,file_sha
from starVLA.rl.flow_grpo.transactions import atomic_json


ARMS=('P0','P1','P2','P3','P4','P5')
CONTRASTS={'P3_minus_P2':('P3','P2'),'P4_minus_P0':('P4','P0'),
           'P5_minus_P3':('P5','P3'),'P5_minus_P4':('P5','P4'),'P5_minus_P1':('P5','P1'),
           'P3_minus_P0':('P3','P0'),'P5_minus_P0':('P5','P0')}
EVENT_FIELDS=('new_zero_seed_events','recovered_zero_seed_events','new_joint_violation_seed_events')


def read(path):return json.loads(Path(path).read_text())


def read_paired_csv(path,expected_sha,*,expected_tokens,seed_count=5):
    path=Path(path)
    if file_sha(path)!=expected_sha:raise ValueError('Published paired CSV hash changed')
    rows=list(csv.DictReader(path.open()));result={}
    for row in rows:
        token=row['scene_token']
        if token in result:raise ValueError('Duplicate scene in published CSV')
        if int(row['inference_seed_count'])!=seed_count:raise ValueError('Changed seed-within-scene population')
        value=dict(scene_token=token,log_id=row['log_id'],inference_seed_count=seed_count)
        for key in ('reference_mean','current_mean','delta_raw'):
            value[key]=float(row[key])
            if not np.isfinite(value[key]):raise ValueError('Nonfinite published scene mean')
        if not (0<=value['reference_mean']<=1 and 0<=value['current_mean']<=1):raise ValueError('Invalid score scale')
        if abs(value['current_mean']-value['reference_mean']-value['delta_raw'])>1e-12:
            raise ValueError('Published CSV delta is not its paired scene mean difference')
        for key in EVENT_FIELDS:
            value[key]=int(row[key])
            if not 0<=value[key]<=seed_count:raise ValueError('Invalid per-scene seed event count')
        result[token]=value
    if set(result)!=set(expected_tokens):raise ValueError('Published test population changed or incomplete')
    return [result[token] for token in sorted(expected_tokens)]


def contrast(left,right,*,repeats,seed):
    if len(left)!=len(right):raise ValueError('Unmatched paired CSV population')
    for a,b in zip(left,right):
        if (a['scene_token'],a['log_id'],a['inference_seed_count'])!=(b['scene_token'],b['log_id'],b['inference_seed_count']):
            raise ValueError('Unmatched scene/log/seeds across published CSVs')
        if a['reference_mean']!=b['reference_mean']:raise ValueError('Contrasts must share exactly the same BASE outputs')
    x=np.asarray([r['current_mean'] for r in left]);y=np.asarray([r['current_mean'] for r in right]);delta=x-y
    logs=[r['log_id'] for r in left]
    result=dict(current_PDMS=float(100*x.mean()),reference_PDMS=float(100*y.mean()),
        delta_PDMS_points=float(100*delta.mean()),
        paired_PDMS_log_cluster_bootstrap=bootstrap(delta,logs,repeats=repeats,seed=seed,kind='score'),
        G_plus_PDMS_points=float(100*np.maximum(delta,0).mean()),G_minus_PDMS_points=float(-100*np.minimum(delta,0).mean()),
        improved_scenes_ge_0_005=int((delta>=.005).sum()),degraded_scenes_ge_0_005=int((delta<=-.005).sum()),
        scene_mean_nonzero_to_zero_count=int(((y>0)&(x==0)).sum()),scene_mean_zero_to_nonzero_count=int(((y==0)&(x>0)).sum()),
        zero_semantics='These two counts concern all-five-seed scene means; they are not any-seed failure transitions.',
        BASE_relative_event_rate_changes={})
    for key in EVENT_FIELDS:
        values=np.asarray([(a[key]-b[key])/a['inference_seed_count'] for a,b in zip(left,right)])
        result['BASE_relative_event_rate_changes'][key]=bootstrap(values,logs,repeats=repeats,seed=seed,kind='rate')
    result['event_semantics']='Change in each model versus the identical fixed BASE draws; not direct model-to-model newly failed/repaired seed counts. Negative new-event change is favorable; positive recovery change is favorable.'
    return result


def load_published(directory,protocol,lock,index):
    tokens=protocol['splits']['NAVTEST'];variants={v['tag']:v for v in lock['variants']}
    if len(tokens)!=12146 or len(set(tokens))!=12146:raise ValueError('Full locked NAVTEST12146 required')
    if lock['protocol_sha256']!=protocol['sha256'] or lock['P_export_index_sha256']!=index['sha256']:
        raise ValueError('Final six-arm lock/index/common protocol mismatch')
    documents={};rows={};provenance={};base_identity=None
    for arm in ARMS:
        path=Path(directory)/(arm+'_vs_BASE.json');document=checked_seal(read(path),'published '+arm)
        full=document['five_seed_mean'];variant=variants[arm]
        if (document['schema']!='full_NAVTEST_only_effect_v1' or document['status']!='COMPLETE' or
                document['arm']!=arm or document['protocol_sha256']!=protocol['sha256'] or
                document['inference_seeds']!=[2101,2102,2103,2104,2105] or document['training_seeds']!=[42] or
                full['status']!='COMPLETE' or full['scenes']!=12146 or full['logs']!=136 or
                full['scene_seed_records']!=60730 or full['readout_family']!='stochastic_root_single'):
            raise ValueError('Only completed matching fixed P-arm full test summaries may enter contrasts')
        if variant['package_identity_sha256']!=index['arms'][arm]['package_metadata_identity_sha256']:
            raise ValueError('Final locked endpoint package differs from CPU export index')
        current=checked_seal(read(Path(full['evaluation'])/'identity.json'),'aggregate identity '+arm)
        if (current['sha256']!=full['evaluation_identity'] or current['sha256']!=document['evaluation_identity'] or
                current['package_identity_sha256']!=variant['package_identity_sha256'] or
                current['deployment_contract']!=variant['configuration']['deployment_contract'] or current['qualification']):
            raise ValueError('Published evaluation is not the fixed pure merged endpoint')
        if base_identity is None:base_identity=full['reference_identity']
        if full['reference_identity']!=base_identity:raise ValueError('Reference BASE outputs differ')
        data=read_paired_csv(document['paired_scene_csv'],document['paired_scene_csv_sha256'],expected_tokens=tokens)
        if len({r['log_id'] for r in data})!=136:raise ValueError('Wrong full test log denominator')
        if abs(100*np.mean([r['current_mean'] for r in data])-full['current_PDMS'])>1e-10:
            raise ValueError('Published summary mean disagrees with published paired CSV')
        if abs(100*np.mean([r['reference_mean'] for r in data])-full['reference_PDMS'])>1e-10:
            raise ValueError('Published BASE mean disagrees with published paired CSV')
        documents[arm]=document;rows[arm]=data
        provenance[arm]=dict(summary=str(path),summary_sha256=file_sha(path),summary_identity_sha256=document['sha256'],
            paired_scene_csv=document['paired_scene_csv'],paired_scene_csv_sha256=document['paired_scene_csv_sha256'])
    return documents,rows,provenance


def self_test():
    checks=[]
    def row(token,log,current,base=.5,new=0):
        return dict(scene_token=token,log_id=log,inference_seed_count=5,reference_mean=base,current_mean=current,
                    delta_raw=current-base,new_zero_seed_events=new,recovered_zero_seed_events=0,new_joint_violation_seed_events=new)
    left=[row('a','L1',.7,new=1),row('b','L1',.7),row('c','L1',.7),row('d','L2',.3)]
    right=[row(t,l,.5) for t,l in [('a','L1'),('b','L1'),('c','L1'),('d','L2')]]
    result=contrast(left,right,repeats=200,seed=42)
    assert abs(result['delta_PDMS_points']-10)<1e-10
    assert abs(result['G_plus_PDMS_points']-15)<1e-10 and abs(result['G_minus_PDMS_points']-5)<1e-10
    assert abs(result['BASE_relative_event_rate_changes']['new_zero_seed_events']['mean_percentage_points']-5)<1e-10
    checks.append('Scene means retain3:1 unequal log scene weights; Gplus-Gminus matches net; BASE-relative event denominator5 retained')
    bad=[dict(r) for r in right];bad[0]['reference_mean']=.4
    try:contrast(left,bad,repeats=20,seed=42)
    except ValueError:pass
    else:raise AssertionError('Changed reference accepted')
    checks.append('Changed BASE reference rejected')
    with tempfile.TemporaryDirectory() as temporary:
        p=Path(temporary)/'rows.csv'
        with p.open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(left[0]));writer.writeheader();writer.writerows(left)
        assert len(read_paired_csv(p,file_sha(p),expected_tokens=['a','b','c','d']))==4
        try:read_paired_csv(p,'0'*64,expected_tokens=['a','b','c','d'])
        except ValueError:pass
        else:raise AssertionError('Changed file hash accepted')
        checks.append('Published CSV SHA and exact membership required')
        with p.open('a') as f:
            writer=csv.DictWriter(f,fieldnames=list(left[0]));writer.writerow(left[0])
        try:read_paired_csv(p,file_sha(p),expected_tokens=['a','b','c','d'])
        except ValueError:pass
        else:raise AssertionError('Duplicate scene accepted')
        checks.append('Duplicate scene rejected')
    return dict(status='PASS',checks=checks,tests=len(checks),CPU_only=True,new_inference_or_scoring_calls=0)


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--analysis-dir');p.add_argument('--protocol');p.add_argument('--lock')
    p.add_argument('--export-index');p.add_argument('--output',required=True);p.add_argument('--self-test',action='store_true')
    p.add_argument('--bootstrap-repeats',type=int,default=2000);p.add_argument('--bootstrap-seed',type=int,default=42);a=p.parse_args()
    if Path(a.output).exists():raise FileExistsError('Keep old analysis; select a new output')
    if a.self_test:
        result=self_test();atomic_json(a.output,result);print(json.dumps(result));return
    if not all((a.analysis_dir,a.protocol,a.lock,a.export_index)):p.error('All four sealed input locations required')
    protocol=checked_seal(read(a.protocol),'common protocol');lock=checked_seal(read(a.lock),'final configuration lock')
    index=checked_seal(read(a.export_index),'P terminal export index')
    documents,rows,provenance=load_published(a.analysis_dir,protocol,lock,index)
    comparisons={name:dict(contrast=left+'-'+right,**contrast(rows[left],rows[right],repeats=a.bootstrap_repeats,seed=a.bootstrap_seed))
                 for name,(left,right) in CONTRASTS.items()}
    logs=[r['log_id'] for r in rows['P0']]
    interaction=np.asarray([r['current_mean'] for r in rows['P5']])-np.asarray([r['current_mean'] for r in rows['P3']])-np.asarray([r['current_mean'] for r in rows['P4']])+np.asarray([r['current_mean'] for r in rows['P0']])
    result=dict(schema='P_published_full_NAVTEST_CSV_contrasts_v1',status='COMPLETE',scenes=12146,logs=136,inference_seeds=[2101,2102,2103,2104,2105],
        training_seeds=[42],protocol_sha256=protocol['sha256'],final_configuration_lock_sha256=lock['sha256'],export_index_identity_sha256=index['sha256'],
        input_provenance=provenance,comparisons=comparisons,
        archive_constraint_interaction=dict(contrast='P5-P3-P4+P0',bootstrap=bootstrap(interaction,logs,repeats=a.bootstrap_repeats,seed=a.bootstrap_seed,kind='score'),meaning='Additional combined gain beyond separate archive and constraint; mechanistic evidence only'),
        scope='Published complete scene-mean CSVs only. No aggregate rebuild, model, actor, scorer, optimizer or new data.',
        safety_limit='Between-P event rate changes are relative to the same BASE draws, not direct P-to-P newly failed/recovered seed counts.',
        uncertainty='Seed means already reduced inside each scene; common log-cluster bootstrap retains scene weight. No training/inference seed uncertainty and no multiplicity correction.',
        P1_not_update_matched=True,replay_gradient_calibration_capped=True,
        source_sha256=file_sha(__file__),new_inference_or_scoring_calls=0)
    result['sha256']=fingerprint(result);atomic_json(a.output,result)
    print(json.dumps({name:value['delta_PDMS_points'] for name,value in comparisons.items()},indent=2))


if __name__=='__main__':main()

"""Join immutable V12 optimized physical targets to the existing World TRAIN.

Observation/teacher records remain original. Only the clean-x planning label
can be overridden; no future image, map, reward or optimization state enters
the actor observation. All original fallback scenes remain in the population.
"""
import argparse
import csv
import json
from pathlib import Path
import pickle
import time

import numpy as np

from starVLA.rl.qwen_world_postcot.contracts import file_sha,fingerprint


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--output',required=True)
    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    export=Path('/mnt/project/optimized-gt-export-20261022/navtrain_v12_full')
    meta=json.loads((export/'MANIFEST.json').read_text());summary=json.loads((export/'SOURCE_RELEASE_SUMMARY.json').read_text())
    for name,spec in meta['files'].items():
        if file_sha(export/name)!=spec['sha256']:raise ValueError('V12 export source changed: '+name)
    data=Path('/mnt/project/qwen-world-postcot-20261025/data/TRAIN');manifest=json.loads((data/'manifest.json').read_text())
    partition=json.loads((data.parent/'DATA_PARTITION.json').read_text())
    if manifest['cohort']!='TRAIN' or file_sha(data/'records.pkl')!=manifest['records_sha256']:
        raise ValueError('Unsealed World TRAIN')
    tokens=manifest['tokens'];records=pickle.load(open(data/'records.pkl','rb'))
    if set(tokens)&set(partition['dev_tokens']+partition['navtest_tokens']):raise ValueError('Held-out tokens in training targets')
    wanted=set(tokens);found={};details={}
    pose_cols=[c for i in range(1,9) for c in (f'x_{i}_m',f'y_{i}_m',f'yaw_{i}_rad')]
    with (export/'optimized_gt_navtrain_103288.csv').open() as f:
        for row in csv.DictReader(f):
            token=row['token']
            if token not in wanted:continue
            if token in found or row['log_name']!=records[token]['log']:raise ValueError('Duplicate token or log mismatch')
            found[token]=np.array([float(row[c]) for c in pose_cols],dtype=np.float64).reshape(8,3)
            details[token]=dict(accepted=row['accepted']=='True',changed=row['changed']=='True',risk_tier=int(row['risk_tier']))
    if set(found)!=wanted:raise ValueError('V12 misses World TRAIN scenes: '+str(len(wanted-set(found))))
    trajectories=np.stack([found[t] for t in tokens]);assert trajectories.shape==(len(tokens),8,3)
    if not np.isfinite(trajectories).all():raise ValueError('Nonfinite optimized trajectories')
    physical_logs={records[t]['log'].rsplit('_',2)[0] for t in tokens}
    if physical_logs&set(partition['dev_physical_logs']+partition['navtest_physical_logs']):raise ValueError('Held-out physical log overlap')
    fallback_diffs=[]
    for t in tokens:
        if not details[t]['accepted']:
            e=found[t]-np.asarray(records[t]['requested_GT']);e[:,2]=(e[:,2]+np.pi)%(2*np.pi)-np.pi
            dt=np.cumsum(records[t]['raw_future_dt_seconds']);valid=np.abs(dt-np.arange(1,9)*.5)<.06
            if valid.any():fallback_diffs.append(float(np.abs(e[valid]).max()))
    max_fallback=max(fallback_diffs,default=0.)
    if max_fallback>1e-4:raise ValueError('Fallback GT coordinate/time contract differs: '+str(max_fallback))
    with (export/'label_metadata.csv').open() as f:
        rows={row['token']:row for row in csv.DictReader(f) if row['token'] in wanted}
    if set(rows)!=wanted:raise ValueError('Missing source score metadata')
    scores=np.array([float(rows[t]['score']) for t in tokens]);gt_scores=np.array([float(rows[t]['gt_score']) for t in tokens])
    if not np.isfinite([scores,gt_scores]).all():raise ValueError('Nonfinite historical label score')
    np.save(out/'targets.npy',trajectories.astype(np.float32),allow_pickle=False)
    (out/'tokens.json').write_text(json.dumps(tokens)+'\n')
    counts=dict(scenes=len(tokens),accepted=sum(details[t]['accepted'] for t in tokens),
        changed=sum(details[t]['changed'] for t in tokens),original_GT_fallback=sum(not details[t]['accepted'] for t in tokens))
    identity=dict(schema='world_optimized_planning_targets_v1',source_version=meta['version'],
        source_run_identity=meta['source_run_identity'],source_export_manifest_sha256=file_sha(export/'MANIFEST.json'),
        source_csv_sha256=meta['files']['optimized_gt_navtrain_103288.csv']['sha256'],
        original_export_scenes=103288,original_historical_label_PDMS=100*summary['export_score'],
        selected_TRAIN_historical_label_PDMS=float(scores.mean()*100),selected_TRAIN_historical_GT_PDMS=float(gt_scores.mean()*100),
        historical_score_warning='Historical V12 nominal scoring, not a newly scored model or automatically certified identical to the current canonical v1.1 backend.',
        current_observation_manifest_sha256=file_sha(data/'manifest.json'),current_records_sha256=manifest['records_sha256'],
        partition_sha256=file_sha(data.parent/'DATA_PARTITION.json'),token_sha256=fingerprint(tokens),counts=counts,
        shape=list(trajectories.shape),frame='current_rear_axle_x_forward_y_left',interval_seconds=.5,
        fields=['x_m','y_m','heading_rad'],float64_source_preserved_at_export=True,training_storage='float32',
        max_valid_fallback_GT_difference=max_fallback,normalization='Unchanged native World ActionCodec, no clipping',
        original_video_teacher_targets_unchanged=True,only_main_planning_supervision_replaced=True,
        DEV_or_Test_labels_used=False,full_TRAIN_population_kept=True,
        files={n:file_sha(out/n) for n in ['targets.npy','tokens.json']},code_sha256=file_sha(__file__),wall_seconds=time.monotonic()-start)
    identity['sha256']=fingerprint(identity);(out/'MANIFEST.json').write_text(json.dumps(identity,indent=2)+'\n')
    (out/'COMPLETE.json').write_text(json.dumps(dict(status='COMPLETE',manifest_sha256=file_sha(out/'MANIFEST.json')))+'\n')
    print(json.dumps(identity,indent=2))


if __name__=='__main__':main()

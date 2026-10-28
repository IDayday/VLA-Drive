"""Aggregate the completed fixed-cohort readout audit without new inference."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from starVLA.rl.qwen_world_postcot.contracts import file_sha,fingerprint


def summarize(root):
    root=Path(root);plan=json.loads((root/'PLAN.json').read_text());body=dict(plan)
    if body.pop('sha256')!=fingerprint(body):raise ValueError('Changed fixed audit plan')
    complete=json.loads((root/'COMPLETE.json').read_text())
    if complete['status']!='COMPLETE' or complete['optimizer_updates'] or complete['new_scorer_calls']:
        raise ValueError('Audit is incomplete or changed its read-only contract')
    rows=[];hashes={}
    for update in plan['checkpoint_updates']:
        path=root/f'update{update:06d}.json';data=json.loads(path.read_text());hashes[path.name]=file_sha(path)
        if data['updates']!=update:raise ValueError('Wrong checkpoint readout')
        for role,tokens in plan['split_tokens'].items():
            sample=[r for r in data['rows'] if r['role']==role]
            if [r['token'] for r in sample]!=tokens:raise ValueError('Fixed cohort changed between checkpoints')
            result=dict(updates=update,role=role,scenes=len(sample))
            for task in ('dino','future'):
                for field in ('MSE','cosine','TRAIN_mean_baseline_MSE'):
                    values=[r[task][field] for r in sample]
                    if not np.isfinite(values).all():raise ValueError('Nonfinite task readout')
                    result[task+'_'+field]=float(np.mean(values))
                result[task+'_MSE_reduction_vs_constant']=1-result[task+'_MSE']/result[task+'_TRAIN_mean_baseline_MSE']
            objects=[r['objects'] for r in sample];count=sum(r['visible_objects'] for r in objects)
            result['visible_objects']=count
            for name,field in [('matched_class_accuracy','matched_class_correct'),('matched_xy_MAE_m','matched_xy_distance_m_sum'),
                    ('matched_heading_MAE_deg','matched_heading_abs_deg_sum'),('visible_positive_recall2m','visible_positive_recall2m_count')]:
                result[name]=sum(r.get(field,0) for r in objects)/count if count else None
            result['object_precision']=None
            result['complete_negative_object_coverage']=all(r['coverage_complete'] for r in objects)
            rows.append(result)
    result=dict(schema='world_aux_progress_summary_v1',plan_sha256=plan['sha256'],source_files=hashes,rows=rows,
        cost=complete,aggregation='Scene means for features; matched visible-object weighting for object metrics.',
        limitations=['Fixed64 TRAIN and64 DEV scenes; auxiliary diagnosis, not planning performance selection.',
            'Object matching and geometry are conditional on visible supplied positives; precision is unknown.',
            'No matched removal ablation: auxiliary learning does not establish a causal PDMS benefit.',
            'Post-head prediction reliability requires its separate scored-candidate audit.'])
    result['sha256']=fingerprint(result);return result


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--input',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True);result=summarize(a.input)
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    with (out/'summary.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(result['rows'][0]));writer.writeheader();writer.writerows(result['rows'])
    print(json.dumps(dict(status='COMPLETE',rows=len(result['rows']),sha256=result['sha256'])))


if __name__=='__main__':main()

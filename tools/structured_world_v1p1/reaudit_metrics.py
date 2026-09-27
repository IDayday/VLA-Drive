"""Read-only V1 NPZ re-audit; separate output, no inference or PDM recomputation."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
import torch
from starVLA.model.modules.structured_world.contracts import WorldTargets
from starVLA.model.modules.structured_world.metrics import diagnostics


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_csv(path,rows):
    keys=sorted({k for r in rows for k in r})
    with Path(path).open('w') as f:
        w=csv.DictWriter(f,keys);w.writeheader();w.writerows(rows)


def summarize(rows):
    total=lambda k:sum(float(r.get(k,0)) for r in rows)
    ratio=lambda n,d:total(n)/total(d) if total(d) else None
    result={'scenes':len(rows),'failed':sum(r['status']!='ok' for r in rows),'gt_targets':total('gt_targets'),'fixed_proposals':total('fixed_k'),
            'no_object_probability_mean':ratio('no_object_probability_sum','fixed_k'),
            'outside_roi_fraction':ratio('outside_roi','fixed_k'),
            'inside_roi_outside_support_fraction':ratio('inside_roi_outside_support','fixed_k'),
            'objectness_predictions':total('objectness_predictions'),'support_predictions':total('support_predictions'),'filtered_predictions':total('filtered_predictions'),
            'legacy_recall_2m':ratio('legacy_matched_detection_targets','gt_targets'),
            'overflow':total('gt_overflow')}
    for mode in ['raw_k','roi','support','objectness','filtered','class_raw_k','class_filtered']:
        x={'recall_2m':ratio(mode+'_tp','gt_targets'),'precision_2m':ratio(mode+'_tp',mode+'_predictions'),
           'tp':total(mode+'_tp'),'fp':total(mode+'_fp'),'centre_error_m':ratio(mode+'_centre_error_sum',mode+'_tp'),
           'yaw_error_rad':ratio(mode+'_yaw_error_sum',mode+'_yaw_targets'),
           'motion_instance_coverage':ratio(mode+'_all_motion_instances','gt_motion_instances'),
           'motion_point_coverage':ratio(mode+'_all_motion_points','gt_motion_points')}
        for group in ['all','dynamic','static']:
            x[group]={m:ratio(f'{mode}_{group}_{n}',f'{mode}_{group}_{d}') for m,n,d in [('ADE','ade_sum','motion_points'),('FDE','fde_sum','fde_targets'),('stationary_ADE','stationary_ade_sum','motion_points'),('stationary_FDE','stationary_fde_sum','fde_targets')]}
        result[mode]=x
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--old-root',required=True);p.add_argument('--targets',required=True);p.add_argument('--output',required=True);p.add_argument('--runs',nargs='+')
    a=p.parse_args();root=Path(a.old_root);targets=Path(a.targets);out=Path(a.output)
    out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1)
    tokens=json.loads((root/'dev_tokens.json').read_text())
    if isinstance(tokens,dict):raise ValueError('Expected explicit token list')
    if len(tokens)!=len(set(tokens)) or len(tokens)!=1696:raise ValueError('Development manifest mismatch')
    cached={t:WorldTargets(**torch.load(targets/'targets'/f'{t}.pt',map_location='cpu',weights_only=True)) for t in tokens}
    fingerprint=hashlib.sha256()
    for t in sorted(tokens):fingerprint.update((t+sha(targets/'targets'/f'{t}.pt')).encode())
    runs=a.runs or [p.name[:-4] for p in sorted(root.glob('*_dev')) if any((p/'predictions').glob('*.npz'))]
    all_rows=[];summaries={};inputs={};raw_summary={}
    for run in runs:
        prediction_root=root/(run+'_dev')/'predictions'
        rows=[];centres=[];noobj=[];hashes=hashlib.sha256()
        for token in tokens:
            path=prediction_root/(token+'.npz');r={'run':run,'token':token,'status':'ok'}
            try:
                hashes.update((token+sha(path)).encode())
                with np.load(path,allow_pickle=False) as z:
                    if 'boxes' not in z:
                        r['world_status']='not_applicable';rows.append(r);continue
                    pred={k:torch.from_numpy(z[k].copy()) for k in ['boxes','logits','future_xy']}
                r.update(diagnostics(pred,cached[token]))
                if r['evaluation_status']!='ok':raise ValueError(r['evaluation_status'])
                centres.append(pred['boxes'][:,:2].numpy());noobj.append(pred['logits'].softmax(-1)[:,-1].numpy())
            except Exception as e:
                r.update(status='failed',error=repr(e))
            rows.append(r)
        write_csv(out/(run+'.csv'),rows)
        summaries[run]=summarize(rows) if centres else {'scenes':len(rows),'world_status':'not_applicable','failed':sum(r['status']!='ok' for r in rows)}
        if centres:
            xy=np.concatenate(centres);prob=np.concatenate(noobj)
            raw_summary[run]={'quantiles':[0,.01,.25,.5,.75,.99,1], 'x':np.quantile(xy[:,0],[0,.01,.25,.5,.75,.99,1]).tolist(), 'y':np.quantile(xy[:,1],[0,.01,.25,.5,.75,.99,1]).tolist(),'no_object_probability':np.quantile(prob,[0,.01,.25,.5,.75,.99,1]).tolist()}
        inputs[run]={'npz_manifest_sha256':hashes.hexdigest(),'pdms_csv_sha256':sha(root/(run+'_pdms')/'scenes.csv')}
        all_rows.extend(rows)
        print(json.dumps({'run':run,'failed':summaries[run]['failed'],'recall':summaries[run].get('filtered',{}).get('recall_2m'),'raw_recall':summaries[run].get('raw_k',{}).get('recall_2m')}),flush=True)
    write_csv(out/'scene_metrics.csv',all_rows)
    evidence={'code_sha':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'targets_manifest_sha256':fingerprint.hexdigest(),'dev_manifest_sha256':sha(root/'dev_tokens.json'),'target_cache':str(targets),'input_banks':inputs,'summaries':summaries,'raw_distributions':raw_summary,
              'matching':'maximum cardinality then minimum distance; strict < 2m; classes optional; no future, size, or yaw',
              'motion':'same geometric assignment; stationary baseline at predicted current centre; dynamic if any valid GT displacement >=1m; terminal FDE; absent denominator=null',
              'planning_trajectories_and_pdms':'read only; unchanged; original NPZ and PDM CSV hashes recorded'}
    (out/'METRIC_REAUDIT.json').write_text(json.dumps(evidence,indent=2))

if __name__=='__main__':main()

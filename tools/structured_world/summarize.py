"""Paired scene analysis with whole-log cluster bootstrap; no failed-row filtering."""
import argparse,csv,json
from pathlib import Path
import numpy as np


def read(path):
 with open(path) as f:rows=list(csv.DictReader(f))
 result={r['token']:r for r in rows}
 if len(result)!=len(rows):raise ValueError('Duplicate evaluation tokens')
 return result


def compare(a,b,seed=20260926,replicates=10000):
 if set(a)!=set(b):raise ValueError('Paired evaluations have different scene sets')
 tokens=sorted(a);logs=sorted({a[t]['log'] for t in tokens});delta=np.array([float(a[t]['score'])-float(b[t]['score']) for t in tokens])
 if any(a[t]['log']!=b[t]['log'] for t in tokens):raise ValueError('Mismatched log identity')
 if any(a[t]['status']!='ok' or b[t]['status']!='ok' for t in tokens):raise ValueError('Invalid paired evaluation: failed scenes retained, result not promoted')
 sums=np.array([sum(delta[i] for i,t in enumerate(tokens) if a[t]['log']==log) for log in logs]);counts=np.array([sum(a[t]['log']==log for t in tokens) for log in logs])
 rng=np.random.default_rng(seed);draws=rng.integers(0,len(logs),size=(replicates,len(logs)));boot=sums[draws].sum(1)/counts[draws].sum(1)
 return {'scenes':len(tokens),'logs':len(logs),'delta_percentage_points':float(delta.mean()*100),'log_cluster_CI95_percentage_points':(np.quantile(boot,[.025,.975])*100).tolist(),'wins':int((delta>1e-10).sum()),'losses':int((delta < -1e-10).sum()),'ties':int((abs(delta)<=1e-10).sum()),'bootstrap_replicates':replicates,'training_seed_uncertainty':'Not represented by this within-seed log-cluster interval'}


def main():
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--runs',nargs='+',required=True);p.add_argument('--output',required=True);a=p.parse_args();root=Path(a.root);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
 banks={};summary=[];all_rows=[]
 for run in a.runs:
  bank=read(root/f'{run}_pdms/scenes.csv');banks[run]=bank;rows=list(bank.values());n=len(rows)
  result={'run':run,'scenes':n,'failed':sum(r['status']!='ok' for r in rows),'PDMS_percent':sum(float(r.get('score',0)) for r in rows)/n*100,'zero_fraction':sum(float(r.get('score',0))==0 for r in rows)/n}
  for metric in ['no_at_fault_collisions','drivable_area_compliance','ego_progress','time_to_collision_within_bound','comfort','driving_direction_compliance']:
   result[metric]=sum(float(r.get(metric) or 0) for r in rows)/n
  diagnostic_lookup={}
  world=root/f'{run}_dev/diagnostics_v6.csv'
  if not world.exists():world=root/f'{run}_dev/scenes_0.csv'
  if world.exists():
   diagnostic_lookup=read(world);diagnostics=list(diagnostic_lookup.values())
   total=lambda key:sum(float(r.get(key) or 0) for r in diagnostics)
   result['mean_inference_seconds']=total('latency_seconds')/len(diagnostics)
   ratio=lambda numerator,denominator:total(numerator)/total(denominator) if total(denominator)>0 else None
   result['class_correct_detection_recall_2m']=ratio('class_correct_detection_targets','gt_targets')
   for cameras in [1,2,3]:result[f'cameras_{cameras}_recall']=ratio(f'cameras_{cameras}_detected',f'cameras_{cameras}_gt')
   result['gt_targets']=total('gt_targets');result['detection_recall_2m']=ratio('matched_detection_targets','gt_targets')
   result['false_positives']=total('false_positives');result['centre_error_matched']=ratio('matched_centre_error_sum','matched_detection_targets')
   result['yaw_error_matched_radians']=ratio('yaw_error_sum','matched_detection_targets')
   result['motion_ADE_detected']=ratio('ade_sum','valid_motion_points')
   result['motion_FDE_detected']=ratio('fde_sum','fde_targets')
   result['assignment_motion_ADE']=ratio('assignment_motion_ade_sum','assignment_motion_points')
   result['assignment_motion_FDE']=ratio('assignment_motion_fde_sum','assignment_motion_fde_targets')
   result['motion_valid_detection_coverage']=ratio('end_to_end_motion_targets','gt_motion_instances')
   for group in ['near','middle','far']:result[group+'_detection_recall']=ratio(group+'_detected',group+'_gt')
   if not any('gt_targets' in r for r in diagnostics):
    for key in list(result):
     if key in ['gt_targets','false_positives'] or any(x in key for x in ['recall','motion_','centre_error','yaw_error']):result[key]=None
  summary.append(result)
  for row in rows:all_rows.append(dict(row,run=run,**{'world_'+k:v for k,v in diagnostic_lookup.get(row['token'],{}).items() if k!='token'}))
 pairs={}
 for run in a.runs:
  if run!='A0' and 'A0' in banks:pairs[run+'__vs__A0']=compare(banks[run],banks['A0'])
 for first,second in [('C_seed42','B_seed42'),('C_seed43','B_seed43'),('E_seed42','D_seed42'),('A2_seed42','A1_seed42'),('E_adapter_support_v6_seed42','E_support_v6_seed42'),('E_support_v6_seed42','C_support_v6_seed42')]:
  if first in banks and second in banks:pairs[first+'__vs__'+second]=compare(banks[first],banks[second])
 for filename,rows in [('summary.csv',summary),('scene_metrics.csv',all_rows)]:
  keys=sorted(set(k for r in rows for k in r))
  with (out/filename).open('w') as f:w=csv.DictWriter(f,keys);w.writeheader();w.writerows(rows)
 paired_rows=[]
 for name in pairs:
  first,second=name.split('__vs__')
  for token in sorted(banks[first]):
   x,y=banks[first][token],banks[second][token]
   paired_rows.append({'comparison':name,'token':token,'log':x['log'],'first_PDMS':float(x['score']),'second_PDMS':float(y['score']),'delta':float(x['score'])-float(y['score'])})
 with (out/'paired_scenes.csv').open('w') as f:
  w=csv.DictWriter(f,['comparison','token','log','first_PDMS','second_PDMS','delta']);w.writeheader();w.writerows(paired_rows)
 (out/'paired_comparisons.json').write_text(json.dumps(pairs,indent=2));(out/'summary.json').write_text(json.dumps(summary,indent=2))
 print(json.dumps({'runs':len(summary),'paired_comparisons':len(pairs)}))
if __name__=='__main__':main()

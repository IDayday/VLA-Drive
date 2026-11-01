from pathlib import Path
import json,time,numpy as np
D=Path('/mnt/project/qwen-world-safety-exploration-20261031/data/probes/FROZEN_SAFETY10K_SUPPORT_20261031/results');summary=json.load(open(D/'SUMMARY.json'));rows=[]
K=['no_at_fault_collisions','drivable_area_compliance','time_to_collision_within_bound','ego_progress']
def credit(records):
 x=np.array([[r['metrics'][k]for k in K]for r in records]);f=(x[:,:3]==1).all(-1);p=f.mean();c=(1-x[:,:3]).mean(-1)
 a=f.astype(float)-p if f.any()else(c-c[0]).mean()-(c-c[0]);ep=np.zeros(8)
 if f.sum()>=2:off=x[:,3]-x[np.flatnonzero(f)[0],3];ep[f]=off[f]-off[f].mean()
 raw=a+ep;strict=f&np.array([r['metrics']['comfort']==1 and r['metrics']['score']>=.9 for r in records]);score=np.array([r['metrics']['score']for r in records])
 return dict(found_quality=bool(strict.any()),quality_candidates=int(strict.sum()),found_feasible=bool(f.any()),feasible_candidates=int(f.sum()),canonical_allzero=bool((score==0).all()),PG0=bool((raw==0).all()),mixed_feasible=bool(0<f.sum()<8),allfail=not bool(f.any()),raw_std=float(raw.std()),mean_PDMS=float(score.mean()*100),positive_unsafe=int(((raw>0)&~f).sum()))
for s in summary['scenes']:
 j=json.load(open(D/'scenes'/f'{s["token"]}.json'));a=j['strategies']['A_late']['records'];b=j['strategies']['B_mixed']['records']
 arms={'late8':a[:8],'early8':b[:8],'split4late4early':a[:4]+b[4:8],'split4early4late':b[:4]+a[4:8]}
 rows.append(dict(token=j['token'],role=j['role'],strategies={k:credit(v)for k,v in arms.items()}))
result={}
for role in ('historical_failure','matched_high_control'):
 rr=[r for r in rows if r['role']==role];result[role]={}
 for arm in rows[0]['strategies']:
  z=[r['strategies'][arm]for r in rr];v={k:sum(float(a[k])for a in z)for k in z[0]};v['raw_std']/=len(z);v['mean_PDMS']/=len(z)
  v['quality_found_not_late8']=sum(r['strategies'][arm]['found_quality']and not r['strategies']['late8']['found_quality']for r in rr)
  v['late8_quality_lost']=sum(r['strategies']['late8']['found_quality']and not r['strategies'][arm]['found_quality']for r in rr)
  v['PG0_late_reactivated']=sum(r['strategies']['late8']['PG0']and not r['strategies'][arm]['PG0']for r in rr)
  result[role][arm]=v
out=dict(schema='world_existing_A_B_G16_to_G8_split_exploration_composition_v1',status='COMPLETE_CPU_EXISTING_RECORDS_ONLY',cohorts=result,rows=rows,queries_added=0,
 limitations=['frozenSafety10k selected156historicalfailure+156highcontrol TRAIN only, not NAV or allpopulation', 'Actual candidates were generated atG16; subsetG8/composition is not actualG8 newforward evidence', 'Comparisons use same existing roots/innovation indices but mixedmode risk distribution and groupbaseline change; not a standalone kernel efficacy claim'])
p=Path('/mnt/project/qwen-world-safety-exploration-20261031/improvement_training_20261101/analysis/MIXED_EXPLORATION_EXISTING_SUPPORT_AUDIT.json');p.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(result,indent=2));print(p)

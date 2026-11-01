"""Finite full12146 existing prediction oracle; no new scoring or training selection."""
import json,hashlib,time,os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from tools.qwen_world_safety_exploration.review import load_full,compatible
from tools.qwen_world_research.evaluate import read_sealed
E=Path('/mnt/project/qwen-world-safety-exploration-20261031');F=E/'standard_training_20261101';G=E/'revival_training_20261101';OUT=E/'analysis/FOUR_ROUTE_STRICTSAFE_ORACLE_20261101';OUT.mkdir(parents=True,exist_ok=True)
SPECS=[('Control10k',F/'evaluations/SAFETY_G8_CONTROL_u010000/NAVTEST'),('Inversion16k',F/'evaluations/STANDARD_INVERSION_G8_u016000/NAVTEST'),('Saturated14k',G/'evaluations/REVIVAL_SATURATED_G8_u014000/NAVTEST'),('Active18k',G/'evaluations/REVIVAL_ACTIVE_DATA_G8_u018000/NAVTEST')]
KEYS=['score','no_at_fault_collisions','drivable_area_compliance','time_to_collision_within_bound','comfort','ego_progress']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def fp(d):return hashlib.sha256(json.dumps(d,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def load(spec):
 label,path=spec;s=read_sealed(path/'SUMMARY.json')
 if s['status']!='COMPLETE'or s['scenes']!=12146 or s['invalid_scenes']!=0:raise ValueError('Partial not usable')
 identity,rows=load_full(path);tokens=sorted(rows);a=np.asarray([[rows[t]['feedback']['metrics'][k]for k in KEYS]for t in tokens],dtype=np.float64)
 # Source rows are validated for full census/protocol/hash seal by native loader.
 seals=[dict(token=t,scene_receipt_sha256=fp(rows[t]))for t in tokens]
 return label,identity,tokens,a,dict(path=str(path),summary_file_sha256=sha(path/'SUMMARY.json'),identity_file_sha256=sha(path/'identity.json'),identity_seal=identity['sha256'],validated_scene_count=len(rows),ordered_scene_receipt_seals_sha256=fp(seals)),seals
with ThreadPoolExecutor(max_workers=4)as pool:loaded=list(pool.map(load,SPECS))
for label,identity,tokens,a,source,seals in loaded:
 compatible(loaded[0][1],identity)
 if tokens!=loaded[0][2]:raise ValueError('Different complete population')
tokens=loaded[0][2];values=np.stack([r[3]for r in loaded]);scores=values[:,:,0];strict=np.all(values[:,:,1:5]==1,axis=2);labels=[r[0]for r in loaded];strict_scores=np.where(strict,scores,-np.inf);has_strict=strict.any(axis=0);strict_oracle=np.where(has_strict,strict_scores.max(axis=0),0.);winner=np.argmax(strict_scores,axis=0);base=scores[0];preserve=np.maximum(base,strict_oracle);ordinary=scores.max(axis=0);n=len(tokens)
rows=[];basezero=base==0
for i,label in enumerate(labels):
 others=np.delete(strict,i,axis=0);other_scores=np.delete(scores,i,axis=0)
 rows.append(dict(label=label,PDMS=100*float(scores[i].mean()),canonical_zero=int((scores[i]==0).sum()),strict_safe_candidates=int(strict[i].sum()),strict_safe_unique_witness=int((strict[i]&~others.any(axis=0)).sum()),canonical_positive_unique_witness=int(((scores[i]>0)&np.all(other_scores==0,axis=0)).sum()),rescued_control_zero_canonical_positive=int((basezero&(scores[i]>0)).sum()),rescued_control_zero_strict_positive=int((basezero&strict[i]&(scores[i]>0)).sum()),strict_witness_when_control_not_strict=int((~strict[0]&strict[i]).sum())))
ledger=OUT/'ledger.jsonl'
with ledger.open('w')as f:
 for j,t in enumerate(tokens):f.write(json.dumps(dict(token=t,cohort='NAVTEST_DIAGNOSTIC_NEVER_TRAINING',scores={labels[i]:float(scores[i,j])for i in range(4)},strict_safe={labels[i]:bool(strict[i,j])for i in range(4)},strict_safe_oracle_score=float(strict_oracle[j]),strict_safe_oracle_winner=None if not has_strict[j]else labels[int(winner[j])]),sort_keys=True)+'\n')
npz=OUT/'verified_metrics.npz';np.savez_compressed(npz,tokens=np.asarray(tokens),labels=np.asarray(labels),metric_names=np.asarray(KEYS),metrics=values,strict=strict)
for r in loaded:
 p=OUT/(r[0]+'_scene_receipt_seals.jsonl')
 with p.open('w')as f:
  for row in r[5]:f.write(json.dumps(row,sort_keys=True)+'\n')
result=dict(schema='world_four_observed_policy_strictsafe_population_oracle_v1',status='COMPLETE_FULL12146_ALL_FOUR_PREDICTIONS_VERIFIED',created_unix=time.time(),population=n,physical_logs=44,policies=rows,common=dict(all_four_canonical_zero=int(np.all(scores==0,axis=0).sum()),all_four_below05=int(np.all(scores<.5,axis=0).sum()),no_strict_safe_policy=int((~has_strict).sum()),any_strict_safe_policy=int(has_strict.sum()),strictsafe_positive_repair_of_control_zero=int((basezero&has_strict&(strict_oracle>0)).sum()),strictsafe_positive_repair_union_among_control_non_strict=int((~strict[0]&has_strict&(strict_oracle>0)).sum())),oracle=dict(strict_safe_definition='NC=DAC=TTC=Comfort=1 exactly',strict_safe_best_PDMS_zero_if_no_strict_candidate=100*float(strict_oracle.mean()),baseline_preserving_strict_safe_improvements_PDMS=100*float(preserve.mean()),baseline_preserving_gain_PDMS=100*float((preserve-base).mean()),baseline_preserving_improved_scenes=int((preserve>base+1e-12).sum()),unconstrained_best_PDMS_diagnostic_only=100*float(ordinary.mean()),no_new_trajectory_generated=True,not_a_deployable_selector=True,selected_using_future_metric_labels=True,not_an_achievable_gain_bound_for_training=True),sources=[r[4]for r in loaded],artifacts=dict(ledger=dict(path=str(ledger),sha256=sha(ledger)),metrics=dict(path=str(npz),sha256=sha(npz))),interpretation_zh=['这是四个已观察策略的逐场景未来指标oracle，只表明该有限策略池有互补见证；不等于可部署选路器、可学习收益或所有物理可解场景上界。','四策略来自不同更新数，不作同预算算法排名；未解决组不等于无安全解。','严格安全oracle无候选时记0；保留Control的改良oracle可能保留原非strict轨迹，不能称最终全场景strict。','NAVTEST token只写诊断ledger，绝不构造TRAIN列表、teacherbank或采样权重。'],scope=dict(new_scores=0,new_bootstrap=0,model_or_image_or_weight_loads=0,GPU=0,training_or_live_source_or_registry_actions=0))
result['sha256']=fp(result)
for p in[OUT/'SUMMARY.json',Path(__file__).parent/'FOUR_ROUTE_STRICTSAFE_ORACLE_20261101.json']:p.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(dict(path=str(OUT/'SUMMARY.json'),oracle=result['oracle'],common=result['common'],policies=rows),ensure_ascii=False,indent=2))

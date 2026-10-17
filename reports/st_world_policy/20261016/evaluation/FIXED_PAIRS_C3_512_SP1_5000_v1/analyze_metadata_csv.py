"""Authorized bounded CSV+metadata fallback; no native/cache/payload loading."""
import csv,hashlib,json,os,sys,time,subprocess
from pathlib import Path
from datetime import datetime,timezone
os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:2])
RUNTIME=Path('/mnt/project/VLA-Drive-st-flow-context-runtime-v3-20261017')
sys.path.insert(0,str(RUNTIME))
import numpy as np
from tools.st_world_policy import evaluate_campaign as e
from tools.st_world_policy.score_world import read_complete_export
ROOT=Path('/mnt/project/st-world-policy-20261016')
PREP=Path('/mnt/project/VLA-Drive-st-evaluation-agent-20261016/reports/st_world_policy/20261016/evaluation')
HERE=Path(__file__).resolve().parent
START=time.monotonic();READS={}
SOURCE='6714b1e9fb2e0c8059b90b8b4d48cd40796fe345'
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=RUNTIME,text=True).strip()==SOURCE
assert e.file_sha256(RUNTIME/'tools/st_world_policy/evaluate_campaign.py')=='3ff851dcabd131d94fc22dc700802a68b4a6efd6fa0f345850ed2d9a8b8628a3'

def pin(p):return dict(path=str(p),sha256=e.file_sha256(p))
def read(p):
 p=Path(p);assert p.suffix in ('.json','.csv') and p.stat().st_size<=32<<20
 READS[str(p)]=p.stat().st_size
 return json.loads(p.read_text())
planpath=PREP/'full_dev_C3_execution_v1/EXECUTION_PLAN.json';plan=read(planpath)
policy=read(PREP/'full_dev_C3_execution_v1/POLICY.json');assert policy['execution_plan']==pin(planpath)
base=read(PREP/'full_dev_execution_v1/POLICY.json')
assert e.pinned_json(policy['parent_policy'])==base
assert policy['statistics']==plan['statistics']
for k in ('population_manifest','inference','score','dedup_resume','resource_default'):assert policy[k]==base[k]
selected=e.load_population(plan,policy['population_manifest']['path'],'FORMAL');expected=e.keyed(selected['scenes'])
assert len(expected)==1696 and len({r['log'] for r in expected.values()})==16
contexts=e.keyed(e.pinned_json(base['score']['metric_index']))

def load(arm,step,stage):
 job=f'{arm}_step{step:06d}_dev1696_seed42_v1';root=ROOT/'formal/evaluation/jobs'/job
 export,scenes=read_complete_export(root/'export');e._formal_binding(plan,arm,export,stage)
 assert e.keyed(scenes)==expected and export['observations']['manifest_sha256']==policy['population_manifest']['sha256']
 assert export['observations']['current_source']==selected['source_student_identity'] and export['observations'].get('history') is None
 assert export['seed']==42 and export['solver']=='original_Euler_10' and export['precision']=='FP32_master_eval_TF32_off'
 assert export['scene_noise']=='foresight-action-v1' and export['checkpoint']['base_binding']==plan['base_binding']['binding_sha256']
 assert e.fingerprint({k:v for k,v in export['checkpoint'].items() if k!='sha256'})==export['checkpoint']['sha256']
 score=read(root/'official_score/IDENTITY.json');complete=read(root/'official_score/COMPLETE.json')
 assert score['schema']=='st_world_official_evaluation_v1' and e.fingerprint({k:v for k,v in score.items() if k!='identity'})==score['identity']
 assert score['export_identity']==export['identity'] and score['export_file_sha256']==e.file_sha256(root/'export/IDENTITY.json') and score['checkpoint']==export['checkpoint']
 assert e.fingerprint({k:v for k,v in score['protocol'].items() if k!='sha256'})==score['protocol']['sha256']==plan['score_protocol_sha256']==score['spec']['score_protocol_hash']
 assert score['metric_index']==base['score']['metric_index']
 assert complete['status']=='COMPLETE' and complete['identity']==score['identity'] and complete['requested_scenes']==complete['valid_scored']==1696
 assert complete['invalid']==complete['unresolved']==0 and complete['official_single_candidate_reference'] is True and complete['deployment_candidates_per_scene']==1
 assert complete['checkpoint_scope']=='FORMAL' and complete['diagnostic_weights'] is False and complete['score_units']=='0..1'
 csvpath=root/'official_score/SCENES.csv';assert e.file_sha256(csvpath)==complete['scenes_csv_sha256']
 with csvpath.open(newline='') as f:
  reader=csv.DictReader(f);assert reader.fieldnames==['scene_id','source_log',*e.METRICS];rows=e.keyed(list(reader),'scene_id','source_log')
 assert set(rows)==set(expected)
 for token,row in rows.items():
  assert row['source_log']==expected[token]['log']
  for k in e.METRICS:row[k]=float(row[k]);assert np.isfinite(row[k]) and 0<=row[k]<=1
 requests=e.keyed(score['request_identities'],'scene_id','source_log');assert set(requests)==set(expected)
 runtime=e.fingerprint({k:export[k] for k in ('environment','seed','solver','precision','scene_noise')})
 for token,r in requests.items():
  context=contexts[token]
  assert r['source_log']==context['log']==expected[token]['log'] and r['metric_cache_sha256']==context['sha256']
  assert str(Path(context['cache_path']).resolve())==r['metric_cache_path'] and r['candidate_id']=='single_policy' and r['generator_kind']=='policy'
  assert r['policy_version']==export['checkpoint']['sha256'] and r['generator_checkpoint_sha256']==export['checkpoint']['native_checkpoint_sha256']
  assert r['generator_source_hash']==e.fingerprint(export['checkpoint']['deployment_source_contract']) and r['inference_runtime_hash']==runtime
  assert r['score_protocol_hash']==plan['score_protocol_sha256'] and r['campaign_id']==plan['campaign_id']
 for k in e.METRICS:assert abs(np.mean([r[k] for r in rows.values()])-complete['aggregate']['metrics'][k])<1e-12
 assert sum(r['score']<=plan['statistics']['zero_epsilon'] for r in rows.values())==complete['aggregate']['zero_count']
 proof=dict(job_id=job,score_complete=pin(root/'official_score/COMPLETE.json'),scenes_csv=pin(csvpath),score_identity=score['identity'],export_identity=export['identity'],checkpoint=export['checkpoint'],aggregate=complete['aggregate'],formal_registration=plan['formal_runs'][arm]['registration'])
 return rows,requests,export,proof

results=[]
for control,treatment,step,stage in [('C0','C3',512,'DIAGNOSTIC_512'),('SP0','SP1',5000,'SCREENING')]:
 a,ar,ae,ap=load(control,step,stage);b,br,be,bp=load(treatment,step,stage)
 assert ae['observations']==be['observations'] and ae['environment']==be['environment']
 assert all(ar[k]['metric_cache_sha256']==br[k]['metric_cache_sha256'] for k in a)
 tokens=sorted(a);logs=[a[t]['source_log'] for t in tokens]
 arrays=[{k:np.array([rows[t][k] for t in tokens]) for k in e.METRICS} for rows in (a,b)]
 stats,flags=e.paired_statistics(*arrays,logs,plan['statistics'])
 delta=arrays[1]['score']-arrays[0]['score'];gain=np.maximum(delta,0);loss=np.maximum(-delta,0)
 positive=float(gain.sum());negative=float(loss.sum());by_log=[]
 for log in sorted(set(logs)):
  mask=np.array(logs)==log;by_log.append(dict(log=log,scenes=int(mask.sum()),gross_gain=float(gain[mask].sum()),gross_loss=float(loss[mask].sum()),net_sum=float(delta[mask].sum())))
 transitions={k:dict(improved=int((arrays[1][k]>arrays[0][k]+1e-8).sum()),degraded=int((arrays[1][k]<arrays[0][k]-1e-8).sum()),equal=int((np.abs(arrays[1][k]-arrays[0][k])<=1e-8).sum())) for k in ['no_at_fault_collisions','drivable_area_compliance','time_to_collision_within_bound','driving_direction_compliance','comfort']}
 concentration=dict(definition='Descriptive only: rank positive scene deltas; log gain sums positive scene deltas before within-log cancellation. Shares use total gross positive gain, never tiny net gain.',positive_scene_count=int((delta>1e-8).sum()),negative_scene_count=int((delta< -1e-8).sum()),gross_gain_sum=positive,gross_loss_sum=negative,net_sum=float(delta.sum()),top_scene_positive_gain_shares={str(k):float(np.sort(gain)[-k:].sum()/positive) if positive else None for k in (1,10,50)},top_log_positive_gain_shares={str(k):sum(sorted([r['gross_gain'] for r in by_log],reverse=True)[:k])/positive if positive else None for k in (1,3)},by_log=by_log)
 results.append(dict(comparison=f'{treatment}_minus_{control}',stage=stage,control=ap,treatment=bp,statistics=stats,safety_transitions=transitions,gain_concentration=concentration))
result=dict(schema='st_fixed_pair_csv_metadata_analysis_v1',status='CSV_METADATA_VERIFIED_PAYLOAD_REVALIDATION_NOT_RUN',created_utc=datetime.now(timezone.utc).isoformat(),source_sha=SOURCE,source_file=pin(RUNTIME/'tools/st_world_policy/evaluate_campaign.py'),plan=pin(planpath),policy=pin(PREP/'full_dev_C3_execution_v1/POLICY.json'),population=policy['population_manifest'],statistics_definitions=plan['statistics'],results=results,requested=1696,paired_valid=1696,logs=16,invalid=0,unresolved=0,bootstrap=2000,bootstrap_seed=20261017,scoring_calls=0,GPU_used=False,payload_bytes_read=0,estimated_full_load_run_payload_read_bytes=370059264,payload_read_cap_bytes=128<<20,verification_boundary='Uses frozen load_population, _formal_binding, read_complete_export and paired_statistics. CSV SHA, full scene/log population, actual approved checkpoint lineage, export prediction receipt hashes, score/request protocol and fixed context metadata verified. load_run payload phase not executed: four ledgers/predictions require about353MiB hash+load, above128MiB cap. Saved scoring arrays, trajectory values and ego payloads not read/revalidated; raw progress/speed, parking proxy and ADE/FDE unavailable this bounded audit.',no_training_stop_or_winner_selection=True,elapsed_seconds=time.monotonic()-START)
with (HERE/'RESULTS.json').open('x') as f:f.write(json.dumps(result,indent=2,sort_keys=True)+'\n')
print(json.dumps([dict(comparison=r['comparison'],score=r['statistics']['metrics']['score'],events={k:v['count'] for k,v in r['statistics']['events'].items() if 'count' in v},safety=r['safety_transitions'],concentration={k:v for k,v in r['gain_concentration'].items() if k!='by_log'}) for r in results],indent=2))

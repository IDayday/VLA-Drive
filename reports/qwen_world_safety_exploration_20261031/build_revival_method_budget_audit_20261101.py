"""Read-only complete-summary/budget audit; never reads weights/images or creates scores."""
import hashlib,json,math,os,time
from pathlib import Path
E=Path('/mnt/project/qwen-world-safety-exploration-20261031');R=Path('/mnt/project/qwen-world-rl-optimization-20261030');N=Path('/mnt/project/qwen-world-abcd-rl-best-20261031');W=Path('/mnt/project/VLA-Drive-qwen-world-safety-exploration-20261031');OUT=E/'analysis/REVIVAL_METHOD_BUDGET_AUDIT_20261101';REPORT=W/'reports/qwen_world_safety_exploration_20261031'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def fp(d):return hashlib.sha256(json.dumps(d,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def atomic(p,d):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.tmp'+str(os.getpid()));t.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n');os.replace(t,p)
def evidence(p):return dict(path=str(p),file_sha256=sha(p))
reg=read(R/'RUNS.json');reviewpaths=[R/'analysis'/n for n in ['MATCHED_1K_PPO_CONTRASTS.json','MATCHED_12K_PPO_CONTRASTS.json','MATCHED_20K_PPO_CONTRASTS.json','NEW_PILOT_MATCHED_REVIEW.json','NOISE03_MATCHED_REVIEW.json','LASTDET_MATCHED_REVIEW.json','PREFERENCE_MIXTURE_MATCHED_REVIEW.json','RESCUE_PAIR_MATCHED_REVIEW.json','LR_STUDY_MATCHED_REVIEW_20261031.json','SAFETY_FRESH_PAIR_MATCHED_REVIEW_20261031.json','FPO_FULLTIME_MATCHED_REVIEW_20261031.json','SAFETY_FEASIBLE_SCORE_MATCHED_REVIEW_20261031.json']]
contrasts=[]
for p in reviewpaths:
 if not p.exists():continue
 d=read(p)
 for label,x in d.get('contrasts',{}).items():
  if not isinstance(x,dict)or x.get('delta_PDMS')is None:continue
  contrasts.append(dict(label=label,target=x.get('target'),baseline=x.get('baseline'),target_PDMS=x.get('target_PDMS'),baseline_PDMS=x.get('baseline_PDMS'),delta_PDMS=x['delta_PDMS'],CI95_PDMS=x.get('CI95_PDMS'),zeros=x.get('zeros',x.get('paired',{})),source=evidence(p),json_pointer='/contrasts/'+label))
classified={
 'PPO_HARD':('LOW_PRIORITY_OLD_RECIPE_REPEAT_EXPOSURE_RISK','10k以上非短试验；Hard仅15171eligible，重复困难场景且matched safety回退，不宜只归因欠训练。'),
 'PPO_FULL':('LONG_RUN_BASELINE_NOT_EARLY_REJECTED','已真实约48.9k，最佳10k后多个完整点未持续提升；不是1k早停。'),
 'PPO_ADAPTIVE':('REDESIGN_BEFORE_RETRY','已约27.5k，固定反馈面板曝光集中与额外3072交互支持分布特化；不等于所有课程无效。'),
 'CISPO_FULL':('LONGER_RECIPE_NO_CLEAR_MATCHED_GAIN','20k同预算+.00393且CI跨0；已有长程，不列本轮优先复活。'),
 'NFT_FULL':('DEPRIORITIZE_OBJECTIVE_NOT_ONLY_BUDGET','约28.6k仍未超IL/控制，属于具体目标较强负证据，不外推所有flow算法。'),
 'PPO_HELDOUT':('NOT_RUN_NOT_FAILED','仅登记，无完整正式评分；不能把未启动说成算法失败。'),
 'PPO_BATCHSTD':('LONGER_RECIPE_NO_CLEAR_MATCHED_GAIN','20k+.01525且CI跨0；不能说所有归一化无效，但不是仅1k没训够。'),
 'PPO_SAFE_COMFORT':('LONG_RUN_SMALL_GAIN_THEN_PLATEAU','20k相对FULL+.15343 CI完全正，已近50k；有效小增量而非早判无效，SafetyEP提供更强起点。'),
 'PPO_ANCHOR40':('LONGER_RECIPE_NO_SUSTAINED_GAIN','10k小正差、20k回退，真实约33.6k；不能把anchor全局压制当既定原因。'),
 'PPO_MEMORYLR20':('INCONCLUSIVE_AT_10K_LOW_PRIORITY','10k−.16067 CI跨0；有实质预算仍未获益，单因子未来可审查，不列首四。'),
 'PPO_SAFETY_EP':('REFERENCE_SUCCESS_NOT_INVALID','10k对FULL+.33040 CI[.04876,.62257]；后续20–30k有退化，固定/轮换零与高分损失并存，非从未学会。'),
 'PPO_PREF_REPLAY':('ONLY_5K_INCONCLUSIVE','5k对FULL−.03614 CI跨0，只有9.68%主流覆盖；旧偏好混合不是已否定全部replay。'),
 'PPO_PREF_CFM':('SPECIFIC_AUX_NEGATIVE_AT_5K','5k对Replay−.14628 CI完全负，需先解释teacher/distribution/强度，不是盲延时必定好。'),
 'PPO_NOISE03':('SHORT_NEGATIVE_TREND_PARTLY_RECOVERING','1k−1.43191到5k−.43958 CI跨0；有学习恢复趋势但采样改动有机制风险，不能说长期无效。'),
 'PPO_LASTDET':('STRONG_RECIPE_NEGATIVE_AT_5K','5k−1.86207 CI完全负；W2窗口/末步处理共同变化，不仅noise减少，不优先精确复活。'),
 'PPO_RESCUE_REPLAY':('WARMSTART_PHASE_CONFOUNDED_BAD_AT_5K','Safety10k weights-only freshAdam/RNG+重复lowhigh数据；phase5k−3.333对原起点，非freshIL同预算，不可混用证据。'),
 'PPO_RESCUE_CFM':('WARMSTART_PHASE_CONFOUNDED_BAD_AT_5K','phase5k对原Safety10k−3.509且对Replay−.176CI跨0；分布/状态重置有共同问题，不能全怪CFM。'),
 'PPO_SAFETY_LRLOW':('USER_STOPPED_INCONCLUSIVE_FORK','原10k exact-native fork，12k对同预算+.15159CI正但14k−.02954CI跨0；停止来自用户聚焦，非长期无效。'),
 'PPO_SAFETY_LRCOS':('USER_STOPPED_INCONCLUSIVE_FORK','从Safety10k native fork到14k，原局部10→20cos；不同于新fresh固定50k调度，用户终止不记科学失败。'),
 'PPO_SAFETY_DATA':('USER_STOPPED_INCONCLUSIVE_FORK','nativefork混合流到14k，12k小正14k小负且CI跨0；不能把用户终止当长期无效。'),
 'PPO_SAFETY_LRDATA':('USER_STOPPED_INCONCLUSIVE_FORK','12k对同预算+.18099CI跨0、对LRLOW差+.02940CI跨0；未证实交互，不自动复活。'),
 'PPO_SAFETY_SUCCESS_MEMORY':('SLOW_LEARNING_NO_CLEAR_10K_GAIN','1k/5k显著小负、10k+.02627CI跨0；证明早期负不等于永远负，但zero净新增19，不能称最终可靠提升。'),
 'PPO_SAFETY_FPO':('STRONG_SPECIFIC_10K_NEGATIVE','1k−.12001、5k−1.02799、10k−2.60511，后两CI完全负；持续恶化支持不原样复活，不泛指所有FPO。'),
 'PPO_FPO_FULLTIME':('SPECIFIC_5K_NEGATIVE','仅sigma上界扩大，5k比原FPO−.25738CI完全负；覆盖节点增加未修复目标/方向问题。'),
 'PPO_SAFETY_FEASIBLE_SCORE':('SPECIFIC_5K_NEGATIVE_MECHANISM_DIFFERS_SAT','5k−.58475CI完全负；对全部feasible稀释EP5/7与窄saturated-only不是同方法，不能互相冒名。')}
rows=[];summary_errors=[];summary_paths=list(Path('/mnt/project/qwen-world-postcot-20261025/rl_optimization_20261030/evaluations').glob('*/SUMMARY.json'))
for run in reg['runs']:
 arm=run['id'];pts=[]
 for p in summary_paths:
  if not p.parent.name.startswith(arm+'_u')or not p.parent.name.endswith('_NAVTEST'):continue
  d=read(p)
  if d.get('status')!='COMPLETE':continue
  b=dict(d);seal=b.pop('sha256',None)
  if d.get('scenes')!=12146 or d.get('expected_scenes')!=12146 or d.get('invalid_scenes')!=0 or d.get('cohort')!='NAVTEST' or fp(b)!=seal:summary_errors.append(dict(path=str(p),reason='complete summary contract mismatch'));continue
  pts.append(dict(updates=d['RL_updates'],PDMS=d['PDMS'],means=d['means'],scene_count=12146,identity_sha256=d['evaluation_identity_sha256'],**evidence(p)))
 pts.sort(key=lambda v:v['updates']);cp=Path(run['last_checkpoint'])if run.get('last_checkpoint')else None;native=None
 if cp is not None and(cp/'COMPLETE').exists():
  z=read(cp/'COMPLETE');native=dict(checkpoint=str(cp),updates=z.get('updates'),next_inner=z.get('next_inner'),world=z.get('world'),COMPLETE=evidence(cp/'COMPLETE'))
 completed=run.get('completed_updates');initial=run.get('initial_updates',10000 if '_LR' in arm or arm=='PPO_SAFETY_DATA'else 0)
 if arm in ('PPO_MEMORYLR20',):initial=0
 if arm in ('PPO_RESCUE_REPLAY','PPO_RESCUE_CFM'):initial=0
 population=run.get('data_scenes');policy=run.get('data_policy');nominal=2*max(0,(completed or 0)-initial)
 exposure=dict(initial_global_updates=initial,completed_updates=completed,approx_main_scene_exposures=nominal,formula='2*(global_updates-initial_global_updates), only complete4inner boundaries exact; pendinginner may have committed one extra8scene collection',not_a_measured_unique_count=True,eligible_population=population,main_stream_policy=policy,full_population_ratio_if_uniform_no_repeat=nominal/population if population and policy=='uniform_full'else None)
 assessment,why=classified.get(arm,('REVIEW_NEEDED','No inferred negative conclusion'))
 selected=[c for c in contrasts if (c.get('target')or'').startswith(arm+'_')or c['label'].startswith(arm+'_')]
 rows.append(dict(id=arm,mechanism=run.get('hypothesis'),algorithm=run.get('algorithm'),registered_status=run['status'],requested_target_updates=run.get('target_updates'),research_budget_updates=run.get('research_budget_updates'),stage_budget_updates=run.get('stage_budget_updates'),completed_updates=completed,exposure=exposure,latest_native=native,full_NAVTEST_points=pts,matched_evidence=selected,assessment=assessment,reason_zh=why,original_registry=evidence(R/'RUNS.json')))
old=read(E/'analysis/STANDARD_LONG_RUN_DESIGN_20261101.json')['old_complete_evidence']
es=[]
roles={'SAFETY_G16':(1,'RETRY_G16','1k正均值且CI跨0；同scene得到更多候选，不等于同算力提升。'), 'SAFETY_DAC_DENSE_G8':(2,'RETRY_DENSE','窄DAC全failtie才激活，只有3k覆盖5.81%，配对区间跨0。'), 'SAFETY_SATURATED_G8':(3,'RETRY_SATURATED','成功饱和但Comfort有差异的窄信用；3k接近负但区间仍跨0，与失败q方案不同。'), 'SAFETY_ACTIVE_DATA_G8':(4,'RETRY_EXACT_ACTIVE_WITH_RETENTION_RISK','原实现1k整体显著负；必须单独审查重复/高分遗忘，不可只按PG更活跃当有效。')}
for x in old['curves']:
 arm=x['arm'];rank,action,why=roles.get(arm,(None,'ALREADY_COVERED_BY_F_OR_NO_EXTRA_RETRY','F已含control/inversion/组合/endpoint低LR；Protected无可信收益且更大信用偏置，不在本轮。'))
 es.append(dict(id=arm,full_NAVTEST_points=x['points'],last_complete_evaluation_updates=x['last_updates'],main_uniform_scene_exposures=2*x['last_updates'],eligible_population=103288,uniform_firstpass_coverage_percent=200*x['last_updates']/103288 if arm!='SAFETY_ACTIVE_DATA_G8'else None,group_size=16 if arm=='SAFETY_G16'else 8,candidate_cost_relative_to_G8=2 if arm=='SAFETY_G16'else 1,priority=rank,action=action,reason_zh=why,active_actual_exposure=dict(total=2000,unique=1520,priority_repeat=480,same_rank_collection_as_control=88)if arm=='SAFETY_ACTIVE_DATA_G8'else None))
ereg={r['id']:r for r in read(E/'RUNS.json')['runs']}
for row in es:
 run=ereg[row['id']];row['registered_runtime_cap']=run['target_updates'];latest_path=Path(run['root'])/'latest.json'
 if latest_path.exists():
  latest=read(latest_path);cp=Path(latest['checkpoint']);seal=read(cp/'COMPLETE');row['actual_latest_native']=dict(updates=latest['updates'],checkpoint=str(cp),next_inner=seal.get('next_inner'),world=seal.get('world'),latest_file_sha256=sha(latest_path),COMPLETE_file_sha256=sha(cp/'COMPLETE'))
result=dict(schema='world_legacy_R_and_E_revival_evidence_audit_v1',status='COMPLETE_READ_ONLY_BUDGET_AND_FULL_SUMMARY_AUDIT',created_unix=time.time(),question_zh='哪些只是短期或欠曝光，哪些有具体目标/分布负证据，哪些已经长训而不能再称早判。',main_conclusions_zh=['优先G16、Dense、Saturated，Active作为第4精确复验并设置覆盖/高分保持诊断。首四旧预算只1k或3k，均不是充分完整数据epoch。','不把所有负面都归因欠训练：旧FPO10k持续显著恶化，LASTDET5k显著负，原反馈ADAPT27k固定面板特化；原样加时不是唯一解释。','SuccessMemory早期显著负在10k转小正但CI仍跨0，说明需要长学习曲线，也说明延时不自动解决zero新增。','原FULL/SafeComfort已接近50k、CISPO/BatchStd已有20k配对；它们不是1k被错误淘汰。用户停掉的fork和未启动Heldout分别标记，不冒充科学失败。','F四个当前长训保持独立；F Combined392的validator浮点微差是执行/验证故障，应修后续训，不记算法负收益。ABCD与HUBER明确永久终止，不属于复活候选。'],E_arms=es,R_arms=rows,E_complete_points=old['snapshot_complete_NAVTEST_summaries'],E_scene_scores=old['total_existing_scene_scores'],R_complete_points=sum(len(r['full_NAVTEST_points'])for r in rows),R_summary_contract_errors=summary_errors,full_summary_rule=dict(scenes=12146,invalid_scenes=0,status='COMPLETE',sealed_sha256=True,partial_means_included=False),new_revival=dict(root=str(E/'revival_training_20261101'),fresh_parent='same original IL70 /fresh Adam/RNG42/update0',scheduler_horizon=50000,warmup=1000,floor=.1,initial_runtime_cap=10000,healthy_default_continuation=20000,external_control=str(E/'standard_training_20261101'),control_evaluation_read_only=True,old_registry_or_ckpt_not_resumed=True),limits_zh=['单训练seed，44physical-logCI只反映该人口场景/日志差异，不是跨seed不确定性。','旧NAVTEST已用于方法选择，全部披露Test-selection-used；其token永不进入训练选择。','新warmup1000/cos50k与旧warm100/constant不同；相对新FControl的同预算差可归于方法，不能把新旧分数差单归于多训练。','2updatesper_scene关系只在当前world8/4inner主流成立；G16候选数翻倍，Active/ADAPT/teacheraux有额外重复，曝光不等于unique。','这里只读取注册/封存summary/已有review/COMPLETE元数据，不再次读scene、图像或权重，不新bootstrap。'],source_reviews=[evidence(p)for p in reviewpaths if p.exists()],scope=dict(GPU=0,new_scores=0,model_loads=0,bootstrap=0,live_state_or_registry_mutations=0,commit=False))
result['sha256']=fp(result);atomic(OUT/'SUMMARY.json',result);atomic(REPORT/'REVIVAL_METHOD_BUDGET_AUDIT_20261101.json',result)
print(json.dumps(dict(path=str(OUT/'SUMMARY.json'),sha256=sha(OUT/'SUMMARY.json'),E_points=result['E_complete_points'],R_points=result['R_complete_points'],R_arms=len(rows),summary_errors=len(summary_errors)),indent=2))

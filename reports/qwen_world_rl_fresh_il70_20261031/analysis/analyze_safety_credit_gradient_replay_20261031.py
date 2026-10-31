"""Read finite completed P0 credit-replay JSON; no forward/scoring/optimizer."""
from pathlib import Path
import collections,hashlib,json,math,os,time,uuid
R=Path('/mnt/project/qwen-world-rl-optimization-20261030');A=R/'analysis';O=A/'SAFETY_EP_NATIVE28K_CREDIT_GRADIENT_REPLAY_20261031'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def atomic(p,v):
 p=Path(p);tmp=p.with_name(p.name+'.writing-'+str(os.getpid()));tmp.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');os.replace(tmp,p)
s=read(O/'SUMMARY.json');protocol=read(O/'PROTOCOL.json');assert s['status']=='COMPLETE' and s['world']==8 and s['native_updates']==28000
assert s['newly_generated_candidates']==s['new_reward_scores']==s['optimizer_steps']==0
cases=[read(O/f'case{i}.json') for i in range(3)]
assert all(c['world']==8 and len(c['groups_by_rank'])==8 for c in cases)
attempts=[]
for i in (1,2,3):
 j=R/f'jobs/SAFETY_EP_CREDIT_GRADIENT_REPLAY/native028000_attempt{i:03}';r=read(j/'RESULT.json');launch=read(j/'LAUNCH.json')
 attempts.append(dict(attempt=i,job=str(j),host=r['host'],lease=launch['lease'],returncode=r['returncode'],child_not_started=r.get('child_not_started',False),reason=r.get('reason'),result_sha256=sha(j/'RESULT.json'),launch_sha256=sha(j/'LAUNCH.json')))
assert attempts[-1]['returncode']==0
allgroups=[g for c in cases for g in c['groups_by_rank']]
counts=dict(groups=24,candidates=192,feasible=0,positive_PPO_unsafe=0,positive_Safety_unsafe=0,negative_PPO_feasible=0,negative_Safety_feasible=0,opposite_nonzero_credit_signs=0)
for g in allgroups:
 for f,p,t in zip(g['feasible'],g['canonical_PPO_advantages'],g['Safety_advantages']):
  counts['feasible']+=bool(f);counts['positive_PPO_unsafe']+=(not f and p>0);counts['positive_Safety_unsafe']+=(not f and t>0)
  counts['negative_PPO_feasible']+=(f and p<0);counts['negative_Safety_feasible']+=(f and t<0);counts['opposite_nonzero_credit_signs']+=(p*t<0)
counts.update(group_classes=dict(collections.Counter(g['classification'] for g in allgroups)),canonical_PPO_PG0_groups=sum(g['canonical_PPO_PG0'] for g in allgroups),Safety_PG0_groups=sum(g['Safety_PG0'] for g in allgroups))
case_metrics=[]
for c in cases:
 scopes={}
 for name,d in c['scopes'].items():
  n=d['global_mean_gradient_norm'];cos=d['cosine']
  def cosine(a,b):return cos['__'.join(sorted((a,b)))]
  def dot(a,b):return n[a]*n[b]*(cosine(a,b) or 0.)
  residual=max(0.,n['safety_pg']**2+n['safety_safe']**2+n['safety_EP']**2+2*dot('safety_safe','safety_EP')-2*dot('safety_pg','safety_safe')-2*dot('safety_pg','safety_EP'))**.5
  scopes[name]=dict(numel=d['numel'],norms=n,cosine=cos,
   Safety_to_PPO_PG_norm_ratio=n['safety_pg']/n['ppo_pg'],anchor_to_Safety_PG_norm_ratio=n['anchor']/n['safety_pg'],anchor_to_PPO_PG_norm_ratio=n['anchor']/n['ppo_pg'],
   EP_to_safe_norm_ratio=n['safety_EP']/n['safety_safe'],safe_EP_cosine=cosine('safety_safe','safety_EP'),
   PPO_Safety_PG_cosine=cosine('ppo_pg','safety_pg'),PPO_Safety_PG_angle_degrees=math.degrees(math.acos(max(-1.,min(1.,cosine('ppo_pg','safety_pg'))))),
   candidate_update_norms=d['candidate_update_norms'],candidate_update_cosine=d['candidate_update_cosine'],
   BF16_separate_attribution_sum_residual_norm_from_Gram=residual,BF16_separate_attribution_sum_residual_relative_to_Safety=residual/n['safety_pg'],
   rank_alignment_ratio=d['rank_alignment_ratio'])
 diag=c['forward_diagnostics_by_rank']
 case_metrics.append(dict(case=c['case'],scope_metrics=scopes,
   all8_current_old_logp_bit_equal=all(x['current_old_logp_bit_equal'] for x in diag),
   max_current_old_logp_abs=max(x['current_old_logp_max_abs'] for x in diag),
   total_advantage_cap_hits=sum(x['total_advantage_cap_hits'] for x in diag),
   actual_total_branch_blocked_candidate_steps=sum(x['shared_total_branch_blocked_candidate_steps'] for x in diag),
   canonical_PPO_blocked_candidate_steps=sum(x['canonical_ppo_blocked_candidate_steps'] for x in diag),
   max_abs_Safety_PG_loss_scalar=max(abs(x['scalars']['safety_pg']) for x in diag),
   source_case=str(O/f"case{c['case']}.json"),source_case_sha256=sha(O/f"case{c['case']}.json")))
comfort=[]
# Only two small, already-scored originals are opened. No rescoring or Qwen.
import torch
P=Path(protocol['source_probe'])
for c in cases:
 for rank,g in enumerate(c['groups_by_rank']):
  if g['Safety_PG0'] and not g['canonical_PPO_PG0']:
   original=P/f"case{c['case']}_rank{rank}.pt";row=torch.load(original,map_location='cpu',weights_only=False)
   values=[float(r['metrics']['comfort']) for r in row['records']]
   assert g['classification']=='success_EP1_saturation' and len(set(values))>1
   local=c['local_gradient_norms_by_rank'][rank]
   comfort.append(dict(case=c['case'],rank=rank,token=g['token'],log=g['log'],metrics_NC_DAC_TTC_EP=g['components_NC_DAC_TTC_EP'],Comfort=values,rewards=g['rewards'],
    canonical_PPO_advantages=g['canonical_PPO_advantages'],Safety_advantages=g['Safety_advantages'],local_component_norms=local,
    unaveraged_local_PPO_norms=local['ppo_pg'],equal_rank_1_over8_contribution_norms={k:v/8 for k,v in local['ppo_pg'].items()},
    original_scored_rollout=str(original),original_scored_rollout_sha256=sha(original),
    conclusion='Canonical PPO has measured nonzero action and Memory PG; Safety PG is exactly zero, while its IL anchor remains active. No group-local gradient vector/cosine was stored, so its exact effect on the global rotation is not identifiable.'))
assert len(comfort)==2
assert all(v==0 for z in comfort for v in z['local_component_norms']['safety_pg'].values())
assert all(z['unaveraged_local_PPO_norms']['action']>0 and z['unaveraged_local_PPO_norms']['memory']>0 for z in comfort)
result=dict(schema='world_Safety_PPO_credit_gradient_replay_interpretation_v1',status='COMPLETE',at_unix=time.time(),scientific_status='FINITE_SAME_THETA_GRADIENT_EVIDENCE_NOT_TRAINING_GAIN',
 source=str(R/'runtime/credit_replay_ca942fd'),output=str(O),attempts=attempts,attempt_evidence=attempts[-1],
 provenance=dict(summary=str(O/'SUMMARY.json'),summary_sha256=sha(O/'SUMMARY.json'),protocol=str(O/'PROTOCOL.json'),protocol_sha256=sha(O/'PROTOCOL.json'),
 native_updates=28000,native_files_sha256=s['native_files_sha256'],policy_sha256=s['policy_sha256'],reference='fixed original complete IL70',world=8,groups=24,candidates=192,scopes=['all','action','memory','scene','predictive'],new_generation=0,new_scoring=0,optimizer_steps=0,gradient_measurement_seconds=s['seconds']),
 gradient_results=dict(case_metrics=case_metrics,source_credit_counts=counts,Comfort_variable_Safety_PG0_groups=comfort),
 paired_training_basis=dict(theta='same sealed SafetyEP28000 for both objectives; not trained PPO theta',
 sample='same previously scored SDE G8 chains, same saved old logp, same current legal observations',
 advantages='original canonical-PPO source group centering/raw64 reward population std and tie mask versus actual saved Safety raw64 structured-A std and mask',
 attribution='A_safe receives total A cap gain and actual total-A PPO clip branch; EP objective is exact SafetyPG−safe complement; neither part independently selects clipping',
 beta50_anchor='identical physical IL-reference mean penalty forward; candidate sums are geometries of separately measured vectors, not new optimizer updates',
 scalar_vs_gradient='At ratio1 centered policy loss value is approximately0 even with nonzero PG; PG0 classification uses actual A/mask and measured gradient, never the scalar loss value',
 all24_current_old_logp_bit_equal=all(x['all8_current_old_logp_bit_equal'] for x in case_metrics),first_inner_proxy_ratio=1,
 cap_hits=sum(x['total_advantage_cap_hits'] for x in case_metrics),Safety_blocked=sum(x['actual_total_branch_blocked_candidate_steps'] for x in case_metrics),canonical_PPO_blocked=sum(x['canonical_PPO_blocked_candidate_steps'] for x in case_metrics)),
 mechanism_interpretation=dict(
 changed_credit_changes_parameters='Measured whole-model and Memory PG rotation demonstrates a parameter-level difference, beyond comparing scalar advantages. Magnitude incorporates both credit definition and its actual normalization.',
 safe_EP_conflict='All3 Memory safe/EP cosines are negative. Shared parameters couple otherwise candidate-local feasibility and within-feasible efficiency preferences; this is partial gradient cancellation, not proof EP is unsafe or should be removed.',
 anchor='Weighted anchor norm is15.2–25.9% of Safety whole-model PG here; its alignment varies. These observations reject a blanket anchor-dominates explanation for this bounded window, not all training.',
 Comfort='The two EP1/strict-feasible groups still have Comfort0/1 and PDMS5/6 versus1. Current Safety ignores that distinction and has measured zero PG; canonical PPO has nonzero action/Memory gradients. Anchor remains, with unknown effect on Comfort.',
 no_clip_explanation='All current/old logp are bit-identical, no A-cap hits or actual PPO blocked candidate-steps; this run isolates first-inner credit differences and cannot settle later-inner clipping pathology.'),
 next_research_conditions=[dict(priority=1,hypothesis='Recover the current canonical Comfort distinction inside feasible quality credit',candidate='A_quality=f*[(EP+0.4*Comfort)−mean_feasible(EP+0.4*Comfort)], leaving A_safe/sourcePG/anchor/sampler unchanged',reason='Restores the observed lost teacher-independent canonical quality distinction while keeping original EP coefficient1; this is an unregistered testable future direction, not an approved recipe or proven gain',caution='Changes normalized total-A population and hence scale; must report zero repair and fixed-high retention alongside matched PDMS and safety.'),
 dict(priority=2,hypothesis='Measure whether partial safe/EP Memory cancellation predicts future high-score degradation',candidate='Reuse sparse real gradient probes at a few matched native milestones before deciding to alter EP coefficient',reason='Only3 same-theta cases are currently measured; negative cosine alone does not determine usefulness.'),
 dict(priority=3,hypothesis='Later-inner transition proxy clipping may differ from first-inner credit effects',candidate='If existing saved native pendingrollout is already available, consider separately authorized same-chain later-inner probe',reason='This diagnostic has ratio1 and zero clip blocking; it neither validates nor rules out Flow-DPPO/FPO clipping ideas.' )],
 numerical_limits=dict(BF16_objective_complement='Scalar safety_safe+safety_EP equals original SafetyPG by construction; separately backpropagated BF16 component gradients are not bitexact additive.',
 observed_all_scope_attribution_Gram_residual_relative=[x['scope_metrics']['all']['BF16_separate_attribution_sum_residual_relative_to_Safety'] for x in case_metrics],
 previous_fresh_P1_fused_vs_component_sum_discrepancy='.34% at another probe; current three-case attribution residual is separately measured and must not be silently replaced by that historical value',
 actual_update='Raw equal-rank-mean gradients before Adam preconditioning/global clip; not an actual parameter displacement. All-five component sum has no update meaning and is omitted.',
 pooling='Each case is one real equal8rank mean. No 24-group pooled gradient vector is stored; do not average cosines/norms and present that as a pooled gradient.',
 inference='Fixed first24 TRAIN-panel scenes,3 cases,1 Safety theta, no population CI/cross-training-seed claim or causal final-score attribution.'),
 conclusions_zh=['同θ同rollout真实PG：Safety全模型范数为原PPO的1.18–1.27倍，方向cos0.836–0.972；Memory方向cos0.712–0.969，证实信用变化实际进入表征梯度。',
 '安全项主导SafetyPG方向；EP与安全项的Memory梯度三次均部分冲突，但EP仅为安全项范数的13.6–21.7%，不能称EP全面主导或据此删除。',
 '两个NC/DAC/TTC/EP全1但Comfort有差异的组：Safety真实PG为0，原PPO全模型本地PG范数0.03253和0.03102；当前Safety信用丢失了canonical PDMS中的这部分区分。',
 'beta50 anchor不是此窗口的梯度主导项；不同case可能同向或反向。scalar loss接近0不等于没有PG。',
 '全部新旧logp逐bit相同，cap与clip阻塞全0；这是first-inner信用诊断，不是后续inner clip或最终提分的因果证据。'])
p=A/'SAFETY_CREDIT_GRADIENT_REPLAY_INTERPRETATION_20261031.json';atomic(p,result)
request=R/'refresh_request.json';rid='Safety-credit-gradient-'+uuid.uuid4().hex
atomic(request,dict(id=rid,scientific_result=False,evidence_kind='completed_supplemental_analysis',evidence=str(p),at_unix=time.time()))
print(json.dumps(dict(status='COMPLETE',path=str(p),sha256=sha(p),refresh_id=rid,groups=counts,all_scope_cases=[{k:x['scope_metrics']['all'][k] for k in ('Safety_to_PPO_PG_norm_ratio','PPO_Safety_PG_cosine','anchor_to_Safety_PG_norm_ratio','safe_EP_cosine')} for x in case_metrics]),indent=2))

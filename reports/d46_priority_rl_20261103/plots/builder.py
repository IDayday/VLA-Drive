#!/usr/bin/env python3
"""Rebuild two figures from sealed complete summaries only (no scene rescans)."""
from datetime import datetime,timezone
import hashlib,json,math,pathlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
R=pathlib.Path('/mnt/project/d46-priority-rl-20261103');O=R/'analysis/plots';OPS=R/'operations/EARLY_EVAL_20261103'
ORDER=['D0','D5','D3_FPO','D3_AWM'];LABEL={'D0':'D0','D5':'D5','D3_FPO':'FPO','D3_AWM':'AWM'}
def fp(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def load(p):return json.loads(pathlib.Path(p).read_text())
def ref(p):return dict(path=str(p),file_sha256=hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest())
def sealed(p):
 d=load(p);b=dict(d);h=b.pop('sha256');assert h==fp(b),('seal',p);return d
plan=sealed(OPS/'PLAN.json');state=load(OPS/'STATE.json');assert state['plan_sha256']==plan['sha256']
bp=pathlib.Path(plan['baseline']).parent/'SUMMARY.json';base=sealed(bp)
assert base['status']=='COMPLETE'and base['scenes']==12146 and base['physical_logs']==44 and base['invalid_scenes']==0 and base['seed']==2101
cases={c['arm']:c for c in plan['cases']};rows=[];evidence=[ref(OPS/'PLAN.json'),ref(OPS/'STATE.json'),ref(bp)]
for arm in ORDER:
 c=cases[arm];st=state['cases'][c['id']];assert st['status']=='COMPLETE'and st['completed_shards']==32
 sp=pathlib.Path(st['summary']);ap=pathlib.Path(st['paired_analysis']);s=sealed(sp);a=sealed(ap);evidence.extend([ref(sp),ref(ap)])
 for obj in (s,a):
  assert obj['status']=='COMPLETE'and obj['scenes']==12146 and obj['invalid_scenes']==0 and obj['physical_logs']==44 and not obj['qualification']
  assert obj['seed']==2101 and obj['token_sha256']==base['token_sha256']and obj['protocol_sha256']==base['protocol_sha256']
 assert a['source_summary_sha256']==s['sha256']and s['model_sha256']==c['policy_signature']
 g=a['gain_loss'];p=a['paired'];high=g['fixed_reference_cohorts']['IL46_high_ge_0.9'];zero=g['fixed_reference_cohorts']['IL46_zero'];other=g['fixed_reference_cohorts']['IL46_other']
 assert high['scenes']==8722 and zero['scenes']==479 and sum(v['scenes']for v in(high,zero,other))==12146
 assert math.isclose(g['G_plus']+g['G_minus'],g['net'],abs_tol=1e-12)
 assert math.isclose(g['net'],s['means']['score']-base['means']['score'],abs_tol=1e-12)
 assert math.isclose(sum(v['population_contribution']for v in(high,zero,other)),g['net'],abs_tol=1e-12)
 delta={k:100*(s['means'][k]-base['means'][k])for k in('no_at_fault_collisions','drivable_area_compliance','time_to_collision_within_bound','ego_progress','comfort')}
 rows.append(dict(arm=arm,label=LABEL[arm],updates=c['updates'],scene_exposures=c['scene_exposures'],exposure_epoch=c['exposure_epoch'],
  PDMS=s['PDMS'],delta_PDMS=100*g['net'],reused_existing_44log_CI95_PDMS=[100*x for x in p['CI95']],
  zero_repaired=p['repaired_zero'],new_zero=p['new_zero'],zero_remaining_from_IL46=479-p['repaired_zero'],total_zero=479-p['repaired_zero']+p['new_zero'],
  gross_gain_PDMS=100*g['G_plus'],gross_loss_magnitude_PDMS=-100*g['G_minus'],
  fixed_IL46_high_ge09=dict(scenes=high['scenes'],cohort_mean_delta_pp=100*high['mean_delta'],
   population_net_contribution_PDMS=100*high['population_contribution'],new_zero=high['new_zero'],retained_ge09=g['initial_high_retained_ge_0_9'],retained_fraction=g['initial_high_retained_ge_0_9']/high['scenes']),
  metric_delta_pp=delta,summary_file=str(sp),paired_analysis_file=str(ap)))
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,
 'axes.labelcolor':'#243347','text.color':'#17283c','xtick.color':'#34465a','ytick.color':'#34465a',
 'figure.facecolor':'white','axes.facecolor':'white','svg.fonttype':'none','savefig.facecolor':'white'})
COL={'repair':'#177b68','new':'#d66150','gain':'#2677b8','loss':'#d78b36'}
x=np.arange(4);width=.33
caption=' | '.join(f"{r['label']}: {r['PDMS']:.4f} PDMS, {r['scene_exposures']:,} scenes ({r['updates']:,} updates)"for r in rows)
caption0=' | '.join(f"{r['label']}: {r['PDMS']:.4f} PDMS, {r['scene_exposures']:,} scenes ({r['updates']:,} updates)"for r in rows[:2])
caption1=' | '.join(f"{r['label']}: {r['PDMS']:.4f} PDMS, {r['scene_exposures']:,} scenes ({r['updates']:,} updates)"for r in rows[2:])
# 1) Parallel zero transitions and cancellation of total gain/loss mass.
f,axs=plt.subplots(1,2,figsize=(14.4,6.5),gridspec_kw={'width_ratios':[1,1.08]})
f.subplots_adjust(left=.065,right=.985,top=.79,bottom=.29,wspace=.27)
f.suptitle('D46 early full NAVTEST: repairs compete with new losses',x=.065,y=.965,ha='left',fontsize=19,fontweight='bold')
f.text(.065,.905,'Same IL46 baseline 89.4955 PDMS | 12,146 scenes / 44 physical logs / invalid 0 | Single B1 ODE',fontsize=11.5,color='#536174')
a=axs[0]
bars=[a.bar(x-width/2,[r['zero_repaired']for r in rows],width,color=COL['repair'],label='IL46 zero repaired'),
 a.bar(x+width/2,[r['new_zero']for r in rows],width,color=COL['new'],label='New zero from IL46 nonzero')]
for b in bars:a.bar_label(b,fmt='%d',padding=4,fontsize=10)
a.set_xticks(x,[r['label']for r in rows]);a.set_ylabel('Scenes');a.set_title('Zero-score transitions (IL46 starts with 479 zeros)',loc='left',fontsize=12,pad=38)
a.set_ylim(0,355);a.grid(axis='y',alpha=.15);a.set_axisbelow(True);a.legend(frameon=False,loc='upper left',bbox_to_anchor=(-.01,1.12),fontsize=9.7)
for i,r in enumerate(rows):a.text(i,-.12,f"{r['total_zero']} zeros after RL",transform=a.get_xaxis_transform(),ha='center',fontsize=10,color='#536174')
a=axs[1]
bars=[a.bar(x-width/2,[r['gross_gain_PDMS']for r in rows],width,color=COL['gain'],label='Gross gain G+'),
 a.bar(x+width/2,[r['gross_loss_magnitude_PDMS']for r in rows],width,color=COL['loss'],label='Gross loss |G−|')]
for b in bars:a.bar_label(b,fmt='%.3f',padding=4,fontsize=10)
a.set_xticks(x,[r['label']for r in rows]);a.set_ylabel('PDMS points (population-normalized)');a.set_title('Gross gains minus gross losses = net improvement',loc='left',fontsize=12,pad=38)
a.set_ylim(0,3.65);a.grid(axis='y',alpha=.15);a.set_axisbelow(True);a.legend(frameon=False,loc='upper left',bbox_to_anchor=(-.01,1.12),fontsize=10)
for i,r in enumerate(rows):a.text(i,-.12,f"net +{r['delta_PDMS']:.3f}",transform=a.get_xaxis_transform(),ha='center',fontsize=10,fontweight='bold')
f.text(.065,.14,'Fixed IL46 high-score cohort (score ≥ 0.9; n=8,722): population net contribution',fontsize=10.5,fontweight='bold')
f.text(.065,.10,'D0 −1.369 | D5 −1.855 | FPO −0.741 | AWM −0.188 PDMS. These are cohort net losses, not gross-loss upper bounds.',fontsize=10,color='#536174')
f.text(.065,.059,caption0,fontsize=9.4)
f.text(.065,.035,caption1,fontsize=9.4)
f.text(.065,.008,'Exposure differs across methods. Four complete checkpoints only; no interpolation, causal ranking, new scoring or bootstrap.',fontsize=9.2,color='#536174')
for ext in('svg','png'):f.savefig(O/('D46_zero_repair_and_gain_loss.'+ext),dpi=190)
plt.close(f)
# 2) Separate component figure, common percentage-point axis.
f,a=plt.subplots(figsize=(12.8,6.7));f.subplots_adjust(left=.085,right=.985,top=.78,bottom=.27)
f.suptitle('Safety and progress move differently',x=.085,y=.96,ha='left',fontsize=19,fontweight='bold')
f.text(.085,.902,'Change from the same IL46 baseline; complete 12,146-scene single-B1 evaluation',fontsize=12,color='#536174')
metrics=['no_at_fault_collisions','drivable_area_compliance','time_to_collision_within_bound','ego_progress'];xx=np.arange(4);w=.18
colors=['#234f85','#8760aa','#168886','#d28b2f']
for i,r in enumerate(rows):
 values=[r['metric_delta_pp'][m]for m in metrics]
 b=a.bar(xx+(i-1.5)*w,values,w,color=colors[i],label=f"{r['label']} · {r['scene_exposures']:,} TRAIN scenes")
 for rect,v in zip(b,values):a.annotate(f'{v:+.2f}',(rect.get_x()+rect.get_width()/2,v),xytext=(0,4 if v>=0 else -5),textcoords='offset points',ha='center',va='bottom'if v>=0 else'top',fontsize=9.5)
a.axhline(0,color='#5c6a7b',lw=1);a.grid(axis='y',alpha=.16);a.set_axisbelow(True);a.set_ylim(-2.35,2.6)
a.set_ylabel('Mean metric change (percentage points)');a.set_xticks(xx,['NC\nNo-at-fault collisions','DAC\nDrivable area','TTC\nCollision-time compliance','EP\nEgo progress'])
a.legend(loc='upper left',bbox_to_anchor=(0,1.12),ncol=2,frameon=False,fontsize=10)
f.text(.085,.131,'D0: NC / DAC / TTC improve, EP −1.552 pp. AWM: EP +0.487 pp, but NC −0.016 pp.',fontsize=10.7,fontweight='bold')
f.text(.085,.093,'Metric changes are descriptive and are not additive PDMS contributions. Different exposure and objectives prevent causal ranking.',fontsize=10,color='#536174')
f.text(.085,.051,caption0,fontsize=9.4)
f.text(.085,.022,caption1,fontsize=9.4)
for ext in('svg','png'):f.savefig(O/('D46_safety_and_progress_delta.'+ext),dpi=190)
plt.close(f)
readout=[
 '四份评测均为完整NAVTEST：12146场景、44物理log、invalid0，同一IL46基线89.49552098，单候选ODE K10/shift5/seed2101。只复用已封存SUMMARY/PAIRED与既有CI，未重扫场景或重采样。',
 'D0：44,800场景曝光、1400更新，90.17112884（+0.675608）。修复248个原零分，同时新增87个零分。总体G+=2.590471、|G−|=1.914863；固定IL46高分≥.9队列8722场景的净贡献−1.368811 PDMS，明显抵消修复收益。安全NC/DAC/TTC分别+0.580/+0.766/+2.001pp，EP−1.551854pp。',
 'FPO：28,800曝光、900更新，90.12278094（+0.627260），与D0相近但曝光较少；固定高分队列净损失−0.740797 PDMS，小于D0。修复164、新增65，EP−0.468990pp。不同曝光且每更新算量不同，不能据此断言因果优越、GPU效率或长期上限。',
 'AWM：25,600曝光、800更新，89.95115511（+0.455634）；EP+0.486705pp，高分保留8660/8722（99.29%），高分队列净损失仅−0.187708。修复72、新增30，显示当前更偏保持/进度，修复幅度较小；NC−0.016466pp，不能概括成所有安全指标都改善。',
 'D5：同D0曝光，89.97889876（+0.483378）；修复288最多，但新增143、grossloss2.588862及高分净损失1.855083也最大，EP−1.905190pp。更大scope当前不是净收益保证；既有44logCI跨0，不据单点宣告方法无效。',
 '高分口径固定用PAIRED_ANALYSIS.gain_loss.fixed_reference_cohorts.IL46_high_ge_0.9（8722），不混用旧SUMMARY.paired中另一个baseline_high_denominator=11199。高分队列净损失不是可实现repair收益，也不是全部grossloss上界。',
 '图表仅供完整测试集上的研究诊断与允许的模型选择；不导出NAVTEST token、不生成训练队列，不改变当前D0/D2恢复及D1/D5资源暂存安排。']
text='\n\n'.join(readout)+'\n';(O/'D46_EARLY_FULL_READOUT_zh.txt').write_text(text)
facts=dict(schema='D46_early_full_plot_facts_v1',status='COMPLETE',generated_utc=datetime.now(timezone.utc).isoformat(),baseline=dict(PDMS=base['PDMS'],scene_count=12146,physical_logs=44,invalid_scenes=0,zero_count=479,fixed_high_ge09_count=8722,means=base['means']),
 rows=rows,definitions=dict(gain_loss='G+ = mean(max(scoreRL-scoreIL,0)); |G−| = mean(max(scoreIL-scoreRL,0)); values×100 are PDMS points.',
  fixed_high='The same IL46 score>=.9 cohort of8722 for all methods; population contribution is its signed delta sum/12146×100.',
  metric_delta='100*(mean_metric_RL-mean_metric_IL46), percentage points; not additive PDMS decomposition.',
  legacy_high_warning='SUMMARY.paired.baseline_high_denominator11199 is not used for >=.9 statistics.'),
 evidence=evidence,limits=['Different exposures; no same-budget causal method ranking.','No new bootstrap, full scene verification, model forward, GPU or training mutation.','Full NAVTEST model selection used; no NAVTEST labels/IDs enter training.'],
 conclusions_zh=readout,builder=ref(__file__),figures={k:ref(O/k)for k in('D46_zero_repair_and_gain_loss.svg','D46_zero_repair_and_gain_loss.png','D46_safety_and_progress_delta.svg','D46_safety_and_progress_delta.png')})
facts['sha256']=fp(facts);(O/'D46_EARLY_FULL_FACTS.json').write_text(json.dumps(facts,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(dict(status='COMPLETE',facts=str(O/'D46_EARLY_FULL_FACTS.json'),figures=list(facts['figures']),readout=str(O/'D46_EARLY_FULL_READOUT_zh.txt')),ensure_ascii=False))

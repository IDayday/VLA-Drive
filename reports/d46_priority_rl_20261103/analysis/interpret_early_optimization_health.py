#!/usr/bin/env python3
"""Add readable conclusions to the finite CPU measurements; no payload reload."""
import hashlib,json,os,pathlib
R=pathlib.Path('/mnt/project/d46-priority-rl-20261103');p=R/'analysis/EARLY_OPTIMIZATION_HEALTH_20261103.json';d=json.loads(p.read_text())
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def fp(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
S=R/'runtime/formal_v1';critical=d['source_audit']
for arm,v in d['arms'].items():
 root=R/'runs'/('D46_'+arm);i=json.loads((root/'identity.json').read_text())
 v['actual_data_identity']=i['data']
 v['critical_source_matches_saved_identity']=all(i['source_files'].get(k)==h for k,h in critical.items())
 v['latest_window_age_seconds_at_audit_finish']=d['finished_unix']-v['recent100_complete_collections']['time']['timestamp_last_optimizer_end']
 v['optimizer_epsilon_unit_note']='Coordinate-count-weighted attenuation; not gradient-mass weighted, not causal performance loss. LoRA_A/B are factorized coordinates.'
 for label in ('equal_exposure_first800','recent100_complete_collections'):
  w=v[label];w['role_gradient_units']='After global maxnorm1 clipping; gradient_norm is before clipping.'
  e=w['exploration_global_once_per_collection'];e['interpretation']='Feasible is strict NC=DAC=TTC=1, not Comfort/score>=.9; allzero/allone are canonical candidate rewards, not canonical deployment B1. rawA zero cannot be fully attributed without reading behavior records.'
 for g,w in v['native_parameter_and_Adam']['actual_Adam_moments_at800'].items():
  w['parameter_mapping_contract']='adapter_state and optimizer groups preserve actor.named_parameters order; per-role count and moment tensor shapes explicitly verified.'
 r=d['cross_arm_scale_comparison'][arm]
 r['text_LoRA_B_delta_RMS_relative_D0']=v['native_parameter_and_Adam']['theta_changes']['text_lora']['lora_B']['delta_RMS']/d['arms']['D0']['native_parameter_and_Adam']['theta_changes']['text_lora']['lora_B']['delta_RMS']
 r['D5_scope_dimension_caveat']='D5 has double text rank and extra action/FiLM trainables; its aggregate L2 is not a controlled functional-distance comparison.'if arm=='D5'else None
 d0=d['arms']['D0']['equal_exposure_first800']['time'];t=v['equal_exposure_first800']['time']
 r['wall_throughput_relative_D0']=t['approx_valid_scenes_per_wall_second']/d0['approx_valid_scenes_per_wall_second']
d['conclusions_zh']=[
 '四条路线在同25600 TRAIN曝光/800 Adam updates、八rank共同前缀上全部loss/gradient finite，first-inner fresh-ratio error全0；原生0与800通过实际文件hash、8rank Boundary、Adam step800和scheduler800验证。未发现停学、未更新或学习率错钟的证据。',
 'FPO/AWM的小梯度不是几百倍更小的参数更新：相对D0，平均grad仅0.317%/0.228%，但相同LoRA_B的实际L2位移仍89.36%/67.57%。不能据gradient norm直接淘汰。',
 'Adam并非完全尺度不变：FPO/AWM的LoRA_A有62.30%/82.00%坐标sqrt(vhat)<=epsilon1e-8，平均epsilon衰减0.430/0.297；LoRA_B平均衰减0.861/0.828。它能解释部分尺度效应，但这些是坐标统计，不能当作已证明的PDMS损失。',
 'FPO实际CFM-energy ratio的clip在全部前800与recent100均从未激活，inner2 ratio_max平均约1.000162，clip边界1.01；普通Gaussian clip约75%仅是另一诊断，不能写成FPO被clip卡住。AWM没有ratio-clipped目标，pg内含正A weighted energy及clean-x reference MSE。',
 'D5扩大scope后所有前800更新都触发global maxnorm1，总norm均10.05；裁剪后action_io norm均0.999714、text约0.02108。recent100参考beta均0.00957、KLcoord均0.01808，说明约束已实际反应于更大policy偏移；缺分量梯度，不把这称为anchor主导或最终退化因果。',
 '四臂原始A非零组均约96.4%；前800 canonical全零组仅22/23/27/26（各25600组），rawA全零约918–934组。总体有训练信用；这些计数不能代替部署B1零分修复。',
 '实测墙钟吞吐D0/D5/FPO/AWM约1.452/1.401/0.992/0.887 scene/s。FPO/AWM额外CFM前向增加优化时间；本窗口记录scoring约占已记录worker分项4%上下，主要成本是generation与optimization，不能靠加scorer就恢复D0吞吐。共享host和存盘时间使其非纯kernel benchmark。',
 '资源倾斜应先等待已启动的完整12146官方B1结果与paired证据。健康证据支持继续观察，既不以小梯度宣判FPO/AWM无效，也不把G16 sampled oracle/候选分或参数移动冒充部署收益。']
d['minimal_falsifiable_followups_not_executed']=[
 dict(priority=1,condition='Only after complete canonical same-budget NAVTEST',proposal='Use full B1 zero/high-retention/paired safety evidence to choose resource priority; keep identical exposure accounting.',reason='Current audit measures optimization health, not method reward outcome.'),
 dict(priority=2,condition='If FPO full-B1 evidence is weak and a new isolated trial is authorized',proposal='One-factor study of energy-ratio scale or optimizer epsilon with explicit KL/reference units, not both; preserve exact baseline.',reason='Actual energy clip never active and epsilon attenuation measurable; neither proves optimal remedy.'),
 dict(priority=3,condition='If D5 shows poor retention at complete B1',proposal='One-factor action-I/O LR or per-role clipping control, keeping text/FiLM and reward fixed.',reason='Actual clipping is dominated by action_io, but scope/LR changes in D5 are confounded; no intervention here.')]
d['interpretation_builder']={'path':__file__,'file_sha256':sha(__file__)}
d.pop('sha256',None);d['sha256']=fp(d);tmp=p.with_suffix('.writing');tmp.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n');os.replace(tmp,p)
print(json.dumps({'path':str(p),'file_sha256':sha(p),'status':d['status'],'generated_utc':d['generated_utc']}))

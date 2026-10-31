"""Four deterministic NAVTEST-only policy illustrations, existing records only."""
import hashlib,json,pickle
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
R=Path('/mnt/project/qwen-world-rl-optimization-20261030');A=R/'analysis';C=A/'evaluation_metric_cache';F=R/'reporting/figures';F.mkdir(parents=True,exist_ok=True)
OUT=A/'SAFETY_EP_POLICY_CASES_20261031.json'
FULL='PPO_FULL_u010000_NAVTEST';SAFE='PPO_SAFETY_EP_u010000_NAVTEST'
OBS=['IL70_NAVTEST','PPO_FULL_u001000_NAVTEST','PPO_FULL_u005000_NAVTEST',FULL,'PPO_SAFETY_EP_u001000_NAVTEST','PPO_SAFETY_EP_u005000_NAVTEST',SAFE]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def fp(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def sealed(p):
 x=read(p);y=dict(x);s=y.pop('sha256',None)
 if s!=fp(y):raise ValueError('Changed sealed input '+str(p))
 return x
D={};inputs={}
for lab in OBS:
 p=C/(lab+'.npz');meta=read(C/(lab+'.json'))
 if sha(p)!=meta['npz_sha256']:raise ValueError('Changed cached metrics')
 with np.load(p,allow_pickle=False) as z:
  if not D:tokens=z['tokens'].tolist();keys=z['metric_keys'].tolist()
  assert tokens==z['tokens'].tolist();D[lab]=z['metrics'].copy()
 inputs[lab]=dict(cache=str(p),cache_sha256=meta['npz_sha256'],identity_sha256=meta['identity_sha256'],summary_sha256=meta['summary_sha256'])
f,s=D[FULL],D[SAFE]
strict=lambda x:(x[:,1:4]==1).all(1)
cohorts=[
 dict(id='safe_ep_gain',title='A  Metric-pass EP gain',title_zh='双方安全指标通过、EP明显增加',
  mask=strict(f)&strict(s)&(f[:,5]==1)&(s[:,5]==1)&(s[:,4]-f[:,4]>=.2),rank=s[:,4]-f[:,4],
  rule='NC=DAC=TTC=Comfort=1 for both; SafetyEP_EP-FULL_EP>=.2; max EP gain, then lexicographic token',
  text_zh='同协议下双方NC/DAC/TTC/Comfort均通过，SafetyEP的EP更高；图只描述请求进度差，不能据此证明必要等待或交通规则判断。'),
 dict(id='zero_repair',title='B  Zero repaired at 10k',title_zh='原PPO零分、SafetyEP修复',
  mask=(f[:,0]<=1e-12)&strict(s)&(s[:,0]>=.9)&(s[:,5]==1),rank=s[:,0],
  rule='FULL_score<=1e-12; SafetyEP strict NC=DAC=TTC=Comfort1 and score>=.9; max SafetyEP score, then lexicographic token',
  text_zh='原PPO的官方零分在SafetyEP10k变为高分；只把修复归于已记录指标，不根据无障碍物轨迹图推断撞谁或模型看懂了什么。'),
 dict(id='high_to_zero',title='C  High score becomes zero',title_zh='原PPO高分、SafetyEP变零',
  mask=(f[:,0]>=.95)&(s[:,0]<=1e-12),rank=f[:,0],
  rule='FULL_score>=.95 and SafetyEP_score<=1e-12; max FULL score, then lexicographic token',
  text_zh='SafetyEP存在由高分变零的反例，不能只展示正例；请求轨迹与官方模拟后的安全结果分开记录。'),
 dict(id='persistent_zero',title='D  Zero across listed observations',title_zh='截至10k的多观测持续零',
  mask=np.stack([D[k][:,0]<=1e-12 for k in OBS]).all(0),rank=np.zeros(len(tokens)),
  rule='score<=1e-12 at all seven explicitly listed observations: IL70 and FULL/SafetyEP1k,5k,10k; lexicographically first token',
  text_zh='此处持续只指IL70及两策略1k/5k/10k共7个已评分观测，不能推出未观测更新、10k以后、不可避免失败或无安全解。'),
]
rawdata=Path('/mnt/project/qwen-sourceflow-native-20261018/data/NAVTEST/records.pkl')
with rawdata.open('rb') as stream:records=pickle.load(stream)
original_hash=sha(rawdata);cases=[]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','savefig.facecolor':'white'})
colors={FULL:'#d67a20',SAFE:'#087f8c','GT':'#7a8490'}
for co in cohorts:
 ids=np.flatnonzero(co['mask']);chosen=sorted(ids,key=lambda i:(-float(co['rank'][i]),tokens[i]))[0];token=tokens[chosen];row=records[token]
 trajectories={};views={};source={}
 for lab in [FULL,SAFE]:
  spec=read(R/'evaluation_specs'/(lab+'.json'));out=Path(spec['output']);ident=sealed(out/'identity.json');summary=sealed(out/'SUMMARY.json');path=out/'scenes'/f'{token}.json';rec=sealed(path)
  if (ident['sha256']!=inputs[lab]['identity_sha256'] or summary['sha256']!=inputs[lab]['summary_sha256'] or summary['status']!='COMPLETE' or summary['scenes']!=12146
      or rec['identity']!=ident['sha256'] or rec['status']!='OK' or rec['token']!=token or rec['log']!=row['log']):raise ValueError('Selected scene ancestry changed')
  req=np.asarray(rec['requested_trajectory'],dtype=np.float64);fb=rec['feedback'];arr=np.ascontiguousarray(req,dtype='<f4')
  if req.shape!=(8,3) or not np.isfinite(req).all() or not np.array_equal(arr.astype(np.float64),req) or hashlib.sha256(arr.tobytes()).hexdigest()!=rec['requested_trajectory_sha256']:
   raise ValueError('Actual physical FP32 requested trajectory changed')
  if not np.array_equal(req,np.asarray(fb['requested_trajectory'])):raise ValueError('Feedback is not exact request')
  idx=chosen;wanted=D[lab][idx]
  if not np.array_equal(np.array([fb['metrics'][k] for k in keys]),wanted):raise ValueError('Metric cache/scene mismatch')
  trajectories[lab]=req;views[lab]=dict(metrics=fb['metrics'],PDMS=100*fb['metrics']['score'],raw_progress_meters=fb['raw_progress_meters'],
   requested_trajectory=req.tolist(),heading_units='radians',official_first_at_fault_collision_seconds=fb['official_first_at_fault_collision_seconds'],official_first_ttc_infraction_seconds=fb['official_first_ttc_infraction_seconds'])
  source[lab]=dict(scene=str(path),scene_file_sha256=sha(path),scene_seal=rec['sha256'],requested_trajectory_sha256=rec['requested_trajectory_sha256'],summary=str(out/'SUMMARY.json'),summary_seal=summary['sha256'],identity_seal=ident['sha256'],canonical_cache=fb['cache'],protocol_sha256=fb['protocol_sha256'])
 if source[FULL]['canonical_cache']!=source[SAFE]['canonical_cache'] or source[FULL]['protocol_sha256']!=source[SAFE]['protocol_sha256']:raise ValueError('Selected canonical scene/context mismatch')
 gt=np.asarray(row['requested_GT'],dtype=np.float32).astype(np.float64)
 if gt.shape!=(8,3) or not np.isfinite(gt).all():raise ValueError('Missing original local GT')
 fig=plt.figure(figsize=(6.0,6.25));fig.suptitle(co['title'],x=.5,y=.985,fontsize=13,fontweight='bold')
 fig.text(.5,.945,f'{token} | 10k FULL vs SafetyEP | ODE seed2101',ha='center',fontsize=8.7)
 axim=fig.add_axes([.07,.68,.86,.23]);camera=Path(row['images']['cam_f0']) if row['images'].get('cam_f0') else None
 camera_proof=dict(path=str(camera) if camera else None,available=bool(camera and camera.exists()))
 if camera_proof['available']:
  with Image.open(camera) as im:
   image=im.convert('RGB');image.thumbnail((840,250));axim.imshow(image)
  camera_proof['sha256']=sha(camera)
 else:axim.text(.5,.5,'Current front image unavailable',ha='center',va='center')
 axim.set_axis_off();axim.set_title('Recorded current front camera (context only)',fontsize=8,pad=3)
 ax=fig.add_axes([.1,.365,.82,.275]);ax.set_title(f"Local requested XY: 8 poses / 0.5s | nav={row['navigation']}",fontsize=9,pad=5)
 for lab,name,marker in [(FULL,'PPO_FULL','o'),(SAFE,'SafetyEP','s')]:
  q=trajectories[lab];ax.plot(q[:,0],q[:,1],color=colors[lab],marker=marker,markersize=4,lw=1.9,label=name)
  ax.annotate('8',q[-1,:2],xytext=(4,4 if lab==SAFE else -10),textcoords='offset points',fontsize=7,color=colors[lab])
 ax.plot(gt[:,0],gt[:,1],color=colors['GT'],marker='.',markersize=4,linestyle='--',lw=1.2,label='Recorded human GT')
 ax.scatter([0],[0],color='#1f2933',s=24,marker='+',label='Ego t=0',zorder=5)
 xy=np.vstack([*trajectories.values(),gt])[:,:2];ax.set_xlim(min(-1.5,float(xy[:,0].min())-1),float(xy[:,0].max())+2)
 ax.set_ylim(min(-1.5,float(xy[:,1].min())-1),max(1.5,float(xy[:,1].max())+1));ax.set_aspect('equal',adjustable='datalim')
 ax.set_xlabel('x forward (m)',fontsize=8);ax.set_ylabel('y left (m)',fontsize=8);ax.grid(alpha=.22);ax.tick_params(labelsize=8)
 ax.legend(loc='best',fontsize=7,framealpha=.87,ncol=2)
 tax=fig.add_axes([.075,.12,.86,.145]);tax.axis('off')
 names=['PDMS','NC','DAC','TTC','EP','Comfort'];values=[]
 for lab in [FULL,SAFE]:
  m=views[lab]['metrics'];values.append([f'{v:.2f}' for v in [100*m['score'],100*m['no_at_fault_collisions'],100*m['drivable_area_compliance'],100*m['time_to_collision_within_bound'],100*m['ego_progress'],100*m['comfort']]])
 table=tax.table(cellText=values,rowLabels=['FULL','SafetyEP'],colLabels=names,loc='center',cellLoc='center');table.auto_set_font_size(False);table.set_fontsize(8.4);table.scale(.94,1.55)
 for (r,c),cell in table.get_celld().items():
  cell.set_edgecolor('#dfe5e8');cell.set_linewidth(.6)
  if r==0:cell.set_facecolor('#edf2f4')
  if c==-1 and r>0:cell.get_text().set_color(colors[[FULL,SAFE][r-1]])
 fig.text(.5,.285,'Metrics shown x100; official simulated outcome; plot is requested plan.',ha='center',fontsize=7.2,color='#46525b')
 fig.text(.5,.079,'Map/obstacles not drawn. Human GT is not the reward reference.',ha='center',fontsize=7.1,color='#46525b')
 fig.text(.5,.054,'Selected diagnostic example: no collision-target or red-light reasoning inference.',ha='center',fontsize=6.9,color='#46525b')
 if co['id']=='persistent_zero':fig.text(.5,.027,'Persistence = IL70 + both policies at 1k/5k/10k only; later states not implied.',ha='center',fontsize=6.7,color='#46525b')
 else:fig.text(.5,.027,f"Rule-selected cohort: {len(ids)} of 12146 scenes; extremes are not representative.",ha='center',fontsize=6.9,color='#46525b')
 artifacts={}
 for ext in ['svg','png']:
  p=F/f"safety_policy_case_{co['id']}.{ext}";fig.savefig(p,dpi=170,metadata={'Date':None} if ext=='svg' else None);artifacts[ext]=dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size)
 plt.close(fig)
 case=dict(id=co['id'],title_zh=co['title_zh'],token=token,log=row['log'],selection_rule=co['rule'],eligible_scene_count=len(ids),selection_tiebreak='largestregisteredcontrastthenlexicographic token, orlexicographicfirstforpersistentzero',interpretation_zh=co['text_zh'],
  metrics=views,local_GT_trajectory=gt.tolist(),coordinate_frame='rear_axle ego xforward/y left/yaw radians;8poses at0.5..4.0s',
  input=dict(current_navigation=row['navigation'],current_proprio=row['proprio'],current_camera=camera_proof),artifacts=artifacts,provenance=source,
  map_obstacles_rendered=False,simulated_executed_trace_rendered=False,GT_role='recorded futureego,notPDMrewardreferenceornormativeoptimaltrajectory',NAVTEST_report_only=True,training_selection_eligible=False)
 if co['id']=='persistent_zero':case['persistent_observations']={lab:float(D[lab][chosen,0]) for lab in OBS}
 cases.append(case)
body=dict(schema='world_SafetyEP_NAVTEST_policy_illustrations_report_only_v1',status='COMPLETE_EXISTING_RECORDS_ONLY',created_at=datetime.now(timezone.utc).isoformat(),
 comparison=dict(FULL=FULL,SafetyEP=SAFE,updates=10000,full_population=12146,inference='singleODE10shift5/seed2101',metrics='officialcanonical',parent='sameIL70freshRLandseed42'),cases=cases,
 limitations_zh=['4个按规则挑选的极端/反例只描述策略差异，不证明总体提升因果或代表性。','NC/DAC/TTC通过不等于真实道路安全或交通规则完整合规。','图画原始8pose请求轨迹而非模拟执行轨迹；不画map/障碍物，不推断撞谁、遮挡目标或是否看懂红灯。','EP零可来自安全门控，不能把零分直接解释为原地不动；必要等待不能仅低EP判定。','GT是记录人类未来轨迹，不是canonical参考规划器轨迹或最优行动证书。','持续零仅指列出的7次观测，不推断全部训练时刻或10k以后。','全部NAVTEST案例只用于报告，未进入训练采样/teacher/preferencebank。'],
 provenance=dict(metric_caches=inputs,raw_records=str(rawdata),raw_records_sha256=original_hash,script=str(Path(__file__).resolve()),script_sha256=sha(__file__)),
 scope=dict(CPU_only=True,new_model_forward=0,new_scores=0,new_GPU_work=0,selected_scene_record_reads=8,original_population_scans=0,weights_loaded=False,official_cache_pickles_unloaded=True,report_renderer_modified=False,git_stage=False,git_commit=False))
body['sha256']=fp(body);OUT.write_text(json.dumps(body,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
print(json.dumps(dict(output=str(OUT),cases=[dict(id=x['id'],token=x['token'],eligible=x['eligible_scene_count'],svg=x['artifacts']['svg']['path']) for x in cases]),ensure_ascii=False))

"""Finite CPU-only analysis of three completed TRAIN4096 populations.

Publishes complete-model interim evidence, then waits on exact diagnostic-owner
pidfd and STATE replacement events. Never probes/dispatches/stops GPU work.
"""
import ctypes,fcntl,hashlib,json,math,os,re,select,sys,time,traceback
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
R=Path('/mnt/project/qwen-world-rl-optimization-20261030');A=R/'analysis';O=R/'owners/TRAIN4096_ANALYSIS';D=R/'owners/TRAIN4096'
MODELS=('IL70','PPO_FULL','PPO_ADAPTIVE');METRICS=('score','no_at_fault_collisions','drivable_area_compliance','time_to_collision_within_bound','ego_progress','comfort','driving_direction_compliance')
now=lambda:datetime.now(timezone.utc).isoformat()
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
fingerprint=lambda v:hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def atomic(path,v):
 p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+f'.tmp.{os.getpid()}');t.write_text(json.dumps(v,sort_keys=True,indent=2,allow_nan=False)+'\n');os.replace(t,p)
def verified(path):
 v=read(path);body=dict(v);seal=body.pop('sha256');assert fingerprint(body)==seal,'Changed sealed JSON: '+str(path);return v
def same_process(owner):
 try:
  f=Path(f"/proc/{owner['pid']}/stat").read_text().split();return int(f[21])==owner['start_ticks'] and f[2]not in ('Z','X')
 except (OSError,ValueError):return False

def refresh(status):
 atomic(R/'reporting/refresh_request.json',{'id':f'TRAIN4096_ANALYSIS_{status}_{now()}','scientific_result':False,'evidence_kind':'completed_supplemental_analysis','reason':'finite TRAIN4096 analysis '+status,'evidence':str(A/'TRAIN4096_DIAGNOSIS.json')})

class Analysis:
 def __init__(self):
  self.panel=verified(R/'data/panels/TRAIN_DIAGNOSTIC_004096.json');self.tokens=self.panel['tokens'];assert len(self.tokens)==4096
  self.small=set(verified(R/'data/panels/TRAIN_DIAGNOSTIC_001024.json')['tokens']);self.pop=read(R/'data/NAVTRAIN/manifest.json');self.physical=lambda s:re.sub(r'_\d+_\d+$','',s)
  self.logs=np.array([self.physical(self.panel['token_logs'][t])for t in self.tokens]);self.Nlog=Counter(self.physical(l)for l in self.pop['token_logs'].values());self.feedback=np.array([t in self.small for t in self.tokens])
  counts=Counter()
  self.stage_sha=[]
  for i in range(10):
   p=R/'adaptive_stages'/f'stage{i:06}.json';stage=verified(p);counts.update(stage['ordered_tokens']);self.stage_sha.append({'index':i,'sha256':stage['sha256']})
  self.exposure=np.array([counts[t]for t in self.tokens]);self.models={};self.identities={};self.cache_info={};self.last_signature=None
 def load_model(self,name,node):
  if name in self.models:return
  summary=node['summary'];assert summary['status']=='COMPLETE' and summary['scenes']==4096 and summary.get('invalid_scenes',0)==0
  output=Path(node['spec']['output']);identity=verified(output/'identity.json')
  assert identity['cohort']=='TRAIN_DIAGNOSTIC' and identity['count']==4096 and set(identity['tokens'])==set(self.tokens)
  assert identity['seed']==2101 and identity['inference']['sampler']=='ODE' and identity['inference']['steps']==10 and identity['inference']['candidates']==1
  assert identity['data_sha256']==self.panel['data_manifest_sha256'] and identity['sha256']==summary['evaluation_identity_sha256']
  assert identity.get('IL_updates')==70000 if name=='IL70' else identity.get('RL_updates')==10000
  if self.identities:
   other=next(iter(self.identities.values()))
   for key in ['cohort','token_sha256','data_sha256','seed','inference','numerics']:assert identity[key]==other[key],'Unmatched diagnostic '+key
   assert identity['protocol']['sha256']==other['protocol']['sha256']
  cache=A/'TRAIN4096_CACHE';cache.mkdir(exist_ok=True);stem=cache/(name+'_'+identity['sha256'][:16]);npz=stem.with_suffix('.npz');meta=stem.with_suffix('.json')
  reused=False
  if meta.exists():
   proof=read(meta);assert proof['evaluation_identity_sha256']==identity['sha256'] and proof['npz_sha256']==sha(npz)
   with np.load(npz,allow_pickle=False)as data:
    assert data['tokens'].tolist()==self.tokens;values=data['metrics'].copy()
   reused=True
  else:
   values=np.empty((4096,len(METRICS)),dtype=float);scene_seals=[]
   for i,t in enumerate(self.tokens):
    row=verified(output/'scenes'/f'{t}.json');assert row['token']==t and row['log']==self.panel['token_logs'][t] and row['identity']==identity['sha256'] and row['status']=='OK'
    feedback=row['feedback'];assert feedback['status']=='OK' and feedback['protocol_sha256']==identity['protocol']['sha256']
    v=[float(feedback['metrics'][k])for k in METRICS];assert all(math.isfinite(x)for x in v) and 0<=v[0]<=1
    values[i]=v;scene_seals.append(row['sha256'])
   tmp=npz.with_suffix('.npz.tmp')
   with tmp.open('wb')as f:np.savez_compressed(f,tokens=np.array(self.tokens),metrics=values)
   os.replace(tmp,npz);atomic(meta,{'schema':'TRAIN4096_normalized_metrics_cache_v1','model':name,'evaluation_identity_sha256':identity['sha256'],'model_sha256':identity['model_sha256'],'scenes':4096,'metrics':METRICS,'scene_seals_sha256':fingerprint(scene_seals),'npz_sha256':sha(npz),'source_output':str(output)})
  assert values.shape==(4096,len(METRICS));assert abs(values[:,0].mean()*100-summary['PDMS'])<1e-9
  self.models[name]=values;self.identities[name]=identity;self.cache_info[name]={'npz':str(npz),'metadata':str(meta),'reused_from_disk':reused,'scene_rows_read_this_process':0 if reused else 4096}
 def aggregate(self,array,mask):
  logs=self.logs[mask];unique,idx=np.unique(logs,return_inverse=True);n=np.bincount(idx).astype(float);sums=np.zeros((len(unique),len(METRICS)));np.add.at(sums,idx,array[mask]);means=sums/n[:,None]
  return unique,n,means
 def points(self,array,mask):
  unique,n,means=self.aggregate(array,mask);weights={'scene_mean':n,'physical_log_equal':np.ones(len(n)),'population_log_reweighted_TRAIN_projection':np.array([self.Nlog[l]for l in unique],dtype=float)}
  return {key:{m:float(v)for m,v in zip(METRICS,100*np.sum(means*w[:,None],axis=0)/w.sum())}for key,w in weights.items()}
 def paired(self,array,mask):
  unique,n,means=self.aggregate(array,mask);rng=np.random.default_rng(42);draw=rng.integers(0,len(n),size=(2000,len(n)));out={}
  for key,w in [('scene_mean',n),('physical_log_equal',np.ones(len(n))),('population_log_reweighted_TRAIN_projection',np.array([self.Nlog[l]for l in unique],dtype=float))]:
   point=100*np.sum(means*w[:,None],axis=0)/w.sum();boot=100*np.sum(means[draw]*w[draw,None],axis=1)/w[draw].sum(axis=1)[:,None];lo,hi=np.quantile(boot,[.025,.975],axis=0)
   out[key]={metric:{'delta_points':float(point[j]),'cluster_bootstrap_95pct':[float(lo[j]),float(hi[j])]}for j,metric in enumerate(METRICS)}
  return out
 def publish(self,state):
  complete=[m for m in MODELS if state['runs'].get(m,{}).get('status')=='COMPLETE'];signature=tuple(complete)
  if signature==self.last_signature:return
  if not all(m in complete for m in ['IL70','PPO_FULL']):return
  for m in complete:self.load_model(m,state['runs'][m])
  final=all(m in complete for m in MODELS);basis='planned sealed stage0..9; ADAPT10k pending'
  native_proof=None
  if final:
   cp=Path(state['runs']['PPO_ADAPTIVE']['spec']['rl_checkpoint']);seal=read(cp/'COMPLETE');txn=read(cp/'transaction.json');ranks=txn['ranks']if isinstance(txn['ranks'],list)else list(txn['ranks'].values())
   assert seal['updates']==10000 and seal['next_inner']==0 and seal['world']==8 and len(ranks)==8 and all(v['scene_cursor']==20000 and v['updates']==10000 for v in ranks)
   basis='actual native10k committed cursor20000 and sealed stage0..9';native_proof={'checkpoint':str(cp),'COMPLETE':seal,'transaction_rank_count':8,'scene_cursor_each_rank':20000}
  cohorts={'all4096':np.ones(4096,dtype=bool),'feedback1024':self.feedback,'extra3072':~self.feedback,'extra_training_unseen10k':(~self.feedback)&(self.exposure==0),'extra_training_seen10k':(~self.feedback)&(self.exposure>0),'feedback_training_unseen10k':self.feedback&(self.exposure==0),'feedback_training_seen10k':self.feedback&(self.exposure>0),'exposure1_to4':(self.exposure>=1)&(self.exposure<=4),'exposure5_to16':(self.exposure>=5)&(self.exposure<=16),'exposure17_plus':self.exposure>=17}
  pairs=[('PPO_FULL','IL70')]+([('PPO_ADAPTIVE','IL70'),('PPO_ADAPTIVE','PPO_FULL')]if final else[]);groups={}
  for label,mask in cohorts.items():
   if not mask.any():continue
   groups[label]={'scenes':int(mask.sum()),'physical_logs':int(len(set(self.logs[mask]))),'feedback1024_scenes':int((mask&self.feedback).sum()),'ADAPT10k_exposures_sum':int(self.exposure[mask].sum()),'model_means_points':{m:self.points(v,mask)for m,v in self.models.items()},'paired':{a+'_minus_'+b:self.paired(self.models[a]-self.models[b],mask)for a,b in pairs}}
  result={'schema':'world_rl_TRAIN4096_diagnosis_v1','status':'COMPLETE'if final else'INTERIM_COMPLETE_MODELS','at':now(),'models_complete':complete,'models_pending':[m for m in MODELS if m not in complete],'owner_status_at_analysis':state['status'],'cohort':'TRAIN_DIAGNOSTIC','native_Test':False,'panel_scenes':4096,'panel_physical_logs':162,'bootstrap':{'unit':'physical log','replicates':2000,'seed':42,'paired_same_scene_and_resampled_log_indices':True},'metric_names':METRICS,'subgroups':groups,'exposure_basis':basis,'exposure_native_proof':native_proof,'stage_census_sha256':self.stage_sha,'model_identities':{m:{k:i.get(k)for k in ['sha256','model_sha256','parent_model_sha256','IL_updates','RL_updates','inference','data_sha256']}for m,i in self.identities.items()},'normalized_metrics_caches':self.cache_info,'estimator_notes':{'scene_mean':'Raw mean over selected scenes; fixed panel is almost equally allocated over physical logs, not103288 scenes.','physical_log_equal':'Mean of per-log subgroup means; broad panels span162 TRAIN physical logs.','population_log_reweighted_TRAIN_projection':'Per-log mean weighted by full NAVTRAIN N_log (equivalent scene weights N_log/n_log). Descriptive TRAIN projection, not a canonical full103288 evaluation or NAVTEST. For exposure-conditioned subsets this is a represented-log projection, not an unbiased population estimator.'},'limits':['TRAIN includes old DEV; none of these groups is an independent heldout Test.','Extra3072 are feedback-unseen, while only the exposure0 subset is training-unseen under the recorded ADAPT census.','Exposure-conditioned contrasts are confounded by deliberate sampling of difficult/regressing scenes; bootstrap is descriptive and exploratory.','No new inference seed, policy or adaptive selection recipe is introduced.'],'GPU_process_actions':0,'runtime_or_controller_modified':False}
  campaign=read(R/'owners/CAMPAIGN/STATE.json');nav=campaign['runs']['PPO_FULL']['stages']['10000'].get('NAVTEST',{}).get('summary');registry=read(R/'RUNS.json')['runs'];parent=next(v['parent_PDMS']for v in registry if v['id']=='PPO_FULL')
  if nav:result['separate_native_NAVTEST_context']={'cohort':'NAVTEST','scenes':nav['scenes'],'FULL10k_PDMS':nav['PDMS'],'IL70_PDMS':parent,'FULL_minus_IL_points':nav['PDMS']-parent,'observed_Test_used':True,'evidence':nav.get('evaluation'),'must_not_merge_with_TRAIN_population_projection':True}
  full=groups['all4096']['paired']['PPO_FULL_minus_IL70']['scene_mean'];result['conclusions']=[{'code':'TRAIN_AND_TEST_DIRECTIONS_CAN_CONFLICT','text_zh':'FULL10k在完整TRAIN4096下降不能据此推断泛化退化或停止控制臂；其完整NAVTEST同期提升。','TRAIN4096_FULL_minus_IL_points':full['score']['delta_points'],'metrics_delta_points':{k:q['delta_points']for k,q in full.items()}}]
  result['next_stage_recommendations']=[{'priority':1,'action':'Keep FULL matched control; finish ADAPT10k4096 diagnosis without changing current adaptive feedback.','automatic_training_action':False}]
  if final:
   delta=self.models['PPO_ADAPTIVE']-self.models['PPO_FULL'];u,n,pm=self.aggregate(delta,self.feedback);u2,n2,em=self.aggregate(delta,~self.feedback);assert u.tolist()==u2.tolist();contrast=pm-em;draw=np.random.default_rng(42).integers(0,len(u),size=(2000,len(u)));boot=100*contrast[draw,0].mean(axis=1);interaction={'panel_minus_extra_log_equal_delta_points':float(100*contrast[:,0].mean()),'physical_log_cluster_bootstrap_95pct':[float(x)for x in np.quantile(boot,[.025,.975])]};result['feedback_vs_extra_interaction']=interaction
   pscore=groups['feedback1024']['paired']['PPO_ADAPTIVE_minus_PPO_FULL']['physical_log_equal']['score'];escore=groups['extra3072']['paired']['PPO_ADAPTIVE_minus_PPO_FULL']['physical_log_equal']['score']
   if pscore['delta_points']>0 and escore['cluster_bootstrap_95pct'][1]<0:code='FEEDBACK_SPECIFIC_ADVANTAGE_WITH_EXTRA_DECLINE';action='Consider a separately registered80/10/10 data candidate; retain current results and obtain matched fullTest context before stopping based on TRAIN.'
   elif escore['cluster_bootstrap_95pct'][0]>0:code='EXTRA_TRAIN_DIAGNOSTIC_GAIN';action='Fixed-panel repetition alone does not establish overfit; retain adaptive and prioritize separately measured workflow efficiency.'
   elif pscore['delta_points']>0 and escore['delta_points']<0:code='POSSIBLE_FEEDBACK_SPECIALIZATION_UNCONFIRMED';action='Treat as a hypothesis; prioritize existing beta40 single-variable trial rather than hot-changing adaptive data.'
   else:code='NO_CLEAR_BROAD_ADAPTIVE_ADVANTAGE';action='Do not declare adaptive winner from feedback panel; use same-budget independent extra scenes and fullTest with safety evidence.'
   result['conclusions'].append({'code':code,'panel_ADAPT_minus_FULL':pscore,'extra_ADAPT_minus_FULL':escore,'text_zh':'以上是TRAIN诊断的场景差异，不能替代原生Test或作曝光改善的因果证明。'});result['next_stage_recommendations'].append({'priority':2,'action':action,'automatic_training_action':False})
  else:result['conclusions'].append({'code':'ADAPT_PENDING_NO_ADAPT_SCORE_IMPUTED','text_zh':'当前仅发布IL70与FULL两份完整4096总体；ADAPT尚未完成，没有填造第三模型或partial-scene均值。'})
  path=A/'TRAIN4096_DIAGNOSIS.json';version=A/('TRAIN4096_DIAGNOSIS_FINAL.json'if final else'TRAIN4096_DIAGNOSIS_INTERIM_IL70_PPO_FULL.json');atomic(version,result);atomic(path,result);self.last_signature=signature
  atomic(O/'STATE.json',{'status':'COMPLETE'if final else'WAITING_ADAPT_AFTER_INTERIM','at':now(),'models_complete':complete,'models_pending':result['models_pending'],'result':str(path)});refresh(result['status']);print(json.dumps({'at':now(),'published':result['status'],'models':complete,'evidence':str(path)}),flush=True)

def main():
 O.mkdir(parents=True,exist_ok=True);lock=(O/'owner.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);target=read(D/'OWNER.json');owner={'pid':os.getpid(),'start_ticks':int(Path('/proc/self/stat').read_text().split()[21]),'source_file':str(Path(__file__).resolve()),'source_sha256':sha(__file__),'started_at':now(),'target_owner':target,'finite_models':MODELS,'CPU_only':True};atomic(O/'OWNER.json',owner)
 analyzer=Analysis();pidfd=None;notify=None
 try:
  state=read(D/'STATE.json');analyzer.publish(state)
  if state['status']=='COMPLETE':return
  if not same_process(target):raise RuntimeError('Exact diagnostic owner exited/replaced before COMPLETE')
  pidfd=(os.pidfd_open(target['pid']) if hasattr(os,'pidfd_open') else ctypes.CDLL(None,use_errno=True).syscall(434,int(target['pid']),0))
  if pidfd<0:raise OSError(ctypes.get_errno(),'pidfd_open failed')
  assert same_process(target),'Diagnostic owner PID identity changed'
  libc=ctypes.CDLL(None,use_errno=True);notify=libc.inotify_init1(os.O_NONBLOCK|os.O_CLOEXEC)
  if notify>=0:libc.inotify_add_watch(notify,os.fsencode(D),0x80|0x8)
  else:notify=None
  while True:
   state=read(D/'STATE.json')
   if state['status']not in ['RUNNING','COMPLETE']:raise RuntimeError('Diagnostic owner failed: '+str(state.get('error',state['status'])))
   analyzer.publish(state)
   if state['status']=='COMPLETE':
    if len(analyzer.models)!=3:raise RuntimeError('COMPLETE owner lacks three complete4096 populations')
    return
   if not same_process(target):
    latest=read(D/'STATE.json')
    if latest['status']=='COMPLETE':analyzer.publish(latest);return
    raise RuntimeError('Exact diagnostic owner exited without COMPLETE')
   ready,_,_=select.select([pidfd]+([notify]if notify is not None else[]),[],[],1800)
   if notify in ready:
    try:os.read(notify,65536)
    except BlockingIOError:pass
 except Exception:
  error=traceback.format_exc();failure={'schema':'world_rl_TRAIN4096_diagnosis_v1','status':'FAILED_REQUIRES_REVIEW','at':now(),'error':error,'no_scores_imputed':True,'target_owner':target,'last_complete_interim_artifact':str(A/'TRAIN4096_DIAGNOSIS_INTERIM_IL70_PPO_FULL.json')if(A/'TRAIN4096_DIAGNOSIS_INTERIM_IL70_PPO_FULL.json').exists()else None,'GPU_process_actions':0};atomic(A/'TRAIN4096_DIAGNOSIS.json',failure);atomic(O/'STATE.json',failure);refresh('FAILED_REQUIRES_REVIEW');print(error,file=sys.stderr,flush=True)
 finally:
  if pidfd is not None:os.close(pidfd)
  if notify is not None:os.close(notify)
if __name__=='__main__':main()

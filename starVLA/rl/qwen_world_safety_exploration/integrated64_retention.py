"""Eight per-scene history slots, one transactional window/rank reservoir."""
import copy,math
import torch
from starVLA.rl.simwam_recipe import checkpoint
from starVLA.rl.simwam_recipe.contracts import seed_for
from starVLA.model.modules.qwen_sourceflow.action_codec import ActionCodec
from . import retention_material as M
from .integrated64_recipe import RETENTION_CONFIG as CONFIG

BANK_SCHEMA='world_integrated64_scene_shadow_reservoir_v1'
WITNESS_SCHEMA='world_integrated64_trueB1_G16carrier_witness_v1'
WINDOW_SCHEMA='world_integrated64_eight_scene_history_transaction_v1'


def new_bank(run,rank):
 return dict(schema=BANK_SCHEMA,config=copy.deepcopy(CONFIG),run=run,rank=rank,windows=0,scenes=0,entries=[],
  rng_state=torch.Generator().manual_seed(seed_for('world_verified_RL_retention_bank',CONFIG['bank_seed'],rank)).get_state(),
  verified_admissions=0,reservoir_replacements=0,reservoir_not_retained=0,upgrades=0,history_slots_selected=0,history_activated=0,
  fresh_ODE_queries_local=0,history_ODE_queries_local=0,extra_forward_seconds_local=0.,extra_scoring_seconds_local=0.)


def good(record):return all(record['metrics'][k]==1 for k in M.STRICT)and record['metrics']['score']>=CONFIG['min_score']
def regression(teacher,current):return any(current['metrics'][k]!=1 for k in M.STRICT)or current['metrics']['score']<teacher['metrics']['score']-CONFIG['regression_margin']


@torch.no_grad()
def probe(actor,conditions,token,log,service,version):
 #Reuse the realB1 deployment arithmetic/officialscore only. The new witness
 #envelope records its truthful G16 carrier and new admission clock explicitly.
 old=M.probe(actor,conditions,token,log,service,version)
 value={k:v for k,v in old.items()if k!='witness_sha256'}
 value.update(schema=WITNESS_SCHEMA,carrier_group_size=16,retention_protocol=copy.deepcopy(CONFIG))
 return M.sealed(value,'witness_sha256')


def validate_witness(e,run,allowed,condition_sha,scoring_protocol,rank,*,require_good=False):
 M.verify_seal(e,'witness_sha256')
 if(e.get('schema')!=WITNESS_SCHEMA or e.get('carrier_group_size')!=16 or e.get('retention_protocol')!=CONFIG
    or allowed.get(e.get('token'))!=e.get('log')or e['source_version']['run']!=run or e.get('source_rank')!=rank
    or type(e['source_version']['update'])is not int or e['source_version']['update']<CONFIG['admission_start_update']or e['source_version']['update']%4
    or e['protocol_sha256']!=scoring_protocol or e.get('single_scored_candidate')is not True or e.get('PG_replay')is not False
    or e['condition']!=dict(schema='world_postcot_trainable_memory_observation_v1',token=e['token'],identity=condition_sha,repeats=16)):
  raise ValueError('Integrated TRAIN witness ancestry/carrier/clock changed')
 for k,shape in dict(chain=(1,11,8,3),timesteps=(10,),deltas=(10,),requested=(8,3)).items():
  x=e[k]
  if x.shape!=shape or x.dtype!=torch.float32 or x.requires_grad or not torch.isfinite(x).all():raise ValueError('Detached trueB1 '+k+' changed')
 root=torch.randn((1,8,3),generator=torch.Generator().manual_seed(seed_for('native_qwen_sourceflow_ODE',CONFIG['ODE_seed'],e['token'])),dtype=torch.float32)
 if not torch.equal(root,e['chain'][:,0].cpu())or M.digest(root)!=e['root_sha256']or not torch.equal(ActionCodec.denorm_odo(e['chain'][:,-1])[0].cpu(),e['requested'].cpu()):raise ValueError('B1 actualroot/action codec changed')
 M.validate_feedback(e['feedback'],e['token'],e['log'],e['requested'],scoring_protocol)
 if e['metrics']!=e['feedback']['metrics']or(require_good and not good(e['feedback'])):raise ValueError('Invalid integrated strictquality witness')
 mean=e['nominal_mean']
 if good(e['feedback']):
  if mean is None or mean.shape!=(1,3,8,3)or mean.dtype!=torch.float32 or mean.requires_grad or not torch.isfinite(mean).all():raise ValueError('Detached historicalW3 means required')
 elif mean is not None:raise ValueError('Failed B1 cannot be a teacher')
 if any(not math.isfinite(e['seconds'][k])or e['seconds'][k]<0 for k in ('forward','scoring')):raise ValueError('Invalid actualB1 time accounting')
 return e


def selection(memory,fresh_token,update):
 #Same-window newlyadmitted/upgraded entries must never masquerade as olderRL.
 ids=[i for i,e in enumerate(memory['entries'])if e['token']!=fresh_token and e['source_version']['update']<update]
 rng=torch.Generator();rng.set_state(memory['rng_state'].cpu())
 return(ids[int(torch.randint(len(ids),(1,),generator=rng))]if ids else None),rng


def advance(memory,event):
 if(event['scene_sequence']!=memory['scenes']+1 or event['window_sequence']!=memory['windows']+1
    or event['memory_before_sha256']!=M.digest(memory)or event['source_version']['run']!=memory['run']):raise ValueError('Integrated shadow scene transaction changed')
 index,rng=selection(memory,event['fresh_token'],event['source_version']['update'])
 if index!=event['selection_index']:raise ValueError('Integrated history RNG/selection changed')
 teacher=None if index is None else memory['entries'][index]
 if(teacher is None)!=(event['teacher']is None)or(teacher is not None and M.digest(teacher)!=M.digest(event['teacher'])):raise ValueError('Historical target not the shadow-bank member')
 current=event['history_witness'];fresh=event['fresh_witness']
 if (teacher is None)!=(current is None):raise ValueError('Missing currentB1 for selectedhistory')
 enabled=teacher is not None and regression(teacher,current['feedback'])
 if enabled!=event['enabled']:raise ValueError('Verified history gate changed')
 m=dict(memory);entries=list(memory['entries']);m['entries']=entries
 if teacher is not None:
  m['history_slots_selected']+=1;m['history_activated']+=int(enabled)
  if good(current['feedback'])and current['metrics']['score']>teacher['metrics']['score']+CONFIG['upgrade_margin']:entries[index]=current;m['upgrades']+=1
 if fresh is not None and good(fresh['feedback']):
  existing=next((i for i,e in enumerate(entries)if e['token']==fresh['token']),None)
  if existing is not None:
   if fresh['metrics']['score']>entries[existing]['metrics']['score']+CONFIG['upgrade_margin']:entries[existing]=fresh;m['upgrades']+=1
  else:
   m['verified_admissions']+=1
   if len(entries)<CONFIG['capacity']:entries.append(fresh)
   else:
    j=int(torch.randint(m['verified_admissions'],(1,),generator=rng))
    if j<CONFIG['capacity']:entries[j]=fresh;m['reservoir_replacements']+=1
    else:m['reservoir_not_retained']+=1
 m['scenes']+=1;m['rng_state']=rng.get_state();m['fresh_ODE_queries_local']+=int(fresh is not None)
 m['history_ODE_queries_local']+=int(event['history_scored_now'])
 for w,scored in ((fresh,fresh is not None),(current,event['history_scored_now'])):
  if scored:m['extra_forward_seconds_local']+=w['seconds']['forward'];m['extra_scoring_seconds_local']+=w['seconds']['scoring']
 return m


@torch.no_grad()
def attach_window(memory,window,actor,conditions,service):
 before=memory;shadow=memory;events=[];history_cache={};version=window['behavior_version'];audit=actor._probability_audit
 try:
  for slot,sample in enumerate(window['samples']):
   index,_=selection(shadow,sample['token'],version['update']);fresh=current=None;scored=False
   if version['update']>=CONFIG['admission_start_update']:fresh=probe(actor,conditions,sample['token'],sample['log'],service,version)
   teacher=None if index is None else shadow['entries'][index]
   if teacher is not None:
    token=teacher['token'];scored=token not in history_cache
    if scored:history_cache[token]=probe(actor,conditions,token,teacher['log'],service,version)
    current=history_cache[token]
   event=dict(schema=WINDOW_SCHEMA,slot=slot,rank=memory['rank'],window_sequence=memory['windows']+1,scene_sequence=shadow['scenes']+1,
    fresh_token=sample['token'],source_version=copy.deepcopy(version),selection_index=index,teacher=teacher,fresh_witness=fresh,history_witness=current,
    history_scored_now=scored,enabled=teacher is not None and regression(teacher,current['feedback']),memory_before=shadow,memory_before_sha256=M.digest(shadow))
   shadow=advance(shadow,event);events.append(event);sample['verified_retention']=event
 finally:actor._probability_audit=audit
 after=dict(shadow,windows=memory['windows']+1)
 material=dict(schema=WINDOW_SCHEMA,config=copy.deepcopy(CONFIG),window_sequence=memory['windows']+1,rank=memory['rank'],
  source_version=copy.deepcopy(version),memory_before=before,events=events,memory_after_sha256=M.digest(after))
 window['integrated_retention']=M.sealed(material,'material_sha256')
 return window


def replay(material):
 M.verify_seal(material,'material_sha256')
 if material.get('schema')!=WINDOW_SCHEMA or material.get('config')!=CONFIG or len(material['events'])!=8:raise ValueError('Integrated8scene bank transaction protocol changed')
 shadow=material['memory_before'];seen={}
 for slot,event in enumerate(material['events']):
  if event['slot']!=slot or event['rank']!=material['rank']or event['source_version']!=material['source_version']or M.digest(event['memory_before'])!=M.digest(shadow):raise ValueError('Scene-shadow order/material changed')
  current=event['history_witness']
  if current is not None:
   token=current['token'];first=token not in seen
   if first!=event['history_scored_now']or(not first and M.digest(current)!=seen[token]):raise ValueError('Historical B1 reuse/query count changed')
   seen[token]=M.digest(current)
  elif event['history_scored_now']:raise ValueError('Unscored historical query claimed')
  shadow=advance(shadow,event)
 after=dict(shadow,windows=shadow['windows']+1)
 if M.digest(after)!=material['memory_after_sha256']:raise ValueError('Integrated finalbank commit digest changed')
 return checkpoint.cpu_tree(after)


def validate_memory(memory,run,rank,windows,allowed,condition_sha,scoring_protocol,maximum_update):
 if(memory.get('schema')!=BANK_SCHEMA or memory.get('config')!=CONFIG or memory.get('run')!=run or memory.get('rank')!=rank
    or memory.get('windows')!=windows or memory.get('scenes')!=8*windows or not 0<=len(memory['entries'])<=CONFIG['capacity']
    or len({e['token']for e in memory['entries']})!=len(memory['entries'])or memory['verified_admissions']<len(memory['entries'])):raise ValueError('Integrated nativebank window/scene accounting changed')
 g=torch.Generator();g.set_state(memory['rng_state'].cpu())
 for e in memory['entries']:
  validate_witness(e,run,allowed,condition_sha,scoring_protocol,rank,require_good=True)
  if e['source_version']['update']>maximum_update:raise ValueError('Future teacher in integrated bank')
 for k in ('verified_admissions','reservoir_replacements','reservoir_not_retained','upgrades','history_slots_selected','history_activated','fresh_ODE_queries_local','history_ODE_queries_local'):
  if type(memory[k])is not int or memory[k]<0:raise ValueError('Invalid integrated counter '+k)
 for k in ('extra_forward_seconds_local','extra_scoring_seconds_local'):
  if not math.isfinite(memory[k])or memory[k]<0:raise ValueError('Invalid integrated actualtime '+k)
 if(memory['fresh_ODE_queries_local']>memory['scenes']or memory['history_ODE_queries_local']>memory['history_slots_selected']or memory['history_slots_selected']>memory['scenes']or memory['history_activated']>memory['history_slots_selected']):raise ValueError('Integrated extraquery/slot counters inconsistent')
 return memory
